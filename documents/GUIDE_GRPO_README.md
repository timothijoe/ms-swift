# Guide-GRPO Design Notes for ms-swift

This note describes the principle of Guide-GRPO and a staged implementation plan for adding it to `ms-swift`.

## 1. Motivation

Standard GRPO trains from a group of sampled completions for the same prompt. For a prompt `q`, the old policy samples `K` completions:

```text
o_1, ..., o_K ~ pi_old(. | q)
```

Each completion receives a reward:

```text
r_i = R(q, o_i)
```

GRPO then computes a group-relative advantage:

```text
A_i = (r_i - mean_q) / (std_q + eps)
```

This works well when the group contains both good and bad completions. Correct completions receive positive advantage, incorrect completions receive negative advantage, and the model learns which trajectories to increase or suppress.

The failure case is when all `K` completions are wrong:

```text
r_1 = r_2 = ... = r_K = 0
```

Then the group has nearly zero reward variance, and the resulting advantages are close to zero. The prompt may be highly valuable for learning, but standard GRPO gets little or no useful gradient from it.

Guide-GRPO addresses exactly this case: when ordinary rollouts all fail, it temporarily adds natural-language guidance to help the model discover a successful trajectory.

## 2. Core Idea

Let the ordinary prompt be:

```text
x_q
```

Let the guided prompt be:

```text
x_tilde_q = x_q + h_q
```

where `h_q` is a problem-specific hint or guidance string. The hint should provide conceptual direction without directly revealing the answer.

Standard GRPO samples:

```text
o_i ~ pi_old(. | x_q)
```

If all ordinary samples fail, Guide-GRPO additionally samples:

```text
o_tilde_i ~ pi_old(. | x_tilde_q)
```

The guided samples and ordinary samples are then compared within the same original prompt group. If a guided completion is correct, it receives positive advantage relative to the failed ordinary completions.

Intuitively:

```text
ordinary rollout fails to find a solution
hint helps the same policy find one
GRPO turns that discovered trajectory into learning signal
```

The important constraint is that the final model should not require hints at test time. Guidance is used for exploration during training, not as an inference-time dependency.

## 3. Importance Weight Comparison

### Standard GRPO

In ordinary GRPO, the completion is sampled from the old policy under the same prompt that will be optimized:

```text
o ~ pi_old(. | q)
```

The token-level importance ratio is:

```text
rho_GRPO_t =
    pi_theta(o_t | q, o_<t)
    /
    pi_old(o_t | q, o_<t)
```

Both numerator and denominator are conditioned on the same prompt `q`.

This ratio answers:

```text
How much more or less likely is the current policy to produce this token
than the old policy under the same prompt?
```

### Guide-GRPO

For a guided rollout, the completion is sampled under the guided prompt:

```text
o_tilde ~ pi_old(. | q, h)
```

But the desired training objective is still the no-hint policy:

```text
pi_theta(. | q)
```

So the guided token-level importance ratio should be:

```text
rho_Guide_t =
    pi_theta(o_tilde_t | q, o_tilde_<t)
    /
    pi_old(o_tilde_t | q, h, o_tilde_<t)
```

The numerator is conditioned on the plain prompt. The denominator is conditioned on the guided prompt.

This ratio answers:

```text
This completion was discovered with a hint.
How much should the no-hint policy increase its probability?
```

That cross-context ratio is the mathematical center of Guide-GRPO.

## 4. Objective Intuition

For ordinary rollouts, Guide-GRPO uses the usual GRPO update.

For guided rollouts, the update should use:

```text
rho_Guide * A_Guide
```

where:

```text
rho_Guide = pi_theta(o_tilde | q) / pi_old(o_tilde | q, h)
```

This lets the model learn from trajectories discovered with hints while optimizing the no-hint model.

Selective guidance is important. Guidance should generally trigger only when all ordinary rollouts fail:

```text
max_i R(q, o_i) == 0
```

If ordinary rollouts already contain a correct answer, GRPO already has useful contrastive signal. Adding hints everywhere can teach the model to rely on guided contexts and can reduce independent reasoning.

## 5. Existing ms-swift GRPO Flow

The main file is:

```text
swift/trainers/rlhf_trainer/grpo_trainer.py
```

The core method is:

```text
GRPOTrainer._generate_and_score_completions
```

The current flow is approximately:

```text
inputs
  -> _generate_completions
  -> _score_completions
  -> _dynamic_sampling, optional
  -> _prepare_batch_inputs
  -> _compute_advantages
  -> compute_loss
```

Guide-GRPO should be inserted after ordinary generation and scoring, but before batch encoding and advantage computation:

```text
inputs
  -> ordinary _generate_completions
  -> ordinary _score_completions
  -> detect failed prompt groups
  -> build guided inputs for failed groups
  -> guided _generate_completions
  -> guided _score_completions
  -> merge ordinary and guided samples
  -> _prepare_batch_inputs
  -> _compute_advantages
  -> compute_loss
```

## 6. Proposed Arguments

Add these fields to `GRPOArgumentsMixin` in:

```text
swift/trainers/arguments.py
```

Suggested arguments:

```python
guide_grpo: bool = False
guide_field: str = "guidance"
guide_trigger: Literal["all_incorrect", "mostly_incorrect", "always"] = "all_incorrect"
guide_incorrect_threshold: float = 0.25
guide_num_generations: Optional[int] = None
guide_train_on_plain_prompt: bool = True
guide_suffix_template: str = (
    "\n\nA hint to the problem is provided below:\n"
    "[HINT_START]\n{guidance}\n[HINT_END]\n"
    "Consider the hint but start your solution from scratch and do not directly reference the hint."
)
```

Recommended defaults:

```text
guide_grpo = False
guide_trigger = all_incorrect
guide_num_generations = None
guide_train_on_plain_prompt = True
```

`guide_num_generations = None` should mean "use the same number as `num_generations`".

## 7. Dataset Format

Each training sample should include a guidance field:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Solve the math problem..."
    }
  ],
  "solution": "42",
  "guidance": "1. Think about...\n2. Notice that...\n3. ..."
}
```

The reward function can continue to use the existing fields such as `solution`. It does not need to know whether a sample was guided.

## 8. Implementation Plan

### Step 1: MVP, Guided Rollout Data Flow

Goal: verify that guided rollouts can be generated and scored without changing the loss.

Implement:

1. Add Guide-GRPO arguments, default disabled.
2. In `_generate_and_score_completions`, preserve the original prompt inputs before ordinary generation.
3. Run ordinary generation and scoring as usual.
4. Detect failed prompt groups.
5. For each failed group, construct a guided input by appending `guide_suffix_template.format(guidance=...)` to the last user message.
6. Run `_generate_completions` and `_score_completions` on guided inputs.
7. Merge ordinary and guided samples.
8. Add basic metrics:

```text
guide/trigger_rate
guide/num_triggered_prompts
guide/num_guided_rollouts
guide/plain_reward_mean
guide/guided_reward_mean
guide/guided_success_rate
```

For MVP, it is acceptable to train guided samples with the guided prompt still present. This does not yet match the paper exactly, but it validates the generation and reward path.

Expected patch size: roughly 150-300 lines.

### Step 2: Proper Guide-GRPO Off-Policy Training

Goal: make the guided rollout optimize the no-hint policy.

For guided samples:

Generation prompt:

```text
q + hint -> guided completion
```

Training prompt:

```text
q -> same guided completion
```

Implementation detail:

1. When constructing guided inputs, store the original plain messages:

```python
data["_guide_original_messages"] = deepcopy(original_messages)
data["_guide_is_guided"] = True
```

2. After guided generation, but before `_prepare_batch_inputs`, replace the guided prompt with the original prompt while keeping the guided assistant response:

```python
guided_response = data["messages"][-1]
plain_messages = deepcopy(data["_guide_original_messages"])
plain_messages.append(guided_response)
data["messages"] = plain_messages
```

3. Preserve rollout logprobs from guided generation:

```text
rollout_logprobs = log pi_old(completion | q, h)
```

4. Let `_prepare_batch_inputs` compute model logprobs under the plain prompt:

```text
log pi_theta(completion | q)
```

5. Reuse existing rollout importance sampling support:

```text
rollout_per_token_logps
rollout_importance_sampling_mode
_apply_rollout_importance_sampling
```

Recommended command option for first tests:

```bash
--log_rollout_offpolicy_metrics true
```

Then enable correction:

```bash
--rollout_importance_sampling_mode sequence_truncate
--rollout_importance_sampling_threshold 2.0
```

This should approximate:

```text
pi_theta(o | q) / pi_old(o | q, h)
```

### Step 3: Paper-Style Enhancements

Goal: add stability and ablation options from the Guide-GRPO paper.

Potential enhancements:

1. Trigger ablations:

```text
all_incorrect
mostly_incorrect
always
```

2. Optional guided-only unclipped objective:

```python
guide_disable_clip: bool = False
```

If enabled, ordinary samples use standard clipped GRPO, while guided samples use an unclipped objective:

```text
-rho * advantage
```

This requires carrying a `guide_mask` into the loss function.

3. Optional adaptive policy reshaping:

```text
f(w_i) = 1 / (P90(w) + w_i)
```

This is more invasive and should be implemented only after the basic off-policy version is stable.

4. More detailed metrics:

```text
guide/is_weight_mean
guide/is_clipped_frac
guide/offpolicy_kl
guide/plain_vs_guided_reward_gap
```

## 9. Advantage Grouping Risk

This is the most important implementation risk.

Standard GRPO assumes each prompt has exactly `num_generations` completions and often reshapes rewards like:

```python
grouped_rewards = rewards.view(-1, num_generations)
```

Guide-GRPO breaks this assumption because some prompts have only ordinary samples:

```text
K samples
```

while failed prompts have ordinary plus guided samples:

```text
K + M samples
```

Therefore, fixed reshape can mis-group rewards and produce wrong advantages.

Recommended solution:

Use request-aware grouping already present in `GRPOTrainer._compute_advantages`.

Each sample should carry:

```python
prompt_id  # same for ordinary and guided samples from the same original prompt
request_id # unique for each rollout
```

Then set or reuse:

```python
self.dynamic_num_samples = True
```

so advantage computation groups by `prompt_id` instead of assuming fixed `num_generations`.

This should be handled carefully because it affects batching, splitting, logging, and loss computation.

## 10. Logprob Alignment Risk

The second major risk is logprob alignment.

For guided samples, the denominator logprobs must correspond to:

```text
log pi_old(completion | q, h)
```

The numerator/training logprobs must correspond to:

```text
log pi_theta(completion | q)
```

The tokenization of the completion must remain aligned with `completion_mask`. If the plain prompt causes the completion to tokenize differently, or if the assistant response boundaries change, rollout logprobs and training logprobs may no longer align.

Before large training runs, test:

```text
len(rollout_logprobs) == number of completion tokens
```

for every guided sample.

If this fails, skip IS correction for that batch and log a warning.

## 11. Suggested Test Plan

Minimum tests:

1. `guide_grpo=False` produces identical behavior to existing GRPO.
2. `guide_grpo=True` with no `guidance` field either skips guidance or raises a clear error.
3. All-incorrect groups trigger guided generation.
4. Non-failed groups do not trigger guided generation under `all_incorrect`.
5. Guided prompt actually contains the hint.
6. With `guide_train_on_plain_prompt=True`, guided samples are trained with the original prompt and the guided completion.
7. Advantage groups are formed by original prompt id, not by fixed `num_generations`.
8. `rollout_logprobs` length matches completion token count.
9. Loss can run forward and backward on a tiny batch.
10. Metrics show nonzero guided rollout counts when expected.

## 12. Minimal Acceptance Criteria

The implementation should satisfy:

```text
--guide_grpo false
```

keeps original GRPO unchanged.

With:

```text
--guide_grpo true
--guide_trigger all_incorrect
```

the trainer:

1. samples ordinary rollouts;
2. identifies prompt groups where all ordinary rollouts fail;
3. appends guidance only for those groups;
4. samples guided rollouts;
5. trains guided completions against the plain prompt when `guide_train_on_plain_prompt=True`;
6. computes advantages by original prompt group;
7. optionally applies rollout importance sampling correction;
8. logs guide-specific metrics.

## 13. Recommended Development Order

Implement in this order:

1. Add arguments and no-op plumbing.
2. Add guide prompt builder.
3. Add failure-group detection.
4. Add guided generation and reward scoring.
5. Add metrics.
6. Add dynamic prompt/request grouping.
7. Add plain-prompt training for guided completions.
8. Validate rollout logprob alignment.
9. Enable existing rollout importance sampling.
10. Consider guided-specific loss modifications only after the above is stable.

The key design principle is:

```text
Use guidance for exploration, but optimize the no-guidance policy.
```

