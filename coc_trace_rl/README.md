# CoC Trace Driving GRPO

This directory contains the first implementation of schema-driven CoC Trace guided GRPO for driving decisions. It is intentionally isolated from the generic `ms-swift` trainer: the original `swift/trainers/rlhf_trainer/grpo_trainer.py` and the global trainer factory are not modified.

## Research Objective

Ordinary GRPO can fail to learn from driving prompts when all sampled completions have weak reasoning. This experiment uses a reference CoC Trace only during training-time exploration:

1. Generate ordinary rollouts from the original driving prompt.
2. Score each ordinary rollout's `<think>` content against a persisted reference CoC schema.
3. When the group reasoning score is low, probabilistically create several guided prompts.
4. Generate additional guided rollouts.
5. Restore the original, unguided prompt before policy optimization.

The trained policy is therefore optimized for the original prompt, not for a prompt containing a reference trace.

## Decisions Made During Design

- `think` is the reference CoC Trace.
- A standalone `coc_trace_reward` measures reasoning-schema coverage. It is used for triggering and metrics, not as an advantage reward.
- A prompt group triggers only when all conditions hold:

  ```text
  coc_trace_enabled == true
  max(coc_trace_reward for ordinary rollouts) < coc_trace_reward_threshold
  deterministic_random(prompt_id, global_step, seed) < coc_trace_probability
  ```

- Defaults are `coc_trace_reward_threshold=0.8` and `coc_trace_probability=0.3`.
- Triggering does not depend on total task reward. A low total reward does not necessarily mean poor reasoning.
- Each triggered group receives diverse guidance variants rather than repeated copies of one prompt.
- Guidance never contains final lateral or longitudinal decisions. It supplies task classification and reasoning checks only.
- CoC Trace is enabled only when at least three guided rollouts are requested, one for each guidance level.

## Guidance Variants

The reference trace is converted into a compact schema:

```text
task_category
task_subcategory
evidence: relative position, visibility, motion trend, timing, safety margin
risk_causes
constraints
```

The ordinary rollout traces are parsed into the same schema. Missing schema fields become guidance gaps.

| Level | Purpose | Prompt content |
| --- | --- | --- |
| 1 | Task locator | Task category and subcategory |
| 2 | Reasoning frame | Task type plus missing evidence dimensions and causal checks |
| 3 | Selective checklist | Three to five schema-relevant reasoning checks |

For example, a blind-spot scenario can prompt the model to verify relative position, visibility limitations, and risk causality without telling it to brake or change lanes.

## Files

```text
coc_trace_rl/
  configs/driving_video_manifest.json       Dataset manifest
  scripts/build_coc_trace_schema.py         Offline JSONL schema preparation
  scripts/run_coc_trace_driving_grpo.sh      Training command
  src/coc_trace_schema.py                   Schema parsing, score, guidance, trigger
  src/driving_dataset.py                    JSONL and driving-row normalization
  src/driving_rewards.py                    Decision accuracy and CoC rewards
  src/driving_formal_rm.py                  Semantic Formal RM plugin
  src/coc_trace_args.py                     Experiment-specific arguments
  src/coc_trace_grpo_trainer.py             GRPO subclass with guided rollouts
  src/coc_trace_main.py                     Manifest registration and trainer launch
```

## Data Preparation

The training JSONL must include a `think` field or an assistant response containing `<think>...</think>`. Before training, persist a reference schema for every row:

```bash
PYTHONPATH=. python coc_trace_rl/scripts/build_coc_trace_schema.py \
  path/to/driving.jsonl \
  my_data/driving_video_style_32_template_v5_with_assistant_coc_schema.jsonl
```

Point `configs/driving_video_manifest.json` at the prepared file. The repository does not copy the reference repository's sample data; dataset paths are deliberately external to the implementation.

## Rewards

The training reward is task-focused:

```text
0.5 * driving_decision_accuracy + 0.5 * driving_formal_rm
```

`coc_trace_score` has weight `0.0` in the GRPO objective. It compares the candidate `<think>` to the persisted `coc_trace_schema` and is used by `CocTraceGRPOTrainer` to decide whether to create guided samples.

`driving_formal_rm` uses `DRIVING_RM_MODEL` as a generative judge. It receives the reference driving schema, target decision, and candidate completion, then emits a score in `[0, 1]`.

## Run

```bash
cd /workspace/docker_mapping/swift_proj/zt-ms-swift
DRIVING_RM_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
bash coc_trace_rl/scripts/run_coc_trace_driving_grpo.sh
```

Useful overrides:

```bash
COC_TRACE_REWARD_THRESHOLD=0.7 \
COC_TRACE_PROBABILITY=0.5 \
COC_TRACE_NUM_GENERATIONS=3 \
DRIVING_NUM_GENERATIONS=2 \
DRIVING_RM_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
bash coc_trace_rl/scripts/run_coc_trace_driving_grpo.sh
```

## Current Implementation Notes

- `CocTraceGRPOTrainer` subclasses `GRPOTrainer`; generic trainer source files are untouched.
- When CoC is disabled, it delegates directly to the parent GRPO implementation.
- When enabled, it generates ordinary samples, finds low-CoC groups, creates Level 1/2/3 guided inputs, generates those samples, restores each guided completion to the plain prompt, and then invokes the parent batch encoding and advantage machinery.
- Ordinary and guided samples use the same `prompt_id` and distinct `request_id` values.
- The implementation records `coc_trace/guided_rollouts` and `coc_trace/triggered_prompts` metrics.

## Validation Status

Focused tests cover schema extraction, safe guidance, deterministic trigger logic, JSONL preparation, CoC scoring, and decision scoring:

```text
5 passed: tests/coc_trace_rl
```

The full upstream GRPO test file requires a GPU BF16 environment, DeepSpeed, and vLLM settings. It cannot pass in the current CPU-only verification environment. Use a remote GPU development environment for real rollout and training debugging.
