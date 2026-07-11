# CoC Trace Driving GRPO Design

## Goal

Add a minimal, self-contained CoC Trace RL training entrypoint to `zt-ms-swift` for the driving GRPO workload. The implementation uses a reference CoC trace only to guide exploration. The policy is optimized and evaluated with the original, unguided driving prompt.

The starting command is the equivalent of:

```bash
DRIVING_RM_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
bash coc_trace_rl/scripts/run_coc_trace_driving_grpo.sh
```

## Scope

### Included

- A new `coc_trace_rl/` package in `zt-ms-swift`.
- A `CocTraceGRPOTrainer` that subclasses the existing `GRPOTrainer` without modifying `swift/trainers/rlhf_trainer/grpo_trainer.py`.
- Manifest-based driving dataset registration for the video-style dataset.
- Driving decision reward, Formal RM, and a new schema-based CoC Trace score.
- Conditional, multi-variant guided rollouts.
- Plain-prompt training and guided-rollout importance-sampling alignment.
- Focused unit tests and a tiny end-to-end smoke test.

### Excluded

- Changes to the existing generic GRPO trainer, trainer factory, or unrelated `ms-swift` features.
- Copying historical manifests, scripts, sample data, path-remapping compatibility code, debug programs, mixed/no-think modes, and unused `inject_gt_*` arguments from `ref-zt-ms-swift`.
- A reward for copying the reference trace verbatim.
- Guidance that contains final lateral or longitudinal decisions.

## Source Boundary

The reference repository is used as a behavioral source, not as a directory to copy wholesale. Only the responsibilities needed by its semantic-RM driving command are retained:

- manifest registration and driving-row normalization;
- driving decision accuracy reward;
- Formal RM construction and scoring utilities;
- video-style manifest shape and shell invocation conventions.

The new implementation is reorganized around the CoC Trace workflow and has one canonical launch script and manifest.

## Components

```text
coc_trace_rl/
  configs/driving_video_manifest.json
  scripts/build_coc_trace_schema.py
  scripts/run_coc_trace_driving_grpo.sh
  src/coc_trace_args.py
  src/coc_trace_main.py
  src/coc_trace_grpo_trainer.py
  src/driving_dataset.py
  src/driving_rewards.py
  src/driving_formal_rm.py
  src/coc_trace_schema.py
```

`coc_trace_main.py` performs the normal manifest and model setup, then constructs `CocTraceGRPOTrainer` directly. This avoids changing `TrainerFactory` and keeps the special trainer local to this experiment.

`CocTraceGRPOTrainer` inherits all ordinary GRPO behavior. It adds only the guided rollout path, its prompt restoration, variable-size grouping metadata, and CoC-specific metrics.

## Data Contract

Each row contains the ordinary driving prompt, the final decision answer, and a reference `think` string. Dataset preprocessing retains the original prompt as `messages`, normalizes the answer for the decision reward, and derives a `coc_trace_schema` from `think`.

The schema contains no target lateral or longitudinal decision. Its fields are:

```text
task_category
task_subcategory
evidence[]: object, relative_position, state_or_trend, temporal_context
risk_causes[]
constraints[]
```

The reference schema is generated during dataset preparation and persisted with the row. At training time, only the candidate `<think>` is parsed. This keeps the training path deterministic and avoids repeatedly parsing the reference text.

`build_coc_trace_schema.py` is an idempotent offline preparation command. It reads a JSONL dataset, adds or validates `coc_trace_schema` from `think`, and writes a new JSONL file. Training rejects rows without a valid persisted schema instead of generating reference schemas inside rollout workers.

## Rewards and Triggering

The policy reward remains task-focused:

```text
task_reward = 0.5 * driving_decision_accuracy + 0.5 * formal_rm
```

`coc_trace_reward` is a separate schema-coverage score. It compares the candidate trace schema with the reference trace schema across task type, evidence, causal risk, and constraints. It is used for triggering and metrics only; it is not added to the GRPO advantage.

For each original prompt group, CoC guidance triggers only when all conditions hold:

```text
coc_trace_enabled == true
max(coc_trace_reward over ordinary rollouts) < coc_trace_reward_threshold
deterministic_random(prompt_id, global_step, seed) < coc_trace_probability
```

Defaults are `coc_trace_enabled=false`, `coc_trace_reward_threshold=0.8`, and `coc_trace_probability=0.3`. No task-reward threshold is used.

The deterministic random draw is shared by all ranks for a prompt group, so distributed training cannot disagree about whether to create guided samples.

## Multi-Variant Guidance

When a group triggers, the trainer derives several guidance contexts from the same reference schema and the schema gaps observed in ordinary rollout traces. The variants are intentionally independent of the numeric CoC score:

1. **Level 1: task locator**. Names the task category and subcategory.
2. **Level 2: reasoning frame**. Names the task type plus uncovered evidence dimensions or causal checks.
3. **Level 3: selective checklist**. Gives three to five schema-relevant checks and asks the model to choose the necessary checks before reasoning.

All variants omit final driving actions. They instruct the model to verify the supplied checks against the video and independently derive the response.

`coc_trace_num_generations` is the total number of guided samples. When CoC Trace is enabled, it must be at least three; the first three samples use one variant per level. Additional samples are allocated with configurable `coc_guidance_level_weights`. Every guided sample records its `guidance_level` and the schema gaps that produced it.

If Level 3 emits a machine-readable selection prefix, the trainer removes that prefix before policy optimization and trims the matching rollout-logprob prefix. Only the resulting reasoning and decision completion is trained on the plain prompt.

## Rollout and Optimization Flow

```text
plain prompt
  -> generate K ordinary rollouts
  -> task reward and CoC Trace score
  -> test CoC trigger
  -> build Level 1/2/3 guidance contexts for triggered prompts
  -> generate M guided rollouts under guided prompts
  -> task reward and CoC Trace score for guided rollouts
  -> restore each guided completion to its original plain prompt
  -> preserve guided rollout logprobs as the old-policy denominator
  -> encode all samples
  -> group advantages by original prompt_id
  -> optimize the plain-prompt policy
```

Ordinary and guided samples share an original `prompt_id`; every rollout has a unique `request_id`. The subclass enables the parent trainer's request-aware, variable-size advantage path. This prevents incorrect fixed-width reshaping when a triggered prompt has `K + M` samples while an untriggered prompt has `K`.

For guided samples, the denominator is the log probability under the guided generation prompt, while the numerator is calculated under the original plain prompt. The existing rollout importance-sampling machinery is used after verifying that the retained completion token count exactly matches the retained rollout logprob count.

## Failures and Observability

- Missing `think` or `coc_trace_schema`: skip guidance for that row and count the skip reason.
- Schema parse failure: assign no CoC score for triggering and skip the guided path for that group.
- Prompt or completion token/logprob mismatch: skip importance-sampling correction for the affected guided batch, log a warning and mismatch metric, but do not silently misalign tokens.
- Empty or invalid guidance variant: omit that variant rather than creating a malformed rollout.

Metrics include trigger rate, triggered prompt count, guided rollout count by level, plain/guided task reward means, plain/guided CoC score means, schema coverage gaps, logprob-alignment failures, importance-sampling weight statistics, and guidance-level reward outcomes.

## Tests and Acceptance Criteria

Unit tests cover:

1. reference and candidate trace schema extraction;
2. schema comparison and guidance-variant construction;
3. all three trigger conditions, including deterministic probability;
4. guidance variants never expose final action labels;
5. Level 3 selection-prefix removal and logprob trimming;
6. ordinary and guided samples sharing one `prompt_id` with distinct `request_id`s;
7. `coc_trace_enabled=false` preserving the parent GRPO path.

A tiny CPU/mock rollout integration test covers trigger, merge, request-aware advantage grouping, batch preparation, and one forward/backward loss call. A GPU smoke command validates the real driving entrypoint with `DRIVING_RM_MODEL` and writes CoC metrics.

The feature is accepted when it can run the semantic driving command, generate all three guided variants for eligible groups, train guided completions against plain prompts with aligned logprobs, and leave existing GRPO behavior unchanged when disabled.
