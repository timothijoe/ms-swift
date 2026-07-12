# CoC Trace Driving GRPO

> Current short conclusions and the cumulative project log are maintained in `../timothi_record/README.md` and `../timothi_record/log.md`.

Schema-driven Chain-of-Cognition Trace guided GRPO for autonomous driving decision-making.

## Research Objective

Ordinary GRPO can fail to learn from driving prompts when all sampled completions have weak reasoning. This experiment uses a reference CoC Trace **only during training-time exploration**:

1. Generate ordinary rollouts from the original driving prompt.
2. Score each ordinary rollout's ` thinking` content against a persisted reference CoC schema.
3. When the group reasoning score is low, probabilistically create several guided prompts.
4. Generate additional guided rollouts (or reuse existing completions with guidance-injected prompts).
5. Replace the worst ordinary rollouts with guided rollouts, then compute advantages.

The trained policy is optimized for the **original unguided prompt** — CoC guidance is an exploration mechanism only.

## Design Decisions

- ` think` is the reference CoC Trace.
- `coc_trace_score` reward measures reasoning-schema coverage. It is used **only for triggering** (weight = 0.0 in advantage computation).
- A prompt group triggers only when all conditions hold:
  ```
  coc_trace_enabled == true
  max(coc_trace_score for ordinary rollouts) < coc_trace_reward_threshold
  deterministic_random(prompt_id, global_step, seed) < coc_trace_probability
  ```
- Defaults: `coc_trace_reward_threshold=0.8`, `coc_trace_probability=0.3`.
- Each triggered group receives diverse guidance variants (Level 1/2/3) rather than repeated copies of one prompt.
- Guidance never contains final lateral or longitudinal decisions. It supplies task classification and reasoning checks only.
- When triggered, the worst ordinary rollouts (by `coc_trace_score`) are **replaced** by guided rollouts, keeping total sample count = `num_generations` so the grouped advantage computation works correctly.

## Guidance Variants

| Level | Purpose | Prompt content |
|-------|---------|----------------|
| 1 | Task locator | Task category and subcategory |
| 2 | Reasoning frame | Task type plus missing evidence dimensions and causal checks |
| 3 | Selective checklist | Three to five schema-relevant reasoning checks |

## Files

```
coc_trace_rl/
  __init__.py                               Package init
  README.md                                 This file
  configs/
    driving_video_manifest.json             Dataset manifest
  scripts/
    build_coc_trace_schema.py               Offline JSONL schema preparation
    run_coc_trace_baseline.sh               Training — CoC Trace disabled
    run_coc_trace_guided.sh                 Training — CoC Trace enabled
  src/
    __init__.py                             Source init
    coc_trace_args.py                       Experiment-specific arguments (CocTraceDrivingArguments)
    coc_trace_main.py                       Manifest registration and trainer launch
    coc_trace_grpo_trainer.py               GRPO subclass with guided rollout replacement
    coc_trace_schema.py                     Schema parsing, scoring, guidance, trigger logic
    driving_dataset.py                      JSONL and driving-row normalization
    driving_rewards.py                      CocTraceScoreReward, DrivingDecisionAccuracyReward
    driving_formal_rm.py                    Generative RM plugin for semantic driving scoring
```

## Quick Start

### 1. Environment

```bash
conda activate sw_312_env
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
```

### 2. Data Preparation

The training JSONL must include a ` think` field or an assistant response containing ` thinking... response`. Before training, persist a reference schema for every row:

```bash
python coc_trace_rl/scripts/build_coc_trace_schema.py \
  path/to/driving.jsonl \
  my_data/driving_video_style_32_template_v5_with_assistant_coc_schema.jsonl
```

Point `configs/driving_video_manifest.json` at the prepared file.

> A minimal 4-sample test dataset is already prepared at `my_data/driving_video_style_32_template_v5_with_assistant_coc_schema.jsonl`.

### 3. Run Training

**Baseline** (CoC Trace disabled):

```bash
DRIVING_RM_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
bash coc_trace_rl/scripts/run_coc_trace_baseline.sh
```

**Guided** (CoC Trace enabled):

```bash
DRIVING_RM_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
bash coc_trace_rl/scripts/run_coc_trace_guided.sh
```

**Useful overrides:**

```bash
COC_TRACE_REWARD_THRESHOLD=0.7 \
COC_TRACE_PROBABILITY=0.5 \
COC_TRACE_NUM_GENERATIONS=3 \
DRIVING_NUM_GENERATIONS=4 \
DRIVING_RM_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
bash coc_trace_rl/scripts/run_coc_trace_guided.sh
```

### 4. VSCode Debugging

Open the project in VSCode and select from the **Run and Debug** dropdown:

| Configuration | Description |
|---------------|-------------|
| **CoC Trace Baseline** | Debug baseline (CoC disabled) |
| **CoC Trace Guided** | Debug guided mode (prob=0.3) |
| **CoC Trace Guided (Force Trigger)** | Debug guided mode with forced trigger (prob=1.0) |
| **CoC Trace Schema — Unit Tests** | Run unit tests |
| **CoC Trace — Prepare JSONL Schema** | Run data preprocessing |

## Rewards

| Reward | Weight | Source | Purpose |
|--------|--------|--------|---------|
| `driving_decision_accuracy` | 0.5 | Exact match of `横向决策`/`纵向决策` vs `gt_answer` | Task accuracy |
| `coc_trace_score` | 0.0 | Schema coverage of ` thinking` vs persisted `coc_trace_schema` | Trigger only |
| `driving_formal_rm` | 0.5 | Generative RM (`DRIVING_RM_MODEL`) scoring semantic agreement | Semantic quality |

The `coc_trace_score` has weight **0.0** in the GRPO objective. It is used solely by `CocTraceGRPOTrainer` to decide whether to create guided samples.

## Coc Trace Trigger Logic

```
def should_trigger_coc_trace(enabled, group_scores, threshold, probability,
                              prompt_id, global_step, seed):
    if not enabled or max(group_scores) >= threshold:
        return False
    draw = SHA256(f"{seed}:{global_step}:{prompt_id}") / 2^64
    return draw < probability
```

The deterministic hash ensures reproducibility across runs with the same seed.

## Coc Trace Guided Rollout Replacement

When CoC is triggered for a prompt group:

1. Parse each ordinary rollout's ` thinking` into a `TraceSchema`.
2. Compare against the reference `TraceSchema` to identify gaps.
3. Build up to `coc_trace_num_generations` guidance variants (Level 1/2/3).
4. For each variant, inject guidance into the user message.
5. Generate a new completion under each guided prompt.
6. Preserve the guided rollout logprobs, then restore the plain prompt for policy optimization.
7. Score the guided samples.
8. Replace the worst ordinary samples (lowest `coc_trace_score`) within the same prompt group.
9. Compute request-aware group advantages on the final set (total = `num_generations`).

## Tests

```bash
python -m pytest tests/coc_trace_rl/ -v
```

Tests cover:
- Schema extraction and safe guidance (no action terms leaked)
- Deterministic trigger logic (flag + threshold + probability)
- JSONL preparation pipeline
- CoC scoring (think vs persisted schema)
- Decision accuracy scoring (both lateral and longitudinal)

## Notes

- `CocTraceGRPOTrainer` subclasses `GRPOTrainer` without modifying upstream source files.
- When CoC is disabled, it delegates directly to the parent GRPO implementation.
- The implementation records `coc_trace/guided_rollouts` and `coc_trace/triggered_prompts` metrics.
- Requires a GPU with BF16 support and at least 24 GiB memory for the 2B policy + 1.5B reward model.

## Change Log

### 2026-07-12: Guided Rollout Correctness Fix

Commit: `f08a942d`

- Replaced copied ordinary completions with real generation under each guided prompt.
- Preserved guided rollout logprobs before restoring the plain prompt for optimization.
- Computed trigger rewards locally and gathered them request-aware for multi-process advantage calculation.
- Limited low-CoC replacement to each sample's original prompt group.
- Restored the three-variant guidance default and removed train/eval JSONL duplication.
- Added trainer regression tests for per-prompt replacement and three-level guided construction.
