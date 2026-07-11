# CoC Trace Driving GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal driving CoC Trace GRPO entrypoint that generates diverse schema-driven guided rollouts while optimizing the unguided policy.

**Architecture:** A new `coc_trace_rl` package owns the driving adapters, schema logic, entrypoint, and `CocTraceGRPOTrainer` subclass. The subclass extends `GRPOTrainer` only through protected orchestration methods; the existing trainer and global trainer factory remain unchanged.

**Tech Stack:** Python 3, PyTorch, Hugging Face datasets/transformers, ms-swift GRPO, pytest, JSONL.

## Global Constraints

- Do not modify `swift/trainers/rlhf_trainer/grpo_trainer.py` or `swift/trainers/trainer_factory.py`.
- Keep new runtime code under `coc_trace_rl/`.
- Do not add final driving actions from `think` to guidance prompts.
- Keep `coc_trace_reward` out of the GRPO advantage reward.
- Trigger only when CoC is enabled, maximum group CoC score is below `0.8`, and deterministic probability is below `0.3`.
- Require `coc_trace_num_generations >= 3` when CoC is enabled.
- Preserve parent GRPO behavior when CoC is disabled.

---

## File Structure

- `coc_trace_rl/src/coc_trace_schema.py`: trace schema, comparison, trigger, and action-free guidance variants.
- `coc_trace_rl/src/driving_dataset.py`: manifest and row preprocessing.
- `coc_trace_rl/scripts/build_coc_trace_schema.py`: idempotent JSONL schema preparation.
- `coc_trace_rl/src/driving_rewards.py`: decision and CoC trace reward functions.
- `coc_trace_rl/src/driving_formal_rm.py`: minimal semantic driving Formal RM plugin.
- `coc_trace_rl/src/coc_trace_args.py`: experiment-only CLI arguments.
- `coc_trace_rl/src/coc_trace_grpo_trainer.py`: derived GRPO trainer.
- `coc_trace_rl/src/coc_trace_main.py`: manifest setup and direct trainer construction.
- `coc_trace_rl/configs/driving_video_manifest.json`: canonical input manifest.
- `coc_trace_rl/scripts/run_coc_trace_driving_grpo.sh`: canonical command.
- `tests/coc_trace_rl/`: schema, reward, trainer, and entrypoint tests.

### Task 1: CoC Trace Schema and Guidance

**Files:**
- Create: `coc_trace_rl/src/coc_trace_schema.py`
- Create: `tests/coc_trace_rl/test_coc_trace_schema.py`

**Interfaces:**
- Produces `TraceSchema`, `GuidanceVariant`, `parse_trace_schema`, `score_trace_schema`, `build_guidance_variants`, and `should_trigger_coc_trace`.

- [ ] **Step 1: Write failing tests**

```python
def test_guidance_has_three_levels_and_no_actions():
    reference = parse_trace_schema('左前方遮挡形成盲区，影响可见性和安全裕度。')
    variants = build_guidance_variants(reference, [parse_trace_schema('前方有车辆。')])
    assert [item.level for item in variants] == [1, 2, 3]
    assert all('左换道' not in item.text and '减速' not in item.text for item in variants)


def test_trigger_requires_flag_threshold_and_probability():
    params = dict(group_scores=[0.2, 0.7], threshold=0.8, probability=1.0,
                  prompt_id='scene-1', global_step=4, seed=9)
    assert should_trigger_coc_trace(enabled=True, **params)
    assert not should_trigger_coc_trace(enabled=False, **params)
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/coc_trace_rl/test_coc_trace_schema.py -v`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the smallest schema API**

```python
@dataclass(frozen=True)
class TraceSchema:
    task_category: str
    task_subcategory: str
    evidence: tuple[str, ...]
    risk_causes: tuple[str, ...]
    constraints: tuple[str, ...]


@dataclass(frozen=True)
class GuidanceVariant:
    level: int
    text: str
    gaps: tuple[str, ...]
```

Use a small driving-risk keyword taxonomy. Remove action phrases before persistence and prompt generation. Score schema-field overlap in `[0, 1]`. Use `sha256(f'{seed}:{global_step}:{prompt_id}')` to derive the shared probability draw.

- [ ] **Step 4: Verify pass**

Run: `pytest tests/coc_trace_rl/test_coc_trace_schema.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add coc_trace_rl/src/coc_trace_schema.py tests/coc_trace_rl/test_coc_trace_schema.py && git commit -m "feat: add CoC trace schema guidance"`

### Task 2: Persisted Schema Dataset Path

**Files:**
- Create: `coc_trace_rl/src/driving_dataset.py`
- Create: `coc_trace_rl/scripts/build_coc_trace_schema.py`
- Modify: `tests/coc_trace_rl/test_coc_trace_schema.py`

**Interfaces:**
- Consumes `parse_trace_schema`.
- Produces `prepare_jsonl(input_path: Path, output_path: Path) -> int` and `DrivingTracePreprocessor.preprocess(row: dict) -> dict | None`.

- [ ] **Step 1: Write failing preparation test**

```python
def test_prepare_jsonl_persists_schema(tmp_path):
    source, target = tmp_path / 'source.jsonl', tmp_path / 'prepared.jsonl'
    source.write_text('{"think":"前方盲区影响可见性。"}\n', encoding='utf-8')
    assert prepare_jsonl(source, target) == 1
    assert json.loads(target.read_text(encoding='utf-8'))['coc_trace_schema']['task_subcategory'] == '盲区'
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/coc_trace_rl/test_coc_trace_schema.py::test_prepare_jsonl_persists_schema -v`

Expected: FAIL because `prepare_jsonl` does not exist.

- [ ] **Step 3: Implement preparation and preprocessor**

Read JSONL line by line, add or replace invalid `coc_trace_schema`, and write exactly one output row per source row. The row preprocessor removes a trailing assistant response, extracts `think` and decision JSON, requires the persisted schema, and preserves media paths without legacy path remapping.

- [ ] **Step 4: Verify pass**

Run: `pytest tests/coc_trace_rl/test_coc_trace_schema.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add coc_trace_rl/src/driving_dataset.py coc_trace_rl/scripts/build_coc_trace_schema.py tests/coc_trace_rl/test_coc_trace_schema.py && git commit -m "feat: prepare driving CoC schemas"`

### Task 3: Driving Reward Surface

**Files:**
- Create: `coc_trace_rl/src/driving_rewards.py`
- Create: `coc_trace_rl/src/driving_formal_rm.py`
- Create: `tests/coc_trace_rl/test_driving_rewards.py`

**Interfaces:**
- Produces ORM registrations `driving_decision_accuracy` and `coc_trace_score`.
- Produces RM plugin registration `driving_formal_rm`.

- [ ] **Step 1: Write failing CoC reward test**

```python
def test_coc_score_uses_think_and_persisted_schema():
    reward = CocTraceScoreReward()
    schema = parse_trace_schema('前方盲区影响可见性。').to_dict()
    result = reward(['<think>前方盲区导致可见性受限。</think>{}'],
                    coc_trace_schema=[schema])
    assert result == [1.0]
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/coc_trace_rl/test_driving_rewards.py -v`

Expected: FAIL because `CocTraceScoreReward` does not exist.

- [ ] **Step 3: Implement reward modules**

Register decision accuracy and CoC schema score ORMs. Migrate only the Formal RM factor/action comparison and plugin interface required for semantic driving scoring. Exclude template catalogs, debug paths, score-file persistence, and reference schema generation.

- [ ] **Step 4: Verify pass**

Run: `pytest tests/coc_trace_rl/test_driving_rewards.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add coc_trace_rl/src/driving_rewards.py coc_trace_rl/src/driving_formal_rm.py tests/coc_trace_rl/test_driving_rewards.py && git commit -m "feat: add driving CoC rewards"`

### Task 4: Derived CoC Trace GRPO Trainer

**Files:**
- Create: `coc_trace_rl/src/coc_trace_grpo_trainer.py`
- Create: `tests/coc_trace_rl/test_coc_trace_trainer.py`

**Interfaces:**
- Consumes `GRPOTrainer`, `GuidanceVariant`, and `should_trigger_coc_trace`.
- Produces `CocTraceGRPOTrainer._generate_and_score_completions` and `_build_guided_inputs`.

- [ ] **Step 1: Write failing delegation and merge tests**

```python
def test_disabled_trainer_delegates_to_parent(monkeypatch):
    trainer = make_fake_trainer(coc_trace_enabled=False)
    parent = monkeypatch.spy(GRPOTrainer, '_generate_and_score_completions')
    trainer._generate_and_score_completions([{'messages': []}])
    assert parent.call_count == 1


def test_guided_rows_share_prompt_and_have_unique_requests():
    trainer = make_fake_trainer(coc_trace_enabled=True, coc_trace_num_generations=3)
    guided = trainer._build_guided_inputs([plain_input('p-1')], torch.tensor([[0.2]]))
    assert {row['prompt_id'] for row in guided} == {'p-1'}
    assert len({row['request_id'] for row in guided}) == 3
    assert [row['guidance_level'] for row in guided] == [1, 2, 3]
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/coc_trace_rl/test_coc_trace_trainer.py -v`

Expected: FAIL because `CocTraceGRPOTrainer` does not exist.

- [ ] **Step 3: Implement the focused subclass**

Delegate to the parent when disabled. When enabled, orchestrate ordinary generation/scoring, group trigger evaluation, three-level guided input creation, guided generation/scoring, parent encoding, and request-aware advantages. Set `dynamic_num_samples=True` for merged groups. Restore `_coc_plain_messages` before encoding guided samples, preserve guided rollout logprobs, trim any Level 3 selection prefix and its logprobs together, and record `coc_trace/` metrics.

- [ ] **Step 4: Verify pass**

Run: `pytest tests/coc_trace_rl/test_coc_trace_trainer.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add coc_trace_rl/src/coc_trace_grpo_trainer.py tests/coc_trace_rl/test_coc_trace_trainer.py && git commit -m "feat: add CoC trace GRPO trainer"`

### Task 5: Arguments and Canonical Entrypoint

**Files:**
- Create: `coc_trace_rl/src/coc_trace_args.py`
- Create: `coc_trace_rl/src/coc_trace_main.py`
- Create: `coc_trace_rl/configs/driving_video_manifest.json`
- Create: `coc_trace_rl/scripts/run_coc_trace_driving_grpo.sh`
- Modify: `tests/coc_trace_rl/test_coc_trace_trainer.py`

**Interfaces:**
- Produces `CocTraceDrivingArguments` and `coc_trace_driving_main(args) -> dict`.

- [ ] **Step 1: Write failing argument test**

```python
def test_enabled_mode_requires_three_guided_samples():
    with pytest.raises(ValueError, match='coc_trace_num_generations'):
        CocTraceDrivingArguments(coc_trace_enabled=True, coc_trace_num_generations=2)


def test_coc_defaults_match_design():
    args = CocTraceDrivingArguments()
    assert (args.coc_trace_enabled, args.coc_trace_reward_threshold, args.coc_trace_probability) == (False, 0.8, 0.3)
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/coc_trace_rl/test_coc_trace_trainer.py -v`

Expected: FAIL because `CocTraceDrivingArguments` does not exist.

- [ ] **Step 3: Implement entrypoint and command**

Define only CoC and required driving arguments. Register one video manifest, import local reward/plugin modules, and instantiate `CocTraceGRPOTrainer` directly with normal SwiftRLHF kwargs. Require `DRIVING_RM_MODEL` in the shell script and pass weights `0.5`, `0.0`, `0.5` for accuracy, CoC score, and Formal RM so the CoC score remains trigger-only.

- [ ] **Step 4: Verify pass**

Run: `pytest tests/coc_trace_rl/test_coc_trace_trainer.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add coc_trace_rl tests/coc_trace_rl/test_coc_trace_trainer.py && git commit -m "feat: add CoC trace driving entrypoint"`

### Task 6: Verification and Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the two commands**

Document the schema-preparation command and canonical `DRIVING_RM_MODEL` launch command. State that CoC guidance is disabled by default and excludes final actions.

- [ ] **Step 2: Run focused tests**

Run: `pytest tests/coc_trace_rl -v`

Expected: PASS.

- [ ] **Step 3: Run syntax and smoke checks**

Run: `python -m compileall coc_trace_rl/src && bash -n coc_trace_rl/scripts/run_coc_trace_driving_grpo.sh`

Expected: exit status `0`.

- [ ] **Step 4: Commit**

Run: `git add README.md tests/coc_trace_rl && git commit -m "docs: document CoC trace driving training"`

