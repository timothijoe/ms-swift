# Timothi Record Log

## 2026-07-12: CoC Trace Driving GRPO

### Design Conclusion

- The target is a minimal driving CoC Trace RL implementation in `zt-ms-swift`, informed by `ref-zt-ms-swift/driving_grpo_clean` but without copying historical files or modifying generic GRPO source.
- Reference `think` text is converted into a persisted CoC schema. The CoC score is a trigger/metric only; decision accuracy and Formal RM remain the training reward.
- Trigger conditions are CoC enabled, group maximum CoC score below threshold, and a deterministic probability draw. The default threshold is `0.8`; the default probability is `0.3`.
- Every triggered group receives three schema-derived guidance variants: task locator, reasoning frame, and selective checklist. Guidance excludes final driving actions.

### Implementation Conclusion

- Added `coc_trace_rl/` with schema preparation, dataset normalization, CoC and decision rewards, Formal RM plugin, derived trainer, manifest entrypoint, baseline/guided scripts, and tests.
- Added `coc_trace_rl/README.md` as the detailed feature document.
- Focused tests cover schema extraction, safe guidance, trigger logic, JSONL preparation, CoC scoring, decision scoring, and prompt-group replacement.

### Debugging Conclusion

- Corrected a remote adaptation that copied the first ordinary completion into every guided sample. Guided samples now call real generation under their own guided prompts.
- Preserved actual guided rollout logprobs before restoring the plain prompt.
- Changed replacement from global ranking to per-`prompt_id` ranking, avoiding cross-scene completion mixing.
- Changed trigger scoring to local reward computation followed by request-aware gathering, avoiding multi-process reward/input index mismatches.
- Removed duplicated train/eval JSONL registration; baseline and guided scripts use `split_dataset_ratio` for an evaluation partition.
- Restored the three-guidance minimum and default (`COC_TRACE_NUM_GENERATIONS=3`).

### Relevant Commits

- `6551eea8`: merge of the initial CoC Trace implementation.
- `d6e67f49`: remote-environment adaptation.
- `f08a942d`: guided rollout correctness fixes.
- `b0b9830e`: CoC README correction and change log.
