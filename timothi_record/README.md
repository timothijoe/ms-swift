# Timothi Record

## Current Conclusion

- CoC Trace driving GRPO is implemented in `coc_trace_rl/` without modifying the generic `GRPOTrainer` source file.
- CoC guidance triggers when enabled, the maximum ordinary group CoC score is below the configured threshold, and a deterministic probability draw succeeds.
- Triggered prompts generate three real guided rollouts, one per schema-derived guidance level. Guidance does not include final driving actions.
- Guided completions are generated with the guided prompt, retain guided rollout logprobs, then train against the restored plain prompt.
- Low-CoC ordinary rollouts are replaced only within their original prompt group.
- Use `run_coc_trace_baseline.sh` and `run_coc_trace_guided.sh` as the paired comparison commands.
- Detailed design, usage, and the latest implementation notes remain in `coc_trace_rl/README.md`.

## Record Policy

- Keep this file short and replace outdated conclusions when the current implementation changes.
- Append each material design, implementation, debugging, or README conclusion to `log.md`.
- Include the relevant commit ID and file paths in each log entry when available.
