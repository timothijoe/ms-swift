# RM Template Prefix Texts

Put long prompt bodies in plain text files to avoid JSON escaping issues.

Recommended files:
- `default_system.txt`
- `default_instruction.txt`
- `default_fewshot.txt`

Then run:

```bash
python3 driving_grpo_clean/sync_rm_prefix.py \
  --type default \
  --system driving_grpo_clean/rm_templates_prefix_texts/default_system.txt \
  --instruction driving_grpo_clean/rm_templates_prefix_texts/default_instruction.txt \
  --fewshot driving_grpo_clean/rm_templates_prefix_texts/default_fewshot.txt
```

This will safely inject text into `driving_grpo_clean/rm_templates_prefix.json`.
