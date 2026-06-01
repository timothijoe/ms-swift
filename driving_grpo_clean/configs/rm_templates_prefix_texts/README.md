# RM Template Prefix Texts

Put long prompt bodies in plain text files to avoid JSON escaping issues.

Recommended files:
- `default_system.txt`
- `default_instruction.txt`
- `default_fewshot.txt`

Then run (write to JSON):

```bash
python3 driving_grpo_clean/src/sync_rm_prefix.py \
  --type default \
  --system driving_grpo_clean/configs/rm_templates_prefix_texts/default_system.txt \
  --instruction driving_grpo_clean/configs/rm_templates_prefix_texts/default_instruction.txt \
  --fewshot driving_grpo_clean/configs/rm_templates_prefix_texts/default_fewshot.txt \
  --compact light
```

This will safely inject text into `driving_grpo_clean/configs/rm_templates_prefix.json`.

Check readability/size without writing:

```bash
python3 driving_grpo_clean/src/sync_rm_prefix.py \
  --type default \
  --system driving_grpo_clean/configs/rm_templates_prefix_texts/default_system.txt \
  --instruction driving_grpo_clean/configs/rm_templates_prefix_texts/default_instruction.txt \
  --fewshot driving_grpo_clean/configs/rm_templates_prefix_texts/default_fewshot.txt \
  --compact light \
  --check-only
```

Compact modes:
- `none`: keep source formatting
- `light`: trim trailing spaces + collapse excessive blank lines (recommended)
- `aggressive`: further remove most extra spaces/newlines for minimal prompt size
