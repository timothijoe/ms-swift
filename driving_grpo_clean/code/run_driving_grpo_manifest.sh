#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

cd "${PROJECT_ROOT}"

set -- "${PYTHON_BIN}" "${SCRIPT_DIR}/driving_manifest_main.py" \
  --dataset_manifest_path "${SCRIPT_DIR}/../data/datasets_manifest.json" \
  --manifest_register_file "${SCRIPT_DIR}/manifest_dataset_register.py" \
  --use_manifest_as_dataset true \
  --reward_funcs driving_no_think_format driving_decision_accuracy \
  --output_dir "${PROJECT_ROOT}/output/GRPO_DRIVING_MANIFEST_CLEAN" \
  --max_length 1024 \
  --eval_strategy steps \
  --eval_on_start true \
  --eval_steps 8 \
  --do_eval true \
  --beta 0.1

# Optional: enable RM-based rubric scoring.
# Example:
# DRIVING_RM_MODEL=Qwen/Qwen2.5-1.5B-Instruct bash driving_grpo_clean/code/run_driving_grpo_manifest.sh
# DRIVING_RM_RUBRIC_JSON='[{"name":"horizontal_decision","desc":"横向决策是否一致","weight":0.5},{"name":"vertical_decision","desc":"纵向决策是否一致","weight":0.5}]' \
# DRIVING_RM_MODEL=Qwen/Qwen2.5-1.5B-Instruct bash driving_grpo_clean/code/run_driving_grpo_manifest.sh
if [ -n "${DRIVING_RM_MODEL:-}" ]; then
  set -- "$@" \
    --external_plugins "${SCRIPT_DIR}/driving_rm_plugin.py" \
    --reward_model "${DRIVING_RM_MODEL}" \
    --reward_model_plugin driving_rubric_rm \
    --reward_weights "${DRIVING_REWARD_W1:-0.4}" "${DRIVING_REWARD_W2:-0.4}" "${DRIVING_REWARD_W3:-0.2}"
fi

"$@"
