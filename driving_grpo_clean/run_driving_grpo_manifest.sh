#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/driving_manifest_main.py" \
  --dataset_manifest_path "${SCRIPT_DIR}/datasets_manifest.json" \
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
