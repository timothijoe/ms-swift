#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

if [ -z "${DRIVING_RM_MODEL:-}" ]; then
  echo "ERROR: DRIVING_RM_MODEL is required, e.g. DRIVING_RM_MODEL=Qwen/Qwen2.5-1.5B-Instruct"
  exit 1
fi

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/driving_manifest_main.py" \
  --dataset_manifest_path "${SCRIPT_DIR}/datasets_manifest_video_style_32_template_v5_with_assistant.json" \
  --manifest_register_file "${SCRIPT_DIR}/manifest_dataset_register.py" \
  --use_manifest_as_dataset true \
  --load_from_cache_file false \
  --external_plugins "${SCRIPT_DIR}/driving_rm_plugin.py" \
  --reward_model "${DRIVING_RM_MODEL}" \
  --reward_model_plugin driving_rubric_rm \
  --reward_weights "${DRIVING_REWARD_W_RM:-1.0}" \
  --output_dir "${PROJECT_ROOT}/output/GRPO_DRIVING_SEMANTIC_RM_THINK" \
  --max_length 1024 \
  --max_completion_length 1024 \
  --per_device_train_batch_size "${DRIVING_PBS_TRAIN:-2}" \
  --per_device_eval_batch_size "${DRIVING_PBS_EVAL:-2}" \
  --num_generations "${DRIVING_NUM_GENERATIONS:-2}" \
  --gradient_accumulation_steps "${DRIVING_GAS:-1}" \
  --eval_strategy steps \
  --eval_on_start false \
  --eval_steps "${DRIVING_EVAL_STEPS:-50}" \
  --do_eval true \
  --beta "${DRIVING_BETA:-0.1}" \
  --num_iterations 1 \
  --dataloader_num_workers 0 \
  --remove_unused_columns false \
  --report_to none
