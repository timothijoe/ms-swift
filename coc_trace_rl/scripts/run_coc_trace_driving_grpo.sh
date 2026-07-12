#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

if [ -z "${DRIVING_RM_MODEL:-}" ]; then
  echo "ERROR: DRIVING_RM_MODEL is required, e.g. Qwen/Qwen2.5-1.5B-Instruct" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m coc_trace_rl.src.coc_trace_main \
  --reward_model "${DRIVING_RM_MODEL}" \
  --reward_model_plugin driving_formal_rm \
  --reward_weights 0.5 0.0 0.5 \
  --coc_trace_enabled true \
  --coc_trace_reward_threshold "${COC_TRACE_REWARD_THRESHOLD:-0.8}" \
  --coc_trace_probability "${COC_TRACE_PROBABILITY:-0.3}" \
  --coc_trace_num_generations "${COC_TRACE_NUM_GENERATIONS:-3}" \
  --num_generations "${DRIVING_NUM_GENERATIONS:-2}" \
  --output_dir "${PROJECT_ROOT}/output/COC_TRACE_DRIVING_GRPO" \
  --max_length "${DRIVING_MAX_LENGTH:-1024}" \
  --max_completion_length "${DRIVING_MAX_COMPLETION_LENGTH:-1024}" \
  --per_device_train_batch_size "${DRIVING_PBS_TRAIN:-2}" \
  --per_device_eval_batch_size "${DRIVING_PBS_EVAL:-2}" \
  --gradient_accumulation_steps "${DRIVING_GAS:-1}" \
  --report_to none
