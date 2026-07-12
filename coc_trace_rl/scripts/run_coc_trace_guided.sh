#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON:-/home/linux/anaconda3/envs/sw_312_env/bin/python}"

# ============================================================
# CoC Trace Driving GRPO — CoC Guided
# ============================================================
# Same as baseline but with CoC Trace guidance enabled.
# When group reasoning scores are low, the worst rollouts are
# replaced with schema-guided rollouts (Level 1/2/3 variants).

DRIVING_RM_MODEL="${DRIVING_RM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

if [ -z "${DRIVING_RM_MODEL}" ]; then
  echo "ERROR: DRIVING_RM_MODEL is required, e.g. Qwen/Qwen2.5-1.5B-Instruct" >&2
  exit 1
fi

"${PYTHON_BIN}" -m coc_trace_rl.src.coc_trace_main \
  --model "${DRIVING_MODEL:-Qwen/Qwen3-VL-2B-Instruct}" \
  --reward_model "${DRIVING_RM_MODEL}" \
  --reward_model_plugin driving_formal_rm \
  --reward_funcs driving_decision_accuracy coc_trace_score \
  --reward_weights 0.5 0.0 0.5 \
  --coc_trace_enabled true \
  --coc_trace_reward_threshold "${COC_TRACE_REWARD_THRESHOLD:-0.8}" \
  --coc_trace_probability "${COC_TRACE_PROBABILITY:-0.3}" \
  --coc_trace_num_generations "${COC_TRACE_NUM_GENERATIONS:-2}" \
  --num_generations "${DRIVING_NUM_GENERATIONS:-4}" \
  --generation_batch_size "${DRIVING_GENERATION_BATCH_SIZE:-4}" \
  --output_dir "${PROJECT_ROOT}/output/COC_TRACE_GUIDED" \
  --max_length "${DRIVING_MAX_LENGTH:-1024}" \
  --max_completion_length "${DRIVING_MAX_COMPLETION_LENGTH:-1024}" \
  --per_device_train_batch_size "${DRIVING_PBS_TRAIN:-2}" \
  --per_device_eval_batch_size "${DRIVING_PBS_EVAL:-2}" \
  --gradient_accumulation_steps "${DRIVING_GAS:-1}" \
  --report_to none \
  --load_from_cache_file false \
  --remove_unused_columns false \
  --dataloader_num_workers 0 \
  --num_train_epochs "${DRIVING_NUM_EPOCHS:-1}" \
  --logging_steps 1