#!/bin/sh
# =============================================================================
# CoC Trace Driving GRPO — Debug: Baseline (CoC Trace disabled)
# =============================================================================
# 用于 VSCode Debug 的 Baseline 入口脚本。
# 不启用 CoC Trace 引导，只使用 driving_decision_accuracy + formal RM 训练。
#
# 用法：
#   bash coc_trace_rl/scripts/debug/run_baseline.sh
#
# 可覆盖的环境变量：
#   DRIVING_RM_MODEL         奖励模型（默认 Qwen/Qwen2.5-1.5B-Instruct）
#   DRIVING_MODEL            策略模型（默认 Qwen/Qwen3-VL-2B-Instruct）
#   DRIVING_NUM_GENERATIONS  每个 prompt 的采样数（默认 4）
#   DRIVING_GAS              梯度累积步数（默认 1）
# =============================================================================

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON:-/home/linux/anaconda3/envs/sw_312_env/bin/python}"
DRIVING_RM_MODEL="${DRIVING_RM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

if [ -z "${DRIVING_RM_MODEL}" ]; then
  echo "ERROR: DRIVING_RM_MODEL is required, e.g. Qwen/Qwen2.5-1.5B-Instruct" >&2
  exit 1
fi

"${PYTHON_BIN}" -m coc_trace_rl.src.coc_trace_main \
  --model "${DRIVING_MODEL:-Qwen/Qwen3-VL-2B-Instruct}" \
  --reward_model "${DRIVING_RM_MODEL}" \
  --reward_model_plugin driving_formal_rm \
  --reward_funcs driving_decision_accuracy \
  --reward_weights 0.5 0.5 \
  --coc_trace_enabled false \
  --num_generations "${DRIVING_NUM_GENERATIONS:-4}" \
  --generation_batch_size "${DRIVING_GENERATION_BATCH_SIZE:-4}" \
  --output_dir "${PROJECT_ROOT}/output/COC_TRACE_DEBUG_BASELINE" \
  --max_length 512 \
  --max_completion_length 512 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${DRIVING_GAS:-1}" \
  --report_to none \
  --load_from_cache_file false \
  --remove_unused_columns false \
  --dataloader_num_workers 0 \
  --num_train_epochs "${DRIVING_NUM_EPOCHS:-1}" \
  --logging_steps 1