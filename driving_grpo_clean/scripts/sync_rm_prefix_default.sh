#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

MODE="${1:-write}"  # write | check
COMPACT="${COMPACT:-light}"  # none | light | aggressive
TYPE="${TYPE:-default}"
SYSTEM_TXT="${SYSTEM_TXT:-driving_grpo_clean/configs/rm_templates_prefix_texts/default_system.txt}"
INSTRUCTION_TXT="${INSTRUCTION_TXT:-driving_grpo_clean/configs/rm_templates_prefix_texts/default_instruction.txt}"
FEWSHOT_TXT="${FEWSHOT_TXT:-driving_grpo_clean/configs/rm_templates_prefix_texts/default_fewshot.txt}"

# Optional local quick-override block:
MODE="check"
MODE="write"

COMPACT="none"
# TYPE=“blind_spot”
# SYSTEM_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/blind_spot/system.txt”
# INSTRUCTION_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/blind_spot/instruction.txt”
# FEWSHOT_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/blind_spot/fewshot.txt”

# TYPE=obstacle_avoidance
# SYSTEM_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/obstacle_avoidance/system.txt”
# INSTRUCTION_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/obstacle_avoidance/instruction.txt”
# FEWSHOT_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/obstacle_avoidance/fewshot.txt”

# TYPE=lane_change
# SYSTEM_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/lane_change/system.txt”
# INSTRUCTION_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/lane_change/instruction.txt”
# FEWSHOT_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/lane_change/fewshot.txt”


# TYPE=yielding
# SYSTEM_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/yielding/system.txt”
# INSTRUCTION_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/yielding/instruction.txt”
# FEWSHOT_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/yielding/fewshot.txt”


# TYPE=traffic_signal
# SYSTEM_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/traffic_signal/system.txt”
# INSTRUCTION_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/traffic_signal/instruction.txt”
# FEWSHOT_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/traffic_signal/fewshot.txt”


# TYPE=road_condition_geometry
# SYSTEM_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/road_condition_geometry/system.txt”
# INSTRUCTION_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/road_condition_geometry/instruction.txt”
# FEWSHOT_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/road_condition_geometry/fewshot.txt”


TYPE=traffic_control
SYSTEM_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/traffic_control/system.txt”
INSTRUCTION_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/traffic_control/instruction.txt”
FEWSHOT_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/traffic_control/fewshot.txt”

sanitize_quotes() {
  # strip leading/trailing ascii/smart quotes if present
  printf '%s' "$1" | sed 's/^[ "\"“”'\''‘’]*//;s/[ "\"“”'\''‘’]*$//'
}

MODE="$(sanitize_quotes "${MODE}")"
COMPACT="$(sanitize_quotes "${COMPACT}")"
TYPE="$(sanitize_quotes "${TYPE}")"
SYSTEM_TXT="$(sanitize_quotes "${SYSTEM_TXT}")"
INSTRUCTION_TXT="$(sanitize_quotes "${INSTRUCTION_TXT}")"
FEWSHOT_TXT="$(sanitize_quotes "${FEWSHOT_TXT}")"



cd "${PROJECT_ROOT}"

CMD="\"${PYTHON_BIN}\" driving_grpo_clean/src/sync_rm_prefix.py \
  --type \"${TYPE}\" \
  --system \"${SYSTEM_TXT}\" \
  --instruction \"${INSTRUCTION_TXT}\" \
  --fewshot \"${FEWSHOT_TXT}\" \
  --compact \"${COMPACT}\""

if [ "${MODE}" = "check" ]; then
  # shellcheck disable=SC2086
  eval ${CMD} --check-only
elif [ "${MODE}" = "write" ]; then
  # shellcheck disable=SC2086
  eval ${CMD}
else
  echo "Usage: bash driving_grpo_clean/scripts/sync_rm_prefix_default.sh [write|check]"
  echo "Optional env:"
  echo "  COMPACT=none|light|aggressive"
  echo "  TYPE=default"
  echo "  SYSTEM_TXT=... INSTRUCTION_TXT=... FEWSHOT_TXT=..."
  exit 1
fi
