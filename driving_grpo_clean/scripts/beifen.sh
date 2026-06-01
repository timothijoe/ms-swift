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

MODE="check"

COMPACT="none"
TYPE=“blind_spot”
SYSTEM_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/blind_spot/system.txt”
INSTRUCTION_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/blind_spot/instruction.txt”
FEWSHOT_TXT=“driving_grpo_clean/configs/rm_templates_prefix_texts/blind_spot/fewshot.txt”



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
