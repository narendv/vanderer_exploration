#!/usr/bin/env bash
set -euo pipefail

# Runs generate_trajectories_basic_agent.py sequentially for each town
# with town-specific rollout counts.
#
# Usage:
#   bash run_generate_trajectories_all_towns.sh [OUTPUT_BASE_DIR] [EXTRA_ARGS...]
#   bash run_generate_trajectories_all_towns.sh --output_dir OUTPUT_BASE_DIR [EXTRA_ARGS...]
#
# Examples:
#   bash run_generate_trajectories_all_towns.sh /path/to/basic_agent
#   bash run_generate_trajectories_all_towns.sh /path/to/basic_agent --offscreen --base_port 2000
#   bash run_generate_trajectories_all_towns.sh --output_dir /path/to/basic_agent --offscreen --base_port 2000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."  # carla_client/, so the generator runs with python -m

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [OUTPUT_BASE_DIR] [EXTRA_ARGS...]
  $(basename "$0") --output_dir OUTPUT_BASE_DIR [EXTRA_ARGS...]
  $(basename "$0") --output_dir=OUTPUT_BASE_DIR [EXTRA_ARGS...]
EOF
}

OUTPUT_BASE_DIR="${SCRIPT_DIR}/generated_trajectories"
EXTRA_ARGS=()
positional_output_dir_set=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -o|--output_dir)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires a value" >&2
        usage >&2
        exit 2
      fi
      OUTPUT_BASE_DIR="$2"
      positional_output_dir_set=1
      shift 2
      ;;
    --output_dir=*)
      OUTPUT_BASE_DIR="${1#*=}"
      positional_output_dir_set=1
      shift
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      # Backward compatibility: first non-flag arg is OUTPUT_BASE_DIR.
      if [[ $positional_output_dir_set -eq 0 && "$1" != -* ]]; then
        OUTPUT_BASE_DIR="$1"
        positional_output_dir_set=1
      else
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

run_for_town() {
  local town="$1"
  local num_rollouts="$2"
  local out_dir="${OUTPUT_BASE_DIR}"

  mkdir -p "${out_dir}"

  echo "============================================================"
  echo "Starting town=${town} num_rollouts=${num_rollouts}"
  echo "Output: ${out_dir}"
  echo "============================================================"

  python -m data_generation.generate_trajectories_basic_agent \
    --towns "${town}" \
    --num_rollouts "${num_rollouts}" \
    --output_dir "${out_dir}" \
    "${EXTRA_ARGS[@]}"
}

# Requested rollout schedule:
# 1000 for towns 1, 2, 4
# 2000 for towns 3, 5
# 3000 for town 10
run_for_town 1 1000
run_for_town 2 1000
run_for_town 4 1000
run_for_town 3 2000
run_for_town 5 2000

echo "All town runs completed."
