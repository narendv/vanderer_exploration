#!/bin/bash
# Runs one exploration experiment. Policy methods start explore/explore_*.py in the
# model env and drive_policy.py in the CARLA env; yamauchi and nbvp need only the CARLA env.
#
# Usage: ./run_experiment.sh --method <vanderer|nomad|nwm|rnd|yamauchi|nbvp> --town N --seed N [--mode guide|select|random]
# Environment: CARLA_BIN (required), MODEL_ENV, CARLA_ENV, OUTPUT_ROOT, MAX_STEPS

CLIENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$CLIENT_DIR")"
MODEL_ENV="${MODEL_ENV:-vanderer}"
CARLA_ENV="${CARLA_ENV:-vanderer-carla}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs}"
MAX_STEPS="${MAX_STEPS:-100000}"
METHOD="" TOWN="" SEED="" MODE=""

usage() {
    echo "Usage: $0 --method <vanderer|nomad|nwm|rnd|yamauchi|nbvp> --town <number> --seed <number> [--mode guide|select|random]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift 2 ;;
        --town) TOWN="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        *) usage ;;
    esac
done
[[ -z "$METHOD" || -z "$TOWN" || -z "$SEED" ]] && usage
: "${CARLA_BIN:?Set CARLA_BIN to <carla>/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping}"

eval "$(conda shell.bash hook)"

# Script and default mode per method. These modes match the evaluated runs.
case $METHOD in
    vanderer) POLICY=explore_carla.py; MODE="${MODE:-guide}" ;;
    nomad) POLICY=explore_nomad.py; MODE="${MODE:-random}" ;;
    nwm) POLICY=explore_nwm.py; MODE="${MODE:-select}" ;;
    rnd) POLICY=explore_rnd.py; MODE="${MODE:-select}" ;;
    yamauchi|nbvp)
        cd "$CLIENT_DIR" || exit 1
        conda activate "$CARLA_ENV"
        python "drive_$METHOD.py" --town "$TOWN" --seed "$SEED" --max_steps "$MAX_STEPS" \
            --root_dir "$OUTPUT_ROOT" --offscreen
        exit $?
        ;;
    *) usage ;;
esac

POLICY_PID=""
cleanup() {
    [[ -z "$POLICY_PID" ]] && return
    kill -INT "$POLICY_PID" 2>/dev/null
    # The policy saves its outputs on the way out.
    for _ in {1..30}; do
        kill -0 "$POLICY_PID" 2>/dev/null || return
        sleep 1
    done
    kill -KILL "$POLICY_PID" 2>/dev/null
}
trap cleanup EXIT

ROOT_DIR="$OUTPUT_ROOT/$METHOD"

cd "$REPO_DIR/explore" || exit 1
conda activate "$MODEL_ENV"
python "$POLICY" --town "$TOWN" --seed "$SEED" --mode "$MODE" --root-dir "$ROOT_DIR" &
POLICY_PID=$!
conda deactivate

sleep 5  # Let the policy bind its ZMQ sockets

# In the foreground, so Ctrl+C reaches it and it can close the CARLA server.
cd "$CLIENT_DIR" || exit 1
conda activate "$CARLA_ENV"
python drive_policy.py --town "$TOWN" --seed "$SEED" --mode "$MODE" --max_steps "$MAX_STEPS" \
    --root_dir "$ROOT_DIR" --offscreen
