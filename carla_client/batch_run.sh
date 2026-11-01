#!/bin/bash
# Runs run_experiment.sh over every method, town and seed, skipping runs that already have output.
#
# Usage: METHODS="vanderer nomad" TOWNS="1 2" SEEDS="1 2 3" ./batch_run.sh

CLIENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(dirname "$CLIENT_DIR")/outputs}"
METHODS="${METHODS:-vanderer nomad nwm rnd yamauchi nbvp}"
TOWNS="${TOWNS:-1 2 3 4 5}"
SEEDS="${SEEDS:-1 2 3}"

declare -A SUBDIR=(
    [vanderer]="vanderer/${MODE:-guide}"
    [nomad]="nomad/${MODE:-random}"
    [nwm]="nwm/${MODE:-select}"
    [rnd]="rnd/${MODE:-select}"
    [yamauchi]=yamauchi
    [nbvp]=nbvp
)

trap 'echo "Interrupted."; exit 130' INT

for METHOD in $METHODS; do
    for TOWN in $TOWNS; do
        for SEED in $SEEDS; do
            TARGET="$OUTPUT_ROOT/${SUBDIR[$METHOD]}/recorded_trajs_town_${TOWN}_seed_${SEED}"
            if [ -d "$TARGET" ]; then
                echo "Skipping $METHOD town $TOWN seed $SEED: $TARGET exists"
                continue
            fi
            echo "Running $METHOD town $TOWN seed $SEED"
            "$CLIENT_DIR/run_experiment.sh" --method "$METHOD" --town "$TOWN" --seed "$SEED" ${MODE:+--mode "$MODE"}
            sleep 10  # Let the previous CARLA server release its port
        done
    done
done
