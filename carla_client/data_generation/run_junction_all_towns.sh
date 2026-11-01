#!/usr/bin/env bash
# Junction-anchored generation for Town12, Town11 and Town06, run strictly one
# town at a time.
#
# Runs generate_trajectories_junction.py in chunks, because the
# CARLA server dies every so often on the large maps, sometimes as an
# uncatchable C++ std::terminate that takes the whole Python process down with
# no chance for a try/except. Each chunk is its own interpreter invocation, so
# the wrapper only has to notice a nonzero exit code and retry that chunk.
#
# A crash mid-chunk re-saves the already-completed seeds of that chunk under new
# timestamps. Dedupe later on `requested_seed` in each trajectory's
# metadata.json - cheaper than tracking partial chunk progress.
#
# Never run two towns at once: each invocation launches its own CARLA server,
# and two large maps resident at the same time is what kills the server.
# CHUNK=0 means "one invocation for the whole town" (retries still apply); used
# for Town06, which is small and has not needed chunking.
#
# Per-town settings, as "town:total_rollouts:chunk_size:junction_uniform_fraction".
# uniform_fraction is the share drawn from the ORDINARY uniform sampler; the
# rest is junction-anchored. Town06 is high (0.6) because its turn inventory is
# only 51 maneuvers deep (18 of them right turns), so a larger junction share
# would just redraw the same corners; Town11/Town12 are low (0.3) because their
# inventories (925 and 2926 turn maneuvers) are nowhere near exhausted.
#
# Usage:
#   conda activate vanderer-carla
#   ./run_junction_all_towns.sh
#   OUTPUT_ROOT=/path/to/junction ./run_junction_all_towns.sh
#   DRY_RUN=1 ./run_junction_all_towns.sh      # print commands, run nothing
#
# Intended to be started inside tmux; it runs for hours.

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1  # carla_client/, so the generator runs with python -m

TOWN_SPECS=(
    "12:3000:100:0.3"
    "11:1000:250:0.3"
    "6:1000:0:0.6"
)

# Written to a NEW root by default. Pointing this at an existing carla_dataset
# town directory would blend two different generation regimes into one split.
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to the dataset output folder}"
BASE_PORT="${BASE_PORT:-2735}"
GAP_SECONDS="${GAP_SECONDS:-60}"
MAX_CHUNK_RETRIES="${MAX_CHUNK_RETRIES:-3}"
CARLA_TIMEOUT="${CARLA_TIMEOUT:-90}"
EXTRA_ARGS="${EXTRA_ARGS:---random_weather}"
GENERATOR="${GENERATOR:-data_generation.generate_trajectories_junction}"
DRY_RUN="${DRY_RUN:-0}"

RUN_LOG_ROOT="$OUTPUT_ROOT/run_logs"
SUMMARY_LOG="$RUN_LOG_ROOT/SUMMARY.txt"

log() { echo "$(date '+%F %T') $*"; }

# ─────────────────────────── preflight ───────────────────────────
# Fail here rather than after burning an hour of failed chunks. The usual cause
# is a forgotten `conda activate vanderer-carla`.
preflight() {
    local ok=0

    if [ ! -f "${GENERATOR//.//}.py" ]; then
        log "FATAL: generator not found: $PWD/$GENERATOR"
        ok=1
    fi

    if ! python -c "import carla" 2>/dev/null; then
        log "FATAL: 'import carla' failed. Did you 'conda activate vanderer-carla'?"
        ok=1
    fi

    if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":${BASE_PORT}\b"; then
        log "FATAL: port $BASE_PORT is already in use. Stop that server, or set BASE_PORT."
        ok=1
    fi

    for spec in "${TOWN_SPECS[@]}"; do
        IFS=':' read -r t total chunk uf <<< "$spec"
        case "$t" in ''|*[!0-9]*) log "FATAL: bad town in spec '$spec'"; ok=1 ;; esac
        case "$total" in ''|*[!0-9]*) log "FATAL: bad total in spec '$spec'"; ok=1 ;; esac
        case "$chunk" in ''|*[!0-9]*) log "FATAL: bad chunk in spec '$spec'"; ok=1 ;; esac
    done

    return $ok
}

# A chunk whose server crashed can leave the process alive holding the RPC port,
# which would then fail every retry. Match the exact port so a CARLA server the
# user is running elsewhere on another port is left alone.
reap_orphan_server() {
    local pids
    pids=$(pgrep -f "carla-rpc-port=${BASE_PORT}" 2>/dev/null)
    if [ -n "$pids" ]; then
        log "  reaping orphaned CARLA server on port $BASE_PORT (pids: $(echo $pids | tr '\n' ' '))"
        # shellcheck disable=SC2086
        kill $pids 2>/dev/null
        sleep 10
        pids=$(pgrep -f "carla-rpc-port=${BASE_PORT}" 2>/dev/null)
        if [ -n "$pids" ]; then
            # shellcheck disable=SC2086
            kill -9 $pids 2>/dev/null
            sleep 5
        fi
    fi
}

# ─────────────────────────── one town ───────────────────────────
run_town() {
    local town="$1" total="$2" chunk="$3" uniform_fraction="$4"
    local output_dir="$OUTPUT_ROOT/carla_town${town}"
    local log_dir="$RUN_LOG_ROOT/town${town}"
    local failed_log="$log_dir/FAILED_CHUNKS.txt"

    mkdir -p "$output_dir" "$log_dir"

    # chunk=0 -> a single invocation covering the whole town.
    if [ "$chunk" -eq 0 ]; then
        chunk="$total"
        log "town=$town: single invocation (no chunking), total=$total uniform_fraction=$uniform_fraction"
    else
        log "town=$town: total=$total chunk_size=$chunk uniform_fraction=$uniform_fraction"
    fi
    log "town=$town: output_dir=$output_dir"

    local town_failures=0
    local start=0
    while [ "$start" -lt "$total" ]; do
        local remaining=$((total - start))
        local n=$chunk
        if [ "$remaining" -lt "$n" ]; then
            n=$remaining
        fi
        local end=$((start + n))

        local attempt=1
        while true; do
            local logfile="$log_dir/chunk_${start}_${end}.attempt${attempt}.log"
            log "town=$town chunk [$start, $end) attempt $attempt/$MAX_CHUNK_RETRIES -> $logfile"

            if [ "$DRY_RUN" = "1" ]; then
                echo "    DRY_RUN: CARLA_QUALITY_LEVEL=low CARLA_TIMEOUT=$CARLA_TIMEOUT python -u -m $GENERATOR \\"
                echo "      --towns $town --start_rollout $start --num_rollouts $n \\"
                echo "      --pair_source junction_anchored \\"
                echo "      --junction_uniform_fraction $uniform_fraction \\"
                echo "      --output_dir $output_dir --offscreen --base_port $BASE_PORT $EXTRA_ARGS"
                break
            fi

            CARLA_QUALITY_LEVEL=low CARLA_TIMEOUT="$CARLA_TIMEOUT" python -u -m "$GENERATOR" \
                --towns "$town" --start_rollout "$start" --num_rollouts "$n" \
                --pair_source junction_anchored \
                --junction_uniform_fraction "$uniform_fraction" \
                --output_dir "$output_dir" --offscreen --base_port "$BASE_PORT" \
                $EXTRA_ARGS > "$logfile" 2>&1
            local status=$?

            if [ "$status" -eq 0 ]; then
                log "town=$town chunk [$start, $end) OK"
                break
            fi

            log "town=$town chunk [$start, $end) FAILED (exit $status), attempt $attempt/$MAX_CHUNK_RETRIES"
            reap_orphan_server

            if [ "$attempt" -ge "$MAX_CHUNK_RETRIES" ]; then
                log "town=$town chunk [$start, $end) giving up after $attempt attempts (exit $status), see $logfile" | tee -a "$failed_log"
                town_failures=$((town_failures + 1))
                break
            fi
            attempt=$((attempt + 1))
            log "  sleeping ${GAP_SECONDS}s before retry (lets the OS release the RPC port cleanly)"
            sleep "$GAP_SECONDS"
        done

        start=$end
        if [ "$start" -lt "$total" ] && [ "$DRY_RUN" != "1" ]; then
            log "  sleeping ${GAP_SECONDS}s before next chunk"
            sleep "$GAP_SECONDS"
        fi
    done

    local saved
    saved=$(find "$output_dir" -mindepth 1 -maxdepth 1 -type d ! -name 'run_logs' 2>/dev/null | wc -l)
    log "town=$town done: $saved trajectory dirs on disk, $town_failures chunk(s) never succeeded"
    echo "town=$town requested=$total saved=$saved failed_chunks=$town_failures uniform_fraction=$uniform_fraction" >> "$SUMMARY_LOG"

    return "$town_failures"
}

# ─────────────────────────── main ───────────────────────────
mkdir -p "$RUN_LOG_ROOT"

if ! preflight; then
    log "preflight failed; nothing was run."
    exit 1
fi

log "starting junction-anchored generation"
log "output_root=$OUTPUT_ROOT generator=$GENERATOR base_port=$BASE_PORT dry_run=$DRY_RUN"
: > "$SUMMARY_LOG"

total_failures=0
for spec in "${TOWN_SPECS[@]}"; do
    IFS=':' read -r town total chunk uf <<< "$spec"
    echo
    log "================ TOWN $town ================"
    run_town "$town" "$total" "$chunk" "$uf"
    total_failures=$((total_failures + $?))
    # Make sure this town's server is really gone before the next map loads.
    [ "$DRY_RUN" = "1" ] || reap_orphan_server
done

echo
log "all towns attempted."
cat "$SUMMARY_LOG"

if [ "$total_failures" -gt 0 ]; then
    log "$total_failures chunk(s) never succeeded - see $RUN_LOG_ROOT/town*/FAILED_CHUNKS.txt"
    exit 1
fi
log "all chunks succeeded."
