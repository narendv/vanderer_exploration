"""
Count left / straight / right decisions at real map junctions in the CARLA dataset.

Companion to get_dataset_heatmap.py, and written for the question that heatmap
cannot answer: *when a dataset trajectory actually reaches an intersection, how
often does it turn?* A coverage heatmap shows where the data went; it does not
separate "drove through the junction straight" from "took the branch", and on
the large maps (Town11 ~21 km across, Town12 ~9 km) a single top-down capture is
~10 m/pixel, so an intersection is not even one pixel.

This script needs no rendering. It queries the OpenDRIVE road network through
carla.Map.get_waypoint(), which is loaded in full even on large maps where only
the *visual* tiles stream in - so Town11/Town12 work here exactly as well as
Town01.

Method, per trajectory:

  1. Snap every recorded pose to the nearest driving waypoint and read
     waypoint.is_junction. Poses whose snap distance exceeds --max_snap_dist_m
     are treated as unmatched (see the z caveat below) and reported separately.
  2. Group contiguous in-junction poses into crossing events, merging two events
     separated by less than --merge_gap_m of path (complex intersections in
     CARLA often have short non-junction slivers inside them).
  3. For each event, take the chord heading over --approach_m of path before the
     entry pose and over --exit_m of path after the exit pose, and difference
     them. Chords rather than per-step yaw because the stored 4 Hz poses are
     noisy over a single step.
  4. Classify |delta| <= --straight_deg as straight, otherwise left or right.

Sign convention: CARLA's world is left-handed (x forward, y right, z up) and yaw
runs from +x toward +y, so a positive heading delta is a RIGHT turn. Stored
traj_data.pkl positions are right-handed, and are flipped to CARLA's frame on
load exactly as get_dataset_heatmap.py does it.

Caveat worth knowing before you read the Town11/Town12 numbers: traj_data.pkl
stores only (x, y) - there is no z - so every pose is snapped at z=0. On maps
with real elevation or stacked roadways that can pick the wrong layer. The
--max_snap_dist_m guard catches the gross cases and the summary prints what
fraction of poses it rejected; if that fraction is large for a town, treat that
town's split as unreliable rather than as a finding.

Usage (dataset is split per town, so point it at the root and it descends):

    conda activate vanderer-carla
    ./CarlaUE4.sh &            # any map; the script loads each town itself
    python -m data_generation.count_junction_turns \
        --dataset_dir /path/to/carla_dataset \
        --towns 1,6,7,10,11,12 \
        --csv /path/to/carla_dataset/junction_turns.csv
"""

import argparse
import csv
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import carla

from data_generation.get_dataset_heatmap import (
    TOWN_TO_WORLD,
    load_train_names,
    parse_towns,
    trajectory_length_m,
    trajectory_straightness,
)


# ─────────────────────────────── loading ───────────────────────────────


def iter_dataset_dirs(dataset_dir: Path) -> List[Path]:
    """
    Rollout-holding directories under dataset_dir.

    The dataset on disk is split per town (carla_town1/, carla_town6/, ...), but
    get_dataset_heatmap.py's loader expects a flat directory of rollouts. Accept
    either: if per-town subdirectories exist, descend into them, otherwise treat
    dataset_dir itself as flat.
    """
    town_dirs = sorted(p for p in dataset_dir.glob("carla_town*") if p.is_dir())
    return town_dirs if town_dirs else [dataset_dir]


def load_rollouts(
    dataset_dir: Path,
    wanted_towns: Optional[set],
    min_traj_frames: int = 0,
    min_traj_length_m: float = 0.0,
    min_straightness: float = 0.0,
    train_split_template: Optional[str] = None,
    limit_per_town: int = 0,
) -> Dict[int, List[Tuple[str, np.ndarray]]]:
    """
    Per town, a list of (rollout_name, positions) with positions in CARLA's
    left-handed world frame.

    Mirrors get_dataset_heatmap.load_rollouts, but keeps the rollout name so
    individual junction events stay traceable back to a directory.
    """
    by_town = defaultdict(list)  # type: Dict[int, List[Tuple[str, np.ndarray]]]
    skipped = defaultdict(int)  # type: Dict[str, int]
    train_names_by_town = {}  # type: Dict[int, Optional[set]]

    for root in iter_dataset_dirs(dataset_dir):
        for rollout_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            metadata_path = rollout_dir / "metadata.json"
            traj_path = rollout_dir / "traj_data.pkl"
            if not metadata_path.exists() or not traj_path.exists():
                continue

            try:
                with open(metadata_path, "r") as f:
                    town = int(json.load(f)["town"])
            except Exception as exc:
                print("[WARN] Skipping {}: bad metadata ({})".format(rollout_dir, exc))
                continue

            if wanted_towns is not None and town not in wanted_towns:
                continue
            if limit_per_town > 0 and len(by_town[town]) >= limit_per_town:
                continue

            if train_split_template is not None:
                if town not in train_names_by_town:
                    train_names_by_town[town] = load_train_names(train_split_template, town)
                train_names = train_names_by_town[town]
                if train_names is not None and rollout_dir.name not in train_names:
                    skipped["not_train"] += 1
                    continue

            try:
                with open(traj_path, "rb") as f:
                    positions_rh = np.asarray(pickle.load(f)["position"], dtype=np.float64)
            except Exception as exc:
                print("[WARN] Skipping {}: bad traj_data.pkl ({})".format(rollout_dir, exc))
                continue

            if positions_rh.ndim != 2 or positions_rh.shape[0] < 2 or positions_rh.shape[1] < 2:
                continue
            if min_traj_frames > 0 and positions_rh.shape[0] < min_traj_frames:
                skipped["few_frames"] += 1
                continue

            # Stored right-handed; CARLA's world is left-handed.
            positions = positions_rh[:, :2].copy()
            positions[:, 1] *= -1.0

            if min_traj_length_m > 0.0 and trajectory_length_m(positions) < min_traj_length_m:
                skipped["short"] += 1
                continue
            if min_straightness > 0.0 and trajectory_straightness(positions) < min_straightness:
                skipped["circular"] += 1
                continue

            by_town[town].append((rollout_dir.name, positions))

    for reason, count in sorted(skipped.items()):
        print("[INFO] Skipped {} trajectories ({})".format(count, reason))

    return by_town


# ─────────────────────────── junction geometry ───────────────────────────


class JunctionLookup:
    """
    Cached is_junction test over a spatial grid.

    Snapping every pose of every rollout is hundreds of thousands of
    get_waypoint() calls per town, and rollouts overlap heavily in space, so
    results are memoised on a --snap_grid_m grid. Junctions are tens of metres
    across, so sub-metre quantisation does not move an entry/exit boundary
    enough to matter once headings are measured metres outside the junction.
    """

    def __init__(self, carla_map: carla.Map, grid_m: float, max_snap_dist_m: float):
        self._map = carla_map
        self._grid = max(grid_m, 1e-3)
        self._max_snap = max_snap_dist_m
        self._cache = {}  # type: Dict[Tuple[int, int], Optional[bool]]
        self.hits = 0
        self.misses = 0

    def is_junction(self, x: float, y: float) -> Optional[bool]:
        """True/False in/out of a junction, or None when the pose cannot be snapped."""
        key = (int(round(x / self._grid)), int(round(y / self._grid)))
        if key in self._cache:
            self.hits += 1
            return self._cache[key]

        self.misses += 1
        waypoint = self._map.get_waypoint(
            carla.Location(x=float(x), y=float(y), z=0.0),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            result = None
        else:
            loc = waypoint.transform.location
            # Horizontal only: traj_data.pkl has no z, so every query is at z=0
            # and the vertical component of the snap distance is meaningless.
            snap_dist = math.hypot(loc.x - x, loc.y - y)
            result = bool(waypoint.is_junction) if snap_dist <= self._max_snap else None

        self._cache[key] = result
        return result


def cumulative_path_length(positions: np.ndarray) -> np.ndarray:
    """Path length from the first pose to each pose."""
    steps = np.hypot(np.diff(positions[:, 0]), np.diff(positions[:, 1]))
    return np.concatenate([[0.0], np.cumsum(steps)])


def find_junction_runs(flags: Sequence[Optional[bool]], arc: np.ndarray, merge_gap_m: float):
    """
    Contiguous in-junction index ranges, merging runs separated by a short gap.

    CARLA models some large intersections as several junction objects with
    non-junction road slivers between them; without merging, one crossing would
    be counted as two or three, each with a fraction of the true heading change.
    """
    runs = []  # type: List[List[int]]
    start = None
    for i, flag in enumerate(flags):
        if flag:
            if start is None:
                start = i
        elif start is not None:
            runs.append([start, i - 1])
            start = None
    if start is not None:
        runs.append([start, len(flags) - 1])

    if merge_gap_m <= 0.0 or len(runs) < 2:
        return runs

    merged = [runs[0]]
    for run in runs[1:]:
        if arc[run[0]] - arc[merged[-1][1]] < merge_gap_m:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return merged


def chord_backward(positions: np.ndarray, arc: np.ndarray, idx: int, span_m: float, min_idx: int):
    """Heading of the chord ending at idx, reaching back span_m of path but not past min_idx."""
    target = arc[idx] - span_m
    j = idx
    while j > min_idx and arc[j] > target:
        j -= 1
    if j == idx or arc[idx] - arc[j] < span_m * 0.5:
        return None
    return math.atan2(positions[idx, 1] - positions[j, 1], positions[idx, 0] - positions[j, 0])


def chord_forward(positions: np.ndarray, arc: np.ndarray, idx: int, span_m: float, max_idx: int):
    """Heading of the chord starting at idx, reaching forward span_m of path but not past max_idx."""
    target = arc[idx] + span_m
    k = idx
    while k < max_idx and arc[k] < target:
        k += 1
    if k == idx or arc[k] - arc[idx] < span_m * 0.5:
        return None
    return math.atan2(positions[k, 1] - positions[idx, 1], positions[k, 0] - positions[idx, 0])


def wrap_deg(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def analyse_trajectory(positions, lookup, args):
    """
    Junction crossing events for one trajectory.

    Returns (events, stats) where each event is a dict and stats counts poses
    that could not be snapped plus crossings dropped for lack of approach or
    exit margin.
    """
    arc = cumulative_path_length(positions)
    flags = [lookup.is_junction(positions[i, 0], positions[i, 1]) for i in range(positions.shape[0])]

    stats = {
        "poses": len(flags),
        "unmatched_poses": sum(1 for f in flags if f is None),
        "runs": 0,
        "truncated": 0,
        "clipped": 0,
    }

    last = len(arc) - 1
    events = []
    for i0, i1 in find_junction_runs(flags, arc, args.merge_gap_m):
        stats["runs"] += 1

        # These routes run spawn-point to spawn-point over 35-60 m, and CARLA
        # spawn points sit in and beside junctions, so most junction encounters
        # in this dataset are clipped by the start or the end of the rollout.
        # Requiring clean approach and exit road outside the junction would
        # throw away the large majority of them, so fall back to a short chord
        # measured *inside* the junction on whichever side is clipped.
        #
        # That fallback under-reads the turn, because part of the turn has
        # already happened inside the chord. The bias therefore runs toward
        # calling turns straight - the same direction as the hypothesis being
        # tested - so partial events are counted separately and never folded
        # into the headline split.
        heading_in = chord_backward(positions, arc, i0, args.approach_m, 0)
        partial_in = heading_in is None
        if partial_in:
            heading_in = chord_forward(positions, arc, i0, args.inside_chord_m, i1)

        heading_out = chord_forward(positions, arc, i1, args.exit_m, last)
        partial_out = heading_out is None
        if partial_out:
            heading_out = chord_backward(positions, arc, i1, args.inside_chord_m, i0)

        if heading_in is None or heading_out is None:
            stats["truncated"] += 1
            continue
        if partial_in and partial_out and (arc[i1] - arc[i0]) < 2.0 * args.inside_chord_m:
            # Both chords are inside the junction and would overlap, so the
            # difference between them measures nothing.
            stats["truncated"] += 1
            continue

        partial = partial_in or partial_out
        if partial:
            stats["clipped"] += 1

        delta = wrap_deg(math.degrees(heading_out - heading_in))
        if abs(delta) <= args.straight_deg:
            label = "straight"
        elif delta > 0.0:
            label = "right"  # CARLA is left-handed: +yaw goes from +x toward +y
        else:
            label = "left"

        events.append(
            {
                "entry_idx": i0,
                "exit_idx": i1,
                "entry_x": float(positions[i0, 0]),
                "entry_y": float(positions[i0, 1]),
                "junction_len_m": float(arc[i1] - arc[i0]),
                "delta_deg": float(delta),
                "label": label,
                "partial": int(partial),
            }
        )

    return events, stats


# ────────────────────────────────── main ──────────────────────────────────


CSV_FIELDS = [
    "town", "rollout", "entry_idx", "exit_idx", "entry_x", "entry_y",
    "junction_len_m", "delta_deg", "label", "partial",
]


def open_csv(path_str: str):
    """Open the per-event CSV for incremental writing, so a mid-run crash keeps what ran."""
    if not path_str:
        return None, None
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(str(path), "w")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
    writer.writeheader()
    return writer, handle


def print_town_row(row: Dict) -> None:
    """One line per town, printed the moment that town finishes."""
    print(
        "  town {}: {} clean crossings ({} straight / {} left / {} right, turn_frac {:.3f}); "
        "{} clipped ({} straight / {} left / {} right, turn_frac {:.3f}); "
        "{} unmeasurable; unmatched {:.1%}".format(
            row["town"],
            row["n_full"], row["full"]["straight"], row["full"]["left"], row["full"]["right"],
            row["turn_frac_full"],
            row["n_part"], row["part"]["straight"], row["part"]["left"], row["part"]["right"],
            row["turn_frac_part"],
            row["truncated"], row["unmatched_frac"],
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count left/straight/right decisions at map junctions in the CARLA dataset."
    )
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--towns", type=str, default="", help="Comma-separated town IDs. Default: all found.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=120.0, help="Large maps take a while to load.")
    parser.add_argument(
        "--straight_deg",
        type=float,
        default=30.0,
        help="Heading change at or below this is 'straight'.",
    )
    parser.add_argument("--approach_m", type=float, default=8.0, help="Chord length before junction entry.")
    parser.add_argument("--exit_m", type=float, default=8.0, help="Chord length after junction exit.")
    parser.add_argument(
        "--inside_chord_m",
        type=float,
        default=4.0,
        help="Fallback chord measured inside the junction when the rollout is clipped there. "
        "Short on purpose: it swallows part of the turn.",
    )
    parser.add_argument(
        "--merge_gap_m",
        type=float,
        default=6.0,
        help="Merge junction runs separated by less than this much path.",
    )
    parser.add_argument("--snap_grid_m", type=float, default=0.5, help="Spatial cache resolution for waypoint lookups.")
    parser.add_argument(
        "--max_snap_dist_m",
        type=float,
        default=6.0,
        help="Poses further than this from the nearest driving waypoint are unmatched.",
    )
    parser.add_argument("--min_traj_frames", type=int, default=0)
    parser.add_argument("--min_traj_length_m", type=float, default=0.0)
    parser.add_argument("--min_straightness", type=float, default=0.0)
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument(
        "--train_split_template",
        type=str,
        default="../train/vint_train/data/data_splits/carla_town{town}/train/traj_names.txt",
    )
    parser.add_argument("--limit_per_town", type=int, default=0, help="Cap rollouts per town (0 = all).")
    parser.add_argument("--csv", type=str, default="", help="Optional path to write every junction event.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError("dataset_dir not found: {}".format(dataset_dir))

    rollouts_by_town = load_rollouts(
        dataset_dir,
        parse_towns(args.towns),
        min_traj_frames=args.min_traj_frames,
        min_traj_length_m=args.min_traj_length_m,
        min_straightness=args.min_straightness,
        train_split_template=args.train_split_template if args.train_only else None,
        limit_per_town=args.limit_per_town,
    )
    if not rollouts_by_town:
        raise RuntimeError("No valid trajectories found for the requested towns.")

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    # CARLA segfaults with some regularity when one process loads several maps
    # in a row, so results are printed and flushed to CSV as each town finishes.
    # A crash then costs the town in flight, not the whole run, and the towns
    # already done can be read off the log. Rerun the survivors with --towns.
    csv_writer, csv_handle = open_csv(args.csv)
    summary = []
    failed = []

    for town in sorted(rollouts_by_town.keys()):
        world_name = TOWN_TO_WORLD.get(town)
        if world_name is None:
            print("[WARN] Skipping unsupported town={}".format(town))
            continue

        trajectories = rollouts_by_town[town]
        print("\n[INFO] town={} world={} rollouts={}".format(town, world_name, len(trajectories)))
        try:
            world = client.load_world(world_name)
            lookup = JunctionLookup(world.get_map(), args.snap_grid_m, args.max_snap_dist_m)
        except Exception as exc:
            print("[FAIL] town={}: could not load {} ({})".format(town, world_name, exc))
            print("       The simulator has most likely died; restart it and rerun with --towns {}".format(town))
            failed.append(town)
            continue

        full = {"straight": 0, "left": 0, "right": 0}
        part = {"straight": 0, "left": 0, "right": 0}
        totals = {"poses": 0, "unmatched_poses": 0, "runs": 0, "truncated": 0, "clipped": 0}
        rollouts_with_junction = 0

        for n, (name, positions) in enumerate(trajectories, 1):
            events, stats = analyse_trajectory(positions, lookup, args)
            for key in totals:
                totals[key] += stats[key]
            if events:
                rollouts_with_junction += 1
            for event in events:
                bucket = part if event["partial"] else full
                bucket[event["label"]] += 1
                if csv_writer is not None:
                    row = {"town": town, "rollout": name}
                    row.update(event)
                    csv_writer.writerow(row)

            if n % 250 == 0:
                print("  ... {}/{} rollouts".format(n, len(trajectories)))

        n_full = sum(full.values())
        n_part = sum(part.values())
        summary.append(
            {
                "town": town,
                "rollouts": len(trajectories),
                "rollouts_with_junction": rollouts_with_junction,
                "runs": totals["runs"],
                "truncated": totals["truncated"],
                "unmatched_frac": (
                    totals["unmatched_poses"] / float(totals["poses"]) if totals["poses"] else 0.0
                ),
                "full": full,
                "part": part,
                "n_full": n_full,
                "n_part": n_part,
                "turn_frac_full": (
                    (full["left"] + full["right"]) / float(n_full) if n_full else float("nan")
                ),
                "turn_frac_part": (
                    (part["left"] + part["right"]) / float(n_part) if n_part else float("nan")
                ),
            }
        )
        print(
            "  cache: {} hits / {} lookups, unmatched poses {:.1%}".format(
                lookup.hits, lookup.hits + lookup.misses, summary[-1]["unmatched_frac"]
            )
        )
        print_town_row(summary[-1])
        if csv_handle is not None:
            csv_handle.flush()

    header = "{:<7}{:>9}{:>11}{:>10}{:>9}{:>7}{:>7}{:>10}"
    print("\n" + "=" * 78)
    print(
        "CLEAN junction crossings - road before and after the junction "
        "(straight = |heading change| <= {:.0f} deg)".format(args.straight_deg)
    )
    print("=" * 78)
    print(header.format("town", "rollouts", "w/junction", "crossings", "straight", "left", "right", "turn_frac"))
    for row in summary:
        print(
            header.format(
                row["town"],
                row["rollouts"],
                "{:.1%}".format(row["rollouts_with_junction"] / float(row["rollouts"])),
                row["n_full"],
                row["full"]["straight"],
                row["full"]["left"],
                row["full"]["right"],
                "{:.3f}".format(row["turn_frac_full"]),
            )
        )

    print("\n" + "=" * 78)
    print("CLIPPED crossings - rollout began or ended inside the junction, one chord")
    print("measured inside it. Biased toward reading turns as straight; read as a floor.")
    print("=" * 78)
    print(header.format("town", "rollouts", "", "crossings", "straight", "left", "right", "turn_frac"))
    for row in summary:
        print(
            header.format(
                row["town"],
                row["rollouts"],
                "",
                row["n_part"],
                row["part"]["straight"],
                row["part"]["left"],
                row["part"]["right"],
                "{:.3f}".format(row["turn_frac_part"]),
            )
        )

    print("\n" + "=" * 78)
    print("Coverage - how much of each town's junction data is usable at all")
    print("=" * 78)
    cov = "{:<7}{:>10}{:>10}{:>10}{:>12}{:>12}"
    print(cov.format("town", "runs", "clean", "clipped", "unmeasurable", "unmatched"))
    for row in summary:
        print(
            cov.format(
                row["town"],
                row["runs"],
                row["n_full"],
                row["n_part"],
                row["truncated"],
                "{:.1%}".format(row["unmatched_frac"]),
            )
        )

    if csv_handle is not None:
        csv_handle.close()
        print("\n[DONE] Junction events written to {}".format(args.csv))

    if failed:
        print(
            "\n[FAIL] {} town(s) not measured: {}. Restart the simulator and rerun "
            "with --towns {}".format(len(failed), failed, ",".join(str(t) for t in failed))
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
