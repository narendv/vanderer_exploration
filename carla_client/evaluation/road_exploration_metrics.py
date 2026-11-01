"""Road-network exploration metrics for recorded CARLA trajectories.

The existing footprint metric (plot_bev_trajectory.compute_exploration_metrics)
counts 4 m grid cells crossed by the path. That number changes with the road's
heading relative to the grid and grows when a vehicle weaves or zigzags inside
the same stretch of road. This script measures exploration on the road network
instead, so neither of those affects it.

For every run:
  * The driven path is resampled every --sample_m metres (within each episode,
    so respawn jumps are not driven distance). A move longer than --max_jump_m
    between consecutive rows of the same episode is treated as a teleport: the
    path is split there and the jump is neither driven distance nor explored road.
  * Each sample is looked up on the town's OpenDRIVE map, built offline with
    carla.Map(name, xodr) -- no CARLA server needed:
      - not on a driving lane  -> off-road
      - inside a junction      -> that junction counts as visited. A junction is
                                  one segment of the network, whose length is the
                                  median length of the paths through it, so it is
                                  covered by driving through it in any direction
                                  (its turn paths all overlap, so counting each
                                  one would weight junctions several times over)
      - on a normal road       -> the road piece (road_id, floor(s / piece_m))
                                  counts as visited
    A road piece ignores the lane: both lanes and both directions of a road
    share the same pieces, so changing lane or driving back the other way is
    not new road.

Per-run metrics:
  path_m            distance driven (segment lengths summed within episodes)
  road_explored_m   distinct road pieces visited x piece_m, plus the length of
                    each junction driven through
  road_coverage     road_explored_m / the town's total road length (roads + junctions)
  road_efficiency   road_explored_m / path_m  (at most ~1; lower = more re-driving)
  junction_coverage junctions visited / junctions in the town (diagnostic; junctions
                    are already part of road_coverage)
  offroad_frac      share of path samples that are not inside any driving lane
  near_road_frac    with --lane_tolerance_m > 0: share of samples outside a driving
                    lane but within the tolerance of its edge (shoulder, parking
                    lane, verge). These still count toward road/junction exploration
                    but are included in offroad_frac.
  jumps             within-episode teleports split out (see --max_jump_m)
  collisions        episodes within the analysed window - 1 (the same rule as
                    aggregate_metrics.py's len(episode_starts) - 1, but restricted to
                    the rows actually scored, so a truncated run is not charged with
                    collisions from the part that was cut off)
  collisions_run    len(episode_starts) - 1 over the whole recorded run
  policy_calls      rows scored / replan_every_steps, or the run's recorded replan
                    count when the log has one
  grid_area_m2, grid_m2_per_m
                    the existing 4 m footprint metric, for side-by-side reference

Usage:
  python -m evaluation.road_exploration_metrics --max_step 100000 \
      --out ../outputs/metrics/road_metrics_100k.csv
"""

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys
from collections import defaultdict

import carla

from evaluation.plot_bev_trajectory import compute_exploration_metrics

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
XODR_DIR = os.path.join(os.getenv("CARLA_ROOT", ""), "CarlaUE4", "Content", "Carla", "Maps", "OpenDrive")
DRIVING = carla.LaneType.Driving

# "method=glob". The method name defaults to the run directory's parent folder,
# which is ambiguous here (NoMaD's runs also live in a folder called "random").
DEFAULT_RUNS = [
    "random=outputs/vanderer/random/recorded_trajs_town_*_seed_*",
    "guide=outputs/vanderer/guide/recorded_trajs_town_*_seed_*",
    "select=outputs/vanderer/select/recorded_trajs_town_*_seed_*",
    "nomad=outputs/nomad/random/recorded_trajs_town_*_seed_*",
    "nwm=outputs/nwm/select/recorded_trajs_town_*_seed_*",
    "rnd_dp=outputs/rnd/select/recorded_trajs_town_*_seed_*",
    "yamauchi=outputs/yamauchi/recorded_trajs_town_*_seed_*",
    "nbvp=outputs/nbvp/recorded_trajs_town_*_seed_*",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                   help="Run-directory globs, relative to the repo root or absolute, optionally "
                        "prefixed with a method name as method=glob.")
    p.add_argument("--max_step", type=int, default=-1,
                   help="Use only the first N trajectory rows (-1 = all), counted "
                        "the same way as plot_bev_trajectory --max_step.")
    p.add_argument("--piece_m", type=float, default=2.0,
                   help="Length of one road piece along the road, metres.")
    p.add_argument("--sample_m", type=float, default=0.5,
                   help="Spacing of path samples looked up on the map, metres.")
    p.add_argument("--lane_tolerance_m", type=float, default=0.0,
                   help="Credit samples up to this far outside a driving lane's edge "
                        "to that lane's road (0 = strict lane membership).")
    p.add_argument("--max_jump_m", type=float, default=2.0,
                   help="Split the path at a within-episode move longer than this between "
                        "consecutive rows (2 m per 0.05 s tick = 40 m/s).")
    p.add_argument("--grid_cell_m", type=float, default=4.0)
    p.add_argument("--grid_interp_m", type=float, default=0.5)
    p.add_argument("--curve_every_m", type=float, default=50.0,
                   help="Record road explored every this many metres driven.")
    p.add_argument("--out", type=str, default=None, help="Per-run results CSV.")
    p.add_argument("--curves_out", type=str, default=None,
                   help="JSON with road explored vs distance driven, per run.")
    p.add_argument("--include_incomplete", action="store_true",
                   help="Include runs shorter than the step budget in the summary.")
    return p.parse_args()


# --------------------------------------------------------------------------- map

_town_cache = {}


def town_model(world_name, piece_m):
    """carla.Map, the town's road pieces, and {junction id: length of one crossing}."""
    name = os.path.basename(world_name)
    key = (name, piece_m)
    if key not in _town_cache:
        with open(os.path.join(XODR_DIR, name + ".xodr"), "r", encoding="utf-8") as f:
            town = carla.Map(name, f.read())
        pieces = set()
        connecting = {}   # junction id -> {road id: length of that path through it}
        # Waypoints much denser than a piece, so every piece is hit.
        for wp in town.generate_waypoints(piece_m / 4.0):
            if wp.lane_type != DRIVING:
                continue
            if wp.is_junction:
                roads = connecting.setdefault(wp.junction_id, {})
                roads[wp.road_id] = max(roads.get(wp.road_id, 0.0), wp.s)
            else:
                pieces.add((wp.road_id, int(wp.s // piece_m)))
        # One length per junction: the paths through it overlap in space, so the
        # median path stands for the junction rather than the sum of all of them.
        junctions = {j: statistics.median(r.values()) for j, r in connecting.items()}
        _town_cache[key] = (town, pieces, junctions)
    return _town_cache[key]


# ----------------------------------------------------------------------- trajectory

def load_episodes(run_dir, max_step, max_jump_m=float("inf")):
    """Path pieces as lists of (x, y, z), plus rows used, teleports and episode count.

    Rows are grouped by episode, sorted by step_idx and truncated to the first
    max_step rows in episode order (same rule as plot_bev_trajectory). Each
    episode is split wherever consecutive rows are more than max_jump_m apart."""
    by_ep = defaultdict(list)
    with open(os.path.join(run_dir, "trajectory.csv"), newline="") as f:
        reader = csv.reader(f)
        h = next(reader)
        ie, isx, ix, iy, iz = (h.index(c) for c in ("episode_idx", "step_idx", "x", "y", "z"))
        for r in reader:
            by_ep[int(r[ie])].append((int(r[isx]), float(r[ix]), float(r[iy]), float(r[iz])))
    episodes, kept, jumps, n_eps = [], 0, 0, 0
    for ep in sorted(by_ep):
        rows = sorted(by_ep[ep], key=lambda t: t[0])
        if max_step > 0:
            if kept >= max_step:
                break
            rows = rows[:max_step - kept]
        kept += len(rows)
        n_eps += 1
        piece = []
        for _, x, y, z in rows:
            if piece and math.hypot(x - piece[-1][0], y - piece[-1][1]) > max_jump_m:
                episodes.append(piece)
                piece = []
                jumps += 1
            piece.append((x, y, z))
        if piece:
            episodes.append(piece)
    return episodes, kept, jumps, n_eps


def resample(points, step):
    """Points every `step` metres along a polyline (x, y, z), plus its length."""
    if not points:
        return [], 0.0
    out, total, carry = [points[0]], 0.0, 0.0
    for (x0, y0, z0), (x1, y1, z1) in zip(points, points[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        total += seg
        if seg <= 0.0:
            continue
        d = step - carry
        while d <= seg:
            t = d / seg
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, z0 + (z1 - z0) * t))
            d += step
        carry = seg - (d - step)
    return out, total


def run_metrics(run_dir, args, method=None):
    meta = json.load(open(os.path.join(run_dir, "metadata.json")))
    town, town_pieces, town_junctions = town_model(meta["world_name"], args.piece_m)  # junctions: id -> length
    episodes, rows_used, jumps, n_eps = load_episodes(run_dir, args.max_step, args.max_jump_m)

    pieces, junctions = set(), set()
    samples = offroad = near_road = 0
    path_m = 0.0
    driven = 0.0
    next_mark = args.curve_every_m
    curve_d, curve_r = [], []

    for pts in episodes:
        res, length = resample(pts, args.sample_m)
        path_m += length
        for i, (x, y, z) in enumerate(res):
            if i > 0:
                driven += args.sample_m
            samples += 1
            loc = carla.Location(x=x, y=y, z=z)
            wp = town.get_waypoint(loc, project_to_road=False, lane_type=DRIVING)
            if wp is None:
                offroad += 1
                if args.lane_tolerance_m > 0.0:
                    near = town.get_waypoint(loc, project_to_road=True, lane_type=DRIVING)
                    if near is not None:
                        nl = near.transform.location
                        # 2D distance to the lane centre; the height check keeps a
                        # sample from being credited to a road above or below it.
                        if (math.hypot(nl.x - x, nl.y - y) <= near.lane_width / 2.0 + args.lane_tolerance_m
                                and abs(nl.z - z) <= 3.0):
                            wp = near
                            near_road += 1
            if wp is not None:
                if wp.is_junction:
                    junctions.add(wp.junction_id)
                else:
                    pieces.add((wp.road_id, int(wp.s // args.piece_m)))
            while driven >= next_mark:
                curve_d.append(round(next_mark, 1))
                curve_r.append(len(pieces & town_pieces) * args.piece_m
                               + sum(town_junctions[j] for j in junctions & set(town_junctions)))
                next_mark += args.curve_every_m

    seen_junctions = junctions & set(town_junctions)
    junction_explored = sum(town_junctions[j] for j in seen_junctions)
    road_explored = len(pieces & town_pieces) * args.piece_m + junction_explored
    total_road = len(town_pieces) * args.piece_m + sum(town_junctions.values())

    grid_rows = {i: [{"x": x, "y": y} for x, y, _ in pts] for i, pts in enumerate(episodes)}
    grid = compute_exploration_metrics(grid_rows, -1, args.grid_cell_m, args.grid_interp_m)["aggregate"]

    budget = args.max_step if args.max_step > 0 else meta.get("max_steps")
    return {
        "method": method or os.path.basename(os.path.dirname(os.path.normpath(run_dir))),
        "town": int(meta.get("town", 0)),
        "seed": int(meta.get("seed", 0)),
        "world": os.path.basename(meta["world_name"]),
        "rows_used": rows_used,
        "complete": bool(budget) and rows_used >= budget,
        "path_m": path_m,
        "road_explored_m": road_explored,
        "town_road_m": total_road,
        "road_coverage": road_explored / total_road if total_road else 0.0,
        "road_efficiency": road_explored / path_m if path_m > 1e-9 else 0.0,
        "junctions_visited": len(seen_junctions),
        "town_junctions": len(town_junctions),
        "junction_road_m": junction_explored,
        "junction_coverage": len(seen_junctions) / float(len(town_junctions)) if town_junctions else 0.0,
        "offroad_frac": offroad / float(samples) if samples else 0.0,
        "near_road_frac": near_road / float(samples) if samples else 0.0,
        "lane_tolerance_m": args.lane_tolerance_m,
        "jumps": jumps,
        "collisions": max(0, n_eps - 1),
        "collisions_run": max(0, len(meta.get("episode_starts", [])) - 1),
        "policy_calls": (meta.get("replans") or len(meta.get("waypoint_generation_times", []))
                         or (rows_used / float(meta["replan_every_steps"]) if meta.get("replan_every_steps") else 0)),
        "grid_area_m2": grid["new_area_m2"],
        "grid_m2_per_m": grid["area_per_meter"],
        "run_dir": os.path.relpath(run_dir, REPO_ROOT),
        "_curve": {"distance_m": curve_d, "road_explored_m": curve_r},
    }


# -------------------------------------------------------------------------- report

COLS = ["method", "town", "seed", "world", "rows_used", "complete", "path_m", "road_explored_m",
        "town_road_m", "road_coverage", "road_efficiency", "junctions_visited", "town_junctions", "junction_road_m",
        "junction_coverage", "offroad_frac", "near_road_frac", "lane_tolerance_m", "jumps", "collisions", "collisions_run", "policy_calls", "grid_area_m2", "grid_m2_per_m", "run_dir"]


def mean_sd(vals):
    if not vals:
        return float("nan"), float("nan")
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def summarize(results, include_incomplete):
    use = [r for r in results if r["complete"] or include_incomplete]
    skipped = [r for r in results if r not in use]
    methods = sorted({r["method"] for r in use})
    towns = sorted({r["town"] for r in use})
    metrics = [("road_coverage", 100, "road cov %"), ("road_efficiency", 1, "road eff"),
               ("junction_coverage", 100, "junc cov %"), ("offroad_frac", 100, "offroad %"),
               ("collisions", 1, "collisions"), ("grid_m2_per_m", 1, "grid m2/m")]

    for key, scale, label in metrics:
        print("\n%s  (mean +/- sd over seeds)" % label)
        print("%-10s" % "town" + "".join("%20s" % m for m in methods))
        for t in towns + ["all"]:
            cells = []
            for m in methods:
                vals = [r[key] * scale for r in use
                        if r["method"] == m and (t == "all" or r["town"] == t)]
                mu, sd = mean_sd(vals)
                cells.append("%12.3f +/-%6.3f" % (mu, sd) if vals else "%20s" % "-")
            print("%-10s" % t + "".join("%20s" % c for c in cells))
    if skipped:
        print("\nExcluded from the summary (shorter than the step budget):")
        for r in skipped:
            print("  %s: %d rows" % (r["run_dir"], r["rows_used"]))


def main():
    args = parse_args()
    dirs = []
    for pattern in args.runs:
        method = None
        head, sep, tail = pattern.partition("=")
        if sep and "/" not in head:
            method, pattern = head, tail
        pattern = pattern if os.path.isabs(pattern) else os.path.join(REPO_ROOT, pattern)
        dirs += [(d, method) for d in sorted(glob.glob(pattern))
                 if os.path.isfile(os.path.join(d, "trajectory.csv"))
                 and os.path.isfile(os.path.join(d, "metadata.json"))]
    if not dirs:
        sys.exit("No run directories matched.")

    results = []
    for i, (d, method) in enumerate(dirs, 1):
        r = run_metrics(d, args, method)
        results.append(r)
        print("[%2d/%d] %-34s cov=%5.1f%% eff=%.3f junc=%5.1f%% offroad=%4.1f%% coll=%3d%s"
              % (i, len(dirs), "%s t%d s%d" % (r["method"], r["town"], r["seed"]),
                 100 * r["road_coverage"], r["road_efficiency"], 100 * r["junction_coverage"],
                 100 * r["offroad_frac"], r["collisions"], "" if r["complete"] else "  [INCOMPLETE]"),
              flush=True)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            for r in results:
                w.writerow({k: r[k] for k in COLS})
        print("\nPer-run results: %s" % args.out)
    if args.curves_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.curves_out)), exist_ok=True)
        with open(args.curves_out, "w") as f:
            json.dump({r["run_dir"]: dict(r["_curve"], method=r["method"], town=r["town"],
                                          seed=r["seed"], town_road_m=r["town_road_m"])
                       for r in results}, f)
        print("Curves: %s" % args.curves_out)

    summarize(results, args.include_incomplete)


if __name__ == "__main__":
    main()
