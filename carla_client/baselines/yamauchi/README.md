# Vanilla Yamauchi frontier exploration — CARLA baseline

Geometric frontier detection over a BEV occupancy grid with nearest-frontier goal selection. Yamauchi (1997), *A Frontier-Based Approach for
Autonomous Exploration*.

Pure numpy/scipy, so it runs in-process in the CARLA env — no ZMQ
server.

## Running

```bash
conda activate vanderer-carla
cd carla_client
python drive_yamauchi.py --town 1 --seed 1 --max_steps 100000 --offscreen
# or: ./run_experiment.sh --method yamauchi --town 1 --seed 1
```

Writes `<root_dir>/yamauchi/recorded_trajs_town_<T>_seed_<S>/` with the same
`trajectory.csv` and `metadata.json` schema as
`drive_policy.py`, so the existing pipeline scores it unchanged:

```bash
python -m evaluation.plot_bev_trajectory --save_dir <run_dir> --port 2000   # needs a server
mkdir -p <out>
python -m evaluation.aggregate_metrics 1 --input_dir <root_dir> --output_dir <out>
```

**Match `--max_steps` to the runs you compare against.** m²/m inflates badly on
short runs.

## Offline checks

```bash
python -m baselines.yamauchi.test_offline    # run from carla_client/, no simulator
```

Covers intrinsics, depth backprojection against a synthetic scene with a known
answer, frontier detection, and Dijkstra reachability/clearance.

## Design notes

**Ground-truth pose.** Poses come from `carla.Image.transform` — no SLAM, no
odometry. The baseline assumes perfect localization.

**Purely geometric.** No road-network prior. Unreachable frontiers (building
interiors, across walls) drop out on their own because Dijkstra over the free
grid never reaches them.

**Corrected intrinsics.** `envs/carla/v0.py` builds `camera_intrinsic` with
`fy = H/(2·tan(fov/2))` = 120 against `fx` = 160. CARLA's `fov` is the
*horizontal* FOV and its cameras have square pixels, so `fy` must equal `fx`
(see `examples/lidar_to_camera.py:120-125`). We compute K locally instead of
editing the shared attribute.
Measured ground-plane residual with the corrected K: mean +0.098 m, std
0.184 m across the image.

**Ego bonnet (`min_range_m` = 2.0).** The front camera
(x=1.0, z=1.7) sees the car's own bonnet across the bottom of the image, about
±33° of bearing. The bonnet sits 1.0–1.4 m ahead of the camera, inside the
obstacle height band. The per-bearing scan keeps the nearest obstacle, so with
the old `min_range_m` = 1.0 free space stopped at the bonnet across the central
~±26° of *every* frame, leaving only the two side slivers to map free space.
The ground isn't visible nearer than 2.27 m, so 2.0 loses no road.

**Forward cone (`--forward_cone_deg` = 90).**
- Only frontier cells within 90° of the vehicle's heading, seen from the
  vehicle, are candidate goals.
- The forward camera leaves the space behind the car unexplored, so the
  nearest frontier is often behind it, and reaching it means a U-turn most
  roads are too narrow for. The same cone on the NBVP baseline cut collisions
  from 8 to 2 on Town01 seed 1.
- If no frontier is reachable inside the cone, the planner falls back to all
  frontiers instead of stalling; Yamauchi has no other way out of a dead end.
  Those replans log `BEHIND`, and `metadata.json` counts them as
  `behind_fallbacks`.
- This is a deviation from Yamauchi (1997); `--forward_cone_deg 0` gives the
  vanilla rule.

## Parameters that were tuned against measurements

These are not guesses; each fixed an observed failure. Town01 seed 1, 4000 steps.

| Parameter | Value | Evidence |
| --- | --- | --- |
| `min_goal_distance_m` | 8.0 | With a 90° forward camera, space beside and behind the vehicle is never observed, so free cells 2 m away permanently border unknown and are permanently "nearest". Without this the vehicle locks onto a goal it is standing on and never travels. |
| `min_obstacle_cells` | 4 | Road grade puts distant ground returns above a flat-world height band, producing isolated obstacle specks mid-road — 54 single-cell blobs, 8% of occupied cells. Each gets inflated into a disk of the vehicle's half-width, and two are enough to seal a road. Dropping them took the reachable area around a stranded vehicle from 0 to 464 m². |
| `resolution_m` | 0.25 | At 0.5 m the inflation disk is 1–3 cells, so one stray cell blocks 3 m of road. |
| `inflation_radius_m` | 1.5 | 1.0 m (roughly the vehicle half-width) is not enough — the PID cuts corners at junctions. 1.0 m → 7 collisions, 1.5 m → 2. |
| `obstacle_min_h` | 0.3 | 0.5 m leaves kerbs invisible, so the map reads far pavement as drivable. |
| `min_frontier_cells` | 20 | Occlusion shadows behind fences produce slivers no car can enter. 20 cells at 0.25 m = 5 m, about a lane width. |
| `progress_timeout_steps` | 200 | `explore_lite`'s `progress_timeout`. Abandons and blacklists a goal the vehicle cannot close on. |
| `replan_retry_steps` | 20 | A failed replan used to re-run Dijkstra every single step for hundreds of steps. |

The fallback creep (when no frontier is reachable) is gated on
`GridMap.is_clear`. Creeping blind drove into walls already in the map and was
a large share of early collisions.
