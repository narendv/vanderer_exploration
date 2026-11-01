# Receding-horizon NBVP — CARLA baseline

Port of the receding-horizon next-best-view planner — Bircher et al.,
*Receding Horizon "Next-Best-View" Planner for 3D Exploration*, ICRA 2016,
[ethz-asl/nbvplanner](https://github.com/ethz-asl/nbvplanner) — onto the
Yamauchi baseline's BEV occupancy grid.

It uses the same map, depth sensor, obstacle inflation, PID and collision
respawn as `baselines/yamauchi`; the only difference is how it picks where to
drive. It's pure numpy/scipy and runs in-process in the CARLA env,
with no ROS, octomap or Gazebo. It imports `yamauchi` as a sibling package.

## Running

```bash
conda activate vanderer-carla
cd carla_client
./run_experiment.sh --method nbvp --town 1 --seed 1
# or: python drive_nbvp.py --town 1 --seed 1 --max_steps 100000 --offscreen
# batch: METHODS=nbvp ./batch_run.sh   (resumable)
```

This writes `<root_dir>/nbvp/recorded_trajs_town_<T>_seed_<S>/` with the same
`trajectory.csv` as the other baselines, so `plot_bev_trajectory.py`
and `aggregate_metrics.py` score it unchanged.
- The `mode` column holds `track`, `backtrack` or `idle`.
- `metadata.json` adds `end_reason`, `backtracks`, `replan_reasons`, tree sizes
  and `nbvp_params`.
- `--seed` also seeds the planner's sampler, so a run is reproducible.

## What was ported

| Upstream | Here |
| --- | --- |
| `rrt.cpp` `iterate()` — sample, nearest node, extend, check, score | `tree.py` `RrtTree.iterate` / `_extend` |
| `rrt.cpp` `gain()` — unmapped volume in the frustum | `views.py` `ViewModel.gain` |
| `volumetric_mapping` `getLineStatusBoundingBox` | `collision.py` `EdgeChecker.edge_clear` |
| `rrt.cpp` `initialize()` / `memorizeBestBranch()` / `getBestEdge()` | `tree.py` `insert_branch` / `best_branch_states` / `best_first_edge` |
| `rrt.cpp` `getPathBackToPrevious()`, `history_` | `explorer.py` `_backtrack`, `history` |
| `nbvp.hpp` `plannerCallback()` stopping rules | `explorer.py` `NBVPExplorer.replan` |
| `interface_nbvp_rotors/src/exploration.cpp` plan-drive-plan loop | `drive_nbvp.py` |

Dropped:
- the ROS/tf/rviz plumbing
- the mesh-inspection gain (`gain/area`, which is 0 in `exploration.yaml`)
- multi-agent segment checks
- logging

## Deviations from upstream

| Upstream | Here | Why |
| --- | --- | --- |
| 3D octomap; state (x, y, z, yaw) | 2D BEV grid; state (x, y, yaw) | Ground vehicle. Sharing Yamauchi's map means only the planner differs between the two baselines. |
| Random yaw sampled per node | Yaw = heading of the edge into the node | A multirotor can point its camera anywhere; the car's camera faces forward. |
| Gain = unmapped voxel volume inside the frustum, occluded only by occupied voxels | Gain = UNKNOWN cell area in the ±45° wedge, from the ground-visibility range (2.27 m: 1.7 m camera, 73.7° vertical FOV) out to `gain_range_m`; any OCC cell occludes | 2D reduction of the frustum. Occlusion is conservative: a low wall blocks the ray, though the real camera sees over it. |
| Bounding box swept along the edge, plus overshoot | Centreline checked against obstacles inflated by 1.5 m, plus overshoot | Same passable mask as Yamauchi. Unknown blocks, as upstream. |
| Scripted initialization motion (spin, back up) | `root_clearance_m`: unobserved non-OCC space within 4 m of the vehicle may be *crossed*; nodes must still land in known space | A car can't spin in place, and its own footprint is never observed. |
| No gain → `ros::shutdown()` | No gain → backtrack one history entry; `finished` only when the history is empty | Option (a). Upstream already backtracks when sampling stalls. |
| `exact_root: true` (root = planned edge end) | Root = measured vehicle pose | The PID doesn't land exactly on the edge end, and collisions respawn the car elsewhere. |
| — | Stalled edge (200 steps without progress) or collision → forget the remembered branch, replan | A multirotor's tracker doesn't fail; the car's turning radius can. |
| Sampling stalls before 15 nodes → backtrack, even with gain found | A stalled tree that has found gain drives its best edge (reason `ok_capped`); it backtracks only with zero gain | At spawn the mapped road fits only a few 8 m edges. Upstream's rule stranded the car at spawn (Town01 seed 2), and its backtracks are U-turns (11 of seed 1's 35 collisions). |
| Viewpoints anywhere around the robot | Viewpoints only within 90° of the vehicle's heading (`--forward_cone_deg`, see below) | A forward camera leaves the space behind unknown, so free-yaw NBVP keeps asking for U-turns the car can't make. |
| Retries an empty plan forever | Run ends after 50 empty plans in a row (`end_reason: stuck`) | The car is stationary throughout, so no area or path is lost — only wall-clock. |

### Forward cone (`--forward_cone_deg`, controller default 90; 0 = upstream)

The controller defaults to 90. The `NBVPExplorer` class
itself defaults to 0, so the library and its offline tests stay upstream NBVP.
The Yamauchi baseline has the same cone on its frontier goals.

With the cone set, a viewpoint only enters the tree if it lies within that
angle of the vehicle's heading, seen from the vehicle. The executed first edge
ends at such a viewpoint, so the car is never sent behind itself. The cone is
re-applied from the new pose at every replan, so it limits each step, not the
route.

It exists because the forward camera leaves the space behind the car unknown.
Free-yaw NBVP then keeps choosing viewpoints behind the car, and the car
can't make those U-turns. It is a variant, not upstream NBVP;
report it as such.

## Parameters

Length parameters are upstream's `exploration.yaml` scaled by the ratio of
sensor ranges: the upstream demo's `sensor_max_range` is 5 m
(`flat_exploration.launch`) and ours is `max_range_m` = 40 m, so ×8. Counts are
unchanged. Vehicle-size parameters are physical and shared with Yamauchi.

| Parameter | Upstream | Here | Basis |
| --- | --- | --- | --- |
| `extension_range_m` | 1.0 | 8.0 | ×8 |
| `gain_range_m` | 1.5 | 12.0 | ×8 |
| `degressive_coeff` (1/m) | 0.5 | 0.0625 | ÷8 |
| `initial_iterations` | 15 | 15 | count |
| `cutoff_iterations` | 200 | 200 | count |
| `zero_gain` | 0 | 0 | — |
| bounding box → `inflation_radius_m` | 0.5×0.5×0.3 m | 1.5 | Yamauchi's measured value (1.0 m → 7 collisions, 1.5 m → 2) |
| `overshoot_m` | 0.5 | 1.0 | The 1.5 m inflation disk plus 1 m covers the car's ~2.4 m half-length |
| `root_clearance_m` | — | 4.0 | Yamauchi's start-cell snap radius |

**These values are derived, not tuned.**

## What to expect

- **Early termination.** NBVP plans locally. In a whole town it may exhaust its
  gain and history before `--max_steps`. m²/m inflates on short runs, so check
  `end_reason` and `final_step_idx`, and report total mapped area alongside m²/m.
- **Replanning cadence.** NBVP replans once per edge (≤ 8 m, about 160 steps at
  3.6 km/h); Yamauchi replans every 20–40 steps. `waypoint_generation_times`
  counts planner calls, so `aggregate_metrics.py`'s collision percentage uses
  a different denominator for the two baselines.
- **Turning radius.** Edges are straight, and consecutive edges can need a
  sharper turn than the car can make; the PID cuts the corner. If collisions
  come out high, the fix is to cap the heading change between consecutive
  edges — one more deviation to document.
- **Backtrack paths are not collision-checked** (as upstream). They retrace edges
  the car has already driven.

## Offline checks

```bash
python -m baselines.nbvp.test_offline    # run from carla_client/, no simulator
```

These cover:
- gain geometry: wedge area, vertical-FOV cut, occlusion
- the collision check: walls, inflation, overshoot, unknown, root clearance
- tree growth and branch re-insertion
- each stopping rule: `finished`, backtracking, stalled sampler
- latency on a 2400×2200-cell town-sized grid

Measured:
- 0.27 s for a typical replan
- 0.31 s for the worst case (runs to cutoff)
- 0.5 ms of gain computation per node
