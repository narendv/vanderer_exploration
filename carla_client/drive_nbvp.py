"""
CARLA controller for the receding-horizon next-best-view (NBVP) baseline.

Port of Bircher et al., "Receding Horizon 'Next-Best-View' Planner for 3D
Exploration" (ICRA 2016), github.com/ethz-asl/nbvplanner, onto the BEV
occupancy grid the Yamauchi baseline builds -- see baselines/nbvp/README.md.
Runs in-process in the CARLA env like drive_yamauchi.py, whose depth
sensor, mount and trajectory.csv schema it imports rather than copies, so the
two baselines differ only in how they choose where to drive.

Execution follows upstream's interface_nbvp_rotors/src/exploration.cpp: call
the planner, drive the single edge it returns, call it again. There is no
periodic replanning -- that is the receding horizon. Differences from the
Yamauchi controller:

  * No creep fallback. When NBVP returns no plan the vehicle stops and the
    planner is retried after --replan_retry_steps (20 steps = 1s, the same
    1s wait exploration.cpp does on an empty path).
  * The run can end before --max_steps. `finished`: no gain anywhere and the
    backtracking history is used up -- where upstream calls ros::shutdown().
    `stuck`: --max_consecutive_failures empty plans in a row, vehicle
    stationary. metadata.json records end_reason either way. m^2/m for such
    a run is a short-run number; compare it with that in mind.
"""

import argparse
import csv
import json
import math
import os
import time
from collections import deque
from typing import Deque, List, Optional

import numpy as np
import gymnasium as gym
import carla

import envs

from common import (
    TargetWaypoint,
    attach_collision_sensor,
    control_to_action_array,
    disable_common_environment_objects,
    dist_2d,
    make_pid_controller,
    nearest_driving_spawn_from_transform,
    set_spectator_to_ego_camera,
    transform_to_dict,
)
# Same depth sensor, mount and CSV schema as the Yamauchi baseline.
from drive_yamauchi import (
    FRONT_SENSOR_TRANSFORM,
    TRAJECTORY_HEADER,
    fetch_latest_depth,
    spawn_depth_camera,
)

from baselines.nbvp.debug_viz import save_tree_png
from baselines.nbvp.explorer import NBVPExplorer
from baselines.yamauchi.grid_map import GridMap


def vertical_fov_deg(img_size, hfov_deg):
    """CARLA's `fov` is horizontal and its pixels are square (yamauchi/grid_map.py)."""
    width, height = img_size
    focal = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    return math.degrees(2.0 * math.atan(height / 2.0 / focal))


def parse_args():
    p = argparse.ArgumentParser(
        description="Receding-horizon next-best-view (NBVP) exploration baseline for CARLA"
    )
    p.add_argument("--town", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 10, 11, 12])
    p.add_argument("--seed", type=int, default=1,
                   help="Selects the spawn and seeds the planner's sampler")
    p.add_argument("--mode", type=str, default="nbvp",
                   help="Output subdirectory under --root_dir")
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--max_time", type=float, default=1e9)

    # Execution -- keep these matched to the runs being compared against.
    p.add_argument("--target_speed_kmh", type=float, default=3.6)
    p.add_argument("--waypoint_reach_thresh_m", type=float, default=1.0)
    p.add_argument("--replan_retry_steps", type=int, default=20,
                   help="Wait after an empty plan before asking again (20 steps = 1s, "
                        "exploration.cpp's wait)")
    p.add_argument("--max_consecutive_failures", type=int, default=50,
                   help="End the run after this many empty plans in a row. The vehicle "
                        "is stationary throughout, so continuing adds no area and no "
                        "path length -- only wall-clock")
    p.add_argument("--warmup_ticks", type=int, default=10)

    # Mapping -- identical to the Yamauchi baseline; see its README for evidence.
    p.add_argument("--resolution_m", type=float, default=0.25)
    p.add_argument("--map_margin_m", type=float, default=100.0)
    p.add_argument("--max_range_m", type=float, default=40.0)
    p.add_argument("--min_range_m", type=float, default=2.0,
                   help="Drops the ego bonnet from the depth frames; see "
                        "drive_yamauchi.py")
    p.add_argument("--obstacle_min_h", type=float, default=0.3)
    p.add_argument("--obstacle_max_h", type=float, default=3.0)
    p.add_argument("--ground_offset_m", type=float, default=0.0,
                   help="Added to the vehicle's z to get assumed road level")
    p.add_argument("--pixel_stride", type=int, default=2)
    p.add_argument("--integrate_every", type=int, default=1)
    p.add_argument("--min_obstacle_cells", type=int, default=4)

    # NBVP. Length parameters are upstream's exploration.yaml scaled by the
    # sensor-range ratio 40m / 5m = 8; see baselines/nbvp/README.md.
    p.add_argument("--extension_range_m", type=float, default=8.0,
                   help="nbvp/tree/extension_range (upstream 1.0m x 8)")
    p.add_argument("--gain_range_m", type=float, default=12.0,
                   help="nbvp/gain/range (upstream 1.5m x 8)")
    p.add_argument("--degressive_coeff", type=float, default=0.0625,
                   help="nbvp/gain/degressive_coeff, per metre (upstream 0.5 / 8)")
    p.add_argument("--zero_gain", type=float, default=0.0, help="nbvp/gain/zero")
    p.add_argument("--initial_iterations", type=int, default=15,
                   help="nbvp/tree/initial_iterations")
    p.add_argument("--cutoff_iterations", type=int, default=200,
                   help="nbvp/tree/cuttoff_iterations")
    p.add_argument("--inflation_radius_m", type=float, default=1.5,
                   help="Vehicle half-width plus margin; stands in for system/bbx. "
                        "Same value as the Yamauchi baseline")
    p.add_argument("--overshoot_m", type=float, default=1.0,
                   help="system/bbx/overshoot. With the 1.5m inflation disk this covers "
                        "the car's ~2.4m half-length past the edge endpoint")
    p.add_argument("--root_clearance_m", type=float, default=4.0,
                   help="Unobserved space this close to the vehicle counts as drivable. "
                        "Stands in for upstream's initialization motion")
    p.add_argument("--forward_cone_deg", type=float, default=90.0,
                   help="Only consider viewpoints within this angle of the vehicle's "
                        "heading, seen from the vehicle (90 = anywhere ahead). 0 = "
                        "upstream NBVP. Stops the planner asking for U-turns towards the "
                        "unknown space behind a forward-facing camera: on the Town01 "
                        "seed-1 smoke test, 90 cut collisions from 8 to 2")
    p.add_argument("--waypoint_spacing_m", type=float, default=2.0)

    # Goal tracking -- identical to the Yamauchi baseline.
    p.add_argument("--max_goal_steps", type=int, default=1200)
    p.add_argument("--progress_timeout_steps", type=int, default=200)
    p.add_argument("--progress_epsilon_m", type=float, default=0.5)

    # Plumbing
    p.add_argument("--offscreen", action="store_true")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_grid_every", type=int, default=0,
                   help="Dump a map + tree PNG every N steps (0 disables)")
    p.add_argument("--root_dir", type=str,
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
    p.add_argument("--trajectory_filename", type=str, default="trajectory.csv")
    p.add_argument("--metadata_filename", type=str, default="metadata.json")
    return p.parse_args()


def main():
    args = parse_args()
    save_dir = os.path.abspath(
        os.path.join(args.root_dir, args.mode, "recorded_trajs_town_%d_seed_%d" % (args.town, args.seed))
    )
    os.makedirs(save_dir, exist_ok=True)

    if args.offscreen:
        os.environ["CARLA_OFFSCREEN"] = "1"
    else:
        os.environ.pop("CARLA_OFFSCREEN", None)

    env = gym.make("carla-town%02d-v1" % args.town)
    base_env = env.unwrapped
    obs, info = env.reset(
        return_info=True, seed=args.seed, spawn_noise_std=0.0, action_exec_noise_std=0.0
    )
    disable_common_environment_objects(base_env.world)

    depth_camera, depth_buffer = spawn_depth_camera(
        base_env, base_env.vehicle, base_env.img_size, base_env.fov, FRONT_SENSOR_TRANSFORM
    )

    grid = GridMap.from_carla_map(
        base_env.world.get_map(),
        resolution_m=args.resolution_m,
        margin_m=args.map_margin_m,
        min_obstacle_cells=args.min_obstacle_cells,
    )
    hfov = float(base_env.fov)
    vfov = vertical_fov_deg(base_env.img_size, hfov)
    explorer = NBVPExplorer(
        grid,
        hfov_deg=hfov,
        vfov_deg=vfov,
        cam_height_m=float(FRONT_SENSOR_TRANSFORM.location.z) - args.ground_offset_m,
        cam_forward_m=float(FRONT_SENSOR_TRANSFORM.location.x),
        max_range_m=args.max_range_m,
        min_range_m=args.min_range_m,
        obstacle_band_m=(args.obstacle_min_h, args.obstacle_max_h),
        pixel_stride=args.pixel_stride,
        extension_range_m=args.extension_range_m,
        gain_range_m=args.gain_range_m,
        degressive_coeff=args.degressive_coeff,
        zero_gain=args.zero_gain,
        initial_iterations=args.initial_iterations,
        cutoff_iterations=args.cutoff_iterations,
        inflation_radius_m=args.inflation_radius_m,
        overshoot_m=args.overshoot_m,
        root_clearance_m=args.root_clearance_m,
        forward_cone_deg=args.forward_cone_deg,
        waypoint_spacing_m=args.waypoint_spacing_m,
        seed=args.seed,
    )
    print("[NBVP] grid %dx%d cells @ %.2fm, origin=(%.1f, %.1f), vfov=%.1fdeg"
          % (grid.nx, grid.ny, grid.res, grid.origin_x, grid.origin_y, vfov))

    pid_controller = make_pid_controller(base_env.vehicle, dt=base_env.dt)
    collision_events: List[carla.CollisionEvent] = []
    collision_sensor = attach_collision_sensor(base_env, collision_events)

    trajectory_fp = open(os.path.join(save_dir, args.trajectory_filename), "w",
                         newline="", encoding="utf-8")
    trajectory_writer = csv.writer(trajectory_fp)
    trajectory_writer.writerow(TRAJECTORY_HEADER)

    local_targets: Deque[carla.Location] = deque()
    current_target: Optional[carla.Location] = None
    plan_kind = None  # "track" or "backtrack" for the edge being driven
    episode_idx = 0
    step_idx = 0
    replans = 0
    backtracks = 0
    next_replan_step = 0
    consecutive_failures = 0
    steps_on_goal = 0
    steps_without_progress = 0
    best_goal_dist = float("inf")
    abandoned_goals = 0
    end_reason = "max_steps"
    replan_reasons = {}
    waypoint_generation_times = []
    episode_starts = [{
        "episode_idx": 0, "step_idx": 0,
        "spawn_transform": transform_to_dict(base_env.start_pos),
    }]

    set_spectator_to_ego_camera(base_env)
    start_time = time.time()

    # Warmup: stationary ticks so the depth sensor produces frames and the map
    # has a wedge of known free space for the first tree to grow into.
    for _ in range(max(1, args.warmup_ticks)):
        env.step(np.array([0.0, 0.0], dtype=np.float32))
    set_spectator_to_ego_camera(base_env)

    print("[NBVP] town=%d seed=%d img_size=%s fov=%s speed=%.2fkm/h extension=%.1fm "
          "gain_range=%.1fm forward_cone=%s"
          % (args.town, args.seed, base_env.img_size, base_env.fov,
             args.target_speed_kmh, args.extension_range_m, args.gain_range_m,
             ("%.0fdeg" % args.forward_cone_deg) if args.forward_cone_deg > 0 else "off"))

    try:
        while step_idx < args.max_steps:
            if (time.time() - start_time) >= args.max_time:
                end_reason = "max_time"
                break
            step_idx += 1

            # ── map update ───────────────────────────────────────────────
            depth, cam_tf = fetch_latest_depth(depth_buffer)
            veh_tf = base_env.vehicle.get_transform()
            ground_z = float(veh_tf.location.z) + args.ground_offset_m
            if depth is not None and cam_tf is not None and step_idx % args.integrate_every == 0:
                explorer.integrate(depth, cam_tf, ground_z)

            # ── edge progress ────────────────────────────────────────────
            pursuing = current_target is not None or bool(local_targets)
            if pursuing:
                steps_on_goal += 1
                if explorer.current_goal is not None:
                    veh_loc = base_env.vehicle.get_location()
                    d = float(np.hypot(veh_loc.x - explorer.current_goal[0],
                                       veh_loc.y - explorer.current_goal[1]))
                    if d < best_goal_dist - args.progress_epsilon_m:
                        best_goal_dist = d
                        steps_without_progress = 0
                    else:
                        steps_without_progress += 1

                stalled = steps_without_progress > args.progress_timeout_steps
                if stalled or steps_on_goal > args.max_goal_steps:
                    # Typically an edge whose heading change the car's turning
                    # radius cannot make. Replan from where it got to.
                    print("[NBVP] Abandoning %s edge (%s) after %d steps: %s"
                          % (plan_kind, "no progress" if stalled else "step cap",
                             steps_on_goal, explorer.current_goal))
                    explorer.abandon_goal()
                    abandoned_goals += 1
                    local_targets.clear()
                    current_target = None
                    next_replan_step = step_idx

            # ── plan: once per edge, when the last one is done ────────────
            out_of_targets = current_target is None and not local_targets
            if out_of_targets and step_idx >= next_replan_step:
                veh_tf = base_env.vehicle.get_transform()
                result = explorer.replan(
                    (float(veh_tf.location.x), float(veh_tf.location.y)),
                    math.radians(float(veh_tf.rotation.yaw)),
                )
                replan_reasons[result.reason] = replan_reasons.get(result.reason, 0) + 1
                waypoint_generation_times.append({
                    "step_idx": int(step_idx),
                    "latency_s": float(result.latency_s),
                    "reason": result.reason,
                    "n_nodes": int(result.n_nodes),
                    "best_gain_m2": float(result.best_gain),
                })

                if result.reason == "finished":
                    end_reason = "finished"
                    print("[NBVP] step=%d no gain anywhere and history exhausted -- "
                          "exploration finished (upstream: ros::shutdown)" % step_idx)
                    break

                if result.waypoints:
                    consecutive_failures = 0
                    z = float(veh_tf.location.z)
                    local_targets = deque(
                        carla.Location(x=float(wx), y=float(wy), z=z) for wx, wy in result.waypoints
                    )
                    current_target = local_targets.popleft()
                    plan_kind = "backtrack" if result.backtrack else "track"
                    steps_on_goal = 0
                    steps_without_progress = 0
                    best_goal_dist = float("inf")
                    replans += 1
                    backtracks += int(result.backtrack)
                    print("[NBVP] Plan #%d %s: goal=(%.1f, %.1f) gain=%.1fm2 nodes=%d "
                          "attempts=%d history=%d %.3fs"
                          % (replans, result.reason, result.goal[0], result.goal[1],
                             result.best_gain, result.n_nodes, result.n_attempts,
                             len(explorer.history), result.latency_s))
                else:
                    consecutive_failures += 1
                    next_replan_step = step_idx + args.replan_retry_steps
                    print("[NBVP] step=%d no plan (%s), nodes=%d, attempts=%d, failures=%d"
                          % (step_idx, result.reason, result.n_nodes, result.n_attempts,
                             consecutive_failures))
                    if consecutive_failures >= args.max_consecutive_failures:
                        end_reason = "stuck"
                        print("[NBVP] %d empty plans in a row; ending run." % consecutive_failures)
                        break

            if current_target is None and local_targets:
                current_target = local_targets.popleft()

            # ── act ──────────────────────────────────────────────────────
            if current_target is not None:
                control = pid_controller.run_step(
                    args.target_speed_kmh, TargetWaypoint(current_target)
                )
                action_arr = control_to_action_array(control)
            else:
                action_arr = np.array([0.0, 0.0], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action_arr)
            set_spectator_to_ego_camera(base_env)
            collision_detected = len(collision_events) > 0

            # ── record ───────────────────────────────────────────────────
            veh_tf = base_env.vehicle.get_transform()
            veh_loc, veh_rot = veh_tf.location, veh_tf.rotation
            veh_vel = base_env.vehicle.get_velocity()
            ctrl = base_env.vehicle.get_control()
            snapshot = base_env.world.get_snapshot()
            sim_time_s = (float(snapshot.timestamp.elapsed_seconds)
                          if snapshot is not None else float(step_idx * base_env.dt))
            mode = plan_kind if (current_target is not None or local_targets) else "idle"
            trajectory_writer.writerow([
                int(episode_idx), int(step_idx), int(info.get("frame", -1)), sim_time_s, mode,
                float(action_arr[0]), float(action_arr[1]),
                float(ctrl.throttle), float(ctrl.brake), float(ctrl.steer),
                float(veh_loc.x), float(veh_loc.y), float(veh_loc.z),
                float(veh_rot.pitch), float(veh_rot.yaw), float(veh_rot.roll),
                float(veh_vel.x), float(veh_vel.y), float(veh_vel.z),
                float(np.sqrt(veh_vel.x ** 2 + veh_vel.y ** 2 + veh_vel.z ** 2)),
                float(reward), int(collision_detected), 0,
                int(terminated), int(truncated),
                (float(current_target.x) if current_target is not None else ""),
                (float(current_target.y) if current_target is not None else ""),
                (float(current_target.z) if current_target is not None else ""),
            ])

            if current_target is not None:
                if dist_2d(base_env.vehicle.get_location(), current_target) < args.waypoint_reach_thresh_m:
                    current_target = None

            if args.save_grid_every and step_idx % args.save_grid_every == 0:
                save_tree_png(
                    os.path.join(save_dir, "grid_%06d.png" % step_idx),
                    explorer.grid,
                    tree=explorer.last_tree,
                    vehicle_xy=(float(veh_loc.x), float(veh_loc.y)),
                    goal_xy=explorer.current_goal,
                    history=explorer.history,
                )

            if step_idx % args.log_every == 0:
                st = explorer.grid.stats()
                print("[NBVP] step=%d ep=%d mode=%s queue=%d plans=%d backtracks=%d history=%d "
                      "free=%.0fm2 occ=%d x=%+.1f y=%+.1f"
                      % (step_idx, episode_idx, mode, len(local_targets), replans, backtracks,
                         len(explorer.history), st["free_cells"] * st["cell_area_m2"],
                         st["occ_cells"], veh_loc.x, veh_loc.y))

            # ── episode restart on collision ─────────────────────────────
            if collision_detected:
                print("[NBVP] Collision at step=%d. Resetting and continuing." % step_idx)
                explorer.abandon_goal()
                try:
                    if collision_sensor is not None:
                        collision_sensor.stop()
                        collision_sensor.destroy()
                        collision_sensor = None
                except Exception:
                    pass
                collision_events.clear()

                recovery_spawn = nearest_driving_spawn_from_transform(
                    base_env.world.get_map(), base_env.vehicle.get_transform()
                )
                obs, info = env.reset(
                    return_info=True, seed=args.seed, spawn_noise_std=0.0,
                    action_exec_noise_std=0.0, spawn_point=recovery_spawn,
                )
                # The env respawns the vehicle, so the depth sensor attached to
                # the old actor is gone with it.
                try:
                    depth_camera.stop()
                    depth_camera.destroy()
                except Exception:
                    pass
                depth_camera, depth_buffer = spawn_depth_camera(
                    base_env, base_env.vehicle, base_env.img_size, base_env.fov,
                    FRONT_SENSOR_TRANSFORM,
                )
                pid_controller = make_pid_controller(base_env.vehicle, dt=base_env.dt)
                collision_sensor = attach_collision_sensor(base_env, collision_events)

                episode_idx += 1
                episode_starts.append({
                    "episode_idx": int(episode_idx), "step_idx": int(step_idx),
                    "spawn_transform": transform_to_dict(base_env.start_pos),
                })
                # Map and history persist across respawns; only the root moves.
                local_targets.clear()
                current_target = None
                next_replan_step = step_idx
                for _ in range(max(1, args.warmup_ticks)):
                    env.step(np.array([0.0, 0.0], dtype=np.float32))

            if terminated or truncated:
                end_reason = "terminated" if terminated else "truncated"
                print("[NBVP] Episode %s at step=%d; stopping." % (end_reason, step_idx))
                break

    except KeyboardInterrupt:
        end_reason = "interrupted"
    finally:
        try:
            trajectory_fp.flush()
            trajectory_fp.close()
        except Exception:
            pass

        st = explorer.grid.stats()
        nodes = [w["n_nodes"] for w in waypoint_generation_times]
        metadata = {
            "town": int(args.town),
            "seed": int(args.seed),
            "mode": args.mode,
            "target_speed_kmh": float(args.target_speed_kmh),
            "waypoint_reach_thresh_m": float(args.waypoint_reach_thresh_m),
            "max_steps": int(args.max_steps),
            "dt_s": float(base_env.dt),
            "world_name": str(base_env.WORLD_NAME),
            "episode_starts": episode_starts,
            "waypoint_generation_times": waypoint_generation_times,
            "final_step_idx": int(step_idx),
            "end_reason": end_reason,
            "replans": int(replans),
            "backtracks": int(backtracks),
            "replan_reasons": replan_reasons,
            "abandoned_goals": int(abandoned_goals),
            "history_len": len(explorer.history),
            "tree_nodes_mean": float(np.mean(nodes)) if nodes else 0.0,
            "tree_nodes_max": int(max(nodes)) if nodes else 0,
            "map_stats": st,
            "nbvp_params": {
                "resolution_m": args.resolution_m,
                "max_range_m": args.max_range_m,
                "min_range_m": args.min_range_m,
                "obstacle_band_m": [args.obstacle_min_h, args.obstacle_max_h],
                "ground_offset_m": args.ground_offset_m,
                "min_obstacle_cells": args.min_obstacle_cells,
                "hfov_deg": hfov,
                "vfov_deg": vfov,
                "extension_range_m": args.extension_range_m,
                "gain_range_m": args.gain_range_m,
                "degressive_coeff": args.degressive_coeff,
                "zero_gain": args.zero_gain,
                "initial_iterations": args.initial_iterations,
                "cutoff_iterations": args.cutoff_iterations,
                "inflation_radius_m": args.inflation_radius_m,
                "overshoot_m": args.overshoot_m,
                "root_clearance_m": args.root_clearance_m,
                "forward_cone_deg": args.forward_cone_deg,
                "waypoint_spacing_m": args.waypoint_spacing_m,
                "replan_retry_steps": args.replan_retry_steps,
                "max_consecutive_failures": args.max_consecutive_failures,
                "max_goal_steps": args.max_goal_steps,
                "progress_timeout_steps": args.progress_timeout_steps,
                "progress_epsilon_m": args.progress_epsilon_m,
            },
        }
        with open(os.path.join(save_dir, args.metadata_filename), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        try:
            if collision_sensor is not None:
                collision_sensor.stop()
                collision_sensor.destroy()
        except Exception:
            pass
        try:
            depth_camera.stop()
            depth_camera.destroy()
        except Exception:
            pass
        env.close()
        print("[NBVP] Finished at step=%d (%s; %d plans, %d backtracks, %d episodes). Saved to %s"
              % (step_idx, end_reason, replans, backtracks, episode_idx + 1, save_dir))


if __name__ == "__main__":
    main()
