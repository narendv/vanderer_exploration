"""
CARLA controller for the vanilla Yamauchi frontier-exploration baseline.

Runs entirely in the CARLA env: there is no policy server and no ZMQ hop, because
the classical pipeline (baselines/yamauchi/) is pure numpy/scipy and imports in-process.

Execution model deliberately mirrors drive_policy.py -- PID waypoint
following, collision sensor, episode restart on collision, and the same
28-column trajectory.csv -- so the resulting m^2/m is directly comparable to
the method's own runs under evaluation/plot_bev_trajectory.py. Helpers are shared
through common.py.

Two deviations from the NoMaD controller, both deliberate:

  * No BasicAgent warmup. NoMaD drives ~10m along the road with BasicAgent to
    fill its context-image buffer. BasicAgent follows the road network, which
    would inject exactly the map prior this baseline is defined not to have.
    The Yamauchi planner needs no context buffer -- one integrated depth frame
    already yields a frontier ahead -- so warmup is just a few stationary
    ticks to let the sensors produce their first frames.

  * Ground-truth pose (see baselines/yamauchi/grid_map.py). No SLAM; the baseline assumes perfect localization.
"""

import argparse
import csv
import json
import os
import time
from collections import deque
from typing import Deque, List, Optional

import numpy as np
import gymnasium as gym
import carla

import envs

# Shared with the policy driver, so all methods drive, collide, respawn and record identically.
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

from baselines.yamauchi.debug_viz import save_grid_png
from baselines.yamauchi.explorer import YamauchiExplorer
from baselines.yamauchi.frontier import detect

# Matches envs/carla/v0.py's _spawn_camera FRONT offset so the depth sensor
# shares an optical centre with the RGB camera the other baselines use.
FRONT_SENSOR_TRANSFORM = carla.Transform(carla.Location(x=1.0, z=1.7))


def decode_depth(image: carla.Image) -> np.ndarray:
    """CARLA's depth camera encodes metric depth (0-1000m) across RGB."""
    image.convert(carla.ColorConverter.Raw)
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4)).astype(np.float32)
    b, g, r = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]  # raw_data is BGRA
    normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 ** 3 - 1.0)
    return (1000.0 * normalized).astype(np.float32)


def spawn_depth_camera(base_env, vehicle, size, fov, transform):
    bp = base_env.world.get_blueprint_library().find("sensor.camera.depth")
    bp.set_attribute("image_size_x", str(size[0]))
    bp.set_attribute("image_size_y", str(size[1]))
    bp.set_attribute("fov", str(fov))
    bp.set_attribute("sensor_tick", str(base_env.dt))
    camera = base_env.world.spawn_actor(bp, transform, attach_to=vehicle)
    buffer = deque(maxlen=base_env.SENSOR_BUFFER_MAXLEN)
    camera.listen(lambda data: buffer.append(data))
    return camera, buffer


def fetch_latest_depth(buffer: deque):
    """Drain the buffer, returning (depth, sensor transform at capture)."""
    latest = None
    while buffer:
        latest = buffer.popleft()
    if latest is None:
        return None, None
    return decode_depth(latest), latest.transform


def run_depth_check(depth, cam_transform, ground_z, hfov_deg, out_path):
    """Verify the backprojection puts the road where it belongs.

    With envs/carla/v0.py's fy=120 the ground plane bows by ~1.3x with image
    row, so a flat road reads as sloped. Printing the spread of ground-height
    residuals across the image is the cheapest way to confirm the corrected
    fx == fy intrinsics before committing to a full run.
    """
    from baselines.yamauchi.grid_map import intrinsics_from_fov

    H, W = depth.shape
    K = intrinsics_from_fov(W, H, hfov_deg)
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]

    vs, us = np.mgrid[0:H, 0:W].astype(np.float64)
    d = depth.ravel()
    us, vs = us.ravel(), vs.ravel()
    keep = (d > 1.0) & (d < 40.0)
    d, us, vs = d[keep], us[keep], vs[keep]

    pts_cam = np.stack(
        [d, (us - cx) * d / f, -(vs - cy) * d / f, np.ones_like(d)], axis=0
    )
    pts_w = np.asarray(cam_transform.get_matrix(), dtype=np.float64).dot(pts_cam)
    wz = pts_w[2]

    # Points near road level: their world z should cluster tightly at ground_z.
    near_ground = np.abs(wz - ground_z) < 1.0
    print("[YAMAUCHI][depth_check] valid points: %d" % d.size)
    if np.any(near_ground):
        resid = wz[near_ground] - ground_z
        print(
            "[YAMAUCHI][depth_check] ground residual: mean=%+.3fm std=%.3fm "
            "p5=%+.3f p95=%+.3f (want std well under 0.30m for a flat road)"
            % (
                resid.mean(),
                resid.std(),
                np.percentile(resid, 5),
                np.percentile(resid, 95),
            )
        )
    else:
        print("[YAMAUCHI][depth_check] no returns near ground level")

    np.savez_compressed(
        out_path, points=pts_w[:3].T.astype(np.float32), ground_z=float(ground_z)
    )
    print("[YAMAUCHI][depth_check] point cloud saved: %s" % out_path)


def parse_args():
    p = argparse.ArgumentParser(
        description="Vanilla Yamauchi nearest-frontier exploration baseline for CARLA"
    )
    p.add_argument("--town", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 10, 11, 12])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--mode", type=str, default="yamauchi",
                   help="Output subdirectory under --root_dir")
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--max_time", type=float, default=1e9)

    # Execution -- keep these matched to the runs being compared against.
    p.add_argument("--target_speed_kmh", type=float, default=3.6)
    p.add_argument("--waypoint_reach_thresh_m", type=float, default=1.0)
    p.add_argument("--replan_every_steps", type=int, default=40)
    p.add_argument("--replan_retry_steps", type=int, default=20,
                   help="Steps to wait before retrying after a failed replan. "
                        "Without it an unreachable-frontier state re-runs Dijkstra "
                        "every single step for hundreds of steps.")
    p.add_argument("--warmup_ticks", type=int, default=10)

    # Mapping
    p.add_argument("--resolution_m", type=float, default=0.25,
                   help="0.5m makes the inflation disk brutally quantised: a 1.5m "
                        "radius is 3 cells, so one stray obstacle cell blocks 3m of "
                        "road. 0.25m keeps the same clearance with finer granularity.")
    p.add_argument("--map_margin_m", type=float, default=100.0)
    p.add_argument("--max_range_m", type=float, default=40.0)
    p.add_argument("--min_range_m", type=float, default=2.0,
                   help="Planar depth below which returns are dropped. The ego bonnet sits "
                        "1.0-1.4m ahead of the camera, inside the obstacle band; at 1.0 it "
                        "capped free space at the bonnet across the central ~+-26deg of "
                        "every frame. The ground is not visible nearer than 2.27m, so 2.0 "
                        "loses no road. Runs before 2026-09-10 used 1.0.")
    p.add_argument("--obstacle_min_h", type=float, default=0.3,
                   help="Returns this far above road level count as obstacles. "
                        "0.5 leaves kerbs and low barriers invisible, so the map "
                        "reads the far pavement as drivable and the vehicle plans "
                        "straight over the kerb.")
    p.add_argument("--obstacle_max_h", type=float, default=3.0)
    p.add_argument("--ground_offset_m", type=float, default=0.0,
                   help="Added to the vehicle's z to get assumed road level")
    p.add_argument("--pixel_stride", type=int, default=2)
    p.add_argument("--integrate_every", type=int, default=1)

    # Frontier / planning
    p.add_argument("--min_obstacle_cells", type=int, default=4,
                   help="Drop obstacle blobs smaller than this (4 cells at 0.25m is "
                        "the area of one 0.5m cell). Removes grade-induced specks "
                        "that would otherwise be inflated into road blockages.")
    p.add_argument("--min_frontier_cells", type=int, default=20,
                   help="Minimum frontier cluster size. Occlusion shadows behind "
                        "fences and building corners produce many 1-3 cell slivers "
                        "that no car can ever drive into; 20 cells at 0.25m (5m of "
                        "frontier) keeps real road openings, which are a lane-width "
                        "or wider.")
    p.add_argument("--inflation_radius_m", type=float, default=1.5,
                   help="Vehicle half-width plus margin. CARLA cars are ~2m wide, "
                        "but 1.0m clearance is not enough: the PID cuts corners at "
                        "junctions and clips buildings. Measured over 4000 steps in "
                        "Town01: 1.0m gave 7 collisions, 1.5m gave 2.")
    p.add_argument("--waypoint_spacing_m", type=float, default=2.0)
    p.add_argument("--min_goal_distance_m", type=float, default=8.0)
    p.add_argument("--forward_cone_deg", type=float, default=90.0,
                   help="Only frontier cells within this angle of the vehicle's heading, "
                        "seen from the vehicle, are candidates (90 = anywhere ahead). If "
                        "none is reachable, fall back to all frontiers. 0 = vanilla "
                        "Yamauchi. The forward camera leaves the space behind unexplored, "
                        "so the nearest frontier is often behind the car and needs a "
                        "U-turn. Runs before 2026-09-10 used 0.")
    p.add_argument("--creep_lookahead_m", type=float, default=6.0,
                   help="Clearance the fallback creep requires before moving")
    p.add_argument("--blacklist_radius_m", type=float, default=6.0)
    p.add_argument("--max_goal_steps", type=int, default=1200,
                   help="Hard cap: abandon and blacklist a goal not reached within N steps")
    p.add_argument("--progress_timeout_steps", type=int, default=200,
                   help="Abandon a goal after N steps without closing distance to it. "
                        "This is explore_lite's progress_timeout; without it the "
                        "vehicle grinds against goals its turning radius cannot reach.")
    p.add_argument("--progress_epsilon_m", type=float, default=0.5,
                   help="Distance reduction that counts as progress")

    # Plumbing
    p.add_argument("--offscreen", action="store_true")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_grid_every", type=int, default=0,
                   help="Dump an occupancy-grid PNG every N steps (0 disables)")
    p.add_argument("--stuck_dump", action="store_true",
                   help="On the first run of consecutive failed replans, dump the "
                        "log-odds grid and vehicle pose for offline diagnosis")
    p.add_argument("--depth_check", action="store_true",
                   help="Dump a backprojected point cloud on the first frame and exit")
    p.add_argument("--root_dir", type=str,
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
    p.add_argument("--trajectory_filename", type=str, default="trajectory.csv")
    p.add_argument("--metadata_filename", type=str, default="metadata.json")
    return p.parse_args()


TRAJECTORY_HEADER = [
    "episode_idx", "step_idx", "frame", "sim_time_s", "mode",
    "action_signed_throttle", "action_steer",
    "applied_throttle", "applied_brake", "applied_steer",
    "x", "y", "z", "pitch", "yaw", "roll",
    "vx", "vy", "vz", "speed_mps",
    "reward", "collision", "lhc_flag", "terminated", "truncated",
    "target_x", "target_y", "target_z",
]


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

    explorer = YamauchiExplorer(
        carla_map=base_env.world.get_map(),
        resolution_m=args.resolution_m,
        margin_m=args.map_margin_m,
        hfov_deg=float(base_env.fov),
        max_range_m=args.max_range_m,
        min_range_m=args.min_range_m,
        obstacle_band_m=(args.obstacle_min_h, args.obstacle_max_h),
        pixel_stride=args.pixel_stride,
        min_frontier_cells=args.min_frontier_cells,
        inflation_radius_m=args.inflation_radius_m,
        waypoint_spacing_m=args.waypoint_spacing_m,
        blacklist_radius_m=args.blacklist_radius_m,
        min_goal_distance_m=args.min_goal_distance_m,
        min_obstacle_cells=args.min_obstacle_cells,
        forward_cone_deg=args.forward_cone_deg,
    )
    print("[YAMAUCHI] grid %dx%d cells @ %.2fm, origin=(%.1f, %.1f)"
          % (explorer.grid.nx, explorer.grid.ny, explorer.grid.res,
             explorer.grid.origin_x, explorer.grid.origin_y))

    pid_controller = make_pid_controller(base_env.vehicle, dt=base_env.dt)
    collision_events: List[carla.CollisionEvent] = []
    collision_sensor = attach_collision_sensor(base_env, collision_events)

    trajectory_fp = open(os.path.join(save_dir, args.trajectory_filename), "w",
                         newline="", encoding="utf-8")
    trajectory_writer = csv.writer(trajectory_fp)
    trajectory_writer.writerow(TRAJECTORY_HEADER)

    local_targets: Deque[carla.Location] = deque()
    current_target: Optional[carla.Location] = None
    episode_idx = 0
    step_idx = 0
    replans = 0
    steps_since_replan = args.replan_every_steps
    steps_on_goal = 0
    consecutive_failures = 0
    stuck_dumped = False
    steps_without_progress = 0
    best_goal_dist = float("inf")
    abandoned_goals = 0
    behind_fallbacks = 0
    replan_reasons = {}
    waypoint_generation_times = []
    episode_starts = [{
        "episode_idx": 0, "step_idx": 0,
        "spawn_transform": transform_to_dict(base_env.start_pos),
    }]

    set_spectator_to_ego_camera(base_env)
    start_time = time.time()

    # Warmup: stationary ticks so both sensors produce a frame and the map has
    # a wedge of free space to find frontiers in. No road-following agent --
    # see module docstring.
    for _ in range(max(1, args.warmup_ticks)):
        env.step(np.array([0.0, 0.0], dtype=np.float32))
    set_spectator_to_ego_camera(base_env)

    print("[YAMAUCHI] town=%d seed=%d img_size=%s fov=%s speed=%.2fkm/h replan_every=%d"
          % (args.town, args.seed, base_env.img_size, base_env.fov,
             args.target_speed_kmh, args.replan_every_steps))

    try:
        while step_idx < args.max_steps and (time.time() - start_time) < args.max_time:
            step_idx += 1
            steps_since_replan += 1

            # ── map update ───────────────────────────────────────────────
            depth, cam_tf = fetch_latest_depth(depth_buffer)
            veh_tf = base_env.vehicle.get_transform()
            ground_z = float(veh_tf.location.z) + args.ground_offset_m

            if depth is not None and cam_tf is not None:
                if args.depth_check:
                    run_depth_check(depth, cam_tf, ground_z, float(base_env.fov),
                                    os.path.join(save_dir, "depth_check.npz"))
                    break
                if step_idx % args.integrate_every == 0:
                    explorer.integrate(depth, cam_tf, ground_z)

            # ── goal progress ────────────────────────────────────────────
            pursuing = current_target is not None or local_targets
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
                # The vehicle cannot close on this frontier -- a kerb, a fence
                # gap, or simply a turning radius the goal does not admit.
                # Blacklist it, or the greedy rule re-selects it forever.
                print("[YAMAUCHI] Abandoning goal (%s) after %d steps: %s"
                      % ("no progress" if stalled else "step cap",
                         steps_on_goal, explorer.current_goal))
                explorer.abandon_goal()
                abandoned_goals += 1
                local_targets.clear()
                current_target = None
                steps_on_goal = 0
                steps_without_progress = 0
                best_goal_dist = float("inf")
                steps_since_replan = args.replan_every_steps

            out_of_targets = current_target is None and not local_targets
            due = steps_since_replan >= args.replan_every_steps
            retry_due = out_of_targets and steps_since_replan >= args.replan_retry_steps
            if due or retry_due:
                veh_loc = base_env.vehicle.get_location()
                result = explorer.replan(
                    (float(veh_loc.x), float(veh_loc.y)),
                    vehicle_yaw_deg=float(base_env.vehicle.get_transform().rotation.yaw),
                )
                steps_since_replan = 0
                replan_reasons[result.reason] = replan_reasons.get(result.reason, 0) + 1
                waypoint_generation_times.append(
                    {"step_idx": int(step_idx), "latency_s": float(result.latency_s)}
                )
                if result.reason == "ok":
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if args.stuck_dump and not stuck_dumped and consecutive_failures >= 5:
                        vl = base_env.vehicle.get_location()
                        vt = base_env.vehicle.get_transform()
                        np.savez_compressed(
                            os.path.join(save_dir, "stuck_grid.npz"),
                            log_odds=explorer.grid.log_odds,
                            origin=np.array([explorer.grid.origin_x, explorer.grid.origin_y]),
                            res=explorer.grid.res,
                            occ_thresh=explorer.grid.occ_thresh,
                            free_thresh=explorer.grid.free_thresh,
                            vehicle=np.array([vl.x, vl.y, vl.z, vt.rotation.yaw]),
                            step=step_idx,
                        )
                        stuck_dumped = True
                        print("[YAMAUCHI] stuck state dumped at step %d (%s)"
                              % (step_idx, result.reason))

                if result.waypoints:
                    local_targets = deque(
                        carla.Location(x=float(wx), y=float(wy), z=float(veh_loc.z))
                        for wx, wy in result.waypoints
                    )
                    current_target = local_targets.popleft()
                    steps_on_goal = 0
                    steps_without_progress = 0
                    best_goal_dist = float("inf")
                    replans += 1
                    behind_fallbacks += int(result.behind)
                    print("[YAMAUCHI] Replan #%d: %d waypoints, goal=(%.1f, %.1f)%s, "
                          "clusters=%d, %.3fs"
                          % (replans, len(result.waypoints), result.goal[0],
                             result.goal[1], " BEHIND (nothing reachable ahead)" if result.behind else "",
                             result.n_clusters, result.latency_s))
                elif out_of_targets:
                    print("[YAMAUCHI] step=%d no plan (%s), clusters=%d"
                          % (step_idx, result.reason, result.n_clusters))

            if current_target is None and local_targets:
                current_target = local_targets.popleft()

            # ── act ──────────────────────────────────────────────────────
            if current_target is not None:
                control = pid_controller.run_step(
                    args.target_speed_kmh, TargetWaypoint(current_target)
                )
                action_arr = control_to_action_array(control)
            else:
                # No reachable frontier: creep so the camera sweeps new space
                # rather than stalling on a dead map -- but only where the map
                # says it is clear. Creeping blind drove into walls the map
                # already knew about, which was most of this baseline's
                # collisions.
                yaw = float(veh_tf.rotation.yaw)
                vx, vy = float(veh_tf.location.x), float(veh_tf.location.y)
                if explorer.grid.is_clear(vx, vy, yaw, args.creep_lookahead_m):
                    action_arr = np.array([0.3, 0.0], dtype=np.float32)
                elif explorer.grid.is_clear(vx, vy, yaw, -args.creep_lookahead_m):
                    action_arr = np.array([-0.3, 0.0], dtype=np.float32)
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
            mode = "track" if (current_target is not None or local_targets) else "idle"
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
                snap = explorer.grid.snapshot()
                save_grid_png(
                    os.path.join(save_dir, "grid_%06d.png" % step_idx),
                    explorer.grid, snapshot=snap,
                    clusters=detect(explorer.grid, snapshot=snap,
                                    min_frontier_cells=args.min_frontier_cells)[:40],
                    vehicle_xy=(float(veh_loc.x), float(veh_loc.y)),
                    goal_xy=explorer.current_goal,
                )

            if step_idx % args.log_every == 0:
                st = explorer.grid.stats()
                print("[YAMAUCHI] step=%d ep=%d mode=%s queue=%d replans=%d "
                      "free=%.0fm2 occ=%d x=%+.1f y=%+.1f"
                      % (step_idx, episode_idx, mode, len(local_targets), replans,
                         st["free_cells"] * st["cell_area_m2"], st["occ_cells"],
                         veh_loc.x, veh_loc.y))

            # ── episode restart on collision ─────────────────────────────
            if collision_detected:
                print("[YAMAUCHI] Collision at step=%d. Resetting and continuing." % step_idx)
                # This goal led us into something; do not re-select it.
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
                # The map persists across respawns -- exploration knowledge is
                # cumulative and the coverage metric spans the whole run. Only
                # the planner's source moves.
                local_targets.clear()
                current_target = None
                steps_on_goal = 0
                steps_without_progress = 0
                best_goal_dist = float("inf")
                steps_since_replan = args.replan_every_steps
                for _ in range(max(1, args.warmup_ticks)):
                    env.step(np.array([0.0, 0.0], dtype=np.float32))

            if terminated or truncated:
                print("[YAMAUCHI] Episode %s at step=%d; stopping."
                      % ("terminated" if terminated else "truncated", step_idx))
                break

    except KeyboardInterrupt:
        pass
    finally:
        try:
            trajectory_fp.flush()
            trajectory_fp.close()
        except Exception:
            pass

        st = explorer.grid.stats()
        metadata = {
            "town": int(args.town),
            "seed": int(args.seed),
            "mode": args.mode,
            "target_speed_kmh": float(args.target_speed_kmh),
            "replan_every_steps": int(args.replan_every_steps),
            "waypoint_reach_thresh_m": float(args.waypoint_reach_thresh_m),
            "max_steps": int(args.max_steps),
            "dt_s": float(base_env.dt),
            "world_name": str(base_env.WORLD_NAME),
            "episode_starts": episode_starts,
            "waypoint_generation_times": waypoint_generation_times,
            "final_step_idx": int(step_idx),
            "replans": int(replans),
            "replan_reasons": replan_reasons,
            "blacklisted_goals": len(explorer.blacklist),
            "abandoned_goals": int(abandoned_goals),
            "behind_fallbacks": int(behind_fallbacks),
            "map_stats": st,
            "yamauchi_params": {
                "resolution_m": args.resolution_m,
                "max_range_m": args.max_range_m,
                "min_range_m": args.min_range_m,
                "obstacle_band_m": [args.obstacle_min_h, args.obstacle_max_h],
                "ground_offset_m": args.ground_offset_m,
                "min_frontier_cells": args.min_frontier_cells,
                "min_obstacle_cells": args.min_obstacle_cells,
                "inflation_radius_m": args.inflation_radius_m,
                "waypoint_spacing_m": args.waypoint_spacing_m,
                "min_goal_distance_m": args.min_goal_distance_m,
                "forward_cone_deg": args.forward_cone_deg,
                "creep_lookahead_m": args.creep_lookahead_m,
                "replan_retry_steps": args.replan_retry_steps,
                "blacklist_radius_m": args.blacklist_radius_m,
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
        print("[YAMAUCHI] Finished at step=%d (%d replans, %d episodes). Saved to %s"
              % (step_idx, replans, episode_idx + 1, save_dir))


if __name__ == "__main__":
    main()
