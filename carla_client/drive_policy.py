"""Runs a policy from explore/ in CARLA: sends camera frames over ZMQ, follows the returned waypoints.

Start the matching explore/explore_*.py first, or use run_experiment.sh, which starts both.
"""
import argparse
import csv
import json
import os
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
import gymnasium as gym
import carla

import envs
from common import (
    TargetWaypoint,
    RoadHeadingTracker,
    attach_collision_sensor,
    control_to_action_array,
    dist_2d,
    make_pid_controller,
    nearest_driving_spawn_from_transform,
    set_spectator_to_ego_camera,
    transform_to_dict,
)
from zmq_comm import ZMQComm



class PaddedContextZMQComm(ZMQComm):
    """ZMQComm that pads a short context instead of needing a warmup drive.

    Frame spacing is the base class's, unchanged: every frame carries its world
    (x, y, z) and sample_by_distance walks backwards until successive frames are
    context_spacing metres apart.

    The only addition is padding. A buffer too short to supply context_size
    spaced frames repeats its *oldest* frame at the front, which is what removes
    the warmup drive - the first plan is made from the spawn view seen
    context_size times. Padding the front rather than the back keeps the newest frame as the
    true current observation, so a short history reads as "the robot was
    stationary, then moved" instead of "the robot slowed to a stop". The copies
    carry strictly increasing timestamps because the policy sorts its
    observations by timestamp.
    """

    def get_context_images(self):
        if not self.image_buffer:
            return []
        context = list(self.sample_by_distance(self.image_buffer))
        pad = self.context_size - len(context)
        if pad > 0:
            rgb, location, timestamp = context[0]
            head = [(rgb, location, timestamp - i * 1e-4) for i in range(pad, 0, -1)]
            context = head + context
            print(
                f"[ZMQComm] Context: {pad} repeat(s) of the oldest "
                f"+ {len(context) - pad} distance-spaced frame(s)"
            )
        else:
            print(f"[ZMQComm] Context: {len(context)} distance-spaced frames")
        return context



def obs_to_rgb(obs: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if obs is None:
        return None
    # Env observation is RGB float32 in CHW and [0, 1].
    rgb_hwc = np.transpose(obs, (1, 2, 0))
    return np.clip(rgb_hwc * 255.0, 0, 255).astype(np.uint8)


def waypoints_flu_to_world(
    vehicle_transform: carla.Transform,
    waypoints_flu: List[Tuple[float, float, float]],
) -> Deque[carla.Location]:
    targets: Deque[carla.Location] = deque()
    for x_fwd, y_left, z_up in waypoints_flu:
        # FLU -> CARLA vehicle frame FRU
        p_local = carla.Location(x=float(x_fwd), y=float(-y_left), z=float(z_up))
        p_world = vehicle_transform.transform(p_local)
        targets.append(p_world)
    return targets




def seed_context(zmq_comm: PaddedContextZMQComm, base_env, obs) -> None:
    """Put the view from the current pose into an emptied context buffer.

    PaddedContextZMQComm pads a short buffer, so the policy can be asked for a
    plan from a standstill. Used at startup and again after a collision, where the
    recovery teleport invalidates the old frames.
    """
    zmq_comm.clear_buffer()
    rgb = obs_to_rgb(obs)
    if rgb is None:
        print("[POLICY] WARNING: no camera frame to seed the context with")
        return
    loc = base_env.vehicle.get_location()
    zmq_comm.add_image(rgb, (loc.x, loc.y, loc.z), time.time())
    print(
        f"[POLICY] Seeded context from the current pose; the policy sees it "
        f"{zmq_comm.context_size}x until real frames replace it"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Drive a waypoint policy in CARLA")
    parser.add_argument("--town", type=int, default=1, choices=[1,2,3,4,5,6,7,10,11,12])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mode", type=str, default="random", choices=["guide", "select", "random"])
    parser.add_argument("--target_speed_kmh", type=float, default=3.6)
    parser.add_argument("--waypoint_reach_thresh_m", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument(
        "--replan_every_steps",
        type=int,
        default=40,
        help="Trigger policy replanning every N env steps",
    )
    parser.add_argument("--context_size", type=int, default=4)
    parser.add_argument("--waypoint_addr", type=str, default="tcp://127.0.0.1:5556")
    parser.add_argument("--image_addr", type=str, default="tcp://127.0.0.1:5555")
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Run CARLA server with off-screen rendering (-RenderOffScreen).",
    )
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument(
        "--max_buffer_frames",
        type=int,
        default=200,
        help="Cap buffered frames to avoid unbounded growth",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"),
        help="Directory to save the trajectory and metadata",
    )
    parser.add_argument(
        "--trajectory_filename",
        type=str,
        default="trajectory.csv",
        help="CSV filename for per-step custom trajectory recording",
    )
    parser.add_argument("--metadata_filename", type=str, default="metadata.json")
    parser.add_argument(
        "--long_horizon_steps",
        type=int,
        default=400,
        help="Set the long-horizon context flag this many simulation steps apart "
             "(400 steps = 20s at dt=0.05)",
    )
    parser.add_argument(
        "--metric_waypoint_spacing",
        type=float,
        default=0.40,
        help="Add an image to the buffer every time the vehicle travels this many meters",
    )
    parser.add_argument(
        "--sample_waypoint_spacing",
        type=int,
        default=2,
        help="Stride, in buffer samples, between context frames and between the "
        "waypoints the policy predicts. Multiplied by --metric_waypoint_spacing to "
        "get the distance in meters.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.save_dir = os.path.join(args.root_dir, f"{args.mode}", f"recorded_trajs_town_{args.town}_seed_{args.seed}")
    initial_spawn: Optional[carla.Transform] = None

    if args.offscreen:
        os.environ["CARLA_OFFSCREEN"] = "1"
    else:
        os.environ.pop("CARLA_OFFSCREEN", None)

    env = gym.make(f"carla-town{args.town:02d}-v1")
    base_env = env.unwrapped

    obs, info = env.reset(
        return_info=True,
        seed=args.seed,
        spawn_noise_std=0.0,
        action_exec_noise_std=0.0,
        spawn_point=initial_spawn,
    )

    zmq_comm = PaddedContextZMQComm(
        waypoint_addr=args.waypoint_addr,
        image_addr=args.image_addr,
        context_size=args.context_size,
        context_spacing=args.sample_waypoint_spacing * args.metric_waypoint_spacing,
    )

    pid_controller = make_pid_controller(base_env.vehicle, dt=base_env.dt)
    collision_events: List[carla.CollisionEvent] = []
    collision_sensor: Optional[carla.Sensor] = attach_collision_sensor(base_env, collision_events)
    road_heading = RoadHeadingTracker(base_env.world.get_map())
    save_enabled = bool(args.save_dir)
    save_dir = os.path.abspath(args.save_dir) if save_enabled else ""
    run_start_wall = time.time()
    trajectory_path: Optional[str] = None
    trajectory_fp = None
    trajectory_writer = None
    episode_idx = 0
    episode_starts = []

    if save_enabled:
        os.makedirs(save_dir, exist_ok=True)
        trajectory_path = os.path.join(save_dir, args.trajectory_filename)
        trajectory_fp = open(trajectory_path, "w", newline="", encoding="utf-8")
        trajectory_writer = csv.writer(trajectory_fp)
        trajectory_writer.writerow(
            [
                "episode_idx",
                "step_idx",
                "frame",
                "sim_time_s",
                "step_duration_s",
                "mode",
                "action_signed_throttle",
                "action_steer",
                "applied_throttle",
                "applied_brake",
                "applied_steer",
                "x",
                "y",
                "z",
                "pitch",
                "yaw",
                "roll",
                "vx",
                "vy",
                "vz",
                "speed_mps",
                "reward",
                "collision",
                "lhc_flag",
                "terminated",
                "truncated",
                "target_x",
                "target_y",
                "target_z",
            ]
        )


    local_targets: Deque[carla.Location] = deque()
    current_target: Optional[carla.Location] = None
    step_idx = 0
    replans = 0
    steps_since_replan = args.replan_every_steps
    gym_hz = 1.0 / float(base_env.dt)
    steps_since_lhc = 0

    print("[POLICY] Started gym controller")
    print(
        f"[POLICY] town={args.town} speed={args.target_speed_kmh:.2f}km/h "
        f"reach_thresh={args.waypoint_reach_thresh_m:.2f}m "
        f"sample_spacing={args.sample_waypoint_spacing * args.metric_waypoint_spacing:.2f}m"
    )
    print(
        f"[POLICY] gym_hz={gym_hz:.2f} "
        f"(image capture gated by distance)"
    )
    print(
        f"[POLICY] replan_every_steps={args.replan_every_steps} "
        f"long_horizon_steps={args.long_horizon_steps} "
        f"({args.long_horizon_steps * base_env.dt:.1f}s)"
    )
    set_spectator_to_ego_camera(base_env)
    spawn_tf = base_env.start_pos
    episode_starts.append(
        {
            "episode_idx": int(episode_idx),
            "step_idx": int(step_idx),
            "spawn_transform": transform_to_dict(spawn_tf),
        }
    )

    # ── Seed the context from the spawn view; the first loop pass replans ──
    seed_context(zmq_comm, base_env, obs)

    try:
        while step_idx < args.max_steps:
            step_idx += 1
            step_t0 = time.perf_counter()
            action_arr: np.ndarray
            add_to_lhc_flag = False
            steps_since_replan += 1
            steps_since_lhc += 1
            should_replan = steps_since_replan >= args.replan_every_steps

            if should_replan:
                if steps_since_lhc >= args.long_horizon_steps:
                    add_to_lhc_flag = True
                    steps_since_lhc = 0
                sent = zmq_comm.send_images(long_horizon_flag=add_to_lhc_flag)
                if sent:
                    new_waypoints = zmq_comm.receive_waypoints()
                    if new_waypoints:
                        veh_tf = base_env.vehicle.get_transform()
                        local_targets = waypoints_flu_to_world(veh_tf, new_waypoints)
                        current_target = local_targets.popleft() if local_targets else None
                        replans += 1
                        steps_since_replan = 0
                        print(
                            f"[POLICY] Replan #{replans}: received {len(new_waypoints)} waypoints"
                        )
                    else:
                        print("[POLICY] Policy response had no usable waypoints")

            if current_target is None and local_targets:
                current_target = local_targets.popleft()
                print(
                    f"[POLICY] New target: x={current_target.x:+.2f}, "
                    f"y={current_target.y:+.2f}, z={current_target.z:+.2f}, "
                    f"remaining={len(local_targets)}"
                )

            if current_target is not None:
                target_wp = TargetWaypoint(current_target)
                control = pid_controller.run_step(args.target_speed_kmh, target_wp)
                action_arr = control_to_action_array(control)
            else:
                action_arr = np.array([0.0, 0.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action_arr)
            set_spectator_to_ego_camera(base_env)
            collision_detected = len(collision_events) > 0
            if not collision_detected:
                road_heading.update(base_env.vehicle)

            if collision_detected:
                print(f"[POLICY] Collision at step={step_idx}. Resetting and continuing.")
                try:
                    if collision_sensor is not None:
                        collision_sensor.stop()
                        collision_sensor.destroy()
                        collision_sensor = None
                except Exception:
                    pass
                collision_events.clear()
                next_episode_idx = episode_idx + 1

                recovery_spawn = nearest_driving_spawn_from_transform(
                    base_env.world.get_map(),
                    base_env.vehicle.get_transform(),
                    ref_dir=road_heading.direction,
                    fallback_wp=road_heading.waypoint,
                )
                
                obs, info = env.reset(
                    return_info=True,
                    seed=args.seed,
                    spawn_noise_std=0.0,
                    action_exec_noise_std=0.0,
                    spawn_point=recovery_spawn,
                )
                pid_controller = make_pid_controller(base_env.vehicle, dt=base_env.dt)
                collision_sensor = attach_collision_sensor(base_env, collision_events)
                road_heading = RoadHeadingTracker(base_env.world.get_map())
                road_heading.update(base_env.vehicle)
                episode_idx = next_episode_idx
                spawn_tf = base_env.start_pos
                episode_starts.append(
                    {
                        "episode_idx": int(episode_idx),
                        "step_idx": int(step_idx),
                        "spawn_transform": transform_to_dict(spawn_tf),
                    }
                )
                seed_context(zmq_comm, base_env, obs)
                local_targets.clear()
                current_target = None
                steps_since_replan = args.replan_every_steps
                steps_since_lhc = 0

            rgb = obs_to_rgb(obs)
            if rgb is not None:
                veh_loc = base_env.vehicle.get_location()
                zmq_comm.add_image(rgb, (veh_loc.x, veh_loc.y, veh_loc.z), time.time())
                if len(zmq_comm.image_buffer) > args.max_buffer_frames:
                    del zmq_comm.image_buffer[:-args.max_buffer_frames]

            if trajectory_writer is not None:
                veh_tf = base_env.vehicle.get_transform()
                veh_loc = veh_tf.location
                veh_rot = veh_tf.rotation
                veh_vel = base_env.vehicle.get_velocity()
                speed_mps = float(np.sqrt(veh_vel.x**2 + veh_vel.y**2 + veh_vel.z**2))
                ctrl = base_env.vehicle.get_control()
                snapshot = base_env.world.get_snapshot()
                sim_time_s = (
                    float(snapshot.timestamp.elapsed_seconds)
                    if snapshot is not None
                    else float(step_idx * base_env.dt)
                )
                mode = "track" if (current_target is not None or local_targets) else "idle"
                trajectory_writer.writerow(
                    [
                        int(episode_idx),
                        int(step_idx),
                        int(info.get("frame", -1)),
                        sim_time_s,
                        float(time.perf_counter() - step_t0),
                        mode,
                        float(action_arr[0]),
                        float(action_arr[1]),
                        float(ctrl.throttle),
                        float(ctrl.brake),
                        float(ctrl.steer),
                        float(veh_loc.x),
                        float(veh_loc.y),
                        float(veh_loc.z),
                        float(veh_rot.pitch),
                        float(veh_rot.yaw),
                        float(veh_rot.roll),
                        float(veh_vel.x),
                        float(veh_vel.y),
                        float(veh_vel.z),
                        speed_mps,
                        float(reward),
                        int(collision_detected),
                        int(add_to_lhc_flag),
                        int(terminated),
                        int(truncated),
                        (float(current_target.x) if current_target is not None else ""),
                        (float(current_target.y) if current_target is not None else ""),
                        (float(current_target.z) if current_target is not None else ""),
                    ]
                )

            if current_target is not None:
                veh_loc = base_env.vehicle.get_location()
                if dist_2d(veh_loc, current_target) < args.waypoint_reach_thresh_m:
                    current_target = None

            if step_idx % args.log_every == 0:
                veh_loc = base_env.vehicle.get_location()
                mode = "track" if (current_target is not None or local_targets) else "idle"
                print(
                    f"[POLICY] step={step_idx} mode={mode} "
                    f"queue={len(local_targets)} replans={replans} lhc_flag={int(add_to_lhc_flag)} reward={reward} "
                    f"x={veh_loc.x:+.2f} y={veh_loc.y:+.2f}"
                )

            if terminated or truncated:
                reason = (
                    "terminated" if terminated else "truncated"
                )
                print(
                    f"[POLICY] Episode ended at step={step_idx} ({reason}). Resetting and continuing."
                )
                try:
                    if collision_sensor is not None:
                        collision_sensor.stop()
                        collision_sensor.destroy()
                        collision_sensor = None
                except Exception:
                    pass
                break

    except KeyboardInterrupt:
        pass
    finally:
        try:
            try:
                if collision_sensor is not None:
                    collision_sensor.stop()
                    collision_sensor.destroy()
                    collision_sensor = None
            except Exception:
                pass
            zmq_comm.close()
            if save_enabled:
                os.makedirs(save_dir, exist_ok=True)
                metadata = {
                    "town": int(args.town),
                    "seed": int(args.seed),
                    "target_speed_kmh": float(args.target_speed_kmh),
                    "replan_every_steps": int(args.replan_every_steps),
                    "long_horizon_steps": int(args.long_horizon_steps),
                    "metric_waypoint_spacing": float(args.metric_waypoint_spacing),
                    "sample_waypoint_spacing": int(args.sample_waypoint_spacing),
                    "sample_spacing": float(
                        args.sample_waypoint_spacing * args.metric_waypoint_spacing
                    ),
                    "waypoint_reach_thresh_m": float(args.waypoint_reach_thresh_m),
                    "max_steps": int(args.max_steps),
                    "dt_s": float(base_env.dt),
                    "world_name": str(base_env.WORLD_NAME),
                    "episode_starts": episode_starts,
                    "steps_completed": int(step_idx),
                    "wall_clock_duration_s": float(time.time() - run_start_wall),
                }
                with open(
                    os.path.join(save_dir, args.metadata_filename),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(metadata, f, indent=2)
                if trajectory_fp is not None:
                    trajectory_fp.flush()
                    trajectory_fp.close()
        finally:
            env.close()


if __name__ == "__main__":
    main()
