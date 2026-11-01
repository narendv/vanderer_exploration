
import os
import json
import math
import pickle
import argparse
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import gymnasium as gym
import carla

import envs
from envs.carla.agents.navigation.basic_agent import BasicAgent
from envs.carla.agents.navigation.global_route_planner import GlobalRoutePlanner
from envs.carla.weather import serialize_weather


def _extract_pose_action(env, info: Dict) -> Optional[Dict[str, float]]:
    """Extract ego pose as (x, y, yaw), preferring env actor transform then info dict."""
    actor_candidates = ("ego_vehicle", "vehicle", "player", "hero")
    for attr in actor_candidates:
        actor = getattr(env, attr, None)
        if actor is not None and hasattr(actor, "get_transform"):
            tf = actor.get_transform()
            return {
                "x": float(tf.location.x),
                "y": float(tf.location.y),
                "yaw": float(tf.rotation.yaw),
            }

    tf = info.get("transform")
    if tf is not None and hasattr(tf, "location") and hasattr(tf, "rotation"):
        return {
            "x": float(tf.location.x),
            "y": float(tf.location.y),
            "yaw": float(tf.rotation.yaw),
        }

    if all(k in info for k in ("x", "y", "yaw")):
        return {"x": float(info["x"]), "y": float(info["y"]), "yaw": float(info["yaw"])}

    return None


def _to_right_handed_xy_yaw(x_lh: float, y_lh: float, yaw_deg_lh: float):
    """CARLA LH (x, y, yaw_deg) -> RH (x, y, yaw_rad)."""
    x_rh = float(x_lh)
    y_rh = float(-y_lh)
    yaw_rh = float(-np.deg2rad(yaw_deg_lh))
    return x_rh, y_rh, yaw_rh


def _sample_stride(dt: float, record_hz: float) -> int:
    """Number of simulation steps between recorded samples."""
    return max(1, int(round((1.0 / record_hz) / dt)))


def _save_frame_as_jpg(frame, save_path: Path):
    """Save CARLA image or ndarray frame as JPG."""
    if hasattr(frame, "save_to_disk"):
        frame.save_to_disk(str(save_path))
        return

    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    img = Image.fromarray(np.transpose(arr, (1, 2, 0)))
    img.save(str(save_path), format="JPEG", quality=95)


class BasicAgentRecorder:
    """Wrap BasicAgent with trajectory recording fields used by saver.

    Observations lag the simulation tick that produced them (the camera buffer is
    drained one step later), so poses are indexed by simulation frame and each
    observation is paired with the pose of *its own* frame rather than with the
    pose that happens to be current when it is dequeued.
    """

    def __init__(
        self,
        env: gym.Env,
        destination: carla.Location,
        seed: int,
        target_speed: float = 7.2,
    ):
        self.env = env
        self.destination = destination
        self.seed = seed
        self.target_speed = target_speed
        self.agent = BasicAgent(
            env.vehicle,
            target_speed=target_speed,  # km/h
            opt_dict={"ignore_traffic_lights": True, "ignore_vehicles": False},
        )
        # Passing a truthy start_location makes this BasicAgent implementation use current vehicle location.
        self.agent.set_destination(destination, start_location=env.vehicle.get_location())
        self.latest_obs_frame = -1
        # (observation, pose) pairs that are known to belong to the same simulation frame.
        self.samples: List[Tuple[object, Dict[str, float]]] = []
        self._pose_by_frame: Dict[int, Dict[str, float]] = {}

    def step(self, obs, info: Dict) -> carla.VehicleControl:
        frame = int(info["frame"])
        obs_frame = int(info["obs_frame"])
        self.latest_obs_frame = max(self.latest_obs_frame, obs_frame)

        # In synchronous mode the actor state readable right now is the state at `frame`.
        pose = _extract_pose_action(self.env, info)
        if pose is not None:
            self._pose_by_frame[frame] = pose

        action = self.agent.run_step()

        if obs is not None and obs_frame >= 0:
            pose_at_obs = self._pose_by_frame.pop(obs_frame, None)
            if pose_at_obs is not None:
                self.samples.append((obs, pose_at_obs))
                # Frames older than the one just consumed can never be matched.
                for stale in [f for f in self._pose_by_frame if f < obs_frame]:
                    del self._pose_by_frame[stale]

        return action

    def recorded_yaws_deg(self, stride: int = 1) -> np.ndarray:
        return np.asarray(
            [pose["yaw"] for _obs, pose in self.samples[::stride]], dtype=np.float64
        )

    def done(self) -> bool:
        return self.agent.done()


def save_trajectory_helper(
    recorder: BasicAgentRecorder,
    path: os.PathLike,
    metadata: Dict,
    record_hz: float = 4.0,
):
    dt = float(recorder.env.dt)
    sample_stride = _sample_stride(dt, record_hz)
    if not recorder.samples[::sample_stride]:
        print("No recorded samples, nothing to save")
        return None

    trajectory_name = datetime.today().strftime("%Y-%m-%d-%H-%M-%S-%f")
    trajectory_dir = Path(path) / trajectory_name
    trajectory_dir.mkdir(parents=True)
    print("Saving trajectory to {}".format(trajectory_dir))

    with open(trajectory_dir / "metadata.json", "w") as f:
        metadata = dict(metadata)
        metadata.update({"record_hz": record_hz, "source_dt": dt, "sample_stride": sample_stride})
        json.dump(metadata, f)

    # Observations and poses are already paired frame-by-frame, so a single stride
    # keeps the image and its pose label together.
    samples = recorder.samples[::sample_stride]

    positions = []
    yaws = []
    for image_idx, (obs, pose) in enumerate(samples):
        save_path = trajectory_dir / f"{image_idx}.jpg"
        _save_frame_as_jpg(obs, save_path)

        x_rh, y_rh, yaw_rh = _to_right_handed_xy_yaw(pose["x"], pose["y"], pose["yaw"])
        positions.append([x_rh, y_rh])
        yaws.append(yaw_rh)

    traj_data = {
        "position": np.asarray(positions, dtype=np.float64),
        "yaw": np.asarray(yaws, dtype=np.float64),
    }
    with open(trajectory_dir / "traj_data.pkl", "wb") as f:
        pickle.dump(traj_data, f)

    print("Saved trajectory to", trajectory_dir)
    return trajectory_dir


def _parse_towns(towns_arg: str) -> Tuple[int, ...]:
    supported_towns = (1, 2, 3, 4, 5, 6, 7, 10, 11, 12)
    towns = []
    for token in towns_arg.split(","):
        token = token.strip()
        if not token:
            continue
        town = int(token)
        if town not in supported_towns:
            raise ValueError(f"Unsupported town {town}. Supported: {supported_towns}")
        towns.append(town)
    if not towns:
        raise ValueError("--towns must contain at least one town")
    return tuple(towns)


def _route_length(route) -> float:
    """Total path length (meters) of a GlobalRoutePlanner route trace."""
    if len(route) < 2:
        return 0.0
    total = 0.0
    prev = route[0][0].transform.location
    for waypoint, _option in route[1:]:
        loc = waypoint.transform.location
        total += float(prev.distance(loc))
        prev = loc
    return total


def _sample_route_pair(
    candidate_points: List[carla.Transform],
    candidate_locations: np.ndarray,
    rng: np.random.RandomState,
    planner: GlobalRoutePlanner,
    min_dist: float,
    max_dist: float,
    reverse_pair_prob: float,
    max_tries: int = 200,
    max_route_probes: int = 20,
    max_endpoint_gap: float = 5.0,
):
    """Sample a start/destination pair whose *driving route* length is in range.

    Straight-line distance only lower-bounds the route, so pairs on opposing
    carriageways or one-way streets can be metres apart yet require a trip around
    the whole block. Candidates are prefiltered on Euclidean distance (cheap) and
    then confirmed with trace_route.

    trace_route snaps both endpoints onto the route graph, and the snap is not
    always tight: the returned route can begin tens of metres from the requested
    origin. The agent would then have to drive to that first waypoint - possibly
    turning around to reach it - so pairs whose route does not actually connect
    them are rejected.
    """
    n = len(candidate_points)
    if n < 2:
        raise RuntimeError("Need at least 2 candidate points to sample start/destination")

    for _ in range(max_tries):
        start_idx = int(rng.randint(0, n))

        # Endpoint snapping means route_len can come in marginally under the
        # straight-line distance, so widen the prefilter by the allowed gap.
        euclid = np.linalg.norm(candidate_locations - candidate_locations[start_idx], axis=1)
        candidates = np.flatnonzero(
            (euclid <= max_dist + max_endpoint_gap) & (euclid > 1e-3)
        )
        if candidates.size == 0:
            continue

        rng.shuffle(candidates)
        for other_idx in candidates[:max_route_probes]:
            src_idx, dst_idx = start_idx, int(other_idx)
            reversed_pair = bool(rng.rand() < reverse_pair_prob)
            if reversed_pair:
                # Reverse before tracing: on one-way roads the two directions have
                # completely different route lengths.
                src_idx, dst_idx = dst_idx, src_idx

            src_loc = candidate_points[src_idx].location
            dst_loc = candidate_points[dst_idx].location
            try:
                route = planner.trace_route(src_loc, dst_loc)
            except Exception:
                continue

            if len(route) < 2:
                continue
            if route[0][0].transform.location.distance(src_loc) > max_endpoint_gap:
                continue
            if route[-1][0].transform.location.distance(dst_loc) > max_endpoint_gap:
                continue

            route_len = _route_length(route)
            if min_dist <= route_len <= max_dist:
                return (
                    src_idx,
                    dst_idx,
                    candidate_points[src_idx],
                    candidate_points[dst_idx],
                    route_len,
                    reversed_pair,
                )

    raise RuntimeError(
        f"Could not find point pair with route length in [{min_dist}, {max_dist}] "
        f"after {max_tries} tries"
    )


def _compute_step_budget(
    route_len_m: float,
    target_speed_kmh: float,
    dt: float,
    slack: float,
    min_steps: int,
    max_steps: int,
) -> int:
    """Steps needed to drive `route_len_m` at `target_speed_kmh`, times a slack factor."""
    speed_mps = max(target_speed_kmh / 3.6, 1e-3)
    nominal_steps = route_len_m / speed_mps / max(dt, 1e-6)
    budget = int(math.ceil(slack * nominal_steps))
    return int(min(max(budget, min_steps), max_steps))


def _heading_change_stats(yaws_deg: np.ndarray, dt: float, window_s: float):
    """Return (total, worst-window) absolute heading change in degrees.

    A stalled local planner orbits its unreachable target waypoint, which shows up
    as a large heading change accumulated over a short window - much tighter than
    any legitimate turn at the recording speed.
    """
    if yaws_deg.size < 2:
        return 0.0, 0.0

    deltas = np.abs((np.diff(yaws_deg) + 180.0) % 360.0 - 180.0)
    total = float(deltas.sum())

    cumulative = np.concatenate(([0.0], np.cumsum(deltas)))
    window_steps = max(1, int(round(window_s / max(dt, 1e-6))))
    if window_steps >= deltas.size:
        return total, total

    worst = float(np.max(cumulative[window_steps:] - cumulative[:-window_steps]))
    return total, worst


def _distance_to_destination(vehicle: carla.Vehicle, destination: carla.Location) -> float:
    return float(vehicle.get_location().distance(destination))


def _generate_map_waypoint_transforms(
    world_map: carla.Map, sampling_resolution: float
) -> List[carla.Transform]:
    waypoints = world_map.generate_waypoints(sampling_resolution)
    return [wp.transform for wp in waypoints]


def _offset_transform_z(transform: carla.Transform, dz: float) -> carla.Transform:
    """Copy a transform, lifting it by `dz` and preserving the waypoint's own elevation.

    The elevation must be preserved: `map.get_waypoint()` snaps in 3D, so flattening
    a point on an overpass to ground level resolves it to the road underneath.
    """
    loc = transform.location
    rot = transform.rotation
    return carla.Transform(carla.Location(x=loc.x, y=loc.y, z=loc.z + dz), rot)


def _drop_random_waypoint_sections(
    points: List[carla.Transform],
    rng: np.random.RandomState,
    num_sections: int,
    section_radius: float,
) -> Tuple[List[carla.Transform], int]:
    if num_sections <= 0 or section_radius <= 0.0 or len(points) == 0:
        return points, 0

    keep_mask = np.ones(len(points), dtype=bool)
    centers = rng.choice(len(points), size=min(num_sections, len(points)), replace=False)

    for center_idx in centers:
        center_loc = points[int(center_idx)].location
        for i, point in enumerate(points):
            if keep_mask[i] and point.location.distance(center_loc) <= section_radius:
                keep_mask[i] = False

    kept_points = [point for i, point in enumerate(points) if keep_mask[i]]
    removed_count = int(len(points) - len(kept_points))
    return kept_points, removed_count


def _reset_env(
    env: gym.Env,
    args_dict: Dict,
    seed: int,
    start_spawn: carla.Transform,
) -> Tuple[np.ndarray, Dict, int]:
    last_error = None
    for attempt in range(args_dict["max_spawn_retries"]):
        # The env reseeds its RNG from `seed` on every reset and draws the spawn
        # perturbation from it, so retrying with the same seed reproduces the exact
        # same failing pose. Vary the seed so each attempt is actually different.
        attempt_seed = seed if attempt == 0 else seed + attempt * 100003
        try:
            obs, info = env.reset(
                return_info=True,
                seed=attempt_seed,
                action_exec_noise_std=args_dict["action_exec_noise_std"],
                spawn_noise_std=args_dict["spawn_noise_std"],
                spawn_point=start_spawn,
            )
            return obs, info, attempt_seed
        except RuntimeError as exc:
            last_error = exc
            if "spawn failed because of collision" not in str(exc).lower():
                raise

    raise RuntimeError(
        f"Failed to reset after {args_dict['max_spawn_retries']} spawn attempts"
    ) from last_error


def _is_server_crash_error(exc: Exception) -> bool:
    """True for the RuntimeError CARLA's client raises when the server has died.

    The server crashing (segfault, OOM) surfaces to Python as a client RPC
    time-out, not a distinguishable exception type - this is the only signal
    available to tell "server is gone" apart from a real bug.
    """
    return "time-out" in str(exc).lower()


def _force_close_env(env: gym.Env):
    """Close env, tolerating a server that already crashed and can't be RPC'd.

    env.close() itself makes RPC calls (_set_synchronous, garbage_collect), which
    raise the same time-out error if the server is already dead. Fall back to
    killing the raw process directly so the port is free for the next server.
    """
    base_env = env.unwrapped
    try:
        env.close()
        return
    except Exception:
        pass
    try:
        base_env.server_proc.kill()
        base_env.server_proc.wait(timeout=10)
    except Exception:
        pass


def _make_town_env(town: int, carla_port: int, args_dict: Dict):
    """Launch a fresh CARLA server + gym env for `town` and apply run config."""
    os.environ["CARLA_PORT"] = str(carla_port)
    if args_dict["offscreen"]:
        os.environ["CARLA_OFFSCREEN"] = "1"
    else:
        os.environ.pop("CARLA_OFFSCREEN", None)

    env = gym.make(f"carla-town{town:02d}-v1")
    base_env = env.unwrapped

    # Large-map streaming overrides (only meaningful on tiled maps like Town12).
    if args_dict["tile_stream_distance"] is not None:
        base_env.TILE_STREAM_DISTANCE = args_dict["tile_stream_distance"]
    if args_dict["actor_active_distance"] is not None:
        base_env.ACTOR_ACTIVE_DISTANCE = args_dict["actor_active_distance"]
    if args_dict["tile_stream_distance"] is not None or args_dict["actor_active_distance"] is not None:
        # Re-apply: the env already called _set_synchronous during construction.
        base_env._set_synchronous(True)
    if base_env.TILE_STREAM_DISTANCE is not None or base_env.ACTOR_ACTIVE_DISTANCE is not None:
        print(
            f"[town={town} port={carla_port}] tile streaming: "
            f"tile_stream_distance={base_env.TILE_STREAM_DISTANCE} "
            f"actor_active_distance={base_env.ACTOR_ACTIVE_DISTANCE} "
            f"spectator_follows_ego={base_env.SPECTATOR_FOLLOWS_EGO}"
        )

    if args_dict["random_weather"]:
        # WEATHER=None switches the env onto its randomized branch; the town classes
        # otherwise pin a fixed clear-noon preset (what evaluation should keep using).
        base_env.WEATHER = None
        base_env.WEATHER_SUN_ALTITUDE_RANGE = (
            args_dict["weather_sun_altitude_min"],
            args_dict["weather_sun_altitude_max"],
        )
        base_env.WEATHER_FOG_PROB = args_dict["weather_fog_prob"]
        base_env.WEATHER_FOG_DENSITY_RANGE = (
            args_dict["weather_min_fog"],
            args_dict["weather_max_fog"],
        )
        print(
            f"[town={town} port={carla_port}] random weather enabled "
            f"(sun_altitude={base_env.WEATHER_SUN_ALTITUDE_RANGE}, "
            f"fog_prob={base_env.WEATHER_FOG_PROB}, "
            f"fog_density={base_env.WEATHER_FOG_DENSITY_RANGE})"
        )

    return env, base_env


def _run_town_worker(
    town: int,
    rollout_indices: List[int],
    args_dict: Dict,
    carla_port: int,
):
    env, base_env = _make_town_env(town, carla_port, args_dict)
    world_map = base_env.world.get_map()

    try:
        if args_dict["pair_source"] == "generated_waypoints":
            candidate_points = _generate_map_waypoint_transforms(
                world_map, args_dict["waypoint_sampling_resolution"]
            )
            preview_count = min(3, len(candidate_points))
            for preview_idx in range(preview_count):
                loc = candidate_points[preview_idx].location
                print(
                    f"[town={town} port={carla_port}] waypoint[{preview_idx}] "
                    f"x={loc.x:.3f}, y={loc.y:.3f}, z={loc.z:.3f}"
                )
            mask_rng = np.random.RandomState(args_dict["waypoint_section_seed"] + town)
            candidate_points, removed_points = _drop_random_waypoint_sections(
                candidate_points,
                mask_rng,
                args_dict["waypoint_section_drop_count"],
                args_dict["waypoint_section_drop_radius"],
            )
            print(
                f"[town={town} port={carla_port}] generated_waypoints={len(candidate_points)} "
                f"(removed={removed_points})"
            )
        else:
            candidate_points = world_map.get_spawn_points()
            removed_points = 0
            print(f"[town={town} port={carla_port}] spawn_points={len(candidate_points)}")

        if len(candidate_points) < 2:
            raise RuntimeError(
                f"[town={town}] not enough candidate points after filtering: {len(candidate_points)}"
            )

        candidate_locations = np.array(
            [[t.location.x, t.location.y, t.location.z] for t in candidate_points],
            dtype=np.float64,
        )
        route_planner = GlobalRoutePlanner(
            world_map, args_dict["waypoint_sampling_resolution"]
        )

        rollout_specs = []
        print(
            f"[town={town} port={carla_port}] Precomputing {len(rollout_indices)} rollout pairs..."
        )
        for i in rollout_indices:
            rng = np.random.RandomState(seed=i)

            try:
                (
                    start_idx,
                    dst_idx,
                    start_spawn,
                    dst_spawn,
                    route_len,
                    reversed_pair,
                ) = _sample_route_pair(
                    candidate_points,
                    candidate_locations,
                    rng,
                    route_planner,
                    min_dist=args_dict["min_route_dist"],
                    max_dist=args_dict["max_route_dist"],
                    reverse_pair_prob=args_dict["reverse_pair_prob"],
                    max_tries=args_dict["max_pair_trials"],
                    max_route_probes=args_dict["max_route_probes"],
                    max_endpoint_gap=args_dict["max_endpoint_gap"],
                )
            except RuntimeError as exc:
                print(f"[town={town} port={carla_port}] rollout={i} skipped in precompute: {exc}")
                continue

            # Lift the spawn clear of the road surface; the destination keeps the
            # waypoint's exact elevation so it resolves to the right road level.
            start_spawn = _offset_transform_z(start_spawn, args_dict["spawn_z_offset"])

            rollout_specs.append(
                {
                    "rollout_idx": i,
                    "start_idx": start_idx,
                    "dst_idx": dst_idx,
                    "start_spawn": start_spawn,
                    "dst_spawn": dst_spawn,
                    "route_len": route_len,
                    "reversed_pair": reversed_pair,
                    "step_budget": _compute_step_budget(
                        route_len,
                        args_dict["target_speed"],
                        float(base_env.dt),
                        args_dict["step_budget_slack"],
                        args_dict["min_steps"],
                        args_dict["max_steps"],
                    ),
                }
            )

        print(f"[town={town} port={carla_port}] Precompute done: {len(rollout_specs)} pairs.")

        for spec in rollout_specs:
            i = spec["rollout_idx"]
            start_idx = spec["start_idx"]
            dst_idx = spec["dst_idx"]
            start_spawn = spec["start_spawn"]
            dst_spawn = spec["dst_spawn"]
            route_len = spec["route_len"]
            reversed_pair = spec["reversed_pair"]
            step_budget = spec["step_budget"]

            base_env.spawns = [start_spawn]
            try:
                obs, info, used_seed = _reset_env(
                    env=env,
                    args_dict=args_dict,
                    seed=i,
                    start_spawn=start_spawn,
                )
            except RuntimeError as exc:
                if _is_server_crash_error(exc):
                    # Fail fast and let the process die: with chunked runs, an
                    # external wrapper restarts the whole chunk on a crash, so
                    # there is no point spending another 90s timeout per
                    # remaining rollout against a server that is already gone.
                    raise
                print(
                    f"[town={town} port={carla_port}] rollout={i} skipped: {exc}"
                )
                continue

            recorder = BasicAgentRecorder(
                base_env,
                dst_spawn.location,
                seed=used_seed,
                target_speed=args_dict["target_speed"],
            )
            done = False
            end_reason = "max_steps"

            for _step_idx in range(step_budget):
                action = recorder.step(obs, info)

                action_arr = np.array([action.throttle - action.brake, action.steer])
                action_arr = np.clip(action_arr, -1, 1)
                obs, _reward, terminated, truncated, info = env.step(action_arr)

                dist_to_goal = _distance_to_destination(base_env.vehicle, dst_spawn.location)
                if dist_to_goal <= args_dict["goal_threshold"]:
                    done = True
                    end_reason = "goal_reached"
                    break
                if recorder.done():
                    done = True
                    end_reason = "agent_done"
                    break
                if terminated:
                    done = True
                    end_reason = "env_terminated"
                    break
                if truncated:
                    done = True
                    end_reason = "env_truncated"
                    break

            last_frame = info["frame"]
            while info["obs_frame"] < last_frame:
                action = recorder.step(obs, info)
                action_arr = np.array([action.throttle - action.brake, action.steer])
                action_arr = np.clip(action_arr, -1, 1)
                obs, _reward, _terminated, _truncated, info = env.step(action_arr)

            # Evaluated on the recorded (strided) series, not the raw sim rate: that is
            # the trajectory that actually gets saved, and it keeps per-tick steering
            # jitter from inflating the cumulative heading change.
            stride = _sample_stride(float(base_env.dt), args_dict["freq"])
            total_yaw_deg, window_yaw_deg = _heading_change_stats(
                recorder.recorded_yaws_deg(stride),
                float(base_env.dt) * stride,
                args_dict["yaw_window_s"],
            )

            print(
                f"rollout={i} town={town} port={carla_port} start={start_idx} dst={dst_idx} "
                f"route_len={route_len:.1f} budget={step_budget} reversed={reversed_pair} "
                f"done={done} reason={end_reason} samples={len(recorder.samples)} "
                f"yaw_total={total_yaw_deg:.0f} yaw_window={window_yaw_deg:.0f}"
            )

            if end_reason == "max_steps" and not args_dict["keep_incomplete"]:
                print(f"rollout={i} town={town} rejected: ran out of steps mid-route")
                continue
            if total_yaw_deg > args_dict["max_total_yaw_change_deg"]:
                print(
                    f"rollout={i} town={town} rejected: total heading change "
                    f"{total_yaw_deg:.0f} deg > {args_dict['max_total_yaw_change_deg']:.0f}"
                )
                continue
            if window_yaw_deg > args_dict["max_window_yaw_change_deg"]:
                print(
                    f"rollout={i} town={town} rejected: looping "
                    f"({window_yaw_deg:.0f} deg in {args_dict['yaw_window_s']:.1f}s > "
                    f"{args_dict['max_window_yaw_change_deg']:.0f})"
                )
                continue

            metadata = {
                "seed": used_seed,
                "requested_seed": i,
                "town": town,
                "start_spawn_idx": start_idx,
                "dest_spawn_idx": dst_idx,
                "route_len": route_len,
                "step_budget": step_budget,
                "end_reason": end_reason,
                "total_yaw_change_deg": total_yaw_deg,
                "max_window_yaw_change_deg": window_yaw_deg,
                "goal_threshold": args_dict["goal_threshold"],
                "min_route_dist": args_dict["min_route_dist"],
                "max_route_dist": args_dict["max_route_dist"],
                "target_speed": args_dict["target_speed"],
                "pair_source": args_dict["pair_source"],
                "reversed_pair": reversed_pair,
                "spawn_z_offset": args_dict["spawn_z_offset"],
                "random_weather": args_dict["random_weather"],
                "weather": serialize_weather(base_env.weather)
                if base_env.weather is not None
                else None,
                "waypoint_sampling_resolution": args_dict["waypoint_sampling_resolution"],
                "waypoint_section_drop_count": args_dict["waypoint_section_drop_count"],
                "waypoint_section_drop_radius": args_dict["waypoint_section_drop_radius"],
                "town_waypoint_removed_points": removed_points,
            }
            save_trajectory_helper(
                recorder,
                args_dict["output_dir"],
                metadata,
                args_dict["freq"],
            )
    finally:
        _force_close_env(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CARLA trajectories with BasicAgent")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/dataset/save/dir",
        help="Directory to save trajectories",
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=500,
        help="Number of rollouts to generate",
    )
    parser.add_argument(
        "--start_rollout",
        type=int,
        default=0,
        help="First rollout index (seed) to generate; rollouts run "
        "start_rollout .. start_rollout + num_rollouts - 1. Use this to split "
        "a run into resumable chunks, e.g. --start_rollout 100 --num_rollouts "
        "100 for the second chunk of 100 after an initial 0-99 chunk.",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=4.0,
        help="Recording frequency in Hz",
    )
    parser.add_argument(
        "--towns",
        type=str,
        default="4",
        help="Comma-separated towns, e.g. 4,10 or 1,2,3,4,5,10",
    )
    parser.add_argument(
        "--min_route_dist",
        type=float,
        default=35.0,
        help="Minimum driving route length between start and destination (meters)",
    )
    parser.add_argument(
        "--max_route_dist",
        type=float,
        default=60.0,
        help="Maximum driving route length between start and destination (meters)",
    )
    parser.add_argument(
        "--max_route_probes",
        type=int,
        default=20,
        help="Max trace_route calls per sampled start point before drawing a new start",
    )
    parser.add_argument(
        "--max_endpoint_gap",
        type=float,
        default=5.0,
        help="Reject a sampled pair if the traced route starts or ends farther than this "
        "(meters) from the requested point; trace_route snaps endpoints onto the route "
        "graph and the snap is not always tight",
    )
    parser.add_argument(
        "--target_speed",
        type=float,
        default=7.2,
        help="BasicAgent target speed in km/h (7.2 km/h = 2 m/s). Also sets the step budget.",
    )
    parser.add_argument(
        "--pair_source",
        type=str,
        choices=["generated_waypoints", "spawn_points"],
        default="generated_waypoints",
        help="Source used to sample start/destination pairs",
    )
    parser.add_argument(
        "--waypoint_sampling_resolution",
        type=float,
        default=2.0,
        help="Sampling resolution (meters) for map.generate_waypoints",
    )
    parser.add_argument(
        "--waypoint_section_drop_count",
        type=int,
        default=0,
        help="Number of random local waypoint sections to remove per town",
    )
    parser.add_argument(
        "--waypoint_section_drop_radius",
        type=float,
        default=10.0,
        help="Radius (meters) for each removed waypoint section",
    )
    parser.add_argument(
        "--waypoint_section_seed",
        type=int,
        default=12345,
        help="Base RNG seed for waypoint section removal (town id is added)",
    )
    parser.add_argument(
        "--reverse_pair_prob",
        type=float,
        default=0.5,
        help="Probability of swapping sampled start/destination before rollout",
    )
    parser.add_argument(
        "--spawn_z_offset",
        type=float,
        default=0.3,
        help="Height (meters) added above the start waypoint when spawning, to clear the "
        "road surface. The waypoint's own elevation is always preserved.",
    )
    parser.add_argument(
        "--max_spawn_retries",
        type=int,
        default=8,
        help="Max retries for env.reset when spawn collision occurs",
    )
    parser.add_argument(
        "--goal_threshold",
        type=float,
        default=5.0,
        help="Distance threshold to consider destination reached (meters)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=2000,
        help="Hard cap on env steps per rollout. The actual budget is derived from the "
        "route length and target speed; this only bounds it.",
    )
    parser.add_argument(
        "--min_steps",
        type=int,
        default=200,
        help="Lower bound on the per-rollout step budget",
    )
    parser.add_argument(
        "--step_budget_slack",
        type=float,
        default=2.5,
        help="Multiplier on the nominal route_len / target_speed step count, to absorb "
        "acceleration, turns and traffic",
    )
    parser.add_argument(
        "--keep_incomplete",
        action="store_true",
        help="Save rollouts that exhausted their step budget mid-route (dropped by default)",
    )
    parser.add_argument(
        "--max_total_yaw_change_deg",
        type=float,
        default=540.0,
        help="Reject a rollout whose cumulative absolute heading change exceeds this",
    )
    parser.add_argument(
        "--yaw_window_s",
        type=float,
        default=5.0,
        help="Window (seconds) for the local loop-detection check",
    )
    parser.add_argument(
        "--max_window_yaw_change_deg",
        type=float,
        default=180.0,
        help="Reject a rollout that turns more than this within any --yaw_window_s window; "
        "catches the local planner orbiting an unreachable waypoint",
    )
    parser.add_argument(
        "--action_exec_noise_std",
        type=float,
        default=0.08,
        help="Action execution noise passed to env.reset",
    )
    parser.add_argument(
        "--spawn_noise_std",
        type=float,
        default=0.20,
        help="Spawn noise (meters) passed to env.reset",
    )
    parser.add_argument(
        "--max_pair_trials",
        type=int,
        default=200,
        help="Max start points drawn while searching for a pair with an in-range route",
    )
    parser.add_argument(
        "--random_weather",
        action="store_true",
        help="Randomize weather per rollout (for training data variance). Off by default, "
        "so evaluation keeps the fixed clear-noon preset. The draw is seeded from the "
        "rollout seed and recorded in metadata.json.",
    )
    parser.add_argument(
        "--weather_sun_altitude_min",
        type=float,
        default=15.0,
        help="Lower bound on sun altitude (degrees) when --random_weather is set. "
        "Negative values put the sun below the horizon (night).",
    )
    parser.add_argument(
        "--weather_sun_altitude_max",
        type=float,
        default=90.0,
        help="Upper bound on sun altitude (degrees) when --random_weather is set",
    )
    parser.add_argument(
        "--weather_fog_prob",
        type=float,
        default=0.50,
        help="Fraction of rollouts that get fog when --random_weather is set; the rest "
        "are completely clear. 0 disables fog entirely.",
    )
    parser.add_argument(
        "--weather_min_fog",
        type=float,
        default=5.0,
        help="Min fog density for the foggy fraction. Scale reference: CARLA's own "
        "presets use 2-7, ~10-15 is clearly visible haze, 25+ washes out the far field.",
    )
    parser.add_argument(
        "--weather_max_fog",
        type=float,
        default=15.0,
        help="Max fog density for the foggy fraction (see --weather_min_fog for scale)",
    )
    parser.add_argument(
        "--tile_stream_distance",
        type=float,
        default=None,
        help="Override the map-tile streaming radius in meters (CARLA default 3000). "
        "Only affects large tiled maps such as Town11/Town12; lower values keep "
        "fewer tiles resident. Both already default to 700 via their env class.",
    )
    parser.add_argument(
        "--actor_active_distance",
        type=float,
        default=None,
        help="Override the actor-active radius in meters (CARLA default 2000). "
        "Large tiled maps only.",
    )
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Run CARLA with off-screen rendering (-RenderOffScreen)",
    )
    parser.add_argument(
        "--base_port",
        type=int,
        default=2000,
        help="Base CARLA RPC port for town worker processes",
    )
    parser.add_argument(
        "--port_stride",
        type=int,
        default=10,
        help="Port increment per worker (e.g., 2000, 2010, ...)",
    )
    args = parser.parse_args()

    if args.min_route_dist < 0 or args.max_route_dist <= args.min_route_dist:
        raise ValueError("Require 0 <= min_route_dist < max_route_dist")
    if args.max_pair_trials <= 0:
        raise ValueError("--max_pair_trials must be > 0")
    if args.max_route_probes <= 0:
        raise ValueError("--max_route_probes must be > 0")
    if args.max_endpoint_gap <= 0:
        raise ValueError("--max_endpoint_gap must be > 0")
    if args.weather_sun_altitude_max < args.weather_sun_altitude_min:
        raise ValueError("Require weather_sun_altitude_min <= weather_sun_altitude_max")
    if args.tile_stream_distance is not None and args.tile_stream_distance <= 0:
        raise ValueError("--tile_stream_distance must be > 0")
    if args.actor_active_distance is not None and args.actor_active_distance <= 0:
        raise ValueError("--actor_active_distance must be > 0")
    if not (0.0 <= args.weather_fog_prob <= 1.0):
        raise ValueError("--weather_fog_prob must be in [0, 1]")
    if not (0.0 <= args.weather_min_fog <= args.weather_max_fog <= 100.0):
        raise ValueError("Require 0 <= weather_min_fog <= weather_max_fog <= 100")
    if args.target_speed <= 0:
        raise ValueError("--target_speed must be > 0")
    if args.min_steps <= 0 or args.max_steps < args.min_steps:
        raise ValueError("Require 0 < min_steps <= max_steps")
    if args.step_budget_slack < 1.0:
        raise ValueError("--step_budget_slack must be >= 1.0")
    if args.yaw_window_s <= 0:
        raise ValueError("--yaw_window_s must be > 0")
    if args.max_total_yaw_change_deg <= 0 or args.max_window_yaw_change_deg <= 0:
        raise ValueError("Yaw change thresholds must be > 0")
    if args.port_stride <= 0:
        raise ValueError("--port_stride must be > 0")
    if args.waypoint_sampling_resolution <= 0:
        raise ValueError("--waypoint_sampling_resolution must be > 0")
    if args.waypoint_section_drop_count < 0:
        raise ValueError("--waypoint_section_drop_count must be >= 0")
    if args.waypoint_section_drop_radius < 0:
        raise ValueError("--waypoint_section_drop_radius must be >= 0")
    if not (0.0 <= args.reverse_pair_prob <= 1.0):
        raise ValueError("--reverse_pair_prob must be in [0, 1]")
    if args.max_spawn_retries <= 0:
        raise ValueError("--max_spawn_retries must be > 0")
    if args.start_rollout < 0:
        raise ValueError("--start_rollout must be >= 0")

    towns = _parse_towns(args.towns)
    town_to_rollouts = defaultdict(list)
    for i in range(args.start_rollout, args.start_rollout + args.num_rollouts):
        town = towns[i % len(towns)]
        town_to_rollouts[town].append(i)

    args_dict = vars(args).copy()
    workers = []
    active_towns = [town for town in towns if len(town_to_rollouts[town]) > 0]
    for idx, town in enumerate(active_towns):
        port = args.base_port + idx * args.port_stride
        p = mp.Process(
            target=_run_town_worker,
            args=(town, town_to_rollouts[town], args_dict, port),
        )
        p.start()
        workers.append((town, port, p))

    failed = []
    for town, port, proc in workers:
        proc.join()
        if proc.exitcode != 0:
            failed.append((town, port, proc.exitcode))

    if failed:
        raise RuntimeError(f"Town worker failures: {failed}")
