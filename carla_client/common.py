"""Driving, collision and recovery helpers shared by the drive_*.py scripts.

All methods use these, so they drive, collide, respawn and record identically.
"""
from typing import List, Optional, Tuple

import numpy as np
import carla

from envs.carla.agents.navigation.controller import VehiclePIDController


class TargetWaypoint:
    """Wrapper for VehiclePIDController, which expects waypoint.transform.location."""

    def __init__(self, location: carla.Location):
        self.transform = carla.Transform(location)


def dist_2d(a: carla.Location, b: carla.Location) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def control_to_action_array(control: carla.VehicleControl) -> np.ndarray:
    signed_throttle = float(np.clip(control.throttle - control.brake, -1.0, 1.0))
    steer = float(np.clip(control.steer, -1.0, 1.0))
    return np.array([signed_throttle, steer], dtype=np.float32)

def transform_to_dict(transform: carla.Transform) -> dict:
    return {
        "location": {
            "x": float(transform.location.x),
            "y": float(transform.location.y),
            "z": float(transform.location.z),
        },
        "rotation": {
            "pitch": float(transform.rotation.pitch),
            "yaw": float(transform.rotation.yaw),
            "roll": float(transform.rotation.roll),
        },
    }

def make_pid_controller(vehicle: carla.Vehicle, dt: float) -> VehiclePIDController:
    args_lateral = {"K_P": 1.95, "K_I": 0.05, "K_D": 0.2, "dt": dt}
    args_longitudinal = {"K_P": 1.0, "K_I": 0.05, "K_D": 0.0, "dt": dt}
    return VehiclePIDController(vehicle, args_lateral, args_longitudinal)


def set_spectator_to_ego_camera(base_env) -> None:
    if not getattr(base_env, "sensors", None):
        return
    cam = base_env.sensors[0].get("sensor")
    if cam is None:
        return
    base_env.world.get_spectator().set_transform(cam.get_transform())


def format_location(location: carla.Location) -> str:
    return f"x={location.x:+.2f} y={location.y:+.2f} z={location.z:+.2f}"


def _dot2d(ax: float, ay: float, bx: float, by: float) -> float:
    return float(ax) * float(bx) + float(ay) * float(by)


def _lane_key(wp: carla.Waypoint) -> Tuple[int, int, int]:
    return (int(wp.road_id), int(wp.section_id), int(wp.lane_id))


def _unit_xy(vec) -> Optional[Tuple[float, float]]:
    n = float(np.hypot(vec.x, vec.y))
    if n < 1e-6:
        return None
    return (float(vec.x) / n, float(vec.y) / n)


class RoadHeadingTracker:
    """Remembers which way *along the road* the ego was travelling.

    The vehicle yaw at the moment of a collision is useless for recovery: the
    policy may have turned it 90 deg into a wall, so the nearest driving lane can
    easily be the oncoming one. We therefore keep the last direction sampled
    while the car was healthy - on a driving lane, near its centre, and actually
    following it - and use that as the reference when snapping back.
    """

    MAX_LANE_OFFSET_M = 3.0  # ignore samples taken while off the lane centre
    MIN_SPEED_MPS = 0.5
    MIN_ALIGNMENT = 0.5  # ~60 deg; ignore samples taken while crossing a lane

    def __init__(self, world_map: carla.Map):
        self.map = world_map
        self.direction: Optional[Tuple[float, float]] = None  # world-frame unit xy
        self.waypoint: Optional[carla.Waypoint] = None  # last known-good lane centre

    def update(self, vehicle: carla.Actor) -> None:
        tf = vehicle.get_transform()
        wp = self.map.get_waypoint(
            tf.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if wp is None:
            return
        if tf.location.distance(wp.transform.location) > self.MAX_LANE_OFFSET_M:
            return  # off-road / mid-crash: not a trustworthy sample

        vel = vehicle.get_velocity()
        speed = float(np.hypot(vel.x, vel.y))
        ref = (
            _unit_xy(vel) if speed >= self.MIN_SPEED_MPS else _unit_xy(tf.get_forward_vector())
        )
        lane_dir = _unit_xy(wp.transform.get_forward_vector())
        if ref is None or lane_dir is None:
            return

        align = _dot2d(lane_dir[0], lane_dir[1], ref[0], ref[1])
        if abs(align) < self.MIN_ALIGNMENT:
            return  # cutting across the lane rather than following it

        sign = 1.0 if align > 0.0 else -1.0
        self.direction = (sign * lane_dir[0], sign * lane_dir[1])
        self.waypoint = wp


def road_lane_candidates(
    world_map: carla.Map, wp: carla.Waypoint, max_lane_id: int = 8
) -> List[carla.Waypoint]:
    """Every driving lane on `wp`'s road at the same `s`, both sides of the centre line.

    Walking get_left_lane()/get_right_lane() is unreliable for this: the hops
    mirror once you cross the centre line, so the oncoming carriageway is never
    reached.
    """
    found = []
    for lane_id in range(-max_lane_id, max_lane_id + 1):
        if lane_id == 0:
            continue
        cand = world_map.get_waypoint_xodr(wp.road_id, lane_id, wp.s)
        if cand is not None and cand.lane_type == carla.LaneType.Driving:
            found.append(cand)
    return found


def expand_anchors(
    world_map: carla.Map, wp: carla.Waypoint, dists: Tuple[float, ...] = (5.0, 10.0, 15.0, 20.0)
) -> List[carla.Waypoint]:
    """`wp` plus the lanes a short walk along the road from it.

    Inside a junction the connector road carries a single one-way lane, so the
    crash road alone offers nothing to match against; walking out of it reaches
    the ordinary road whose oncoming lanes we need.
    """
    anchors = [wp]
    seen = {(wp.road_id, wp.lane_id)}
    for d in dists:
        for hop in ("next", "previous"):
            for cand in getattr(wp, hop)(d):
                key = (cand.road_id, cand.lane_id)
                if key in seen:
                    continue
                seen.add(key)
                anchors.append(cand)
    return anchors


def pick_lane_matching_direction(
    world_map: carla.Map,
    anchors: List[carla.Waypoint],
    location: carla.Location,
    ref_dir: Tuple[float, float],
    min_alignment: float = 0.25,
    max_distance_m: float = 25.0,
) -> Optional[carla.Waypoint]:
    """Nearest lane on the anchors' roads whose travel direction agrees with `ref_dir`."""
    best: Optional[carla.Waypoint] = None
    best_dist: Optional[float] = None
    for anchor in anchors:
        for cand in road_lane_candidates(world_map, anchor):
            lane_dir = _unit_xy(cand.transform.get_forward_vector())
            if lane_dir is None:
                continue
            if _dot2d(lane_dir[0], lane_dir[1], ref_dir[0], ref_dir[1]) <= min_alignment:
                continue
            d = float(location.distance(cand.transform.location))
            if d > max_distance_m:
                continue
            if best_dist is None or d < best_dist:
                best, best_dist = cand, d
    return best


def nearest_driving_spawn_from_transform(
    world_map: carla.Map,
    transform: carla.Transform,
    ref_dir: Optional[Tuple[float, float]] = None,
    fallback_wp: Optional[carla.Waypoint] = None,
) -> Optional[carla.Transform]:
    wp = world_map.get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if wp is None:
        print("[RECOVERY] Could not find a valid driving waypoint.")
        return None

    # The policy ignores lane rules, so the lane nearest the crash often runs
    # against the way the car was going. Respawn in the nearest lane (on the
    # crash road, or the last road the car was cleanly on) whose own direction
    # matches - the yaw comes from a lane that genuinely runs that way, not from
    # flipping a lane's yaw, so the car is left legally placed on the road.
    if ref_dir is not None:
        anchors = expand_anchors(world_map, wp)
        if fallback_wp is not None:
            anchors += expand_anchors(world_map, fallback_wp)
        chosen = pick_lane_matching_direction(world_map, anchors, transform.location, ref_dir)
        if chosen is None:
            chosen = wp
            print(
                "[RECOVERY] No driving lane here runs the pre-collision way "
                "(one-way road?); keeping the nearest lane."
            )
        elif _lane_key(chosen) != _lane_key(wp):
            print(
                f"[RECOVERY] Nearest lane {_lane_key(wp)} opposes the pre-collision "
                f"heading; using lane {_lane_key(chosen)} instead."
            )
        wp = chosen

    # 1. Get the base location and rotation
    spawn_loc = wp.transform.location
    spawn_rot = wp.transform.rotation

    # 2. Create a brand new transform with the adjusted Z value
    safe_drop_height = 2.0
    new_transform = carla.Transform(
        carla.Location(x=spawn_loc.x, y=spawn_loc.y, z=spawn_loc.z + safe_drop_height),
        spawn_rot
    )

    print(
        f"[RECOVERY] Snapping to nearest driving lane at "
        f"{format_location(new_transform.location)} yaw={spawn_rot.yaw:+.1f}"
    )

    return new_transform


def disable_label(world: carla.World, label) -> int:
    objs = world.get_environment_objects(label)
    ids = {o.id for o in objs}
    if ids:
        world.enable_environment_objects(ids, False)
    return len(ids)


def disable_common_environment_objects(world: carla.World) -> None:
    # Use names to stay compatible with CARLA versions that may expose different enums.
    label_names = ("Poles", "StreetLights") # "Trees"
    for name in label_names:
        label = getattr(carla.CityObjectLabel, name, None)
        if label is None:
            print(f"[ENV] CityObjectLabel.{name} not found; skipping")
            continue
        count = disable_label(world, label)
        print(f"[ENV] Disabled {count} environment objects for label={name}")


def attach_collision_sensor(base_env, collision_events: List[carla.CollisionEvent]):
    bp = base_env.world.get_blueprint_library().find("sensor.other.collision")
    sensor = base_env.world.spawn_actor(
        bp, carla.Transform(), attach_to=base_env.vehicle
    )

    def _on_collision(event: carla.CollisionEvent):
        other_actor = event.other_actor.type_id if event.other_actor else "unknown"
        if "sidewalk" in other_actor:
            print(f"[COLLISION] with={other_actor}, ignore")
            pass
        else:
            collision_events.append(event)
            impulse = event.normal_impulse
            intensity = float(np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2))
            
            print(f"[COLLISION] with={other_actor} intensity={intensity:.2f}")

    sensor.listen(_on_collision)
    return sensor
