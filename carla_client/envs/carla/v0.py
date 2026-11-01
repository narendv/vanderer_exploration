import carla
import numpy as np
import gymnasium as gym
import subprocess
from collections import deque
from typing import Tuple, Optional, List
import socket
import time
import atexit
import os
import shlex
from .weather import (
    DEFAULT_FOG_DENSITY_RANGE,
    DEFAULT_SUN_ALTITUDE_RANGE,
    get_random_weather_config,
)
from .utils import (
    deserialize_control_from_arr,
    preproc_carla_img,
    add_noise_to_transform,
    add_noise_to_action,
)


class CarlaMultipathEnvBase(gym.Env):
    CARLA_BIN = os.getenv("CARLA_BIN")
    WORLD_NAME = "Town10HD_Opt"
    CARLA_ARGV = ["CarlaUE4", "--world", WORLD_NAME]
    # Town12 in particular can take longer than 40s to answer an RPC under load
    # (tile streaming, GC pauses); override with CARLA_TIMEOUT if still too tight.
    TIMEOUT = int(os.getenv("CARLA_TIMEOUT", "90"))
    R_CLOSE = 4.0
    # list of (spawn, waypoints); last waypoint is destination
    ROUTES: List[Tuple[carla.Transform, List[carla.Location]]] = []
    SPECTATOR_VIEWPOINT = carla.Transform(
        carla.Location(x=-49.796131, y=22.167667, z=129.373856),
        carla.Rotation(pitch=-82.899261, yaw=-12.960452, roll=-0.001222),
    )
    # Large-map (tiled) streaming controls. None leaves the CARLA default in place,
    # which is correct for the single-tile towns where streaming never happens.
    #   tile_stream_distance : radius (m) within which map tiles stay resident
    #   actor_active_distance: radius (m) within which actors stay active
    # CARLA's defaults are 3000/2000 m, which on a 9.4 x 8.3 km map like Town12 keeps
    # a large fraction of all tiles loaded at once.
    TILE_STREAM_DISTANCE: Optional[float] = None
    ACTOR_ACTIVE_DISTANCE: Optional[float] = None
    # CARLA streams tiles around the spectator too (spectator_as_ego defaults True).
    # Parking the spectator far from the ego therefore pins a *second* region of tiles
    # for the whole run. Following the ego keeps it to a single region.
    SPECTATOR_FOLLOWS_EGO = False
    SPECTATOR_FOLLOW_HEIGHT = 40.0
    # Safety net on the per-sensor frame buffer (frames, ~307 KB each at 320x240).
    SENSOR_BUFFER_MAXLEN = 120

    WEATHER = None  # None for random weather
    # Weather is drawn from its own RNG stream, offset from the episode seed, so
    # that turning randomization on does not shift the spawn/action noise for a seed.
    WEATHER_SEED_OFFSET = 7919
    WEATHER_SUN_ALTITUDE_RANGE = DEFAULT_SUN_ALTITUDE_RANGE
    WEATHER_FOG_PROB = 0.0
    WEATHER_FOG_DENSITY_RANGE = DEFAULT_FOG_DENSITY_RANGE

    def __init__(
        self,
        dt: float = 0.05,
    ):
        super().__init__()
        self.dt = dt
        self.img_size = (320, 240)
        self.fov = 90

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, self.img_size[1], self.img_size[0]), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.frame = None
        self.rng = None
        self.weather = None
        self.carla_port = int(os.getenv("CARLA_PORT", "2000"))
        if not self.CARLA_BIN:
            raise RuntimeError(
                "Set CARLA_BIN to the server binary: <carla>/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
            )
        self.server_proc = subprocess.Popen([self.CARLA_BIN, *self._build_carla_argv()])
        atexit.register(self.server_proc.kill)
        start = time.time()
        while not self._is_open("localhost", self.carla_port, timeout=1):
            if (time.time() - start) > self.TIMEOUT:
                raise socket.timeout("Failed to connect to CARLA.")
        self.client = carla.Client("localhost", self.carla_port)
        self.client.set_timeout(self.TIMEOUT)
        self.client.load_world(self.WORLD_NAME)
        self.world = self.client.get_world()
        self.spawns = self.world.get_map().get_spawn_points()
        self.spectator = self.world.get_spectator()
        self.vehicle = None
        self.sensors = []
        self.camera_intrinsic = None
        self._set_synchronous(True)

    def _build_carla_argv(self):
        """
        Build CARLA server args, with optional off-screen rendering.

        Controls:
        - CARLA_OFFSCREEN=1|true|yes : append -RenderOffScreen
        - CARLA_SERVER_ARGS="<extra args>" : append parsed extra args
        """
        argv = ["CarlaUE4", "--world", self.WORLD_NAME]

        offscreen = os.getenv("CARLA_OFFSCREEN", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if offscreen:
            argv.append("-RenderOffScreen")
        argv.append(f"-carla-rpc-port={self.carla_port}")

        # Add quality level here
        if os.getenv("CARLA_QUALITY_LEVEL", "").strip().lower() == "low":
            argv.append("-quality-level=Low") 
            # Add -nosound as well to save extra CPU/Memory
            argv.append("-nosound")

        extra_args = os.getenv("CARLA_SERVER_ARGS", "").strip()
        if extra_args:
            argv.extend(shlex.split(extra_args))

        return argv

    def _is_open(self, host: str, port: int, timeout: float = 1.0):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((host, port))
            s.settimeout(timeout)
            s.shutdown(socket.SHUT_RDWR)
            return True
        except:
            return False

    def garbage_collect(self):
        """Tear down the ego and its sensors.

        Order matters. A sensor must be stopped before it is destroyed - destroying one
        that is still listening leaves frames in flight against a callback whose Python
        side is going away, which crashes the server - and sensors must go before the
        vehicle they are attached to, never after.
        """
        for elem in self.sensors:
            sensor = elem["sensor"]
            try:
                if sensor.is_listening:
                    sensor.stop()
            except RuntimeError:
                pass  # already gone; nothing to unsubscribe
            try:
                if sensor.is_alive:
                    sensor.destroy()
            except RuntimeError:
                pass
            elem["buffer"].clear()
        self.sensors = []

        if self.vehicle is not None:
            try:
                if self.vehicle.is_alive:
                    self.vehicle.destroy()
            except RuntimeError:
                pass
        self.vehicle = None

    def reset(
        self,
        reload: bool = False,
        return_info: bool = False,
        seed: Optional[int] = None,
        spawn_noise_std: float = 0.5,  # N(0, std) additive noise to spawn location
        action_exec_noise_std: float = 0.0,  # N(0, std) additive, but guaranteed throttle/brake axis does not flip sign
        spawn_point: Optional[carla.Transform] = None,
    ):
        if self.ROUTES is None:
            raise ValueError("No routes specified.")
        if reload:
            self.client.set_timeout(15)
            self.client.load_world(self.WORLD_NAME)
            self.world = self.client.get_world()
            self.spectator = self.world.get_spectator()
            self._set_synchronous(True)
        else:
            self.garbage_collect()
        self.vehicle = None
        self.sensors = []
        self.spawn_noise_std = spawn_noise_std
        self.action_exec_noise_std = action_exec_noise_std

        if seed is None:
            self.seed = np.random.randint(0, 4294967295)  # 0..2^32-1
        else:
            self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        self.reset_spawn = spawn_point if spawn_point is not None else self.rng.choice(self.spawns)
        # Weather uses a separate RNG stream keyed off the same seed, so it stays
        # reproducible without consuming draws from the spawn/action noise stream.
        if self.WEATHER is None:
            weather_rng = np.random.RandomState(
                int(self.seed + self.WEATHER_SEED_OFFSET) % (2 ** 32)
            )
            weather = get_random_weather_config(
                weather_rng,
                sun_altitude_range=self.WEATHER_SUN_ALTITUDE_RANGE,
                fog_prob=self.WEATHER_FOG_PROB,
                fog_density_range=self.WEATHER_FOG_DENSITY_RANGE,
            )
        else:
            weather = self.WEATHER
        self.weather = weather
        self.world.set_weather(weather)
        vehicle_blueprint = self._get_vehicle_blueprint_by_id(1)
        spawn_transform = add_noise_to_transform(
            self.reset_spawn, self.spawn_noise_std, self.rng
        )

        # Move the spectator to the spawn *before* spawning the vehicle: CARLA streams
        # tiles around the spectator, so this pages in the tiles we are about to need
        # and keeps the resident set to one region instead of two.
        if self.SPECTATOR_FOLLOWS_EGO:
            self.spectator.set_transform(self._overhead_transform(spawn_transform))

        self._spawn_vehicle(vehicle_blueprint, spawn_transform)
        self._spawn_camera(
            self.vehicle,
            self.img_size,
            self.fov,
        )

        # set spectator camera
        if not self.SPECTATOR_FOLLOWS_EGO:
            self.spectator.set_transform(self.SPECTATOR_VIEWPOINT)

        # make sure camera sensors receive the first image, and the car is on the ground
        for i in range(20):
            obs, _, _, _, info = self.step(np.array([0, 0]))

        if return_info:
            return obs, info
        else:
            return obs

    def step(self, action: np.ndarray):
        vel = self.vehicle.get_velocity().length() * 3.6
        if not (-1 <= action[0] <= 1):  # throttle/brake
            print("Throttle/brake action out of bounds: {}".format(action[0]))
        if not (-1 <= action[1] <= 1):  # steer
            print("Steer action out of bounds: {}".format(action[1]))
        action = np.clip(action, -1, 1)
        action = add_noise_to_action(action, self.action_exec_noise_std, self.rng)
        action = deserialize_control_from_arr(action)
        self.vehicle.apply_control(action)
        self.frame = self.world.tick()
        self._green_light_to_vehicle(self.vehicle)
        state = []  # observations
        obs_frame = -1
        for sensor in self.sensors:
            if len(sensor["buffer"]) > 0:
                elem = sensor["buffer"].popleft()
            else:
                elem = None
            state.append(elem)
        if len(state) == 1:
            state = state[0]
        if state is not None:
            obs_frame = state.frame
            state = preproc_carla_img(state, channel_first=True).copy()

        # reward, routes_reached = self._calc_reward(self.vehicle, self.active_routes)
        # for i in range(len(self.active_routes_used)):
        #     self.active_routes_used[i] = self.active_routes_used[i] or routes_reached[i]
        # terminated = reward == 1
        terminated = False
        truncated = False
        reward = 0
        info = {
            "frame": self.frame,
            "obs_frame": obs_frame,
            # "routes_used": self.active_routes_used,
        }
        return state, reward, terminated, truncated, info

    def _calc_reward(self, vehicle, routes):
        dests = [x[1][-1] for x in routes]  # destinations
        ego_location = vehicle.get_location()
        routes_reached = [False] * len(routes)
        for i, route in enumerate(routes):
            # update whether ego vehicle has reached the waypoints
            # NOTE: assuming only destination overlaps between routes! Really should be counting
            for p in route[1][:-1]:  # not including the destination
                if p.distance(ego_location) < self.R_CLOSE:
                    routes_reached[i] = True
        # so _calc_reward really becomes _update_route_checkpoints
        # and done can be determined by checking if any destination flag is True
        best_distance = min([ego_location.distance(x) for x in dests])
        if best_distance < self.R_CLOSE:
            return 1, routes_reached
        else:
            return 0, routes_reached

    def render(self, mode):
        pass

    def set_state(self):
        pass

    def close(self):
        self._set_synchronous(False)
        self.garbage_collect()
        self.server_proc.kill()
        self.server_proc.wait()

    def _overhead_transform(self, transform: carla.Transform) -> carla.Transform:
        """Top-down view above `transform`, for when the spectator follows the ego."""
        loc = transform.location
        return carla.Transform(
            carla.Location(x=loc.x, y=loc.y, z=loc.z + self.SPECTATOR_FOLLOW_HEIGHT),
            carla.Rotation(pitch=-90.0, yaw=transform.rotation.yaw, roll=0.0),
        )

    def _set_synchronous(self, synchronous: bool = True):
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous
        settings.fixed_delta_seconds = self.dt
        if self.TILE_STREAM_DISTANCE is not None:
            settings.tile_stream_distance = float(self.TILE_STREAM_DISTANCE)
        if self.ACTOR_ACTIVE_DISTANCE is not None:
            settings.actor_active_distance = float(self.ACTOR_ACTIVE_DISTANCE)
        self.world.apply_settings(settings)

    def _get_vehicle_blueprint_by_id(self, blueprint_id: int, randomize: bool = False):
        bps = self.world.get_blueprint_library().filter("vehicle.*")
        bps = sorted(bps, key=lambda bp: bp.id)
        bp = bps[blueprint_id]
        for attr in ["color", "driver_id"]:
            if bp.has_attribute(attr):
                vals = bp.get_attribute(attr).recommended_values
                if randomize:
                    val = self.rng.choice(vals)
                else:
                    val = vals[0]
                bp.set_attribute(attr, val)
        bp.set_attribute("role_name", "hero")
        return bp

    def _spawn_vehicle(
        self, vehicle_blueprint: carla.ActorBlueprint, spawn_point: carla.Transform
    ):
        self.vehicle = self.world.spawn_actor(vehicle_blueprint, spawn_point)
        light_state = (
            carla.VehicleLightState.NONE
            | carla.VehicleLightState.LowBeam
            | carla.VehicleLightState.Position
        )
        self.vehicle.set_light_state(carla.VehicleLightState(light_state))
        return self.vehicle

    def _register_sensor_buffer(self, sensor: carla.Sensor, buffer: deque):
        def callback(data):
            buffer.append(data)

        sensor.listen(callback)

    def _spawn_camera(
        self,
        vehicle: carla.Vehicle,
        size: Tuple[int, int] = (320, 240),
        fov: float = 90,
        transform: Optional[carla.Transform] = None,
    ):
        FRONT = carla.Transform(carla.Location(x=1.0, z=1.7))
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(size[0]))
        bp.set_attribute("image_size_y", str(size[1]))
        bp.set_attribute("fov", str(fov))
        bp.set_attribute("sensor_tick", str(self.dt))
        self.camera_intrinsic = np.array(
            [
                [size[0] / (2 * np.tan(fov * np.pi / 360)), 0, size[0] / 2],
                [0, size[1] / (2 * np.tan(fov * np.pi / 360)), size[1] / 2],
                [0, 0, 1],
            ]        )
        if transform is None:
            transform = FRONT
        camera = self.world.spawn_actor(bp, transform, attach_to=vehicle)
        # Bounded: step() pops one frame per tick and sensor_tick == dt, so this should
        # never bind. It caps the damage if a frame ever arrives late, since each frame
        # is ~307 KB and an unbounded buffer is held alive by the callback closure.
        buffer = deque(maxlen=self.SENSOR_BUFFER_MAXLEN)
        self._register_sensor_buffer(camera, buffer)
        self.sensors.append(
            {
                "sensor": camera,
                "buffer": buffer,
            }
        )
        return camera

    def _green_light_to_vehicle(self, vehicle: carla.Vehicle):
        if vehicle.is_at_traffic_light():
            traffic_light = vehicle.get_traffic_light()
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                traffic_light.set_state(carla.TrafficLightState.Green)
