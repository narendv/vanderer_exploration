import carla
from typing import Optional

from .v0 import CarlaMultipathEnvBase

CLEAR_NOON_WEATHER = carla.WeatherParameters(
    cloudiness=0.0,
    precipitation=0.0,
    precipitation_deposits=0.0,
    wind_intensity=0.0,
    sun_azimuth_angle=0.0,
    sun_altitude_angle=90.0,
    fog_density=0.0,
    wetness=0.0,
)


class CarlaNoRouteEnvBase(CarlaMultipathEnvBase):
    """
    Route-free CARLA env.

    Differences from v0:
    - No predefined ROUTES are required.
    - Spawns default to all map spawn points.
    - Reward/termination can optionally use `destination`; otherwise reward is always 0.
    """

    ROUTES = []
    destination: Optional[carla.Location] = None

    def __init__(self, dt: float = 0.05):
        super().__init__(dt=dt)
        self.spawns = self.world.get_map().get_spawn_points()

    def set_destination(self, destination: Optional[carla.Location]):
        self.destination = destination

    def reset(
        self,
        reload: bool = False,
        return_info: bool = False,
        seed: Optional[int] = None,
        spawn_noise_std: float = 0.0,
        action_exec_noise_std: float = 0.0,
        spawn_point: Optional[carla.Transform] = None,
        destination: Optional[carla.Location] = None,
    ):
        if destination is not None:
            self.destination = destination

        obs = super().reset(
            reload=reload,
            return_info=return_info,
            seed=seed,
            spawn_noise_std=spawn_noise_std,
            action_exec_noise_std=action_exec_noise_std,
            spawn_point=spawn_point,
        )
        self.start_pos = spawn_point if spawn_point is not None else self.reset_spawn

        return obs

    def _calc_reward(self, vehicle, routes):
        # Compatibility with v0 info schema.
        routes_reached = [False] * len(routes)

        if self.destination is None:
            return 0, routes_reached

        dest = self.destination
        if hasattr(dest, "location"):
            dest = dest.location

        ego_location = vehicle.get_location()
        best_distance = ego_location.distance(dest)
        if best_distance < self.R_CLOSE:
            return 1, routes_reached
        return 0, routes_reached


class CarlaTown04V1(CarlaNoRouteEnvBase):
    WORLD_NAME = "Town04_Opt"
    WEATHER = CLEAR_NOON_WEATHER


class CarlaTown10V1(CarlaNoRouteEnvBase):
    WORLD_NAME = "Town10HD_Opt"
    WEATHER = CLEAR_NOON_WEATHER


class CarlaTown01V1(CarlaNoRouteEnvBase):
    WORLD_NAME = "Town01_Opt"
    WEATHER = CLEAR_NOON_WEATHER


class CarlaTown02V1(CarlaNoRouteEnvBase):
    WORLD_NAME = "Town02_Opt"
    WEATHER = CLEAR_NOON_WEATHER


class CarlaTown03V1(CarlaNoRouteEnvBase):
    WORLD_NAME = "Town03_Opt"
    WEATHER = CLEAR_NOON_WEATHER


class CarlaTown05V1(CarlaNoRouteEnvBase):
    WORLD_NAME = "Town05_Opt"
    WEATHER = CLEAR_NOON_WEATHER


# Towns 06/07/12 ship without an "_Opt" map variant, so they use the plain name.
class CarlaTown06V1(CarlaNoRouteEnvBase):
    WORLD_NAME = "Town06"
    WEATHER = CLEAR_NOON_WEATHER


class CarlaTown07V1(CarlaNoRouteEnvBase):
    WORLD_NAME = "Town07"
    WEATHER = CLEAR_NOON_WEATHER


class CarlaTown12V1(CarlaNoRouteEnvBase):
    # Large tiled map (37 streamed tiles); loading and routing are much heavier
    # than the small towns.
    WORLD_NAME = "Town12"
    WEATHER = CLEAR_NOON_WEATHER
    # CARLA's 3000/2000 m defaults keep much of this 9.4 x 8.3 km map resident at
    # once, and rollouts teleport across it, so tiles churn constantly.
    TILE_STREAM_DISTANCE = 700.0
    ACTOR_ACTIVE_DISTANCE = 700.0
    # Without this the spectator stays parked at SPECTATOR_VIEWPOINT and pins a
    # second region of tiles for the whole run.
    SPECTATOR_FOLLOWS_EGO = True


class CarlaTown11V1(CarlaNoRouteEnvBase):
    # Even larger tiled map than Town12 (132 streamed tiles, ~20 x 21.5 km vs
    # Town12's 9.4 x 8.3 km); same streaming problem, likely worse.
    WORLD_NAME = "Town11"
    WEATHER = CLEAR_NOON_WEATHER
    # Starting point carried over from Town12's tuning (CARLA's 3000/2000 m
    # defaults are the same here) - not yet independently verified for this
    # map's size, so a smoke test + memory-plateau check are worth (re)running
    # before a long unattended job. Override via --tile_stream_distance /
    # --actor_active_distance if this turns out too loose (or too tight).
    TILE_STREAM_DISTANCE = 700.0
    ACTOR_ACTIVE_DISTANCE = 700.0
    SPECTATOR_FOLLOWS_EGO = True
