from typing import Dict, Tuple

import numpy as np
import carla


# Sun altitudes below the horizon give unusably dark frames; widen this (the
# original range was -20..50) if night-time data is wanted.
DEFAULT_SUN_ALTITUDE_RANGE = (15.0, 90.0)

# Measured on Town03 at 320x240: density 3-6 is barely perceptible (CARLA's own
# presets sit here - ClearNoon=2, HardRainNoon=7), 10-15 is clearly visible haze
# with the far field softened but everything still legible, 25+ starts washing the
# far field out, and 60+ is close to a whiteout. This range is deliberately light.
DEFAULT_FOG_DENSITY_RANGE = (5.0, 15.0)

# CARLA pairs fog_density with these two values in every built-in preset; fog reads
# wrong without them (fog_distance is where fog starts, in meters).
FOG_DISTANCE = 0.75
FOG_FALLOFF = 0.1


def get_random_weather_config(
    rng: np.random.RandomState,
    sun_altitude_range: Tuple[float, float] = DEFAULT_SUN_ALTITUDE_RANGE,
    fog_prob: float = 0.0,
    fog_density_range: Tuple[float, float] = DEFAULT_FOG_DENSITY_RANGE,
) -> carla.WeatherParameters:
    """Sample randomized weather.

    Roughly half the samples are forced dry; the rest keep puddles and wetness
    correlated with precipitation so the scene stays physically plausible. Fog is
    applied to only a `fog_prob` fraction of samples, so most frames stay clear.

    Every parameter is drawn unconditionally so the number of RNG draws does not
    depend on the arguments.
    """
    result = carla.WeatherParameters()

    dry = rng.rand() < 0.5
    result.cloudiness = rng.uniform(low=0, high=100)
    result.precipitation = rng.uniform(low=0, high=100)
    result.precipitation_deposits = max(
        rng.uniform(low=0, high=100), result.precipitation - 10
    )  # puddles
    result.wind_intensity = rng.uniform(low=0, high=100)
    result.sun_azimuth_angle = rng.uniform(low=0, high=360)
    result.sun_altitude_angle = rng.uniform(
        low=sun_altitude_range[0], high=sun_altitude_range[1]
    )
    result.wetness = max(rng.uniform(low=0, high=100), result.precipitation - 10)

    # Both fog draws happen unconditionally to keep the draw count fixed.
    foggy = rng.rand() < fog_prob
    fog_density = rng.uniform(low=fog_density_range[0], high=fog_density_range[1])
    if foggy:
        result.fog_density = fog_density
        result.fog_distance = FOG_DISTANCE
        result.fog_falloff = FOG_FALLOFF
    else:
        result.fog_density = 0.0

    if dry:
        result.precipitation = 0
        result.precipitation_deposits = 0
        result.wetness = 0

    return result


def serialize_weather(weather: carla.WeatherParameters) -> Dict[str, float]:
    """Weather parameters as a plain dict, for recording in trajectory metadata."""
    return {
        "cloudiness": float(weather.cloudiness),
        "precipitation": float(weather.precipitation),
        "precipitation_deposits": float(weather.precipitation_deposits),
        "wind_intensity": float(weather.wind_intensity),
        "sun_azimuth_angle": float(weather.sun_azimuth_angle),
        "sun_altitude_angle": float(weather.sun_altitude_angle),
        "fog_density": float(weather.fog_density),
        "fog_distance": float(weather.fog_distance),
        "fog_falloff": float(weather.fog_falloff),
        "wetness": float(weather.wetness),
    }


def get_weather_config(rng: np.random.RandomState) -> carla.WeatherParameters:
    """Fixed clear/dry/calm weather. Consumes no RNG draws."""
    result = carla.WeatherParameters()

    # Simplest fixed weather (clear, dry, calm)
    result.cloudiness = 0.0
    result.precipitation = 0.0
    result.precipitation_deposits = 0.0
    result.wind_intensity = 0.0
    result.sun_azimuth_angle = 0.0
    result.sun_altitude_angle = 90.0
    result.fog_density = 0.0
    result.wetness = 0.0

    return result
