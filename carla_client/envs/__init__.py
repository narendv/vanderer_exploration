import logging
from gymnasium.envs.registration import register

try:
    import carla

    register(
        id="carla-town10-v1",
        entry_point="envs.carla.v1:CarlaTown10V1",
        max_episode_steps=200000,
        reward_threshold=1.0,
    )
    register(
        id="carla-town04-v1",
        entry_point="envs.carla.v1:CarlaTown04V1",
        max_episode_steps=200000,
        reward_threshold=1.0,
    )
    register(
        id="carla-town01-v1",
        entry_point="envs.carla.v1:CarlaTown01V1",
        max_episode_steps=200000,
        reward_threshold=1.0,
    )
    register(
        id="carla-town02-v1",
        entry_point="envs.carla.v1:CarlaTown02V1",
        max_episode_steps=200000,
        reward_threshold=1.0,
    )
    register(
        id="carla-town03-v1",
        entry_point="envs.carla.v1:CarlaTown03V1",
        max_episode_steps=200000,
        reward_threshold=1.0,
    )
    register(
        id="carla-town05-v1",
        entry_point="envs.carla.v1:CarlaTown05V1",
        max_episode_steps=200000,
        reward_threshold=1.0,
    )
    register(
        id="carla-town06-v1",
        entry_point="envs.carla.v1:CarlaTown06V1",
        max_episode_steps=200000,
        reward_threshold=1.0,
    )
    register(
        id="carla-town07-v1",
        entry_point="envs.carla.v1:CarlaTown07V1",
        max_episode_steps=200000,
        reward_threshold=1.0,
    )
    register(
        id="carla-town12-v1",
        entry_point="envs.carla.v1:CarlaTown12V1",
        max_episode_steps=200000,
        reward_threshold=1.0,
    )
    register(
        id="carla-town11-v1",
        entry_point="envs.carla.v1:CarlaTown11V1",
        max_episode_steps=200000,
        reward_threshold=1.0,
    )
except ImportError:
    logging.warning("Carla not installed, skipping")
    pass
