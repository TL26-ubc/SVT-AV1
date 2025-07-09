from gymnasium.envs.registration import register
from .environment import Av1GymEnv

register(
    id="Av1Env-v0",
    entry_point="av1gym.environment:Av1GymEnv",
)

__all__ = ["Av1GymEnv"]