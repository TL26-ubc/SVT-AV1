from gymnasium.envs.registration import register
from .environment import Av1GymEnv
from .norm import Av1GymObsNormWrapper

register(
    id="Av1Env-v0",
    entry_point="av1gym.environment:Av1GymEnv",
)

__all__ = ["Av1GymEnv", "Av1GymObsNormWrapper"]