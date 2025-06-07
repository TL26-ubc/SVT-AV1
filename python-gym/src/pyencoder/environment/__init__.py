from gymnasium.envs.registration import register

register(
    id="Av1Env-v0",
    entry_point="src.environment:Av1Env"
)