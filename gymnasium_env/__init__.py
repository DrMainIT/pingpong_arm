"""Custom Gymnasium environments for the ping pong arm project."""

from gymnasium.envs.registration import register

register(
    id="gymnasium_env/PingPongEnv-v0",
    entry_point="gymnasium_env.envs:PingPongEnv",
)
