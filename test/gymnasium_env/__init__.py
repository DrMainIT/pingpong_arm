from gymnasium.envs.registration import register

register(
    id="gymnasium_env/GridWorld-v0",
    entry_point="gymnasium_env.envs:GridWorldEnv",
)


register(
    id="gymnasium_env/Pusher-v0",
    entry_point="gymnasium_env.envs:PusherEnv",
)

register(
    id="gymnasium_env/Baunce-v0",
    entry_point="gymnasium_env.envs:BaunceEnv",
)