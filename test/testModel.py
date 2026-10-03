import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import gymnasium_env
from icecream import ic
import numpy as np
# Carica il modello salvato
model = PPO.load("logs/ppo/gymnasium_env-PingPongEnv-v0_39/gymnasium_env-PingPongEnv-v0.zip")


eval_env = gym.make("gymnasium_env/PingPongEnv-v0", render_mode="human")

obs, _ = eval_env.reset()  # Unpack the tuple
episode_over = False
for _ in range(1000):
    ic(obs)
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = eval_env.step(action)  # Update unpacking here
    ic("Action: ", action)
    if terminated:
        print("Terminated")
    if truncated:
        print("Truncated")

    episode_over = terminated or truncated
    if episode_over:
        osb, _ = eval_env.reset()


eval_env.close()
