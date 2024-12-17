import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium_env
from icecream import ic

# Carica il modello salvato
model = PPO.load("ppo_pusherB")


eval_env = gym.make("gymnasium_env/Pusher-v0", render_mode="human")
obs, _ = eval_env.reset()  # Unpack the tuple

episode_over = False
for _ in range(10000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = eval_env.step(action)  # Update unpacking here
    if terminated:
        print("Terminated")
    if truncated:
        print("Truncated")
    
    episode_over = terminated or truncated
    if episode_over:
        osb, _ = eval_env.reset()
    

eval_env.close()


#95343580527
