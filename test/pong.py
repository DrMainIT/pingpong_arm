"""
task1: allenare un modello da pesi esistenti, done
task2: capire multiprocessing 
task3: benchmarking modello allenato 
task4: modificare il centro dell'end effector 

"""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium
import gymnasium_env
from icecream import ic
from stable_baselines3.common.env_util import make_vec_env


train_env = make_vec_env("gymnasium_env/Pusher-v0", n_envs=20)

model = PPO("MlpPolicy", train_env, verbose=1)

# Aggiungi del codice di debug per stampare le dimensioni delle osservazioni
obs = train_env.reset()
# Addestra il modello
model.learn(total_timesteps=200000)
model.save("ppo_pusher")

train_env.close()

# # Evaluation: With screen
# eval_env = gym.make("gymnasium_env/Pusher-v0", render_mode="human")
# obs, _ = eval_env.reset()  # Unpack the tuple

# episode_over = False
# for _ in range(10000):
#     action, _ = model.predict(obs)
#     obs, reward, terminated, truncated, info = eval_env.step(action)  # Update unpacking here
#     if terminated:
#         print("Terminated")
#     if truncated:
#         print("Truncated")
#     print(reward)
#     episode_over = terminated or truncated
#     if episode_over:
#         osb, _ = eval_env.reset()
    

# eval_env.close()