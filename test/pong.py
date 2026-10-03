"""
task1: allenare un modello da pesi esistenti, done
task2: capire multiprocessing
task3: benchmarking modello allenato
task4: modificare il centro dell'end effector

"""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import gymnasium
import gymnasium_env
from icecream import ic
import torch as th

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps, BaseCallback


train_env = make_vec_env("gymnasium_env/PingPongEnv-v0", n_envs=4)
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
#policy_kwargs = dict(net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])])

model = PPO(
    "MlpPolicy",
    train_env,
    #policy_kwargs=policy_kwargs,
    n_steps = 1024,
    batch_size = 64,
    n_epochs = 4,
    gamma = 0.999,
    gae_lambda = 0.98,
    ent_coef = 0.01,
    verbose=1)

# checkpoint_callback = CheckpointCallback(
#   save_freq=50000,
#   save_path="./logs",
#   name_prefix="model",
#   #save_replay_buffer=True,
#   #save_vecnormalize=True,
# )

# Aggiungi del codice di debug per stampare le dimensioni delle osservazioni
obs = train_env.reset()
# Addestra il modello
model.learn(total_timesteps=100000,progress_bar=True) # callback=checkpoint_callback
model.save("ppo_pusher")

train_env.close()

# Evaluation: With screen
eval_env = gym.make("gymnasium_env/PingPongEnv-v0", render_mode="human")
obs, _ = eval_env.reset()  # Unpack the tuple
episode_over = False
for _ in range(10000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = eval_env.step(action)  # Update unpacking here
    if terminated:
        print("Terminated")
    if truncated:
        print("Truncated")
    print(reward)
    episode_over = terminated or truncated
    if episode_over:
        osb, _ = eval_env.reset()

eval_env.close()
