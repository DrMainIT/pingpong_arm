import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium_env
from icecream import ic

# Crea l'ambiente di addestramento
train_env = make_vec_env("gymnasium_env/Pusher-v0", n_envs=10)

# Carica il modello salvato
model = PPO.load("ppo_pusherB", env=train_env)

# Aggiungi del codice di debug per stampare le dimensioni delle osservazioni
obs = train_env.reset()
ic(obs.shape)  # Stampa la forma delle osservazioni per il debug
# Continua l'addestramento del modello
model.learn(total_timesteps=100000)

# Salva il modello addestrato
model.save("ppo_pusher")

train_env.close()

# Valutazione: Con schermo
eval_env = gym.make("gymnasium_env/Pusher-v0", render_mode="human")
obs, _ = eval_env.reset()  # Scompatta la tupla
episode_over = False
for _ in range(10000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = eval_env.step(action)  # Aggiorna lo unpacking
    if terminated:
        print("Terminated")
    if truncated:
        print("Truncated")
    print(reward)
    episode_over = terminated or truncated
    if episode_over:
        obs, _ = eval_env.reset()

