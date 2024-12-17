import gymnasium
import gymnasium_env
env = gymnasium.make('gymnasium_env/Pusher-v0',
                     render_mode='human')

observation, info = env.reset()

episode_over = False
for _ in range(10000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated:
        print("Terminated")
    if truncated:
        print("Truncated")
    
    episode_over = terminated or truncated
    if episode_over:
        observation, info = env.reset()

env.close()
