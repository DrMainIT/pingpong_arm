"""Run a saved PPO checkpoint in the interactive MuJoCo viewer."""
import argparse
from pathlib import Path

import gymnasium as gym
import gymnasium_env  # noqa: F401 — registers the custom environment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ENV_ID = "gymnasium_env/PingPongEnv-v0"
DEFAULT_MODEL = Path("results/models/ppo_pingpong_run_60.zip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--normalization", type=Path, help="Optional VecNormalize statistics")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    env = DummyVecEnv([lambda: gym.make(ENV_ID, render_mode="human")])
    stats_path = args.normalization
    if stats_path is None:
        candidate = args.model.parent / "vecnormalize.pkl"
        if candidate.exists():
            stats_path = candidate
    if stats_path is not None:
        env = VecNormalize.load(str(stats_path), env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(str(args.model), env=env, device="cpu")
    obs = env.reset()
    completed = 0
    try:
        while completed < args.episodes:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, _, dones, _ = env.step(action)
            env.render()
            if dones[0]:
                completed += 1
    finally:
        env.close()


if __name__ == "__main__":
    main()
