"""Train a PPO policy for the custom simulated ping pong arm."""
import argparse
from pathlib import Path

import gymnasium as gym
import gymnasium_env  # noqa: F401 — registers the custom environment
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

ENV_ID = "gymnasium_env/PingPongEnv-v0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/current_run"))
    parser.add_argument("--resume", type=Path, help="Optional Stable-Baselines3 PPO checkpoint")
    parser.add_argument("--normalization", type=Path, help="VecNormalize statistics for a resumed checkpoint")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    vector_env = make_vec_env(ENV_ID, n_envs=args.envs, seed=args.seed)
    stats_path = args.normalization
    if stats_path is None and args.resume is not None:
        candidate = args.resume.parent / "vecnormalize.pkl"
        if candidate.exists():
            stats_path = candidate

    if stats_path is not None:
        vector_env = VecNormalize.load(str(stats_path), vector_env)
        vector_env.training = True
    elif args.resume is None:
        vector_env = VecNormalize(vector_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if args.resume:
        model = PPO.load(str(args.resume), env=vector_env, device=args.device)
    else:
        model = PPO(
            "MlpPolicy",
            vector_env,
            n_steps=1024,
            batch_size=64,
            n_epochs=4,
            gamma=0.999,
            gae_lambda=0.98,
            ent_coef=0.01,
            tensorboard_log=str(args.output / "tensorboard"),
            verbose=1,
            seed=args.seed,
            device=args.device,
        )

    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=args.resume is None)
    model.save(args.output / "ppo_pingpong")
    if isinstance(vector_env, VecNormalize):
        vector_env.save(str(args.output / "vecnormalize.pkl"))
    vector_env.close()
    print(f"Saved policy and normalization statistics under {args.output}")


if __name__ == "__main__":
    main()
