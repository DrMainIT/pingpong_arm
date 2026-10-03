# Robot Table Tennis with Reinforcement Learning

An experimental project exploring whether a tabletop robot arm could learn to return a ping pong ball. It combines a custom MuJoCo scene and Gymnasium environment with PPO training, ONNX policy exports, and an early remote-inference-to-motor-control prototype.

**Status: unfinished research prototype.** I built and iterated the robot and table models, trained and evaluated many PPO runs, exported policies to ONNX, and experimented with remote inference and motor commands. I did not achieve reliable rallies or a complete game against a person.

## Simulation

![Robot arm and ball in the simulated table-tennis scene](media/simulation-scene.png)

The robot and racket geometry were iterated alongside the simulation:

![Robot arm and racket model](media/robot-racket-model.png)

[Watch the early simulation recording](media/pingpong-simulation.mp4)

## Project highlights

- Custom `PingPongEnv` environment in `gymnasium_env/`, with robot joint state, ball, racket, and target information in the observation.
- PPO training and interactive policy playback from root-level `train.py` and `play.py`.
- 60 archived ping pong training runs, including checkpoints, evaluation files, monitor logs, and TensorBoard event data under `results/ppo/`.
- Four exported PPO policy variants and an ONNX Runtime inference service under `deployment/onnx/`.
- A FastAPI inference experiment and Raspberry Pi GPIO client exploring how policy outputs could reach servo motors. The motor client is dry-run by default; the full real-time hardware loop was not validated.
- URDF and MuJoCo XML robot, racket, ball, and table models under `urdf/` and mesh assets under `stl/`. An optional PyBullet model viewer is in `scripts/view_in_bullet.py`.

## Repository layout

```text
.
├── gymnasium_env/       # Project's custom Gymnasium environments
├── train.py             # PPO training and optional checkpoint continuation
├── play.py              # Interactive checkpoint playback
├── scripts/             # Optional PyBullet viewer
├── deployment/onnx/     # ONNX models, inference service, and Pi prototype
├── results/
│   ├── models/          # Curated checkpoints
│   └── ppo/             # Archived runs, evaluations, and TensorBoard events
├── media/               # Simulation recording and screenshots
├── urdf/                # Robot, racket, and table descriptions
└── stl/                 # Robot mesh assets
```

The ping pong project is kept separate from the air hockey experiments in the original workspace. Generic Gymnasium tutorial code and the unrelated demo notebook were removed from this repository layout.

## Install

Python 3.10 or newer is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

MuJoCo's viewer also needs a working graphics environment.

## Train

```bash
python train.py --timesteps 100000 --envs 4 --seed 0
```

The new run, normalization statistics, and TensorBoard logs are saved to `results/current_run/`. To continue from a Stable-Baselines3 checkpoint:

```bash
python train.py --resume results/models/ppo_pingpong_run_60_best.zip --timesteps 100000
```

If the checkpoint was trained with observation/reward normalization, pass its saved statistics with `--normalization path/to/vecnormalize.pkl`.

## Play a saved policy

The default is the best checkpoint from archived run 60:

```bash
python play.py
```

Choose another Stable-Baselines3 checkpoint with `--model path/to/model.zip`. If the run used `VecNormalize`, also pass `--normalization path/to/vecnormalize.pkl`.

## ONNX and hardware experiments

See [`deployment/onnx/README.md`](deployment/onnx/README.md) for the local FastAPI/ONNX Runtime prototype and the experimental Raspberry Pi client. The deployment artifacts show an inference and motor-command exploration; they do not demonstrate reliable or safe human-versus-robot play.

## What I learned

Table tennis is a difficult contact-rich control problem: the policy must predict a fast ball, time a strike, produce a useful return, and map simulation actions onto calibrated hardware. The project gave me hands-on work in robot modeling, custom RL environments, PPO training, model export, and deployment prototyping, while making clear how much remained between simulation experiments and reliable play.
