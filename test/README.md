# Training and experiment archive

This folder contains the custom Gymnasium environments, PPO scripts, saved ping pong checkpoints, evaluation data, and TensorBoard event files for the robot table-tennis project. The air hockey experiments from the original mixed workspace are not included here. ONNX export/runtime tests are represented by the project-owned code and models in `cloud_deployment/`; the upstream [SB3-to-Coral example](https://github.com/chunky/sb3_to_coral) was used as a reference.

Start with the repository-level [README](../README.md) for project context, installation, and the simulation gallery. See [`cloud_deployment/README.md`](cloud_deployment/README.md) for the ONNX Runtime and remote inference prototype.

Archived PPO logs preserve iterations rather than a single claimed final policy. Several scripts originated as experiments and may need local configuration changes before use.
