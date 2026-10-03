# ONNX deployment prototype

This directory contains the project's four exported PPO ONNX models, an ONNX Runtime inference service, a small HTTP client, and an experimental Raspberry Pi GPIO client.

Install and run the FastAPI service locally:

```bash
cd deployment/onnx
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

The service loads `models/my_ppo_model4.onnx` by default. Set `ONNX_MODEL_PATH` to select another model. Use `python a.py` for a local request. Set `PINGPONG_INFERENCE_URL` to configure the client address.

The Raspberry Pi client runs in dry-run mode by default; `--enable-motors` opts into GPIO output. The observation format, action scaling, servo calibration, timing, and hardware loop have not been validated together for reliable play. Treat the API and Dockerfile as deployment experiments, not production robot control. Do not expose the service to an untrusted network.
