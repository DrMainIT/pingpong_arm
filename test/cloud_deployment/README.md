# ONNX inference and motor-control prototype

This folder preserves the experimental deployment path: exported PPO policies, an ONNX Runtime inference endpoint, a small HTTP client, and Raspberry Pi GPIO servo code.

The endpoint accepts JSON shaped as `{"observation": [[...]]}` at `POST /process-data`. It loads `models/my_ppo_model4.onnx` by default; set `ONNX_MODEL_PATH` to select another export. Start the service with:

```bash
cd test/cloud_deployment
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

Use `python a.py` for a local inference request. The Raspberry Pi client is dry-run by default. Hardware output requires `--enable-motors` and local calibration. The observation sample, ONNX variants, output-to-servo mapping, and Denavit–Hartenberg model come from exploratory work; they have not been validated as a complete real-time robot control loop. Do not expose the prototype endpoint to an untrusted network.

The Dockerfile is provided as a packaging experiment; the model and API have not been validated as a production deployment.
