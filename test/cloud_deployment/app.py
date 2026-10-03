"""Experimental FastAPI wrapper around an exported PPO ONNX policy.

Run from this directory with ONNX_MODEL_PATH pointing at the desired export.
This endpoint is a research prototype, not a production robot-control service.
"""
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from roboticstoolbox import DHRobot, RevoluteDH

MODEL_PATH = Path(os.environ.get("ONNX_MODEL_PATH", Path(__file__).parent / "models" / "my_ppo_model4.onnx"))
onnx.checker.check_model(onnx.load(str(MODEL_PATH)))
SESSION = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
ROBOT = DHRobot([
    RevoluteDH(a=0.10, alpha=0),
    RevoluteDH(a=0.07, alpha=0),
    RevoluteDH(a=0.80, alpha=0),
    RevoluteDH(a=1.20, alpha=0),
])

app = FastAPI(title="Ping Pong Arm ONNX Inference Prototype")


class InferenceRequest(BaseModel):
    observation: list[list[float]]


@app.get("/")
async def read_root():
    return {"service": "ping-pong-arm-onnx-prototype", "model": MODEL_PATH.name}


@app.post("/process-data")
async def process_data(request: InferenceRequest):
    observation = np.asarray(request.observation, dtype=np.float32)
    input_name = SESSION.get_inputs()[0].name
    try:
        outputs = SESSION.run(None, {input_name: observation})
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"ONNX inference failed: {exc}") from exc
    actions = outputs[0]
    result = {"actions": actions.tolist()}
    if len(outputs) > 1:
        result["values"] = outputs[1].tolist()
    if len(outputs) > 2:
        result["log_prob"] = outputs[2].tolist()
    if actions.ndim == 2 and actions.shape[0] and actions.shape[1] == 4:
        result["end_effector_position"] = ROBOT.fkine(actions[0]).t.tolist()
    return result
