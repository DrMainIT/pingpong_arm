"""Check and run a single inference against one of the archived ONNX exports."""
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

model_path = Path(__file__).resolve().parent / "models" / "my_ppo_model.onnx"
onnx.checker.check_model(onnx.load(str(model_path)))
session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
shape = session.get_inputs()[0].shape
input_shape = [1, *(int(size) if isinstance(size, int) else 1 for size in shape[1:])]
observation = np.zeros(input_shape, dtype=np.float32)
print(session.run(None, {session.get_inputs()[0].name: observation}))
