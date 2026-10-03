"""Small client for exercising the experimental inference endpoint."""
import os

import requests

url = os.environ.get("PINGPONG_INFERENCE_URL", "http://127.0.0.1:8000/process-data")
observation = [[0.0] * 23]  # PingPongEnv currently emits 23 values
response = requests.post(url, json={"observation": observation}, timeout=10)
response.raise_for_status()
print(response.json())
