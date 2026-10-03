"""Small client for exercising the experimental inference endpoint."""
import os

import requests

url = os.environ.get("PINGPONG_INFERENCE_URL", "http://127.0.0.1:8000/process-data")
observation = [[0.3, 1.0, 4.5, 0.0, 0.0, 0.0, 2.10034773, -0.56972212, 1.62007707]]
response = requests.post(url, json={"observation": observation}, timeout=10)
response.raise_for_status()
print(response.json())
