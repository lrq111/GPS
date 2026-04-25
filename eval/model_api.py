"""Compatibility import for shared external LLM API helpers."""

import importlib.util
from pathlib import Path

MODEL_API = Path(__file__).resolve().parents[1] / "verl" / "utils" / "model_api.py"
spec = importlib.util.spec_from_file_location("_gps_model_api", MODEL_API)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

for name, value in vars(module).items():
    if not name.startswith("_"):
        globals()[name] = value
