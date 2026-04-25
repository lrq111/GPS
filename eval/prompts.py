"""Compatibility import for the canonical shared prompts."""

import importlib.util
from pathlib import Path

SHARED_PROMPTS = Path(__file__).resolve().parents[1] / "verl" / "utils" / "shared_prompts.py"
spec = importlib.util.spec_from_file_location("_gps_shared_prompts", SHARED_PROMPTS)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

for name, value in vars(module).items():
    if not name.startswith("_"):
        globals()[name] = value
