import json
import os
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    if not config_path:
        return {}
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.lower().endswith(('.yaml', '.yml')):
            return yaml.safe_load(f) or {}
        if config_path.lower().endswith('.json'):
            return json.load(f)
        raise ValueError("Unsupported config format. Use .yaml, .yml, or .json")


def apply_config(args: Any, config: Dict[str, Any]) -> None:
    for key, value in config.items():
        if hasattr(args, key) and value is not None:
            setattr(args, key, value)
