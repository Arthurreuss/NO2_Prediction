from typing import Any, Dict

import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)
