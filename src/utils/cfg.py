from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file into a Python dictionary.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the parsed configuration data.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)
