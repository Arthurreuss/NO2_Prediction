from typing import Any, Dict

import optuna


def set_hyperparameter_ranges(
    cfg: Dict[str, Any], model_type: str, trial: optuna.Trial
) -> Dict[str, Any]:
    """Sample hyperparameters for a given model type using Optuna.

    This function reads the hyperparameter search space for `model_type`
    from the configuration dictionary and uses the provided Optuna trial
    to sample concrete hyperparameter values.

    Supported hyperparameter types in the configuration:
      - "int": sampled via `trial.suggest_int`
      - "float": sampled via `trial.suggest_float`
      - "categorical": sampled via `trial.suggest_categorical`

    Args:
        cfg: Configuration dictionary containing model optimization ranges
            under `cfg["models"]["optimization_ranges"]`.
        model_type: Key identifying the model whose hyperparameter space
            should be used.
        trial: Active Optuna trial used to sample hyperparameters.

    Returns:
        A dictionary mapping hyperparameter names to the sampled values
        for the current Optuna trial.

    Raises:
        ValueError: If an unknown hyperparameter type is encountered in
            the configuration.
    """
    space = cfg["models"]["optimization_ranges"][model_type]

    hp: Dict[str, Any] = {}

    for param, spec in space.items():
        if spec["type"] == "int":
            hp[param] = trial.suggest_int(
                param,
                spec["min"],
                spec["max"],
                step=spec.get("step"),
            )

        elif spec["type"] == "float":
            hp[param] = trial.suggest_float(
                param,
                float(spec["min"]),
                float(spec["max"]),
                log=spec.get("log", False),
            )

        elif spec["type"] == "categorical":
            hp[param] = trial.suggest_categorical(
                param,
                spec["values"],
            )

        else:
            raise ValueError(f"Unknown hyperparameter type: {spec['type']}")

    return hp
