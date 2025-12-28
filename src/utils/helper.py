from typing import Any, Dict

import mlflow
import optuna
import torch


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


def load_torch_model_from_registry(
    model_name: str, device: str = "cpu"
) -> torch.nn.Module:
    """Load a registered PyTorch model from the MLflow Model Registry.

    This function loads a model that was logged using
    `mlflow.pytorch.log_model` and registered in the MLflow Model Registry.
    The model is loaded from the `production` stage, moved to the specified
    device, and set to evaluation mode.

    Args:
        model_name: Name of the registered MLflow model.
        device: Device identifier to load the model onto (e.g., "cpu",
            "cuda", or "mps").

    Returns:
        A PyTorch model loaded from the MLflow registry, moved to the
        specified device, and set to evaluation mode.
    """
    model_uri = f"models:/{model_name}@production"
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    model.to(device)
    model.eval()
    return model
