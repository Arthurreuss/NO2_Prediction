from typing import Any, Dict, Optional, Tuple

import joblib
import mlflow
import pandas as pd
from torch.utils.data import DataLoader

from src.data.dataloaders import make_dataloader, make_multitarget_dataloader
from src.utils.cfg import load_config
from src.utils.device import get_device


def setup_experiment(
    config_path: str = "config.yaml",
    experiment_name: Optional[str] = None,
    single_sensor: bool = True,
) -> Dict[str, Any]:
    """Set up an experiment context (config, MLflow, device, scaler, numeric columns).

    This function:
      - loads the project configuration from disk
      - initializes the MLflow experiment
      - selects a compute device
      - loads the fitted scaler used for inverse-scaling metrics
      - infers numeric columns from the training parquet file

    Args:
        config_path: Path to the YAML configuration file.
        experiment_name: Optional MLflow experiment name override. If not
            provided, the name is taken from the configuration.
        single_sensor: Whether to use the single-sensor data paths. If False,
            multi-sensor paths are used.

    Returns:
        A dictionary containing the initialized experiment context, including:
            - "cfg": Loaded configuration dictionary.
            - "device": Selected torch device identifier.
            - "scaler": Loaded scaler object (joblib).
            - "numeric_cols": List of numeric column names inferred from train data.
            - "data_cfg": Data configuration section from the config.
    """
    cfg = load_config(config_path)

    # MLflow Setup
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    else:
        mlflow.set_experiment(cfg["training"]["mlflow"]["experiment_name"])

    # Device
    device = get_device(cfg["training"].get("device", "auto"))

    # Data Config
    data_cfg = cfg["data"]
    data_paths = data_cfg["single" if single_sensor else "multi‚"]

    # Load Scaler
    scaler = joblib.load(data_paths["scaler_path"])

    # Load Numeric Cols (metadata)
    df_train = pd.read_parquet(data_paths["train_path"])
    numeric_cols = df_train.select_dtypes(include=[float, int]).columns.tolist()

    return {
        "cfg": cfg,
        "device": device,
        "scaler": scaler,
        "numeric_cols": numeric_cols,
        "data_cfg": data_cfg,
    }


def get_dataloaders(
    cfg: Dict[str, Any], multi_target: bool = False, multi_city: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders based on configuration flags.

    This helper selects the appropriate data split paths (single vs multi-city)
    and the appropriate loader constructor (single-target vs multi-target), then
    returns DataLoaders for train, validation, and test.

    Args:
        cfg: Loaded configuration dictionary.
        multi_target: If True, uses the multi-target DataLoader constructor and
            expects `target_cols` in the data configuration. If False, uses the
            single-target constructor and expects `target_col`.
        multi_city: If True, uses the multi-city data paths. If False, uses the
            single-city paths.

    Returns:
        A tuple of (train_loader, val_loader, test_loader).
    """
    data_cfg = cfg["data"]
    data_paths = data_cfg["multi" if multi_city else "single"]
    batch_size = cfg["training"]["batch_size"]
    loader_func = make_multitarget_dataloader if multi_target else make_dataloader

    common_args: Dict[str, Any] = {
        "feature_cols": data_cfg["feature_cols"],
        ("target_cols" if multi_target else "target_col"): (
            data_cfg.get("target_cols") if multi_target else data_cfg["target_col"]
        ),
        "input_length": data_cfg["input_length"],
        "horizon": data_cfg["horizon"],
        "batch_size": batch_size,
    }

    train_loader = loader_func(data_paths["train_path"], shuffle=True, **common_args)
    val_loader = loader_func(data_paths["val_path"], shuffle=False, **common_args)
    test_loader = loader_func(data_paths["test_path"], shuffle=False, **common_args)

    return train_loader, val_loader, test_loader


def log_hyperparameters(
    cfg: Dict[str, Any],
    model_params: Dict[str, Any] = None,
    additional_params: Dict[str, Any] = None,
) -> None:
    """Log training, data, and model hyperparameters to MLflow.

    This function logs:
      - standard training parameters (lr, batch_size, weight_decay, grad_clip)
      - key data parameters (input_length, horizon, number of features)
      - the feature list as a comma-separated string
      - any provided model-specific parameters
      - any additional parameters provided by the caller

    Args:
        cfg: Loaded configuration dictionary.
        model_params: Optional dictionary of model-specific parameters to log.
        additional_params: Optional dictionary of additional parameters to log.
    """
    train_cfg = cfg.get("training", {})
    mlflow.log_param("lr", train_cfg.get("lr"))
    mlflow.log_param("batch_size", train_cfg.get("batch_size"))
    mlflow.log_param("weight_decay", train_cfg.get("weight_decay", 0))
    mlflow.log_param("grad_clip", train_cfg.get("grad_clip"))

    data_cfg = cfg.get("data", {})
    mlflow.log_param("input_length", data_cfg.get("input_length"))
    mlflow.log_param("horizon", data_cfg.get("horizon"))

    features = data_cfg.get("feature_cols", [])
    mlflow.log_param("n_features", len(features))
    mlflow.log_param("features", ",".join(features) if features else "")

    if model_params:
        for k, v in model_params.items():
            mlflow.log_param(k, v)

    if additional_params:
        for k, v in additional_params.items():
            mlflow.log_param(k, v)
