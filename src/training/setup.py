from typing import Any, Dict, Optional

import joblib
import mlflow
import pandas as pd
import yaml

from src.data.dataloaders import make_dataloader, make_multitarget_dataloader
from src.utils.cfg import load_config
from src.utils.device import get_device


def setup_experiment(
    config_path: str = "config.yaml",
    experiment_name: Optional[str] = None,
    single: bool = True,
) -> Dict[str, Any]:
    """
    Loads config, sets up MLflow, device, scaler, and numeric columns.
    Returns a context dictionary with everything needed to start building models.
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
    data_cfg = cfg["data"]["single" if single else "multi‚"]

    # Load Scaler
    scaler = None
    if "scaler_path" in data_cfg and data_cfg["scaler_path"]:
        scaler = joblib.load(data_cfg["scaler_path"])

    # Load Numeric Cols (metadata)
    df_train = pd.read_parquet(data_cfg["train_path"])
    numeric_cols = df_train.select_dtypes(include=[float, int]).columns.tolist()

    return {
        "cfg": cfg,
        "device": device,
        "scaler": scaler,
        "numeric_cols": numeric_cols,
        "data_cfg": data_cfg,
    }


def get_dataloaders(cfg: Dict[str, Any], multi_target: bool = False):
    """
    Helper to generate train/val/test loaders based on the context.
    """
    data_cfg = cfg["data"]
    batch_size = cfg["training"]["batch_size"]
    loader_func = make_multitarget_dataloader if multi_target else make_dataloader

    common_args = {
        "feature_cols": (
            data_cfg["feature_cols"] if multi_target else data_cfg["feature_col"]
        ),
        "input_length": data_cfg["input_length"],
        "horizon": data_cfg["horizon"],
        "batch_size": batch_size,
    }

    train_loader = loader_func(data_cfg["train_path"], shuffle=True, **common_args)
    val_loader = loader_func(data_cfg["val_path"], shuffle=False, **common_args)
    test_loader = loader_func(data_cfg["test_path"], shuffle=False, **common_args)

    return train_loader, val_loader, test_loader


def log_hyperparameters(
    cfg: Dict[str, Any],
    model_params: Dict[str, Any] = None,
    additional_params: Dict[str, Any] = None,
):
    """
    Logs standard config params and model-specific params to MLflow.
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
