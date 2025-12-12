import argparse
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.data.dataloaders import make_dataloader, make_multitarget_dataloader
from src.models.gru import GRUForecast
from src.models.multi_gru import MultiGRUForecast
from src.utils.device import get_device
from src.utils.metrics import rmse, smape
from src.utils.scale import inverse_target
from src.utils.train_gru import evaluate_model as evaluate_gru_model


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_test_loaders(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]

    gru_test_loader = make_dataloader(
        data_cfg["test_path"],
        data_cfg["feature_cols"],
        data_cfg["target_col"],
        data_cfg["input_length"],
        data_cfg["horizon"],
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    multi_gru_test_loader = make_multitarget_dataloader(
        data_cfg["test_path"],
        data_cfg["feature_cols"],
        data_cfg["target_cols"],
        data_cfg["input_length"],
        data_cfg["horizon"],
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    return gru_test_loader, multi_gru_test_loader


def prepare_scaler_and_numeric(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]
    scaler = joblib.load(data_cfg["scaler_path"])
    df_train = pd.read_parquet(data_cfg["train_path"])
    numeric_cols = df_train.select_dtypes(include=[float, int]).columns.tolist()
    return scaler, numeric_cols


def evaluate_gru_trial(
    trial_id: int,
    cfg: Dict[str, Any],
    device: torch.device,
    test_loader,
    scaler,
    numeric_cols: List[str],
):
    data_cfg = cfg["data"]
    feature_cols = data_cfg["feature_cols"]
    input_size = len(feature_cols)
    horizon = data_cfg["horizon"]
    target_col = data_cfg["target_col"]

    ckpt_path = Path("results") / f"gru_best_trial_{trial_id}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"GRU checkpoint not found: {ckpt_path}")

    # hard code model parameters here
    model = GRUForecast(
        input_size=input_size,
        hidden_size=112,
        num_layers=1,
        horizon=horizon,
        dropout=0.38,
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    loss_fn = nn.MSELoss()

    test_metrics = evaluate_gru_model(
        model,
        test_loader,
        loss_fn,
        device=device,
        scaler=scaler,
        numeric_cols=numeric_cols,
        target_col=target_col,
    )

    print(f"\n[GRU] Trial {trial_id} – Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {float(v):.4f}")


def evaluate_multi_gru_trial(
    trial_id: int,
    cfg: Dict[str, Any],
    device: torch.device,
    test_loader,
    scaler,
    numeric_cols: List[str],
):
    data_cfg = cfg["data"]
    feature_cols = data_cfg["feature_cols"]
    target_cols = data_cfg["target_cols"]
    input_size = len(feature_cols)
    horizon = data_cfg["horizon"]

    no2_col = "nitrogen_dioxide"
    if no2_col not in target_cols:
        raise ValueError(f"{no2_col} not in target_cols: {target_cols}")
    no2_idx = target_cols.index(no2_col)

    ckpt_path = Path("results") / f"multigru_best_trial_{trial_id}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"MultiGRU checkpoint not found: {ckpt_path}")

    # hard code model parameters here
    model = MultiGRUForecast(
        input_size=input_size,
        target_cols=target_cols,
        horizon=horizon,
        shared_hidden_size=112,
        branch_hidden_size=56,
        num_layers=1,
        dropout=0.11,
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
        y_true_all = []
        y_pred_all = []

        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            y_true_all.append(y.cpu())
            y_pred_all.append(y_hat.cpu())

        y_true = torch.cat(y_true_all, dim=0)  # [N, H, T]
        y_pred = torch.cat(y_pred_all, dim=0)

        y_true_no2 = y_true[..., no2_idx]  # [N, H]
        y_pred_no2 = y_pred[..., no2_idx]  # [N, H]

        y_true_no2_den = inverse_target(scaler, y_true_no2, numeric_cols, no2_col)
        y_pred_no2_den = inverse_target(scaler, y_pred_no2, numeric_cols, no2_col)

        test_rmse_no2 = rmse(y_true_no2_den, y_pred_no2_den)
        test_smape_no2 = smape(y_true_no2_den, y_pred_no2_den)

    print(f"\n[MultiGRU] Trial {trial_id} – Test NO2 metrics:")
    print(f"  rmse:  {float(test_rmse_no2):.4f}")
    print(f"  smape: {float(test_smape_no2):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gru-trials",
        nargs="*",
        type=int,
        default=[],
        help="List of GRU trial IDs to evaluate (e.g. --gru-trials 3 7 14)",
    )
    parser.add_argument(
        "--multi-gru-trials",
        nargs="*",
        type=int,
        default=[],
        help="List of MultiGRU trial IDs to evaluate (e.g. --multi-gru-trials 8 11)",
    )
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    device = get_device(cfg["training"].get("device", "auto"))
    print(f"[evaluate_trials] Using device: {device}")

    gru_test_loader, multigru_test_loader = build_test_loaders(cfg)
    scaler, numeric_cols = prepare_scaler_and_numeric(cfg)

    for trial_id in args.gru_trials:
        evaluate_gru_trial(
            trial_id,
            cfg,
            device=device,
            test_loader=gru_test_loader,
            scaler=scaler,
            numeric_cols=numeric_cols,
        )

    for trial_id in args.multi_gru_trials:
        evaluate_multi_gru_trial(
            trial_id,
            cfg,
            device=device,
            test_loader=multigru_test_loader,
            scaler=scaler,
            numeric_cols=numeric_cols,
        )


if __name__ == "__main__":
    main()
