from pathlib import Path

import joblib
import pandas as pd
import torch
import yaml

from src.data.dataloaders import make_dataloader
from src.interpret.feature_importance import permutation_importance_gru
from src.models.gru import GRUForecast
from src.utils.device import get_device


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    feature_cols = data_cfg["feature_cols"]
    target_col = data_cfg["target_col"]
    input_length = data_cfg["input_length"]
    horizon = data_cfg["horizon"]

    device = get_device(cfg["training"].get("device", "auto"))
    print(f"[feature_importance_gru] Using device: {device}")

    val_loader = make_dataloader(
        data_cfg["val_path"],
        feature_cols,
        target_col,
        input_length,
        horizon,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    # load trained GRU
    results_dir = Path("results")
    model_path = results_dir / "gru_best_trial_14.pt"

    model = GRUForecast(
        input_size=len(feature_cols),
        hidden_size=112,  # must match training config
        num_layers=1,
        horizon=horizon,
        dropout=0.38,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    baseline_rmse, importances = permutation_importance_gru(
        model, val_loader, device, feature_cols, n_repeats=5
    )

    print(f"Baseline normalized RMSE (val): {baseline_rmse:.4f}")
    print("Permutation importances (ΔRMSE):")
    for name, imp in sorted(importances.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:30s}  {imp:+.5f}")


if __name__ == "__main__":
    main()
