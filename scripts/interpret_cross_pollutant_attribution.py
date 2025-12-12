from pathlib import Path

import torch
import yaml

from src.data.dataloaders import make_multitarget_dataloader
from src.interpret.cross_pollutant_attribution import (
    cross_pollutant_importance_multi_gru,
)
from src.models.multi_gru import MultiGRUForecast
from src.utils.device import get_device


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    feature_cols = data_cfg["feature_cols"]
    target_cols = data_cfg["target_cols"]
    input_length = data_cfg["input_length"]
    horizon = data_cfg["horizon"]

    device = get_device(cfg["training"].get("device", "auto"))
    print(f"[cross_pollutant_multi_gru] Using device: {device}")

    val_loader = make_multitarget_dataloader(
        data_cfg["val_path"],
        feature_cols,
        target_cols,
        input_length,
        horizon,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    # load trained Multi GRU
    model_path = Path("results") / "multi_gru_best_trial_8.pt"
    model = MultiGRUForecast(
        input_size=len(feature_cols),
        target_cols=target_cols,
        horizon=horizon,
        shared_hidden_size=112,  # must match your training config
        branch_hidden_size=64,  # must match your training config
        num_layers=1,
        dropout=0.44,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # pollutants to inspect (you can extend this list)
    pollutants = ["pm10", "pm2_5", "ozone", "nitrogen_dioxide"]

    importance = cross_pollutant_importance_multi_gru(
        model=model,
        loader=val_loader,
        device=device,
        feature_cols=feature_cols,
        target_cols=target_cols,
        pollutants_of_interest=pollutants,
        max_batches=10,
    )

    print("Cross-pollutant gradient-based importance for NO2 prediction:")
    for p, val in sorted(importance.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {p:20s}  {val:.6f}")


if __name__ == "__main__":
    main()
