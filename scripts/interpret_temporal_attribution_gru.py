from pathlib import Path

import torch
import yaml

from src.data.dataloaders import make_dataloader
from src.interpret.temporal_attribution import compute_saliency_gru, summarize_saliency
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
    print(f"[temporal_importance_gru] Using device: {device}")

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
    model_path = Path("results") / "gru_best_trial_14.pt"
    model = GRUForecast(
        input_size=len(feature_cols),
        hidden_size=112,  # must match training config
        num_layers=1,
        horizon=horizon,
        dropout=0.38,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    out = compute_saliency_gru(model, val_loader, device, max_batches=10)
    ft_imp = out["feature_time_importance"]  # [L, F]

    per_feature = summarize_saliency(ft_imp, feature_cols)

    print("Average saliency per feature (normalized, larger = more influential):")
    for name, val in sorted(per_feature.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:30s}  {val:.6f}")

    # optional: show time-step importance
    per_time = ft_imp.mean(dim=1)  # [L]
    print("\nMean saliency per time step (0 = oldest, L-1 = most recent):")
    for t, v in enumerate(per_time):
        print(f"  t={t:2d}: {float(v):.6f}")


if __name__ == "__main__":
    main()
