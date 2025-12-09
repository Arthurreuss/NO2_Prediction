from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam

from src.data.dataloaders import make_dataloader
from src.models.gru import GRUForecast
from src.utils.device import get_device
from src.utils.train_gru import evaluate_model, train_one_epoch


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg["training"].get("device", "auto"))
    print(f"[run_gru] Using device: {device}")

    data_cfg = cfg["data"]

    train_loader = make_dataloader(
        data_cfg["train_path"],
        data_cfg["feature_cols"],
        data_cfg["target_col"],
        data_cfg["input_length"],
        data_cfg["horizon"],
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = make_dataloader(
        data_cfg["val_path"],
        data_cfg["feature_cols"],
        data_cfg["target_col"],
        data_cfg["input_length"],
        data_cfg["horizon"],
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )
    test_loader = make_dataloader(
        data_cfg["test_path"],
        data_cfg["feature_cols"],
        data_cfg["target_col"],
        data_cfg["input_length"],
        data_cfg["horizon"],
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    feature_cols = data_cfg["feature_cols"]
    input_size = len(feature_cols)
    horizon = data_cfg["horizon"]
    target_col = data_cfg["target_col"]

    scaler_path = data_cfg.get("scaler_path")
    scaler = joblib.load(scaler_path) if scaler_path is not None else None

    df_train = pd.read_parquet(data_cfg["train_path"])
    numeric_cols = df_train.select_dtypes(include=[float, int]).columns.tolist()

    if target_col not in numeric_cols:
        raise ValueError(f"target_col '{target_col}' not found in numeric_cols")

    model = GRUForecast(
        input_size=input_size,
        hidden_size=32,
        num_layers=1,
        horizon=horizon,
        dropout=0.2,
    ).to(device)

    lr = float(cfg["training"]["lr"])
    weight_decay = float(cfg["training"].get("weight_decay", 0.0))
    grad_clip = cfg["training"].get("grad_clip", None)
    patience = int(cfg["training"].get("patience", 5))

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = results_dir / "gru_best.pt"

    best_val_rmse = float("inf")
    epochs_no_improve = 0

    for epoch in range(cfg["training"]["num_epochs"]):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device=device,
            grad_clip=grad_clip,
        )

        val_metrics = evaluate_model(
            model,
            val_loader,
            loss_fn,
            device=device,
            scaler=scaler,
            numeric_cols=numeric_cols,
            target_col=target_col,
        )

        parts = [
            f"Epoch {epoch+1}",
            f"train_loss={train_loss:.4f}",
            f"val_loss={val_metrics['loss']:.4f}",
            f"val_rmse_norm={val_metrics['rmse_norm']:.4f}",
            f"val_smape_norm={val_metrics['smape_norm']:.2f}",
        ]
        if "rmse" in val_metrics:
            parts.append(f"val_rmse={val_metrics['rmse']:.4f}")
        if "smape" in val_metrics:
            parts.append(f"val_smape={val_metrics['smape']:.2f}")
        print(", ".join(parts))

        current_rmse = val_metrics.get("rmse", val_metrics["rmse_norm"])

        if current_rmse < best_val_rmse:
            best_val_rmse = current_rmse
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"[run_gru] Early stopping triggered after {epoch+1} epochs.")
            break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = evaluate_model(
        model,
        test_loader,
        loss_fn,
        device=device,
        scaler=scaler,
        numeric_cols=numeric_cols,
        target_col=target_col,
    )
    print("[run_gru] Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
