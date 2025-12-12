from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam

from src.data.dataloaders import make_multitarget_dataloader
from src.models.hgru import HierarchicalGRUForecast
from src.utils.device import get_device
from src.utils.metrics import rmse, smape
from src.utils.scale import inverse_target


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    mlflow_cfg = cfg["training"]["mlflow"]
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    device = get_device(cfg["training"].get("device", "auto"))
    print(f"[run_hier_hgru] Using device: {device}")

    data_cfg = cfg["data"]
    models_cfg = cfg["models"]
    feature_cols = data_cfg["feature_cols"]
    target_cols = data_cfg["target_cols"]
    input_length = data_cfg["input_length"]
    horizon = data_cfg["horizon"]

    train_loader = make_multitarget_dataloader(
        data_cfg["train_path"],
        feature_cols,
        target_cols,
        input_length,
        horizon,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = make_multitarget_dataloader(
        data_cfg["val_path"],
        feature_cols,
        target_cols,
        input_length,
        horizon,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )
    test_loader = make_multitarget_dataloader(
        data_cfg["test_path"],
        feature_cols,
        target_cols,
        input_length,
        horizon,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    input_size = len(feature_cols)
    scaler = joblib.load(data_cfg["scaler_path"])

    df_train = pd.read_parquet(data_cfg["train_path"])
    numeric_cols = df_train.select_dtypes(include=[float, int]).columns.tolist()

    downsample_factor = models_cfg.get("hgru", {}).get("downsample_factor", 24)
    short_hidden_size = models_cfg.get("hgru", {}).get("short_hidden_size", 64)
    long_hidden_size = models_cfg.get("hgru", {}).get("long_hidden_size", 32)
    num_layers_short = models_cfg.get("hgru", {}).get("num_layers_short", 1)
    num_layers_long = models_cfg.get("hgru", {}).get("num_layers_long", 1)
    dropout = models_cfg.get("hgru", {}).get("dropout", 0.3)

    model = HierarchicalGRUForecast(
        input_size=input_size,
        target_cols=target_cols,
        horizon=horizon,
        downsample_factor=downsample_factor,
        short_hidden_size=short_hidden_size,
        long_hidden_size=long_hidden_size,
        num_layers_short=num_layers_short,
        num_layers_long=num_layers_long,
        dropout=dropout,
    ).to(device)

    lr = float(cfg["training"]["lr"])
    weight_decay = float(cfg["training"].get("weight_decay", 0.0))
    grad_clip = cfg["training"].get("grad_clip", None)
    patience = int(cfg["training"].get("patience", 5))

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = results_dir / "hier_hgru_best.pt"

    best_val_rmse = float("inf")
    epochs_no_improve = 0

    with mlflow.start_run(run_name=f"HierHGRU_{mlflow_cfg['run_postfix']}"):
        mlflow.log_param("model_type", "HierarchicalHGRU")
        mlflow.log_param("downsample_factor", downsample_factor)
        mlflow.log_param("short_hidden_size", short_hidden_size)
        mlflow.log_param("long_hidden_size", long_hidden_size)
        mlflow.log_param("num_layers_short", num_layers_short)
        mlflow.log_param("num_layers_long", num_layers_long)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("horizon", horizon)
        mlflow.log_param("input_length", input_length)
        mlflow.log_param("lr", lr)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("grad_clip", grad_clip)
        mlflow.log_param("features", ",".join(feature_cols))
        mlflow.log_param("target_cols", ",".join(target_cols))
        mlflow.log_param("device", str(device))

        no2_col = "nitrogen_dioxide"
        if no2_col not in target_cols:
            raise ValueError(f"{no2_col} not in target_cols: {target_cols}")
        no2_idx = target_cols.index(no2_col)

        for epoch in range(cfg["training"]["num_epochs"]):
            model.train()
            running_loss = 0.0
            n_batches = 0

            for x, y in train_loader:
                x = x.to(device)  # [B, L, F]
                y = y.to(device)  # [B, H, T]

                optimizer.zero_grad()
                y_hat = model(x)  # [B, H, T]
                loss = loss_fn(y_hat, y)
                loss.backward()

                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()

                running_loss += loss.item()
                n_batches += 1

            train_loss = running_loss / max(n_batches, 1)

            model.eval()
            with torch.no_grad():
                val_losses = []
                y_true_all = []
                y_pred_all = []

                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    y_hat = model(x)
                    val_losses.append(loss_fn(y_hat, y).item())

                    y_true_all.append(y.cpu())
                    y_pred_all.append(y_hat.cpu())

                val_loss = sum(val_losses) / max(len(val_losses), 1)
                y_true = torch.cat(y_true_all, dim=0)  # [N, H, T]
                y_pred = torch.cat(y_pred_all, dim=0)  # [N, H, T]

                y_true_no2 = y_true[..., no2_idx]
                y_pred_no2 = y_pred[..., no2_idx]

                y_true_no2_den = inverse_target(
                    scaler, y_true_no2, numeric_cols, no2_col
                )
                y_pred_no2_den = inverse_target(
                    scaler, y_pred_no2, numeric_cols, no2_col
                )

                val_rmse_no2 = rmse(y_true_no2_den, y_pred_no2_den)
                val_smape_no2 = smape(y_true_no2_den, y_pred_no2_den)

            print(
                f"Epoch {epoch+1}, "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_rmse_no2={float(val_rmse_no2):.4f}, "
                f"val_smape_no2={float(val_smape_no2):.2f}"
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_rmse_no2", float(val_rmse_no2), step=epoch)
            mlflow.log_metric("val_smape_no2", float(val_smape_no2), step=epoch)

            current_rmse = float(val_rmse_no2)
            if current_rmse < best_val_rmse:
                best_val_rmse = current_rmse
                torch.save(model.state_dict(), best_model_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"[run_hier_hgru] Early stopping after {epoch+1} epochs.")
                break

        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            y_true_all = []
            y_pred_all = []

            for x, y in test_loader:
                x = x.to(device)
                y_hat = model(x)
                y_true_all.append(y.cpu())
                y_pred_all.append(y_hat.cpu())

            y_true = torch.cat(y_true_all, dim=0)
            y_pred = torch.cat(y_pred_all, dim=0)

            y_true_no2 = y_true[..., no2_idx]
            y_pred_no2 = y_pred[..., no2_idx]

            y_true_no2_den = inverse_target(scaler, y_true_no2, numeric_cols, no2_col)
            y_pred_no2_den = inverse_target(scaler, y_pred_no2, numeric_cols, no2_col)

            test_rmse_no2 = rmse(y_true_no2_den, y_pred_no2_den)
            test_smape_no2 = smape(y_true_no2_den, y_pred_no2_den)

        print(
            "[run_hier_hgru] Test NO2:",
            {"rmse": float(test_rmse_no2), "smape": float(test_smape_no2)},
        )

        mlflow.log_metric("test_rmse_no2", float(test_rmse_no2))
        mlflow.log_metric("test_smape_no2", float(test_smape_no2))

        mlflow.pytorch.log_model(model, name="hier_hgru_model")
        mlflow.log_artifact("config.yaml")


if __name__ == "__main__":
    main()
