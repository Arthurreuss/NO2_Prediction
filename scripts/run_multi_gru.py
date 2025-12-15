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
from src.models.multi_gru import MultiGRUForecast
from src.utils.device import get_device
from src.utils.metrics import rmse, smape
from src.utils.scale import inverse_target
from src.utils.train_gru import evaluate_model, train_one_epoch


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    mlflow_cfg = cfg["training"]["mlflow"]
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    device = get_device(cfg["training"].get("device", "auto"))
    print(f"[run_multi_gru] Using device: {device}")

    data_cfg = cfg["data"]
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
    n_targets = len(target_cols)

    scaler = joblib.load(data_cfg["scaler_path"])
    df_train = pd.read_parquet(data_cfg["train_path"])
    numeric_cols = df_train.select_dtypes(include=[float, int]).columns.tolist()

    model = MultiGRUForecast(
        input_size=input_size,
        target_cols=target_cols,
        horizon=horizon,
        shared_hidden_size=112,
        branch_hidden_size=64,
        num_layers=1,
        dropout=0.44,
    ).to(device)

    lr = float(cfg["training"]["lr"])
    weight_decay = float(cfg["training"].get("weight_decay", 0.0))
    grad_clip = cfg["training"].get("grad_clip", None)
    patience = int(cfg["training"].get("patience", 5))

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = results_dir / "multi_gru_best.pt"

    best_val_rmse = float("inf")
    epochs_no_improve = 0

    with mlflow.start_run(run_name=f"Multi_GRU_{mlflow_cfg['run_postfix']}"):
        mlflow.log_param("model_type", "Multi GRU")
        mlflow.log_param("shared_hidden_size", 32)
        mlflow.log_param("branch_hidden_size", 16)
        mlflow.log_param("num_layers", 1)
        mlflow.log_param("dropout", 0.2)
        mlflow.log_param("horizon", horizon)
        mlflow.log_param("input_length", input_length)
        mlflow.log_param("lr", lr)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("grad_clip", grad_clip)
        mlflow.log_param("features", ",".join(feature_cols))
        mlflow.log_param("target_cols", ",".join(target_cols))
        mlflow.log_param("device", str(device))

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
                scaler=None,
            )

            with torch.no_grad():
                y_true_all, y_pred_all = [], []
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    y_hat = model(x)
                    y_true_all.append(y.cpu())  # [B, H, T]
                    y_pred_all.append(y_hat.cpu())

                y_true = torch.cat(y_true_all, dim=0)  # [N, H, T]
                y_pred = torch.cat(y_pred_all, dim=0)

                no2_col = "nitrogen_dioxide"
                t_idx = target_cols.index(no2_col)

                y_true_no2 = y_true[..., t_idx]  # [N, H]
                y_pred_no2 = y_pred[..., t_idx]  # [N, H]

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
                f"val_loss={val_metrics['loss']:.4f}, "
                f"val_rmse_norm={val_metrics['rmse_norm']:.4f}, "
                f"val_smape_norm={val_metrics['smape_norm']:.2f}, "
                f"val_rmse_no2={val_rmse_no2:.4f}, "
                f"val_smape_no2={val_smape_no2:.2f}"
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_metrics["loss"], step=epoch)
            mlflow.log_metric("val_rmse_norm", val_metrics["rmse_norm"], step=epoch)
            mlflow.log_metric("val_smape_norm", val_metrics["smape_norm"], step=epoch)
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
                print(f"[run_multi_gru] Early stopping after {epoch+1} epochs.")
                break

        model.load_state_dict(torch.load(best_model_path, map_location=device))

        with torch.no_grad():
            y_true_all, y_pred_all = [], []
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                y_true_all.append(y.cpu())
                y_pred_all.append(y_hat.cpu())

            y_true = torch.cat(y_true_all, dim=0)
            y_pred = torch.cat(y_pred_all, dim=0)

            no2_col = "nitrogen_dioxide"
            t_idx = target_cols.index(no2_col)

            y_true_no2 = y_true[..., t_idx]
            y_pred_no2 = y_pred[..., t_idx]

            y_true_no2_den = inverse_target(scaler, y_true_no2, numeric_cols, no2_col)
            y_pred_no2_den = inverse_target(scaler, y_pred_no2, numeric_cols, no2_col)

            test_rmse_no2 = rmse(y_true_no2_den, y_pred_no2_den)
            test_smape_no2 = smape(y_true_no2_den, y_pred_no2_den)

        print(
            "[run_multi_gru] Test NO2:",
            {"rmse": test_rmse_no2, "smape": test_smape_no2},
        )

        mlflow.log_metric("test_rmse_no2", float(test_rmse_no2))
        mlflow.log_metric("test_smape_no2", float(test_smape_no2))

        mlflow.pytorch.log_model(model, name="multi_gru_model")
        mlflow.log_artifact("config.yaml")


if __name__ == "__main__":
    main()
