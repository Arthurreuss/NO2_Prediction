from pathlib import Path
from typing import Any, Dict

import joblib
import mlflow
import mlflow.pytorch
import optuna
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


def build_dataloaders(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]

    train_loader = make_multitarget_dataloader(
        data_cfg["train_path"],
        data_cfg["feature_cols"],
        data_cfg["target_cols"],
        data_cfg["input_length"],
        data_cfg["horizon"],
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )

    val_loader = make_multitarget_dataloader(
        data_cfg["val_path"],
        data_cfg["feature_cols"],
        data_cfg["target_cols"],
        data_cfg["input_length"],
        data_cfg["horizon"],
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    test_loader = make_multitarget_dataloader(
        data_cfg["test_path"],
        data_cfg["feature_cols"],
        data_cfg["target_cols"],
        data_cfg["input_length"],
        data_cfg["horizon"],
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def create_model(
    trial: optuna.Trial, cfg: Dict[str, Any], input_size: int, target_cols
):
    data_cfg = cfg["data"]
    horizon = data_cfg["horizon"]

    # Hyperparameters to optimize
    downsample_factor = trial.suggest_categorical("downsample_factor", [12, 24, 48])
    short_hidden_size = trial.suggest_categorical(
        "short_hidden_size", [32, 64, 96, 128]
    )
    long_hidden_size = trial.suggest_categorical("long_hidden_size", [32, 64, 96, 128])
    num_layers_short = trial.suggest_int("num_layers_short", 1, 2)
    num_layers_long = trial.suggest_int("num_layers_long", 1, 2)
    dropout = trial.suggest_float("dropout", 0.1, 0.6)

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
    )

    # Log these as params in MLflow inside the objective
    hparams = {
        "downsample_factor": downsample_factor,
        "short_hidden_size": short_hidden_size,
        "long_hidden_size": long_hidden_size,
        "num_layers_short": num_layers_short,
        "num_layers_long": num_layers_long,
        "dropout": dropout,
    }

    return model, hparams


def objective(trial: optuna.Trial) -> float:
    # Load config once per trial
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    mlflow_cfg = cfg["training"]["mlflow"]
    mlflow.set_experiment("HierarchicalHGRU_Optimization")

    device = get_device(cfg["training"].get("device", "auto"))
    print(f"[optimize_hier_hgru] Trial {trial.number} using device: {device}")

    data_cfg = cfg["data"]
    feature_cols = data_cfg["feature_cols"]
    target_cols = data_cfg["target_cols"]
    input_length = data_cfg["input_length"]
    horizon = data_cfg["horizon"]
    input_size = len(feature_cols)

    train_loader, val_loader, _ = build_dataloaders(cfg)

    scaler = joblib.load(data_cfg["scaler_path"])
    df_train = pd.read_parquet(data_cfg["train_path"])
    numeric_cols = df_train.select_dtypes(include=[float, int]).columns.tolist()

    no2_col = "nitrogen_dioxide"
    if no2_col not in target_cols:
        raise ValueError(f"{no2_col} not in target_cols: {target_cols}")
    no2_idx = target_cols.index(no2_col)

    model, hparams = create_model(trial, cfg, input_size, target_cols)
    model = model.to(device)

    # Optimizer hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    # Training setup
    num_epochs = cfg["training"]["num_epochs"]
    patience = int(cfg["training"].get("patience", 5))
    grad_clip = cfg["training"].get("grad_clip", None)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = results_dir / f"hier_hgru_trial{trial.number}_best.pt"

    best_val_rmse = float("inf")
    epochs_no_improve = 0

    with mlflow.start_run(run_name=f"HierHGRU_optuna_t{trial.number}", nested=True):
        mlflow.log_param("model_type", "HierarchicalHGRU")
        mlflow.log_param("input_length", input_length)
        mlflow.log_param("horizon", horizon)
        mlflow.log_param("features", ",".join(feature_cols))
        mlflow.log_param("target_cols", ",".join(target_cols))
        mlflow.log_param("device", str(device))
        mlflow.log_param("lr", lr)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("grad_clip", grad_clip)

        for k, v in hparams.items():
            mlflow.log_param(k, v)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            n_batches = 0

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                y_hat = model(x)
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
                y_true = torch.cat(y_true_all, dim=0)
                y_pred = torch.cat(y_pred_all, dim=0)

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
                f"[Trial {trial.number}] Epoch {epoch+1}, "
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
            trial.report(current_rmse, step=epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

            if current_rmse < best_val_rmse:
                best_val_rmse = current_rmse
                torch.save(model.state_dict(), best_model_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"[Trial {trial.number}] Early stopping after {epoch+1} epochs.")
                break

        mlflow.log_metric("best_val_rmse_no2", best_val_rmse)
        return best_val_rmse


def main():
    study = optuna.create_study(
        direction="minimize",
        study_name="HierHGRU_Optimization",
    )
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    best = study.best_trial
    print(f"  Trial number: {best.number}")
    print(f"  Value (val_rmse_no2): {best.value}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
