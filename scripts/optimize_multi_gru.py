from functools import partial
from pathlib import Path

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
from src.models.multi_gru import HGRUForecast
from src.utils.device import get_device
from src.utils.metrics import rmse, smape
from src.utils.scale import inverse_target
from src.utils.train_gru import evaluate_model, train_one_epoch


def build_context():
    """Load config, data, scaler, etc. once and return a context dict."""
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    mlflow_cfg = cfg["training"]["mlflow"]
    device = get_device(cfg["training"].get("device", "auto"))
    print(f"[optimize_hgru] Using device: {device}")

    data_cfg = cfg["data"]
    feature_cols = data_cfg["feature_cols"]
    target_cols = data_cfg["target_cols"]
    input_length = data_cfg["input_length"]
    horizon = data_cfg["horizon"]

    # Data loaders (only train/val needed for Optuna)
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

    # Scaler + numeric columns for denorm
    scaler = joblib.load(data_cfg["scaler_path"])
    df_train = pd.read_parquet(data_cfg["train_path"])
    numeric_cols = df_train.select_dtypes(include=[float, int]).columns.tolist()

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    ctx = {
        "cfg": cfg,
        "mlflow_cfg": mlflow_cfg,
        "device": device,
        "data_cfg": data_cfg,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "input_length": input_length,
        "horizon": horizon,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "scaler": scaler,
        "numeric_cols": numeric_cols,
        "results_dir": results_dir,
    }
    return ctx


def objective(trial: optuna.Trial, ctx: dict) -> float:
    """
    Optuna objective:
    - sample hyperparams
    - train HGRU with early stopping
    - log everything to MLflow
    - return best validation RMSE on NO2 (denormalized)
    """
    cfg = ctx["cfg"]
    mlflow_cfg = ctx["mlflow_cfg"]
    device = ctx["device"]
    feature_cols = ctx["feature_cols"]
    target_cols = ctx["target_cols"]
    input_length = ctx["input_length"]
    horizon = ctx["horizon"]
    train_loader = ctx["train_loader"]
    val_loader = ctx["val_loader"]
    scaler = ctx["scaler"]
    numeric_cols = ctx["numeric_cols"]
    results_dir = ctx["results_dir"]

    # --- Hyperparameter search space ---
    shared_hidden_size = trial.suggest_int("shared_hidden_size", 16, 128, step=16)
    branch_hidden_size = trial.suggest_int("branch_hidden_size", 8, 64, step=8)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    grad_clip = cfg["training"].get("grad_clip", None)
    patience = int(cfg["training"].get("patience", 5))
    num_epochs = int(cfg["training"]["num_epochs"])

    # Fresh model + optimizer per trial
    model = HGRUForecast(
        input_size=len(feature_cols),
        target_cols=target_cols,
        horizon=horizon,
        shared_hidden_size=shared_hidden_size,
        branch_hidden_size=branch_hidden_size,
        num_layers=1,
        dropout=dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_model_path = results_dir / f"hgru_best_trial_{trial.number}.pt"
    best_val_rmse_no2 = float("inf")
    epochs_no_improve = 0

    experiment_name = "HGRU_Optimization_Optuna"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"HGRU_optuna_t{trial.number}"):
        # log trial hyperparameters
        mlflow.log_param("model_type", "HGRU")
        mlflow.log_param("shared_hidden_size", shared_hidden_size)
        mlflow.log_param("branch_hidden_size", branch_hidden_size)
        mlflow.log_param("num_layers", 1)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("horizon", horizon)
        mlflow.log_param("input_length", input_length)
        mlflow.log_param("lr", lr)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("grad_clip", grad_clip)
        mlflow.log_param("features", ",".join(feature_cols))
        mlflow.log_param("target_cols", ",".join(target_cols))
        mlflow.log_param("device", str(device))

        for epoch in range(num_epochs):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                device=device,
                grad_clip=grad_clip,
            )

            # normalized multi-target metrics
            val_metrics = evaluate_model(
                model,
                val_loader,
                loss_fn,
                device=device,
                scaler=None,  # NO2 denorm handled below
            )

            # denormalized NO2 metrics
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
                f"[trial {trial.number}] Epoch {epoch+1}, "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"val_rmse_norm={val_metrics['rmse_norm']:.4f}, "
                f"val_smape_norm={val_metrics['smape_norm']:.2f}, "
                f"val_rmse_no2={val_rmse_no2:.4f}, "
                f"val_smape_no2={val_smape_no2:.2f}"
            )

            # log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_metrics["loss"], step=epoch)
            mlflow.log_metric("val_rmse_norm", val_metrics["rmse_norm"], step=epoch)
            mlflow.log_metric("val_smape_norm", val_metrics["smape_norm"], step=epoch)
            mlflow.log_metric("val_rmse_no2", float(val_rmse_no2), step=epoch)
            mlflow.log_metric("val_smape_no2", float(val_smape_no2), step=epoch)

            # early stopping on NO2 RMSE
            current_rmse = float(val_rmse_no2)
            if current_rmse < best_val_rmse_no2:
                best_val_rmse_no2 = current_rmse
                torch.save(model.state_dict(), best_model_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # report to Optuna (for pruning etc.)
            trial.report(best_val_rmse_no2, step=epoch)

            if trial.should_prune():
                print(f"[trial {trial.number}] Pruned at epoch {epoch+1}")
                raise optuna.TrialPruned()

            if epochs_no_improve >= patience:
                print(
                    f"[trial {trial.number}] Early stopping after {epoch+1} epochs "
                    f"(best_val_rmse_no2={best_val_rmse_no2:.4f})."
                )
                break

        mlflow.log_metric("best_val_rmse_no2", float(best_val_rmse_no2))

    return best_val_rmse_no2


def main():
    ctx = build_context()
    cfg = ctx["cfg"]

    opt_cfg = cfg["training"].get("optuna", {})
    n_trials = int(opt_cfg.get("n_trials", 20))
    timeout = opt_cfg.get("timeout", None)  # seconds or None
    study_name = opt_cfg.get("study_name", "hgru_optuna_study")

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
        ),
    )

    # pass context via partial to avoid globals
    study.optimize(partial(objective, ctx=ctx), n_trials=n_trials, timeout=timeout)

    print("Optuna optimization finished.")
    print("  Best value (val_rmse_no2):", study.best_value)
    print("  Best params:", study.best_params)


if __name__ == "__main__":
    main()
