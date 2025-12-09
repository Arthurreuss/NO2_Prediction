from functools import partial
from pathlib import Path

import joblib
import mlflow
import optuna
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam

from src.data.dataloaders import make_dataloader
from src.models.gru import GRUForecast
from src.utils.device import get_device
from src.utils.train_gru import evaluate_model, train_one_epoch


def build_context(config_path: str = "config.yaml") -> dict:
    """Load config, data, scaler, etc. once and return a context dict."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mlflow_cfg = cfg["training"]["mlflow"]
    device = get_device(cfg["training"].get("device", "auto"))
    print(f"[optimize_gru] Using device: {device}")

    data_cfg = cfg["data"]
    feature_cols = data_cfg["feature_cols"]
    target_col = data_cfg["target_col"]
    input_length = data_cfg["input_length"]
    horizon = data_cfg["horizon"]

    # Train/val loaders (Optuna uses validation as objective; test is for final model)
    train_loader = make_dataloader(
        data_cfg["train_path"],
        feature_cols,
        target_col,
        input_length,
        horizon,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = make_dataloader(
        data_cfg["val_path"],
        feature_cols,
        target_col,
        input_length,
        horizon,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    # Scaler + numeric columns (for de-normalized metrics)
    scaler_path = data_cfg.get("scaler_path")
    scaler = joblib.load(scaler_path) if scaler_path is not None else None

    df_train = pd.read_parquet(data_cfg["train_path"])
    numeric_cols = df_train.select_dtypes(include=[float, int]).columns.tolist()
    if target_col not in numeric_cols:
        raise ValueError(f"target_col '{target_col}' not found in numeric_cols")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    ctx = {
        "cfg": cfg,
        "mlflow_cfg": mlflow_cfg,
        "device": device,
        "data_cfg": data_cfg,
        "feature_cols": feature_cols,
        "target_col": target_col,
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
      - sample GRU hyperparameters
      - train with early stopping
      - log everything to MLflow
      - return best validation RMSE (denormalized if scaler exists)
    """
    cfg = ctx["cfg"]
    mlflow_cfg = ctx["mlflow_cfg"]
    device = ctx["device"]
    data_cfg = ctx["data_cfg"]
    feature_cols = ctx["feature_cols"]
    target_col = ctx["target_col"]
    input_length = ctx["input_length"]
    horizon = ctx["horizon"]
    train_loader = ctx["train_loader"]
    val_loader = ctx["val_loader"]
    scaler = ctx["scaler"]
    numeric_cols = ctx["numeric_cols"]
    results_dir = ctx["results_dir"]

    # --- Hyperparameter search space ---
    hidden_size = trial.suggest_int("hidden_size", 16, 128, step=16)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    grad_clip = cfg["training"].get("grad_clip", None)
    patience = int(cfg["training"].get("patience", 5))
    num_epochs = int(cfg["training"]["num_epochs"])

    # Fresh model + optimizer per trial
    model = GRUForecast(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=1,
        horizon=horizon,
        dropout=dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_model_path = results_dir / f"gru_best_trial_{trial.number}.pt"
    best_val_rmse = float("inf")
    epochs_no_improve = 0

    experiment_name = "GRU_Optimization_Optuna"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"GRU_optuna_t{trial.number}"):
        # log hyperparameters
        mlflow.log_param("model_type", "GRU")
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("num_layers", 1)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("horizon", horizon)
        mlflow.log_param("input_length", input_length)
        mlflow.log_param("lr", lr)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("grad_clip", grad_clip)
        mlflow.log_param("features", ",".join(feature_cols))
        mlflow.log_param("target_col", target_col)
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

            val_metrics = evaluate_model(
                model,
                val_loader,
                loss_fn,
                device=device,
                scaler=scaler,
                numeric_cols=numeric_cols,
                target_col=target_col,
            )

            # choose denormalized RMSE if available, otherwise normalized
            current_rmse = float(val_metrics.get("rmse", val_metrics["rmse_norm"]))

            print(
                f"[trial {trial.number}] Epoch {epoch+1}, "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"val_rmse_norm={val_metrics['rmse_norm']:.4f}, "
                f"val_smape_norm={val_metrics['smape_norm']:.2f}, "
                f"val_rmse={val_metrics.get('rmse', float('nan')):.4f}, "
                f"val_smape={val_metrics.get('smape', float('nan')):.2f}"
            )

            # log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_metrics["loss"], step=epoch)
            mlflow.log_metric("val_rmse_norm", val_metrics["rmse_norm"], step=epoch)
            mlflow.log_metric("val_smape_norm", val_metrics["smape_norm"], step=epoch)
            if "rmse" in val_metrics:
                mlflow.log_metric("val_rmse", val_metrics["rmse"], step=epoch)
            if "smape" in val_metrics:
                mlflow.log_metric("val_smape", val_metrics["smape"], step=epoch)

            # early stopping on (de-)normalized RMSE
            if current_rmse < best_val_rmse:
                best_val_rmse = current_rmse
                torch.save(model.state_dict(), best_model_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # report to Optuna (for pruning etc.)
            trial.report(best_val_rmse, step=epoch)
            if trial.should_prune():
                print(f"[trial {trial.number}] Pruned at epoch {epoch+1}")
                raise optuna.TrialPruned()

            if epochs_no_improve >= patience:
                print(
                    f"[trial {trial.number}] Early stopping after {epoch+1} epochs "
                    f"(best_val_rmse={best_val_rmse:.4f})."
                )
                break

        mlflow.log_metric("best_val_rmse", float(best_val_rmse))

    return best_val_rmse


def main():
    ctx = build_context("config.yaml")
    cfg = ctx["cfg"]

    opt_cfg = cfg["training"].get("optuna", {})
    n_trials = int(opt_cfg.get("n_trials", 20))
    timeout = opt_cfg.get("timeout", None)  # seconds or None
    study_name = opt_cfg.get("study_name", "gru_optuna_study")

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
        ),
    )

    study.optimize(partial(objective, ctx=ctx), n_trials=n_trials, timeout=timeout)

    print("Optuna optimization finished.")
    print("  Best value (val_rmse):", study.best_value)
    print("  Best params:", study.best_params)


if __name__ == "__main__":
    main()
