from functools import partial
from pathlib import Path

import mlflow
import optuna
import torch
import torch.nn as nn
from torch.optim import Adam

from src.models.multi_gru import MultiGRUForecast
from src.training.engine import fit_model
from src.training.evaluate import evaluate_model
from src.training.setup import get_dataloaders, setup_experiment


def objective(trial, ctx):
    cfg = ctx["cfg"]
    device = ctx["device"]
    data_cfg = cfg["data"]
    target_cols = data_cfg["target_cols"]

    hp = {
        "shared_hidden_size": trial.suggest_int("shared_hidden_size", 16, 128, step=16),
        "branch_hidden_size": trial.suggest_int("branch_hidden_size", 8, 64, step=8),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
    }

    train_loader, val_loader, _ = get_dataloaders(cfg, multi_target=True)

    model = MultiGRUForecast(
        input_size=len(data_cfg["feature_cols"]),
        target_cols=target_cols,
        horizon=data_cfg["horizon"],
        shared_hidden_size=hp["shared_hidden_size"],
        branch_hidden_size=hp["branch_hidden_size"],
        num_layers=hp["num_layers"],
        dropout=hp["dropout"],
    ).to(device)

    optimizer = Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = nn.MSELoss()

    validate_fn = partial(
        evaluate_model,
        loss_fn=loss_fn,
        scaler=ctx["scaler"],
        numeric_cols=ctx["numeric_cols"],
        target_col="nitrogen_dioxide",
        all_target_cols=target_cols,
    )

    results_dir = Path("results/optuna_multigru")
    results_dir.mkdir(parents=True, exist_ok=True)
    trial_model_path = results_dir / f"multigru_trial_{trial.number}.pt"

    with mlflow.start_run(nested=True):
        mlflow.log_params(hp)

        best_score = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epochs=cfg["training"]["num_epochs"],
            patience=int(cfg["training"].get("patience", 5)),
            save_path=trial_model_path,
            validate_fn=validate_fn,
            grad_clip=cfg["training"].get("grad_clip"),
            trial=trial,
            log_mlflow=True,
        )

    return best_score


def main():
    ctx = setup_experiment(
        "config.yaml", experiment_name="MultiGRU_Optimization_Optuna"
    )
    cfg = ctx["cfg"]
    device = ctx["device"]

    study = optuna.create_study(
        direction="minimize",
        study_name="multigru_optuna_study",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    study.optimize(partial(objective, ctx=ctx), n_trials=20)

    print("Best params:", study.best_params)
    print("Best NO2 RMSE:", study.best_value)

    print("\n--- Evaluating Best Model on Test Set ---")
    best_trial = study.best_trial
    best_hp = best_trial.params
    target_cols = cfg["data"]["target_cols"]

    model = MultiGRUForecast(
        input_size=len(cfg["data"]["feature_cols"]),
        target_cols=target_cols,
        horizon=cfg["data"]["horizon"],
        shared_hidden_size=best_hp["shared_hidden_size"],
        branch_hidden_size=best_hp["branch_hidden_size"],
        num_layers=1,
        dropout=best_hp["dropout"],
    ).to(device)

    best_model_path = (
        Path("results/optuna_multigru") / f"multigru_trial_{best_trial.number}.pt"
    )
    print(f"Loading weights from: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    _, _, test_loader = get_dataloaders(cfg, multi_target=True)
    loss_fn = nn.MSELoss()

    test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        loss_fn=loss_fn,
        device=device,
        scaler=ctx["scaler"],
        numeric_cols=ctx["numeric_cols"],
        target_col="nitrogen_dioxide",
        all_target_cols=target_cols,
    )

    print("Test Results:", test_metrics)

    with mlflow.start_run(run_name="MultiGRU_Best_Test_Results"):
        mlflow.log_params(best_hp)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})


if __name__ == "__main__":
    main()
