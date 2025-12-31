from functools import partial
from pathlib import Path

import mlflow
import optuna
import torch
import torch.nn as nn
from torch.optim import Adam

from src.models.gru import GRUForecast
from src.training.engine import fit_model
from src.training.evaluate import evaluate_model
from src.training.setup import get_dataloaders, setup_experiment
from src.utils.helper import set_hyperparameter_ranges


def objective(trial: "optuna.trial.Trial", ctx: dict) -> float:
    """
    Objective function for Optuna hyperparameter optimization of a GRU-based forecasting model.

    This function sets up the model, optimizer, loss function, and data loaders based on the provided configuration.
    It then trains the model using the specified hyperparameters, logs relevant information to MLflow, and returns
    the best validation score achieved during training.

    Args:
        trial (optuna.trial.Trial): The Optuna trial object used for suggesting hyperparameters.
        ctx (dict): Context dictionary containing configuration, device, scaler, and other necessary objects.

    Returns:
        float: The best validation score (e.g., lowest validation loss) achieved during the trial.
    """

    print("Starting trial:", trial.number)
    cfg = ctx["cfg"]
    device = ctx["device"]
    data_cfg = cfg["data"]

    hp = set_hyperparameter_ranges(cfg, "gru", trial)

    train_loader, val_loader, _ = get_dataloaders(cfg, multi_target=False)

    model = GRUForecast(
        input_size=len(data_cfg["feature_cols"]),
        hidden_size=hp["hidden_size"],
        num_layers=hp["num_layers"],
        horizon=data_cfg["horizon"],
        dropout=hp["dropout"],
    ).to(device)

    optimizer = Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = nn.MSELoss()

    validate_fn = partial(
        evaluate_model,
        loss_fn=loss_fn,
        scaler=ctx["scaler"],
        numeric_cols=ctx["numeric_cols"],
        target_col=data_cfg["target_col"],
    )

    results_dir = Path("results/optuna_gru")
    results_dir.mkdir(parents=True, exist_ok=True)
    trial_model_path = results_dir / f"gru_trial_{trial.number}.pt"

    with mlflow.start_run(run_name=f"GRU_Trial_{trial.number}", nested=True) as run:
        trial.set_user_attr("mlflow_run_id", run.info.run_id)
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


def main() -> None:
    """
    Runs the hyperparameter optimization for a GRU forecasting model using Optuna.

    This function performs the following steps:
        1. Sets up the experiment context and configuration.
        2. Initializes an Optuna study for hyperparameter optimization.
        3. Runs the optimization process for a specified number of trials.
        4. Loads the best model checkpoint and evaluates it on the test set.
        5. Logs the best model and test metrics to MLflow.
        6. Cleans up model checkpoint files.

    Returns:
        None
    """

    ctx = setup_experiment(
        "config_training.yaml",
        experiment_name="GRU_Optimization_Optuna",
    )
    cfg = ctx["cfg"]
    device = ctx["device"]

    study = optuna.create_study(
        direction="minimize",
        study_name="gru_optuna_study",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    study.optimize(
        partial(objective, ctx=ctx),
        n_trials=cfg["models"]["optimization_ranges"].get("trials", 20),
    )

    print("Best params:", study.best_params)
    print("Best RMSE:", study.best_value)

    print("\n--- Evaluating Best Model on Test Set ---")
    best_trial = study.best_trial
    best_run_id = best_trial.user_attrs["mlflow_run_id"]
    best_hp = best_trial.params

    model = GRUForecast(
        input_size=len(cfg["data"]["feature_cols"]),
        hidden_size=best_hp["hidden_size"],
        num_layers=best_hp["num_layers"],
        horizon=cfg["data"]["horizon"],
        dropout=best_hp["dropout"],
    ).to(device)

    best_model_path = Path("results/optuna_gru") / f"gru_trial_{best_trial.number}.pt"
    print(f"Loading weights from: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    _, _, test_loader = get_dataloaders(cfg, multi_target=False)
    loss_fn = nn.MSELoss()

    test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        loss_fn=loss_fn,
        device=device,
        scaler=ctx["scaler"],
        numeric_cols=ctx["numeric_cols"],
        target_col=cfg["data"]["target_col"],
    )

    print("Test Results:", test_metrics)
    with mlflow.start_run(run_id=best_run_id):
        model_name = "gru_best_model"
        model_uri = f"runs:/{best_run_id}/{model_name}"
        mlflow.pytorch.log_model(model, model_name)
        mlflow.register_model(model_uri, model_name)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    for f in Path("results/optuna_gru").glob("gru_trial_*.pt"):
        f.unlink()


if __name__ == "__main__":
    main()
