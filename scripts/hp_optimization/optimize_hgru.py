from functools import partial
from pathlib import Path

import mlflow
import optuna
import torch
import torch.nn as nn
from torch.optim import Adam

from src.models.hgru import HierarchicalGRUForecast
from src.training.engine import fit_model
from src.training.evaluate import evaluate_model
from src.training.setup import get_dataloaders, setup_experiment
from src.utils.helper import set_hyperparameter_ranges


def objective(trial: "optuna.trial.Trial", ctx: dict) -> float:
    """
    Objective function for Optuna hyperparameter optimization of a Hierarchical GRU model.

    This function sets up the model, optimizer, loss function, and data loaders based on the provided
    configuration and context. It then trains the model using the specified hyperparameters from the
    Optuna trial, evaluates its performance, and logs relevant information to MLflow. The best validation
    score achieved during training is returned as the objective value for optimization.

    Args:
        trial (optuna.trial.Trial): The Optuna trial object used to suggest hyperparameters.
        ctx (dict): Context dictionary containing configuration, device, scaler, and other necessary objects.

    Returns:
        float: The best validation score (e.g., loss) achieved during training, to be minimized by Optuna.
    """
    print("Starting trial:", trial.number)
    cfg = ctx["cfg"]
    device = ctx["device"]
    data_cfg = cfg["data"]
    target_cols = data_cfg["target_cols"]

    hp = set_hyperparameter_ranges(cfg, "hgru", trial)

    train_loader, val_loader, _ = get_dataloaders(cfg, multi_target=True)

    model = HierarchicalGRUForecast(
        input_size=len(data_cfg["feature_cols"]),
        target_cols=target_cols,
        horizon=data_cfg["horizon"],
        downsample_factor=hp["downsample_factor"],
        short_hidden_size=hp["short_hidden_size"],
        long_hidden_size=hp["long_hidden_size"],
        num_layers_short=hp["num_layers_short"],
        num_layers_long=hp["num_layers_long"],
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

    results_dir = Path("results/optuna_hgru")
    results_dir.mkdir(parents=True, exist_ok=True)
    trial_model_path = results_dir / f"hgru_trial_{trial.number}.pt"

    with mlflow.start_run(run_name=f"HGRU_Trial_{trial.number}", nested=True) as run:
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
    Runs the hyperparameter optimization workflow for the Hierarchical GRU model using Optuna.

    This function sets up the experiment context, creates and runs an Optuna study to optimize
    the model's hyperparameters, evaluates the best model on the test set, logs the results and
    model to MLflow, and cleans up saved model checkpoints.

    Args:
        None

    Returns:
        None
    """

    ctx = setup_experiment(
        "config_training.yaml", experiment_name="HierHGRU_Optimization_Optuna"
    )
    cfg = ctx["cfg"]
    device = ctx["device"]

    study = optuna.create_study(
        direction="minimize",
        study_name="hgru_optuna_study",
    )

    study.optimize(
        partial(objective, ctx=ctx),
        n_trials=cfg["models"]["optimization_ranges"].get("trials", 20),
    )
    print("Best params:", study.best_params)
    print("Best NO2 RMSE:", study.best_value)

    print("\n--- Evaluating Best Model on Test Set ---")
    best_trial = study.best_trial
    best_run_id = best_trial.user_attrs["mlflow_run_id"]
    best_hp = best_trial.params
    target_cols = cfg["data"]["target_cols"]

    model = HierarchicalGRUForecast(
        input_size=len(cfg["data"]["feature_cols"]),
        target_cols=target_cols,
        horizon=cfg["data"]["horizon"],
        downsample_factor=best_hp["downsample_factor"],
        short_hidden_size=best_hp["short_hidden_size"],
        long_hidden_size=best_hp["long_hidden_size"],
        num_layers_short=best_hp["num_layers_short"],
        num_layers_long=best_hp["num_layers_long"],
        dropout=best_hp["dropout"],
    ).to(device)

    best_model_path = Path("results/optuna_hgru") / f"hgru_trial_{best_trial.number}.pt"
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

    with mlflow.start_run(run_id=best_run_id):
        model_name = "hgru_best_model"
        model_uri = f"runs:/{best_run_id}/{model_name}"
        mlflow.pytorch.log_model(model, model_name)
        mlflow.register_model(model_uri, model_name)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    for f in Path("results/optuna_hgru").glob("hgru_trial_*.pt"):
        f.unlink()


if __name__ == "__main__":
    main()
