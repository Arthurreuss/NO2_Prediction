from functools import partial
from math import log
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.optim import Adam

from src.models.multi_gru import MultiGRUForecast
from src.training.engine import fit_model
from src.training.evaluate import evaluate_model
from src.training.setup import get_dataloaders, log_hyperparameters, setup_experiment


def main() -> None:
    """
    Runs the training, validation, and testing pipeline for the Multi-GRU forecasting model.

    This function sets up the experiment context, loads configuration and data, initializes the model,
    optimizer, and loss function, and manages the training loop with early stopping and model checkpointing.
    It also logs metrics and artifacts to MLflow for experiment tracking.

    Steps performed:
        1. Loads experiment configuration and device setup.
        2. Prepares data loaders for training, validation, and testing.
        3. Initializes the Multi-GRU model with hyperparameters from the config.
        4. Sets up optimizer and loss function.
        5. Defines the validation function with partial arguments.
        6. Creates results directory and sets model checkpoint path.
        7. Starts an MLflow run, logs hyperparameters, and trains the model.
        8. Loads the best model checkpoint and evaluates on the test set.
        9. Logs test metrics and model artifacts to MLflow.

    Raises:
        FileNotFoundError: If the configuration file or required data files are missing.
        Exception: For any errors during model training, evaluation, or logging.
    """
    ctx = setup_experiment("config.yaml")
    cfg = ctx["cfg"]
    device = ctx["device"]
    train_loader, val_loader, test_loader = get_dataloaders(cfg, multi_target=True)
    data_cfg = cfg["data"]
    feature_cols = data_cfg["feature_cols"]
    target_cols = data_cfg["target_cols"]
    horizon = data_cfg["horizon"]
    lr = cfg["training"]["lr"]
    weight_decay = cfg["training"].get("weight_decay", 0.0)

    hp_multi_gru = cfg["models"]["multi_gru"]
    hp_multi_gru["input_size"] = len(feature_cols)
    hp_multi_gru["target_cols"] = target_cols
    hp_multi_gru["horizon"] = horizon

    model = MultiGRUForecast(**hp_multi_gru).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    validate_fn = partial(
        evaluate_model,
        loss_fn=loss_fn,
        scaler=ctx["scaler"],
        numeric_cols=ctx["numeric_cols"],
        target_col="nitrogen_dioxide",
        all_target_cols=target_cols,
    )

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = results_dir / "multi_gru_best.pt"

    with mlflow.start_run(
        run_name=f"Multi_GRU_{cfg['training']['mlflow']['run_postfix']}"
    ):
        log_hyperparameters(cfg, hp_multi_gru, {"device": str(device)})

        fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epochs=cfg["training"]["num_epochs"],
            patience=int(cfg["training"].get("patience", 5)),
            save_path=best_model_path,
            validate_fn=validate_fn,
            grad_clip=cfg["training"].get("grad_clip", None),
            log_mlflow=True,
        )

        print(f"Loading best model: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_metrics = validate_fn(model=model, dataloader=test_loader, device=device)
        print("Test Results:", test_metrics)

        mlflow.log_metric("test_rmse", test_metrics["rmse"])
        mlflow.log_metric("test_smape", test_metrics["smape"])
        mlflow.pytorch.log_model(model, "multi_gru_model")
        mlflow.log_artifact("config.yaml")


if __name__ == "__main__":
    main()
