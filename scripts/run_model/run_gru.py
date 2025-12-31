from functools import partial
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.optim import Adam

from src.models.gru import GRUForecast
from src.training.engine import fit_model
from src.training.evaluate import evaluate_model
from src.training.setup import get_dataloaders, log_hyperparameters, setup_experiment


def main() -> None:
    """
    Runs the GRU forecasting model training, validation, and testing pipeline.

    This function sets up the experiment context, prepares data loaders, initializes the GRU model,
    configures the optimizer and loss function, and manages the training loop with early stopping.
    It logs hyperparameters and metrics to MLflow, saves the best model checkpoint, and evaluates
    the model on the test set.

    Returns:
        None
    """
    ctx = setup_experiment("config_training.yaml")
    cfg, device = ctx["cfg"], ctx["device"]
    train_loader, val_loader, test_loader = get_dataloaders(cfg, multi_target=False)
    data_cfg = cfg["data"]
    feature_cols = data_cfg["feature_cols"]
    target_col = data_cfg["target_col"]
    horizon = data_cfg["horizon"]

    model_params = cfg["models"]["gru"]
    model_params["input_size"] = len(feature_cols)
    model_params["horizon"] = horizon

    model = GRUForecast(**model_params).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )
    loss_fn = nn.MSELoss()

    validate_fn = partial(
        evaluate_model,
        loss_fn=loss_fn,
        scaler=ctx["scaler"],
        numeric_cols=ctx["numeric_cols"],
        target_col=target_col,
    )

    save_path = Path("results") / "gru_best.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"GRU_{cfg['training']['mlflow']['run_postfix']}"):
        log_hyperparameters(cfg, model_params, {"device": str(device)})

        fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epochs=cfg["training"]["num_epochs"],
            patience=cfg["training"].get("patience", 5),
            grad_clip=cfg["training"].get("grad_clip"),
            save_path=save_path,
            validate_fn=validate_fn,
            log_mlflow=True,
        )

        print(f"Loading best model from {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))

        test_metrics = validate_fn(model=model, dataloader=test_loader, device=device)

        print("\nTest Metrics:", test_metrics)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.pytorch.log_model(model, "gru_model")
        mlflow.log_artifact("config_training.yaml")


if __name__ == "__main__":
    main()
