import mlflow
import torch.nn as nn

from src.models.persistence import PersistenceModel
from src.training.evaluate import evaluate_persistence
from src.training.setup import get_dataloaders, setup_experiment


def main() -> None:
    """
    Runs the persistence model experiment using the provided configuration.

    This function sets up the experiment context, loads validation and test data loaders,
    initializes the persistence model, and evaluates it on both validation and test sets.
    The results, including metrics and parameters, are logged to MLflow.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        KeyError: If required keys are missing in the configuration or context.
        Exception: For any errors during model evaluation or logging.
    """
    ctx = setup_experiment("configs/config_training.yaml")
    cfg, device = ctx["cfg"], ctx["device"]

    _, val_loader, test_loader = get_dataloaders(ctx, multi_target=False)

    feature_cols = cfg["data"]["feature_cols"]
    target_col = cfg["data"]["target_col"]
    target_idx = feature_cols.index(target_col)

    model = PersistenceModel(target_feature_index=target_idx)

    if hasattr(model, "to"):
        model = model.to(device)

    loss_fn = nn.MSELoss()

    with mlflow.start_run(run_name="Persistence_Refactored"):
        mlflow.log_param("model_type", "Persistence")
        mlflow.log_param("target_col", target_col)
        mlflow.log_param("horizon", cfg["data"]["horizon"])

        val_metrics = evaluate_persistence(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            horizon=cfg["data"]["horizon"],
            device=device,
            scaler=ctx["scaler"],
            numeric_cols=ctx["numeric_cols"],
            target_col=target_col,
        )

        print("\nVal Metrics (Persistence):")
        print(val_metrics)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        test_metrics = evaluate_persistence(
            model=model,
            loader=test_loader,
            loss_fn=loss_fn,
            horizon=cfg["data"]["horizon"],
            device=device,
            scaler=ctx["scaler"],
            numeric_cols=ctx["numeric_cols"],
            target_col=target_col,
        )

        print("\nTest Metrics (Persistence):")
        print(test_metrics)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})


if __name__ == "__main__":
    main()
