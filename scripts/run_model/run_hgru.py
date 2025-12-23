from functools import partial
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.optim import Adam

from src.models.hgru import HierarchicalGRUForecast
from src.training.engine import fit_model
from src.training.evaluate import evaluate_model
from src.training.setup import get_dataloaders, log_hyperparameters, setup_experiment


def main():
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

    hgru_cfg = cfg["models"]["hgru"]
    hgru_cfg["input_size"] = len(feature_cols)
    hgru_cfg["target_cols"] = target_cols
    hgru_cfg["horizon"] = horizon

    model = HierarchicalGRUForecast(**hgru_cfg).to(device)

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
    best_model_path = results_dir / "hier_hgru_best.pt"

    with mlflow.start_run(
        run_name=f"HierHGRU_{cfg['training']['mlflow']['run_postfix']}"
    ):
        log_hyperparameters(cfg, hgru_cfg, {"device": str(device)})

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
        mlflow.pytorch.log_model(model, "hier_hgru_model")
        mlflow.log_artifact("config.yaml")


if __name__ == "__main__":
    main()
