from math import log
from pathlib import Path
from typing import Callable, Optional

import mlflow
import mlflow.pytorch
import optuna
import torch
import torch.nn as nn


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: Optional[float] = None,
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


def fit_model(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int,
    patience: int,
    save_path: Path,
    validate_fn: Optional[Callable],
    grad_clip: Optional[float] = None,
    trial: Optional[optuna.Trial] = None,
    log_mlflow: bool = True,
) -> float:
    """
    Generic training loop with Early Stopping, MLflow logging, and Optuna pruning.

    """
    best_score = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, grad_clip
        )

        metrics = validate_fn(model=model, dataloader=val_loader, device=device)
        current_score = metrics.get(
            "rmse", metrics.get("score", metrics.get("val_loss"))
        )

        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Score: {current_score:.4f} | RMSE: {metrics.get('rmse', float('nan')):.4f} | SMAPE: {metrics.get('smape', float('nan')):.4f}"
        )

        if log_mlflow:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v), step=epoch)

        # Optuna Pruning
        if trial:
            trial.report(current_score, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Early Stopping
        if current_score < best_score:
            best_score = current_score
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return best_score
