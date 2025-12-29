from pathlib import Path
from typing import Callable, Optional

import mlflow
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
    """Train a model for a single epoch.

    This function iterates over a dataloader, performs forward and backward
    passes, applies optional gradient clipping, and updates model parameters.

    Args:
        model: PyTorch model to train.
        dataloader: Iterable yielding batches of (x, y) tensors.
        optimizer: Optimizer used to update model parameters.
        loss_fn: Loss function used to compute training loss.
        device: Device on which training is performed.
        grad_clip: Optional maximum norm for gradient clipping. If provided,
            gradients are clipped using `torch.nn.utils.clip_grad_norm_`.

    Returns:
        The average loss over all batches in the epoch.
    """
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
    """Fit a model using a training loop with early stopping, MLflow logging, and Optuna pruning.

    The loop trains for up to `epochs` epochs, evaluates on a validation loader
    via `validate_fn`, tracks the best validation score, and applies early
    stopping based on `patience`. When a new best score is found, the model
    state dict is saved to `save_path`.

    If `trial` is provided, validation scores are reported to Optuna and the
    trial may be pruned.

    If `log_mlflow` is True, training loss and validation metrics are logged
    to MLflow each epoch.

    Args:
        model: PyTorch model to train.
        train_loader: Iterable yielding training batches.
        val_loader: Iterable yielding validation batches.
        optimizer: Optimizer used to update model parameters.
        loss_fn: Loss function used during training.
        device: Device on which training and validation are performed.
        epochs: Maximum number of epochs to train.
        patience: Number of consecutive epochs without improvement allowed
            before early stopping triggers.
        save_path: File path where the best model state dict is saved.
        validate_fn: Validation function called as
            `validate_fn(model=model, dataloader=val_loader, device=device)`.
            It is expected to return a dictionary of metrics.
        grad_clip: Optional maximum norm for gradient clipping. If provided,
            gradients are clipped during training.
        trial: Optional Optuna trial used for reporting and pruning.
        log_mlflow: Whether to log metrics to MLflow.

    Returns:
        The best validation score observed during training.
    """
    best_score = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, grad_clip
        )

        metrics = validate_fn(model=model, dataloader=val_loader, device=device)
        current_score = metrics["score"]

        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Score: {current_score:.4f}"
        )

        if log_mlflow:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            for k, v in metrics.items():
                mlflow.log_metric(f"val_{k}", float(v), step=epoch)

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
