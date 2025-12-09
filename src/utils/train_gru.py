from typing import Dict, List, Optional

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .metrics import rmse, smape
from .scale import inverse_target


def train_one_epoch(
    model: Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    loss_fn,
    device: str,
    grad_clip: Optional[float] = None,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_model(
    model: Module,
    dataloader: DataLoader,
    loss_fn,
    device: str,
    scaler=None,
    numeric_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    y_true_all, y_pred_all = [], []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        total_loss += loss.item()
        n_batches += 1

        y_true_all.append(y.cpu())
        y_pred_all.append(y_hat.cpu())

    y_true = torch.cat(y_true_all, dim=0)  # [N, H], normalized
    y_pred = torch.cat(y_pred_all, dim=0)

    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "rmse_norm": rmse(y_true, y_pred),
        "smape_norm": smape(y_true, y_pred),
    }

    # Denormalized metrics in original units (µg/m³) if possible
    if scaler is not None and numeric_cols is not None and target_col is not None:
        y_true_denorm = inverse_target(scaler, y_true, numeric_cols, target_col)
        y_pred_denorm = inverse_target(scaler, y_pred, numeric_cols, target_col)

        metrics["rmse"] = rmse(y_true_denorm, y_pred_denorm)
        metrics["smape"] = smape(y_true_denorm, y_pred_denorm)

    return metrics
