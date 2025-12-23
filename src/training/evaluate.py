from typing import Dict, List, Optional

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from src.utils.metrics import rmse, smape
from src.utils.scale import inverse_target


@torch.no_grad()
def evaluate_model(
    model: Module,
    dataloader: DataLoader,
    loss_fn,
    device: str,
    scaler=None,
    numeric_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    all_target_cols: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Generic evaluation loop that supports both single-target and multi-target models.

    Args:
        all_target_cols: List of all target names corresponding to the model output channels.
                         Required if the model outputs multiple targets but you only want
                         to denormalize/measure 'target_col'.
    """
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

    y_true = torch.cat(y_true_all, dim=0)  # [N, H] or [N, H, F]
    y_pred = torch.cat(y_pred_all, dim=0)

    metrics = {
        "val_loss": total_loss / max(n_batches, 1),
        "rmse_norm": rmse(y_true, y_pred),
        "smape_norm": smape(y_true, y_pred),
    }

    if scaler is not None and numeric_cols is not None and target_col is not None:
        if all_target_cols is not None:
            if target_col not in all_target_cols:
                raise ValueError(
                    f"{target_col} not found in model outputs: {all_target_cols}"
                )

            t_idx = all_target_cols.index(target_col)
            # Slice: [N, H, F] -> [N, H]
            y_true_slice = y_true[..., t_idx]
            y_pred_slice = y_pred[..., t_idx]
        else:
            y_true_slice = y_true
            y_pred_slice = y_pred

        y_true_denorm = inverse_target(scaler, y_true_slice, numeric_cols, target_col)
        y_pred_denorm = inverse_target(scaler, y_pred_slice, numeric_cols, target_col)

        denorm_rmse = rmse(y_true_denorm, y_pred_denorm)
        metrics["rmse"] = denorm_rmse
        metrics["score"] = denorm_rmse
        metrics["smape"] = smape(y_true_denorm, y_pred_denorm)

    return metrics


def evaluate_persistence(
    loader,
    model,
    horizon: int,
    *,
    scaler=None,
    numeric_cols=None,
    target_col: str = None,
):
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for xb, yb in loader:
            y_pred = model.predict_batch(xb, horizon)  # [B, H]

            # Ensure yb is [B, H]
            if yb.ndim == 3 and yb.shape[-1] == 1:
                yb = yb[..., 0]

            y_true_all.append(yb.detach().cpu())
            y_pred_all.append(y_pred.detach().cpu())

    y_true = torch.cat(y_true_all, dim=0)  # [N, H]
    y_pred = torch.cat(y_pred_all, dim=0)  # [N, H]

    if scaler is not None:
        if numeric_cols is None or target_col is None:
            raise ValueError(
                "If scaler is provided, numeric_cols and target_col must be provided."
            )
        y_true = inverse_target(scaler, y_true, numeric_cols, target_col)
        y_pred = inverse_target(scaler, y_pred, numeric_cols, target_col)

    rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()

    denom = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
    smape = (
        torch.where(
            denom == 0, torch.zeros_like(denom), torch.abs(y_pred - y_true) / denom
        ).mean()
        * 100.0
    ).item()

    return {"rmse": rmse, "smape": smape}
