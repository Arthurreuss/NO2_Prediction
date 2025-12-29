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
    device: str,
    loss_fn,
    scaler=None,
    numeric_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    all_target_cols: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Evaluate a forecasting model on a DataLoader.

    This evaluation function supports both single-target and multi-target models.

    Behavior:
      - Always runs the model in eval mode and accumulates predictions/targets
        across the full `dataloader`.
      - If `all_target_cols` is provided, computes denormalized RMSE/SMAPE for
        each target channel and stores them under keys:
            - "rmse_<col>", "smape_<col>"
        It also computes macro averages across targets:
            - "rmse_mean", "smape_mean"
        If `target_col` is present in `all_target_cols`, it additionally sets:
            - "score" = "rmse_<target_col>"
      - If `all_target_cols` is not provided, computes denormalized RMSE/SMAPE
        only for `target_col` and stores them under keys:
            - "rmse_<target_col>", "smape_<target_col>", and "score"

    Args:
        model: PyTorch model to evaluate.
        dataloader: DataLoader yielding (x, y) batches.
        device: Device identifier used by `Tensor.to(device)`.
        loss_fn: Loss function used to compute batch loss.
        scaler: Optional fitted scaler used to inverse-transform targets.
        numeric_cols: Optional list of numeric column names used when fitting the scaler.
        target_col: Optional target name used for single-target denormalization and/or
            for selecting the "score" metric when `all_target_cols` is provided.
        all_target_cols: Optional list of all target names corresponding to the model
            output channels. When provided, the function expects `y_true` and `y_pred`
            to be shaped [N, H, F], where F == len(all_target_cols), and computes per-target
            denormalized metrics for each target.

    Returns:
        A dictionary of denormalized evaluation metrics. Keys depend on whether
        `all_target_cols` is provided (per-target + macro averages) or not
        (single-target only).

    Raises:
        ValueError: If `all_target_cols` is provided but `y_true`/`y_pred` are not 3D
            or if their channel dimension does not match `len(all_target_cols)`.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    y_true_all, y_pred_all = [], []
    metrics = {}

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        total_loss += float(loss.item())
        n_batches += 1

        y_true_all.append(y.detach().cpu())
        y_pred_all.append(y_hat.detach().cpu())

    y_true = torch.cat(y_true_all, dim=0)  # [N,H] or [N,H,F]
    y_pred = torch.cat(y_pred_all, dim=0)

    if all_target_cols is not None:
        # y must be [N,H,F] in this mode
        if y_true.ndim != 3 or y_pred.ndim != 3:
            raise ValueError(
                f"all_target_cols was provided, but model outputs have shape "
                f"y_true={tuple(y_true.shape)}, y_pred={tuple(y_pred.shape)}; expected [N,H,F]."
            )

        f = y_true.shape[-1]
        if len(all_target_cols) != f:
            raise ValueError(
                f"len(all_target_cols)={len(all_target_cols)} does not match "
                f"number of output channels F={f}."
            )

        denorm_rmses = []
        denorm_smapes = []

        for j, col in enumerate(all_target_cols):
            y_true_slice = y_true[..., j]  # [N,H]
            y_pred_slice = y_pred[..., j]  # [N,H]

            y_true_den = inverse_target(scaler, y_true_slice, numeric_cols, col)
            y_pred_den = inverse_target(scaler, y_pred_slice, numeric_cols, col)

            r = rmse(y_true_den, y_pred_den)
            s = smape(y_true_den, y_pred_den)

            metrics[f"rmse_{col}"] = r
            metrics[f"smape_{col}"] = s

            denorm_rmses.append(r)
            denorm_smapes.append(s)

        metrics["rmse_mean"] = float(sum(denorm_rmses) / max(len(denorm_rmses), 1))
        metrics["smape_mean"] = float(sum(denorm_smapes) / max(len(denorm_smapes), 1))

        if target_col in all_target_cols:
            metrics["score"] = metrics[f"rmse_{target_col}"]

        return metrics

    # Case 2: single-target denorm metrics
    y_true_denorm = inverse_target(scaler, y_true, numeric_cols, target_col)
    y_pred_denorm = inverse_target(scaler, y_pred, numeric_cols, target_col)

    denorm_rmse = rmse(y_true_denorm, y_pred_denorm)
    metrics[f"rmse_{target_col}"] = denorm_rmse
    metrics["score"] = denorm_rmse
    metrics[f"smape_{target_col}"] = smape(y_true_denorm, y_pred_denorm)

    return metrics


def evaluate_persistence(
    loader,
    model,
    horizon: int,
    *,
    scaler=None,
    numeric_cols=None,
    target_col: str = None,
) -> Dict[str, float]:
    """Evaluate a persistence baseline on a dataset.

    This function runs a persistence-style model that implements
    `predict_batch(xb, horizon)` and computes RMSE and SMAPE over all samples.
    If a scaler is provided, predictions and targets are inverse-transformed
    before metric computation.

    Args:
        loader: Iterable yielding (xb, yb) batches.
        model: Persistence model implementing `predict_batch`.
        horizon: Forecast horizon (number of future time steps predicted).
        scaler: Optional fitted scaler used to inverse-transform the target.
        numeric_cols: Optional list of numeric column names used by the scaler.
        target_col: Optional name of the target column to denormalize/measure.

    Returns:
        A dictionary containing:
            - "rmse": Root mean squared error.
            - "smape": Symmetric mean absolute percentage error (in percent).

    Raises:
        ValueError: If `scaler` is provided but `numeric_cols` or `target_col`
            are not provided.
    """
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

    return {f"rmse_{target_col}": rmse, f"smape_{target_col}": smape}
