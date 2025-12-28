import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def _rmse_from_residuals(residuals: torch.Tensor) -> float:
    """Compute RMSE from a tensor of residuals.

    Args:
        residuals: Tensor containing residuals computed as (y_pred - y_true).

    Returns:
        Root mean squared error as a Python float.
    """
    mse = (residuals**2).mean().item()
    return math.sqrt(mse)


def evaluate_rmse_norm(
    model: nn.Module,
    loader,
    device: torch.device,
    target_cols: List[str],
    target_of_interest: str = "nitrogen_dioxide",
) -> float:
    """Compute RMSE on the normalized target for a specific pollutant.

    This function evaluates RMSE in normalized space for a single target of
    interest. It supports:
      - multi-target models that output tensors with shape [B, H, T]
      - single-target models that output tensors with shape [B, H]

    The ground-truth `y` is assumed to come from a multi-target loader and to
    have shape [B, H, T].

    Args:
        model: PyTorch model to evaluate.
        loader: Iterable yielding (x, y) batches.
        device: Torch device on which computation is performed.
        target_cols: Ordered list of target names corresponding to the last
            dimension of `y` (and `y_hat` if multi-target).
        target_of_interest: Name of the target variable to evaluate.

    Returns:
        RMSE in normalized space as a Python float.

    Raises:
        ValueError: If `target_of_interest` is not present in `target_cols`.
    """
    model.eval()
    all_residuals: List[torch.Tensor] = []

    if target_of_interest not in target_cols:
        raise ValueError(
            f"{target_of_interest} not found in target columns: {target_cols}"
        )
    t_idx = target_cols.index(target_of_interest)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)  # Shape: [B, H, T] (since loader is multi_target=True)

            y_hat = model(x)
            y_true_no2 = y[..., t_idx]  # [B, H]

            if y_hat.ndim == 3:
                y_pred_no2 = y_hat[..., t_idx]
            else:
                y_pred_no2 = y_hat

            residuals = (y_pred_no2 - y_true_no2).detach().cpu()
            all_residuals.append(residuals)

    all_residuals_cat = torch.cat(all_residuals, dim=0)  # [N, H]
    return _rmse_from_residuals(all_residuals_cat)


def permutation_importance(
    model: nn.Module,
    loader,
    device: torch.device,
    feature_names: List[str],
    target_cols: List[str],
    target_of_interest: str = "nitrogen_dioxide",
    n_repeats: int = 5,
    seed: int = 42,
) -> Tuple[float, Dict[str, float]]:
    """Compute permutation feature importance based on RMSE degradation.

    This method estimates feature importance by measuring the increase in
    normalized RMSE when a single feature is permuted across the batch
    dimension (for each time step). The process is repeated `n_repeats`
    times and averaged.

    Args:
        model: PyTorch model to evaluate.
        loader: Iterable yielding (x, y) batches.
        device: Torch device on which computation is performed.
        feature_names: Ordered list of feature names corresponding to the last
            dimension of `x`.
        target_cols: Ordered list of target names corresponding to the last
            dimension of `y` (and `y_hat` if multi-target).
        target_of_interest: Name of the target variable used to compute RMSE.
        n_repeats: Number of permutation repeats per feature.
        seed: Random seed used for permutation reproducibility.

    Returns:
        A tuple of:
            - baseline_rmse: RMSE in normalized space without permutation.
            - importances: Dictionary mapping each feature name to the average
              RMSE increase caused by permuting that feature.

    Raises:
        ValueError: If `target_of_interest` is not present in `target_cols`.
    """
    torch.manual_seed(seed)
    model.eval()

    baseline_rmse = evaluate_rmse_norm(
        model, loader, device, target_cols, target_of_interest
    )

    importances: Dict[str, float] = {name: 0.0 for name in feature_names}

    for _ in range(n_repeats):
        for j, name in enumerate(feature_names):
            all_residuals: List[torch.Tensor] = []
            t_idx = target_cols.index(target_of_interest)

            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                idx = torch.randperm(x.size(0), device=x.device)
                x_perm = x.clone()
                x_perm[:, :, j] = x_perm[idx, :, j]

                with torch.no_grad():
                    y_hat = model(x_perm)

                y_true_no2 = y[..., t_idx]

                if y_hat.ndim == 3:
                    y_pred_no2 = y_hat[..., t_idx]
                else:
                    y_pred_no2 = y_hat

                residuals = (y_pred_no2 - y_true_no2).detach().cpu()
                all_residuals.append(residuals)

            all_residuals_cat = torch.cat(all_residuals, dim=0)
            rmse_perm = _rmse_from_residuals(all_residuals_cat)

            importances[name] += rmse_perm - baseline_rmse

    for name in importances:
        importances[name] /= float(n_repeats)

    return baseline_rmse, importances
