import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def _rmse_from_residuals(residuals: torch.Tensor) -> float:
    """
    Computes RMSE from a tensor of residuals (y_pred - y_true).
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
    """
    Compute RMSE on the normalized target for a specific pollutant.
    Handles both Single-Target (2D output) and Multi-Target (3D output) models.
    """
    model.eval()
    all_residuals = []

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

    all_residuals = torch.cat(all_residuals, dim=0)  # [N, H]
    return _rmse_from_residuals(all_residuals)


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
    """
    Compute permutation feature importance based on RMSE degradation.
    """
    torch.manual_seed(seed)
    model.eval()

    baseline_rmse = evaluate_rmse_norm(
        model, loader, device, target_cols, target_of_interest
    )

    importances = {name: 0.0 for name in feature_names}

    for _ in range(n_repeats):
        for j, name in enumerate(feature_names):
            all_residuals = []
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

            all_residuals = torch.cat(all_residuals, dim=0)
            rmse_perm = _rmse_from_residuals(all_residuals)

            importances[name] += rmse_perm - baseline_rmse

    for name in importances:
        importances[name] /= float(n_repeats)

    return baseline_rmse, importances
