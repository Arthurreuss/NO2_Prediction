import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def _rmse_from_residuals(residuals: torch.Tensor) -> float:
    """
    residuals: tensor of shape [N, H] (y_pred - y_true)
    """
    mse = (residuals**2).mean().item()
    return math.sqrt(mse)


def evaluate_rmse_norm_gru(
    model: nn.Module,
    loader,
    device: torch.device,
) -> float:
    """
    Compute RMSE on the normalized target (GRU, single target).
    loader yields (x, y) with:
        x: [B, L, F]  normalized inputs
        y: [B, H]     normalized target
    """
    model.eval()
    all_residuals = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)  # [B, H]
            residuals = (y_hat - y).detach().cpu()
            all_residuals.append(residuals)

    all_residuals = torch.cat(all_residuals, dim=0)  # [N, H]
    return _rmse_from_residuals(all_residuals)


def permutation_importance_gru(
    model: nn.Module,
    loader,
    device: torch.device,
    feature_names: List[str],
    n_repeats: int = 5,
    seed: int = 42,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute permutation feature importance on the *normalized* RMSE.

    For each feature j:
      - for each batch, permute feature j across the batch dimension
      - compute RMSE
      - importance = permuted_rmse - baseline_rmse

    Returns:
      baseline_rmse, dict(feature_name -> importance)
    """
    torch.manual_seed(seed)
    model.eval()

    baseline_rmse = evaluate_rmse_norm_gru(model, loader, device)

    F = len(feature_names)
    importances = {name: 0.0 for name in feature_names}

    for _ in range(n_repeats):
        for j, name in enumerate(feature_names):
            all_residuals = []
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                idx = torch.randperm(x.size(0), device=x.device)
                x_perm = x.clone()
                x_perm[:, :, j] = x_perm[idx, :, j]

                with torch.no_grad():
                    y_hat = model(x_perm)  # [B, H]
                residuals = (y_hat - y).detach().cpu()
                all_residuals.append(residuals)

            all_residuals = torch.cat(all_residuals, dim=0)
            rmse_perm = _rmse_from_residuals(all_residuals)
            importances[name] += rmse_perm - baseline_rmse

    for name in importances:
        importances[name] /= float(n_repeats)

    return baseline_rmse, importances
