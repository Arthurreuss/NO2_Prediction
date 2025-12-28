from typing import Dict, List

import torch
import torch.nn as nn


def cross_pollutant_importance(
    model: nn.Module,
    loader,
    device: torch.device,
    feature_cols: List[str],
    target_cols: List[str],
    pollutants_of_interest: List[str],
    max_batches: int = 10,
) -> Dict[str, float]:
    """Estimate cross-pollutant importance via input-gradient attribution.

    This function computes gradient-based importance scores for a set of
    pollutant features with respect to the model's NO2 predictions. For each
    batch (up to `max_batches`), it:
      - enables gradients on the input tensor `x`
      - computes model predictions `y_hat`
      - selects the NO2 output channel and sums it to form a scalar score
      - backpropagates to obtain gradients of the score w.r.t. inputs
      - aggregates absolute gradients for each pollutant-of-interest feature

    The returned importance for each pollutant is the mean absolute gradient
    per input element (averaged over batch and time dimensions).

    Args:
        model: A PyTorch model producing outputs shaped [B, H, T], where T
            corresponds to `target_cols`.
        loader: Iterable yielding (x, y) batches.
        device: Torch device on which computation is performed.
        feature_cols: Ordered list of feature names corresponding to the last
            dimension of `x`.
        target_cols: Ordered list of target names corresponding to the last
            dimension of `y_hat`.
        pollutants_of_interest: List of pollutant feature names to score. Only
            those present in `feature_cols` are used.
        max_batches: Maximum number of batches from `loader` to process.

    Returns:
        A dictionary mapping pollutant feature names to mean absolute gradient
        importance scores.
    """
    model.eval()
    no2_name = "nitrogen_dioxide"
    no2_idx = target_cols.index(no2_name)

    pollutant_to_idx = {
        p: feature_cols.index(p) for p in pollutants_of_interest if p in feature_cols
    }

    grad_sums: Dict[str, float] = {p: 0.0 for p in pollutant_to_idx}
    total_elements = 0

    for b_idx, (x, y) in enumerate(loader):
        if b_idx >= max_batches:
            break

        x = x.to(device)
        x.requires_grad_(True)

        model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()

        y_hat = model(x)  # [B, H, T]
        y_no2 = y_hat[..., no2_idx]  # [B, H]

        score = y_no2.sum()
        score.backward()
        grad = x.grad.detach().abs().cpu()

        B, L, F = grad.shape
        total_elements += B * L

        for p, j in pollutant_to_idx.items():
            grad_sums[p] += grad[:, :, j].sum().item()

    pollutant_importance = {p: val / total_elements for p, val in grad_sums.items()}

    return pollutant_importance
