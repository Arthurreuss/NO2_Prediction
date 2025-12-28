from typing import Dict, List

import torch
import torch.nn as nn


def compute_saliency_gru(
    model: nn.Module,
    loader,
    device: torch.device,
    target_cols: List[str],
    target_of_interest: str = "nitrogen_dioxide",
    max_batches: int = 10,
) -> Dict[str, torch.Tensor]:
    """Compute gradient-based saliency scores for GRU-style forecasting models.

    This function computes input-gradient saliency by accumulating the absolute
    gradient of the model's target prediction with respect to the input
    features. Gradients are aggregated over batches (up to `max_batches`) and
    averaged over the number of samples processed.

    For multi-target models (3D output), the target channel specified by
    `target_of_interest` is selected. For single-target models (2D output),
    the output is used directly.

    Args:
        model: PyTorch model whose predictions are analyzed.
        loader: Iterable yielding (x, y) batches.
        device: Torch device on which computation is performed.
        target_cols: Ordered list of target names corresponding to the last
            dimension of the model output (for multi-target models).
        target_of_interest: Name of the target variable used to compute saliency.
        max_batches: Maximum number of batches from `loader` to process.

    Returns:
        A dictionary containing:
            - "feature_time_importance": A tensor of shape [L, F] representing
              the average absolute input-gradient per time step and feature.

    Raises:
        ValueError: If `target_of_interest` is not present in `target_cols`.
        RuntimeError: If no batches are processed (e.g., empty loader).
    """
    model.eval()
    grads_sum: torch.Tensor | None = None
    count = 0

    if target_of_interest not in target_cols:
        raise ValueError(
            f"{target_of_interest} not found in target columns: {target_cols}"
        )
    t_idx = target_cols.index(target_of_interest)

    for b_idx, (x, y) in enumerate(loader):
        if b_idx >= max_batches:
            break

        x = x.to(device)
        x.requires_grad_(True)
        model.zero_grad()

        y_hat = model(x)

        if y_hat.ndim == 3:
            y_target = y_hat[..., t_idx]  # [B, H]
        else:
            y_target = y_hat  # [B, H]

        score = y_target.sum()
        score.backward()

        grad = x.grad.detach().abs().cpu()

        if grads_sum is None:
            grads_sum = grad.sum(dim=0)  # Sum over batch -> [L, F]
        else:
            grads_sum += grad.sum(dim=0)

        count += x.size(0)

    if grads_sum is None:
        raise RuntimeError("No batches in loader for saliency computation.")

    feature_time_importance = grads_sum / float(count)

    return {
        "feature_time_importance": feature_time_importance,  # [L, F]
    }


def summarize_saliency(
    feature_time_importance: torch.Tensor,
    feature_names: List[str],
) -> Dict[str, float]:
    """Aggregate time-wise saliency into per-feature importance scores.

    This function collapses the time dimension by taking the mean over time,
    producing a single importance value per feature.

    Args:
        feature_time_importance: Saliency tensor of shape [L, F], typically
            returned by `compute_saliency_gru`.
        feature_names: Ordered list of feature names corresponding to the
            feature dimension F.

    Returns:
        A dictionary mapping each feature name to its mean saliency value.
    """
    per_feature = feature_time_importance.mean(dim=0)  # [F]
    return {name: float(val.item()) for name, val in zip(feature_names, per_feature)}
