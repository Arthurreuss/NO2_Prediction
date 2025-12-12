from typing import Dict, List

import torch
import torch.nn as nn


def compute_saliency_gru(
    model: nn.Module,
    loader,
    device: torch.device,
    max_batches: int = 10,
) -> Dict[str, torch.Tensor]:
    """
    Compute gradient-based saliency for GRU on the validation set.

    For each batch:
      - x.requires_grad_(True)
      - y_hat = model(x)
      - take scalar = mean over horizon for each sample, then mean over batch
      - backprop to x; accumulate |grad| over batches.

    Returns dict with:
      "feature_time_importance": [L, F] tensor of mean |grad| per time step and feature
    """
    model.eval()
    grads_sum = None
    count = 0

    for b_idx, (x, y) in enumerate(loader):
        if b_idx >= max_batches:
            break

        x = x.to(device)
        x.requires_grad_(True)

        model.zero_grad()

        y_hat = model(x)  # [B, H]
        # scalar objective: mean prediction over batch and horizon
        score = y_hat.mean()
        score.backward()

        grad = x.grad.detach().abs().cpu()  # [B, L, F]

        if grads_sum is None:
            grads_sum = grad.sum(dim=0)  # [L, F]
        else:
            grads_sum += grad.sum(dim=0)

        count += x.size(0)

    if grads_sum is None:
        raise RuntimeError("No batches in loader for saliency computation.")

    # average over samples
    feature_time_importance = grads_sum / float(count)

    return {
        "feature_time_importance": feature_time_importance,  # [L, F]
    }


def summarize_saliency(
    feature_time_importance: torch.Tensor,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Collapse over time dimension to get per-feature importance.
    feature_time_importance: [L, F]
    Returns dict(feature_name -> importance)
    """
    # mean over time steps
    per_feature = feature_time_importance.mean(dim=0)  # [F]
    return {name: float(val.item()) for name, val in zip(feature_names, per_feature)}
