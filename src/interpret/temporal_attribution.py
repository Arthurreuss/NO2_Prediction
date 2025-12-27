from typing import Dict, List

import torch
import torch.nn as nn

from src.training.setup import get_dataloaders, setup_experiment
from src.utils.helper import load_torch_model_from_registry


def compute_saliency_gru(
    model: nn.Module,
    loader,
    device: torch.device,
    target_cols: List[str],
    target_of_interest: str = "nitrogen_dioxide",
    max_batches: int = 10,
) -> Dict[str, torch.Tensor]:
    """
    Compute gradient-based saliency (Input Gradients).
    Accumulates |grad| of the target prediction w.r.t input features.
    """
    model.eval()
    grads_sum = None
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
    """
    Collapse over time dimension to get per-feature importance.
    feature_time_importance: [L, F]
    """
    per_feature = feature_time_importance.mean(dim=0)  # [F]
    return {name: float(val.item()) for name, val in zip(feature_names, per_feature)}
