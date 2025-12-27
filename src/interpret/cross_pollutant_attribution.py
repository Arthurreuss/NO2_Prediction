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

    model.eval()
    no2_name = "nitrogen_dioxide"
    no2_idx = target_cols.index(no2_name)

    pollutant_to_idx = {
        p: feature_cols.index(p) for p in pollutants_of_interest if p in feature_cols
    }

    grad_sums = {p: 0.0 for p in pollutant_to_idx}
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
