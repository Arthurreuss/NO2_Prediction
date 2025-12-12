from typing import Dict, List

import torch
import torch.nn as nn


def cross_pollutant_importance_multi_gru(
    model: nn.Module,
    loader,
    device: torch.device,
    feature_cols: List[str],
    target_cols: List[str],
    pollutants_of_interest: List[str],
    max_batches: int = 10,
) -> Dict[str, float]:
    """
    Compute gradient-based cross-pollutant importance for NO2 prediction.

    Assumes:
      - model(x) -> [B, H, T] where T = len(target_cols)
      - NO2 is one of target_cols ("nitrogen_dioxide")
      - pollutants_of_interest are column names in feature_cols

    Returns:
      dict: pollutant_name -> average |∂y_NO2 / ∂x_pollutant|
            aggregated over batch and time steps.
    """
    model.eval()

    no2_name = "nitrogen_dioxide"
    if no2_name not in target_cols:
        raise ValueError(f"{no2_name} not found in target_cols: {target_cols}")
    no2_idx = target_cols.index(no2_name)

    # map pollutant name -> feature index in feature_cols
    pollutant_to_idx: Dict[str, int] = {}
    for p in pollutants_of_interest:
        if p not in feature_cols:
            # silently skip or raise depending on your preference
            # here we skip and warn later if dict is empty
            continue
        pollutant_to_idx[p] = feature_cols.index(p)

    if not pollutant_to_idx:
        raise ValueError(
            f"None of the pollutants_of_interest are in feature_cols. "
            f"pollutants_of_interest={pollutants_of_interest}, feature_cols={feature_cols}"
        )

    grad_sums: Dict[str, float] = {p: 0.0 for p in pollutant_to_idx}
    sample_count = 0

    for b_idx, (x, y) in enumerate(loader):
        if b_idx >= max_batches:
            break

        x = x.to(device)
        x.requires_grad_(True)
        model.zero_grad()

        y_hat = model(x)  # [B, H, T]
        y_no2 = y_hat[..., no2_idx]  # [B, H]
        score = y_no2.mean()
        score.backward()

        grad = x.grad.detach().abs().cpu()  # [B, L, F]
        B, L, F = grad.shape
        sample_count += B

        # sum |grad| over batch and time for each pollutant feature
        for p, j in pollutant_to_idx.items():
            grad_sums[p] += grad[:, :, j].sum().item()

    if sample_count == 0:
        raise RuntimeError("No batches processed for cross-pollutant attribution.")

    # normalize by number of samples and timesteps for comparability
    norm_factor = float(sample_count * L)
    pollutant_importance = {p: grad_sums[p] / norm_factor for p in pollutant_to_idx}

    return pollutant_importance
