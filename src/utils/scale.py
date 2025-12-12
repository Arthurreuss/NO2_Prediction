from typing import List

import numpy as np
import torch
from sklearn.base import BaseEstimator


def inverse_target(
    scaler: BaseEstimator,
    y_norm: torch.Tensor,
    numeric_cols: List[str],
    target_col: str,
) -> torch.Tensor:
    """
    Accepts y_norm shaped [N,H] or [N,H,1]. Returns [N,H] in original units.
    """
    y_norm = y_norm.detach().cpu()

    # allow [N,H,1]
    if y_norm.ndim == 3 and y_norm.shape[-1] == 1:
        y_norm = y_norm[..., 0]

    if y_norm.ndim != 2:
        raise ValueError(
            f"inverse_target expected [N,H] (or [N,H,1]) but got shape {tuple(y_norm.shape)}"
        )

    y_np = y_norm.numpy()
    n, h = y_np.shape
    flat = y_np.reshape(-1)

    n_features = len(numeric_cols)
    target_idx = numeric_cols.index(target_col)

    dummy = np.zeros((n * h, n_features), dtype=np.float32)
    dummy[:, target_idx] = flat

    inv = scaler.inverse_transform(dummy)
    target_inv = inv[:, target_idx].reshape(n, h)

    return torch.from_numpy(target_inv)
