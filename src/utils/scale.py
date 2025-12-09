# src/utils/scale.py
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
    Inverse-transform the target column using the scaler that was
    fitted on df[numeric_cols] in PreProcessingPipeline.normalize_data.

    Strategy:
      - create dummy array of shape (N*H, len(numeric_cols))
      - fill ONLY the target column with flattened y_norm
      - apply scaler.inverse_transform
      - extract the target column and reshape back to [N, H]
    """
    y_np = y_norm.numpy()  # [N, H]
    n, h = y_np.shape
    flat = y_np.reshape(-1)  # [N*H]

    n_features = len(numeric_cols)
    target_idx = numeric_cols.index(target_col)

    dummy = np.zeros((n * h, n_features), dtype=np.float32)
    dummy[:, target_idx] = flat

    inv = scaler.inverse_transform(dummy)
    target_inv = inv[:, target_idx].reshape(n, h)

    return torch.from_numpy(target_inv)
