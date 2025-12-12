from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesWindowDataset(Dataset):
    """
    Creates (X, y) pairs from a time series with sliding windows.

    X shape: [input_length, num_features]
    y shape: [horizon] or [horizon, 1] for single target
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        input_length: int,
        horizon: int,
        group_col: Optional[str] = "location",
    ) -> None:
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.input_length = input_length
        self.horizon = horizon
        self.group_col = group_col

        sort_cols = [c for c in [group_col, "time"] if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)

        self.df = df

        self.indices = self._compute_indices()

    def _compute_indices(self):
        idxs = []
        if self.group_col and self.group_col in self.df.columns:
            for _, g in self.df.groupby(self.group_col):
                n = len(g)
                max_start = n - (self.input_length + self.horizon) + 1
                if max_start <= 0:
                    continue
                base = g.index[0]
                for s in range(max_start):
                    idxs.append(base + s)
        else:
            n = len(self.df)
            max_start = n - (self.input_length + self.horizon) + 1
            for s in range(max_start):
                idxs.append(s)
        return idxs

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        end_x = start + self.input_length
        end_y = end_x + self.horizon

        window = self.df.iloc[start:end_x]
        target_window = self.df.iloc[end_x:end_y]

        x = torch.tensor(window[self.feature_cols].values, dtype=torch.float32)
        y = torch.tensor(target_window[self.target_col].values, dtype=torch.float32)

        # shapes: [L, F], [H]
        return x, y
