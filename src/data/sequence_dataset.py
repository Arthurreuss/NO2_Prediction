from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesWindowDataset(Dataset):
    """Create (X, y) pairs from a time series using sliding windows.

    Each sample consists of:
      - X: a window of input features of length `input_length`
      - y: the subsequent target values of length `horizon`

    Shapes:
        X: [input_length, num_features]
        y: [horizon] (single target)
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
        """Initialize the dataset and pre-compute valid window start indices.

        The DataFrame is sorted by `[group_col, "time"]` where those columns
        exist. If `group_col` is present, windows are computed independently
        per group to prevent windows from crossing group boundaries.

        Args:
            df: Input DataFrame containing the time series.
            feature_cols: Column names used as input features.
            target_col: Column name used as the prediction target.
            input_length: Number of past time steps in each input window.
            horizon: Number of future time steps in each target window.
            group_col: Optional column name used to group the data (e.g.,
                by location). If present in `df`, windows are generated
                separately within each group.
        """
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.input_length = input_length
        self.horizon = horizon
        self.group_col = group_col

        sort_cols = [c for c in [group_col, "time"] if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)

        self.df = df

        self.indices = self._compute_indices()

    def _compute_indices(self) -> List[int]:
        """Compute valid starting indices for sliding windows.

        If `group_col` is available, indices are computed per group such that
        windows do not cross group boundaries. Otherwise, indices are computed
        across the full DataFrame.

        Returns:
            A list of integer start indices for all valid (X, y) windows.
        """
        idxs: List[int] = []
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
        """Return the number of available sliding windows.

        Returns:
            The number of samples in the dataset.
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a single (X, y) sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple (x, y) where:
              - x is a float32 tensor of shape [input_length, num_features]
              - y is a float32 tensor of shape [horizon]
        """
        start = self.indices[idx]
        end_x = start + self.input_length
        end_y = end_x + self.horizon

        window = self.df.iloc[start:end_x]
        target_window = self.df.iloc[end_x:end_y]

        x = torch.tensor(window[self.feature_cols].values, dtype=torch.float32)
        y = torch.tensor(target_window[self.target_col].values, dtype=torch.float32)

        # shapes: [L, F], [H]
        return x, y
