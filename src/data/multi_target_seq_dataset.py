from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiTargetSequenceDataset(Dataset):
    """PyTorch Dataset for multi-target time-series sequence forecasting.

    This dataset creates sliding windows over a time-series stored in a
    parquet file. For each window, it returns:
      - an input sequence of features with length `input_length`
      - a target sequence of multiple targets with length `horizon`

    The windows are generated with a configurable step size.
    """

    def __init__(
        self,
        path: str,
        feature_cols: List[str],
        target_cols: List[str],
        input_length: int,
        horizon: int,
        step: int = 1,
    ) -> None:
        """Initialize the multi-target sequence dataset.

        Args:
            path: Path to the parquet file containing the time-series data.
            feature_cols: List of column names to be used as input features.
            target_cols: List of column names to be used as prediction targets.
            input_length: Number of past time steps in each input sequence.
            horizon: Number of future time steps to predict.
            step: Step size between consecutive sliding windows.
        """
        df = pd.read_parquet(path).reset_index(drop=True)
        self.features = df[feature_cols].values.astype("float32")
        self.targets = df[target_cols].values.astype("float32")
        self.input_length = input_length
        self.horizon = horizon
        self.step = step

        self.n = len(df)
        self.indices = []
        max_start = self.n - (input_length + horizon) + 1
        for start in range(0, max_start, step):
            self.indices.append(start)

    def __len__(self) -> int:
        """Return the number of available sliding windows.

        Returns:
            The number of samples in the dataset.
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a single input/target sequence pair.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple (x, y) where:
              - x is a tensor of shape [input_length, n_features]
              - y is a tensor of shape [horizon, n_targets]
        """
        start = self.indices[idx]
        end_in = start + self.input_length
        end_out = end_in + self.horizon

        x = self.features[start:end_in, :]  # [L, F]
        y = self.targets[end_in:end_out, :]  # [H, T]

        x = torch.from_numpy(x)  # [L, F]
        y = torch.from_numpy(y)  # [H, T]

        return x, y
