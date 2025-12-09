from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiTargetSequenceDataset(Dataset):
    def __init__(
        self,
        path: str,
        feature_cols: List[str],
        target_cols: List[str],
        input_length: int,
        horizon: int,
        step: int = 1,
    ):
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
        return len(self.indices)

    def __getitem__(self, idx: int):
        start = self.indices[idx]
        end_in = start + self.input_length
        end_out = end_in + self.horizon

        x = self.features[start:end_in, :]  # [L, F]
        y = self.targets[end_in:end_out, :]  # [H, T]

        x = torch.from_numpy(x)  # [L, F]
        y = torch.from_numpy(y)  # [H, T]

        return x, y
