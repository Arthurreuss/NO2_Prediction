from typing import List

import pandas as pd
from torch.utils.data import DataLoader

from .multi_target_seq_dataset import MultiTargetSequenceDataset
from .sequence_dataset import TimeSeriesWindowDataset


def make_dataloader(
    path: str,
    feature_cols: List[str],
    target_col: str,
    input_length: int,
    horizon: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    df = pd.read_parquet(path)
    ds = TimeSeriesWindowDataset(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        input_length=input_length,
        horizon=horizon,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return loader


def make_multitarget_dataloader(
    path: str,
    feature_cols: List[str],
    target_cols: List[str],
    input_length: int,
    horizon: int,
    batch_size: int,
    shuffle: bool,
    step: int = 1,
) -> DataLoader:
    ds = MultiTargetSequenceDataset(
        path=path,
        feature_cols=feature_cols,
        target_cols=target_cols,
        input_length=input_length,
        horizon=horizon,
        step=step,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)
