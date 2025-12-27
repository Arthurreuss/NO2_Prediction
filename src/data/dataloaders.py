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
    """Create a PyTorch DataLoader for a single-target time-series dataset.

    This function loads a parquet file into a pandas DataFrame, constructs
    a `TimeSeriesWindowDataset` with sliding windows, and wraps it in a
    PyTorch `DataLoader`.

    Args:
        path: Path to the parquet file containing the time-series data.
        feature_cols: List of column names to be used as input features.
        target_col: Name of the target column to predict.
        input_length: Number of past time steps used as input.
        horizon: Forecast horizon (number of future steps to predict).
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the dataset at each epoch.

    Returns:
        A PyTorch DataLoader yielding batches from the constructed
        `TimeSeriesWindowDataset`.
    """
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
    """Create a PyTorch DataLoader for a multi-target time-series dataset.

    This function constructs a `MultiTargetSequenceDataset` directly from
    a parquet file path and wraps it in a PyTorch `DataLoader`.

    Args:
        path: Path to the parquet file containing the time-series data.
        feature_cols: List of column names to be used as input features.
        target_cols: List of target column names to predict.
        input_length: Number of past time steps used as input.
        horizon: Forecast horizon (number of future steps to predict).
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the dataset at each epoch.
        step: Step size between consecutive sliding windows.

    Returns:
        A PyTorch DataLoader yielding batches from the constructed
        `MultiTargetSequenceDataset`.
    """
    ds = MultiTargetSequenceDataset(
        path=path,
        feature_cols=feature_cols,
        target_cols=target_cols,
        input_length=input_length,
        horizon=horizon,
        step=step,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)
