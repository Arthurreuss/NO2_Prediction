import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def diurnal_profile(
    df: pd.DataFrame,
    time_col: str = "time",
    cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute average diurnal (hour-of-day) profile for selected columns.
    """
    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not found in DataFrame.")
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        raise TypeError(f"{time_col} must be datetime64 dtype.")

    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    tmp = df.copy()
    tmp["hour"] = tmp[time_col].dt.hour
    profile = tmp.groupby("hour")[cols].mean()

    print("=== Diurnal profile (mean by hour of day) ===")
    print(profile)
    return profile


def weekly_profile(
    df: pd.DataFrame,
    time_col: str = "time",
    cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute average weekly (day-of-week) profile for selected columns.
    Monday=0, Sunday=6.
    """
    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not found in DataFrame.")
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        raise TypeError(f"{time_col} must be datetime64 dtype.")

    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    tmp = df.copy()
    tmp["dayofweek"] = tmp[time_col].dt.dayofweek
    profile = tmp.groupby("dayofweek")[cols].mean()

    print("=== Weekly profile (mean by day of week, Mon=0) ===")
    print(profile)
    return profile
