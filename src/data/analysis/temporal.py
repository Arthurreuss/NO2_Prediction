from typing import List, Optional

import pandas as pd


def diurnal_profile(
    df: pd.DataFrame,
    time_col: str = "time",
    cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute an average diurnal (hour-of-day) profile.

    This function groups the data by hour of day (0–23) based on the
    specified time column and computes the mean for the selected columns.

    Args:
        df: Input DataFrame containing a datetime column.
        time_col: Name of the datetime column used to extract the hour
            of day.
        cols: Optional list of column names for which the diurnal profile
            should be computed. If None, pandas will attempt to compute
            the mean for all columns supported by `groupby().mean()`.

    Returns:
        A DataFrame indexed by hour of day (0–23) containing the mean
        values of the selected columns.
    """
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
    """Compute an average weekly (day-of-week) profile.

    This function groups the data by day of week based on the specified
    time column and computes the mean for the selected columns.

    Day-of-week encoding follows pandas convention:
        Monday = 0, ..., Sunday = 6.

    Args:
        df: Input DataFrame containing a datetime column.
        time_col: Name of the datetime column used to extract the day of
            week.
        cols: Optional list of column names for which the weekly profile
            should be computed. If None, pandas will attempt to compute
            the mean for all columns supported by `groupby().mean()`.

    Returns:
        A DataFrame indexed by day of week (0–6, where 0 = Monday)
        containing the mean values of the selected columns.
    """
    tmp = df.copy()
    tmp["dayofweek"] = tmp[time_col].dt.dayofweek
    profile = tmp.groupby("dayofweek")[cols].mean()

    print("=== Weekly profile (mean by day of week, Mon=0) ===")
    print(profile)
    return profile
