from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def detect_outliers_zscore(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    z_thresh: float = 3.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect outliers using a z-score threshold.

    For each selected numeric column, the z-score is computed as:
        (x - mean) / std

    A value is flagged as an outlier if its absolute z-score exceeds
    the given threshold.

    Args:
        df: Input DataFrame containing the data to analyze.
        cols: Optional list of column names to check for outliers.
            If None, all numeric columns in `df` are used.
        z_thresh: Z-score threshold above which a value is considered
            an outlier.

    Returns:
        A tuple containing:
            - outlier_mask: A DataFrame of boolean values with the same
              shape as the selected columns, where True indicates an
              outlier.
            - summary: A DataFrame summarizing the number and percentage
              of outliers per column, sorted by percentage descending.
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    df_num = df[cols].copy()
    z = (df_num - df_num.mean()) / df_num.std(ddof=0)
    outlier_mask = z.abs() > z_thresh

    total = len(df)
    counts = outlier_mask.sum()
    summary = pd.DataFrame(
        {
            "n_outliers": counts,
            "pct_outliers": (counts / total * 100).round(2),
        }
    ).sort_values("pct_outliers", ascending=False)

    print(f"\n=== Z-score Outlier Summary (threshold = {z_thresh}) ===")
    print(summary)

    return outlier_mask, summary


def detect_outliers_iqr(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    iqr_multiplier: float = 1.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect outliers using the Interquartile Range (IQR) rule.

    For each selected numeric column, values are flagged as outliers if
    they fall outside the range:
        [Q1 - iqr_multiplier * IQR, Q3 + iqr_multiplier * IQR]

    where:
        IQR = Q3 - Q1

    Args:
        df: Input DataFrame containing the data to analyze.
        cols: Optional list of column names to check for outliers.
            If None, all numeric columns in `df` are used.
        iqr_multiplier: Multiplier applied to the IQR to determine the
            lower and upper bounds for outlier detection.

    Returns:
        A tuple containing:
            - outlier_mask: A DataFrame of boolean values with one column
              per analyzed feature, where True indicates an outlier.
            - summary: A DataFrame summarizing the number and percentage
              of outliers per column, sorted by percentage descending.
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_mask = pd.DataFrame(False, index=df.index, columns=cols)

    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        outlier_mask[col] = (df[col] < lower) | (df[col] > upper)

    total = len(df)
    counts = outlier_mask.sum()
    summary = pd.DataFrame(
        {
            "n_outliers": counts,
            "pct_outliers": (counts / total * 100).round(2),
        }
    ).sort_values("pct_outliers", ascending=False)

    print(f"\n=== IQR Outlier Summary (multiplier = {iqr_multiplier}) ===")
    print(summary)

    return outlier_mask, summary
