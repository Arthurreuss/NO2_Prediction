from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def detect_outliers_zscore(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    z_thresh: float = 3.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect outliers using a z-score threshold.

    Returns:
        - outlier_mask: DataFrame of boolean values (True = outlier)
        - summary: DataFrame with counts and percentage for each column
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
    """
    Detect outliers using the IQR rule.

    Returns:
        - outlier_mask: DataFrame of boolean values (True = outlier)
        - summary: DataFrame with counts and percentage for each column
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
