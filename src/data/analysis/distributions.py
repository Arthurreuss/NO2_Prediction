import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def summarize_distributions(
    df: pd.DataFrame, numeric_only: bool = True
) -> pd.DataFrame:
    """
    Return basic descriptive statistics for numeric columns.
    """
    if numeric_only:
        df_num = df.select_dtypes(include=[np.number])
    else:
        df_num = df

    desc = df_num.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    print("=== Distribution summary (numeric columns) ===")
    print(desc)
    return desc


def plot_histograms(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    bins: int = 50,
) -> None:
    """
    Plot histograms for selected numeric columns (or all numeric columns if cols is None).
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    n = len(cols)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))

    plt.figure(figsize=(14, n_rows * 4))
    for i, col in enumerate(cols, start=1):
        plt.subplot(n_rows, n_cols, i)
        df[col].hist(bins=bins)
        plt.title(col)
        plt.tight_layout()
    plt.show()
