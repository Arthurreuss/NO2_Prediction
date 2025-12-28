from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize_distributions(
    df: pd.DataFrame, numeric_only: bool = True
) -> pd.DataFrame:
    """Return basic descriptive statistics for columns in a DataFrame.

    By default, this function computes descriptive statistics only for
    numeric columns. If `numeric_only` is set to False, statistics are
    computed for all columns supported by pandas `describe`.

    Args:
        df: Input DataFrame containing the data to summarize.
        numeric_only: If True, restricts the summary to numeric columns
            only. If False, includes all columns.

    Returns:
        A DataFrame containing descriptive statistics (count, mean, std,
        min, selected percentiles, and max) for each included column.
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
    """Plot histograms for selected numeric columns.

    If no columns are explicitly provided, histograms are generated for
    all numeric columns in the DataFrame. Each column is plotted in its
    own subplot.

    Args:
        df: Input DataFrame containing the data to visualize.
        cols: Optional list of column names to plot. If None, all numeric
            columns in `df` are used.
        bins: Number of histogram bins to use for each plot.
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
