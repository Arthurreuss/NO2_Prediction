from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def compute_correlation_matrix(
    df: pd.DataFrame, method: str = "pearson"
) -> pd.DataFrame:
    """
    Compute correlation matrix for numeric features.
    """
    df_num = df.select_dtypes(include=[np.number])
    corr = df_num.corr(method=method)
    print(f"=== {method.capitalize()} correlation matrix (numeric features) ===")
    print(corr)
    return corr


def plot_correlation_heatmap(
    df: pd.DataFrame,
    method: str = "pearson",
    figsize: Tuple[int, int] = (12, 10),
    vmax: float = 1.0,
) -> None:
    """
    Plot correlation heatmap for numeric features.
    """
    corr = df.select_dtypes(include=[np.number]).corr(method=method)
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=False, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    plt.title(f"{method.capitalize()} correlation heatmap")
    plt.tight_layout()
    plt.show()


def correlations_with_target(
    df: pd.DataFrame,
    target_col: str,
    method: str = "pearson",
) -> pd.Series:
    """
    Compute correlation of all numeric features with a given target column.
    Returns a Series sorted by absolute correlation descending.
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in DataFrame.")

    df_num = df.select_dtypes(include=[np.number]).copy()
    # ensure target is included in numeric subset
    if target_col not in df_num.columns:
        df_num[target_col] = df[target_col].astype(float)

    corr = df_num.corr(method=method)[target_col].drop(target_col)
    corr_sorted = corr.reindex(corr.abs().sort_values(ascending=False).index)

    print(f"=== {method.capitalize()} correlations with target '{target_col}' ===")
    print(corr_sorted)
    return corr_sorted
