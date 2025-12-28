from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def compute_correlation_matrix(
    df: pd.DataFrame, method: str = "pearson"
) -> pd.DataFrame:
    """Compute a correlation matrix for numeric features in a DataFrame.

    This function selects all numeric columns from the input DataFrame and
    computes their pairwise correlations using the specified method.

    Args:
        df: Input DataFrame containing the data.
        method: Correlation method to use. Common options include
            "pearson", "spearman", and "kendall".

    Returns:
        A pandas DataFrame representing the correlation matrix of all
        numeric features.
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
    """Plot a correlation heatmap for numeric features.

    This function computes the correlation matrix for numeric columns and
    visualizes it as a heatmap using seaborn.

    Args:
        df: Input DataFrame containing the data.
        method: Correlation method to use. Common options include
            "pearson", "spearman", and "kendall".
        figsize: Size of the matplotlib figure as (width, height).
        vmax: Maximum absolute value for the color scale. The minimum is
            set to -vmax.
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
    """Compute correlations of numeric features with a target column.

    This function computes the correlation between each numeric feature
    and a specified target column, then sorts the results by absolute
    correlation value in descending order.

    Args:
        df: Input DataFrame containing the data.
        target_col: Name of the target column to correlate against.
        method: Correlation method to use. Common options include
            "pearson", "spearman", and "kendall".

    Returns:
        A pandas Series containing correlations between each numeric
        feature and the target column, sorted by absolute correlation
        descending.

    Raises:
        ValueError: If `target_col` is not present in the input DataFrame.
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in DataFrame.")

    df_num = df.select_dtypes(include=[np.number]).copy()
    if target_col not in df_num.columns:
        df_num[target_col] = df[target_col].astype(float)

    corr = df_num.corr(method=method)[target_col].drop(target_col)
    corr_sorted = corr.reindex(corr.abs().sort_values(ascending=False).index)

    print(f"=== {method.capitalize()} correlations with target '{target_col}' ===")
    print(corr_sorted)
    return corr_sorted
