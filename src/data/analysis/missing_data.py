import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def assess_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute missing data statistics per column.
    Returns a DataFrame with:
      - n_missing
      - missing_pct
      - dtype
    """
    n_rows = len(df)
    missing_counts = df.isna().sum()
    missing_pct = missing_counts / n_rows * 100

    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_missing": missing_counts,
            "missing_pct": missing_pct.round(2),
        }
    ).sort_values("missing_pct", ascending=False)

    print("=== Missing data summary (sorted by missing_pct) ===")
    print(summary)

    return summary


def plot_missing_heatmap(df: pd.DataFrame) -> None:
    """
    Visualize missingness as a heatmap with ALL feature names shown on the Y axis.
    Rows = features, Columns = samples.
    """

    missing = df.isna().T
    n_features = missing.shape[0]

    plt.figure(figsize=(18, n_features * 0.3))
    ax = sns.heatmap(
        missing,
        cmap="viridis",
        cbar=False,
        yticklabels=df.columns,
        xticklabels=1000,
    )

    ax.set_title("Missing Data Heatmap\n(rows = features, cols = samples)")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Features")

    plt.tight_layout()
    plt.show()
