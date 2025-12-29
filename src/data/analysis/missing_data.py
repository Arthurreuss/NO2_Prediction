import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def assess_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Compute missing-data statistics per column.

    This function calculates the absolute and relative amount of missing
    values for each column in the input DataFrame and reports the column
    data types. The resulting summary is sorted by missing percentage in
    descending order.

    Args:
        df: Input DataFrame for which missing data should be analyzed.

    Returns:
        A DataFrame indexed by column name with the following columns:
            - dtype: Data type of the column (as string).
            - n_missing: Number of missing (NaN) values in the column.
            - missing_pct: Percentage of missing values relative to the
              total number of rows.
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
    """Visualize missing values in a DataFrame as a heatmap.

    The heatmap shows missingness (NaN values) across the entire DataFrame:
        - Rows correspond to features (columns in the original DataFrame).
        - Columns correspond to samples (rows in the original DataFrame).

    All feature names are displayed on the Y-axis to provide a complete
    overview of missing data patterns.

    Args:
        df: Input DataFrame whose missing values should be visualized.
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
