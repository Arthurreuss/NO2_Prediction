import re

import numpy as np
import pandas as pd


def calculate_smape(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
) -> np.ndarray | pd.Series:
    """Calculates Symmetric Mean Absolute Percentage Error (0-100%).

    Computes the SMAPE between truth and prediction values. Handles division by
    zero by replacing those instances with 0.0.

    Args:
        y_true: Array-like of ground truth values.
        y_pred: Array-like of predicted values.

    Returns:
        Array-like containing the SMAPE scores (percentage 0-100).
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * diff


def get_horizon_metrics(
    df_history: pd.DataFrame,
    preds_sequence: pd.DataFrame,
    pollutant: str = "nitrogen_dioxide",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregates error metrics by forecast horizon step (1-72 hours).

    Calculates RMSE and SMAPE for each step in the forecast horizon to analyze
    how model performance degrades as the prediction window increases.

    Args:
        df_history: DataFrame containing historical ground truth data.
        preds_sequence: DataFrame containing sequence of predictions.
        pollutant: The pollutant column name to evaluate. Defaults to
            "nitrogen_dioxide".

    Returns:
        A tuple containing two DataFrames:
            1. horizon_stats: Aggregated statistics (mean RMSE, SMAPE, etc.)
               indexed by horizon step.
            2. merged: The joined DataFrame containing matched predictions and
               actuals used for calculation.
    """
    df_pred_all = preds_sequence.copy()

    df_pred_all["time"] = pd.to_datetime(df_pred_all["time"], utc=True)
    df_pred_all["prediction_generated_at"] = pd.to_datetime(
        df_pred_all["prediction_generated_at"], utc=True
    )
    diff = df_pred_all["time"] - df_pred_all["prediction_generated_at"]
    df_pred_all["step"] = diff.dt.total_seconds() / 3600
    df_pred_all["step"] = df_pred_all["step"].round().astype(int)

    df_history = df_history.copy()
    df_history["time"] = pd.to_datetime(df_history["time"], utc=True)

    merged = pd.merge(
        df_pred_all,
        df_history[["time", pollutant]],
        on="time",
        how="inner",
        suffixes=("_pred", "_actual"),
    )

    pred_col = f"{pollutant}_pred"
    actual_col = f"{pollutant}_actual"

    merged["sq_error"] = (merged[pred_col] - merged[actual_col]) ** 2
    merged["smape"] = calculate_smape(merged[actual_col], merged[pred_col])

    horizon_stats = (
        merged.groupby("step")
        .agg(
            MSE_mean=("sq_error", "mean"),
            MSE_std=("sq_error", "std"),
            SMAPE_mean=("smape", "mean"),
            SMAPE_std=("smape", "std"),
            Count=("sq_error", "count"),
        )
        .reset_index()
    )

    horizon_stats["RMSE_mean"] = np.sqrt(horizon_stats["MSE_mean"])

    horizon_stats["RMSE_upper"] = np.sqrt(
        horizon_stats["MSE_mean"] + horizon_stats["MSE_std"].fillna(0)
    )
    horizon_stats["RMSE_lower"] = np.sqrt(
        (horizon_stats["MSE_mean"] - horizon_stats["MSE_std"].fillna(0)).clip(lower=0)
    )

    return horizon_stats, merged


def get_performance_over_time(
    df_history: pd.DataFrame,
    preds_all: dict[str, pd.DataFrame],
    pollutant: str = "nitrogen_dioxide",
) -> pd.DataFrame:
    """Calculates aggregated metrics (RMSE, SMAPE) for each COMPLETE forecast run.

    Aggregates performance over the entire 72h horizon into single scalar values
    per forecast generation time. Used to track model stability over time.
    Only includes runs where a full 72-hour forecast exists.

    Args:
        df_history: DataFrame containing historical ground truth data.
        preds_all: Dictionary mapping model names to their raw prediction DataFrames.
        pollutant: The pollutant column name to evaluate. Defaults to
            "nitrogen_dioxide".

    Returns:
        DataFrame containing performance metrics (RMSE, SMAPE) for each model
        and generation timestamp, sorted by generation time.
    """
    results = []

    df_history = df_history.copy()
    df_history["time"] = pd.to_datetime(df_history["time"], utc=True)

    for model_name, df_pred in preds_all.items():
        if pollutant != "nitrogen_dioxide" and model_name == "GRU":
            continue

        df_pred = df_pred.copy()
        df_pred["time"] = pd.to_datetime(df_pred["time"], utc=True)
        df_pred["prediction_generated_at"] = pd.to_datetime(
            df_pred["prediction_generated_at"], utc=True
        )

        merged = pd.merge(
            df_pred,
            df_history[["time", pollutant]],
            on="time",
            how="inner",
            suffixes=("_pred", "_actual"),
        )

        pred_col = f"{pollutant}_pred"
        actual_col = f"{pollutant}_actual"

        merged["sq_error"] = (merged[pred_col] - merged[actual_col]) ** 2
        merged["smape"] = calculate_smape(merged[actual_col], merged[pred_col])

        grouped = merged.groupby("prediction_generated_at")

        for gen_time, group in grouped:
            if len(group) == 72:
                rmse = np.sqrt(group["sq_error"].mean())
                smape = group["smape"].mean()

                results.append(
                    {
                        "Model": model_name,
                        "prediction_generated_at": gen_time,
                        "RMSE": rmse,
                        "SMAPE": smape,
                        "Count": len(group),
                    }
                )
        if len(results) == 0:
            return pd.DataFrame()

    return pd.DataFrame(results).sort_values("prediction_generated_at")
