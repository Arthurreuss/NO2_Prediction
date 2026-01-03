import numpy as np
import pandas as pd


def calculate_smape(y_true, y_pred):
    """Calculates Symmetric Mean Absolute Percentage Error (0-100%)."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * diff


def get_horizon_metrics(df_history, preds_sequence, pollutant="nitrogen_dioxide"):
    """
    Aggregates metrics by forecast step (1-72).
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


def get_performance_over_time(df_history, preds_all, pollutant="nitrogen_dioxide"):
    """
    Calculates aggregated metrics (RMSE, SMAPE) for each COMPLETE forecast run.
    Aggregates the 72h horizon into single scalar values per run.
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

    return pd.DataFrame(results).sort_values("prediction_generated_at")
