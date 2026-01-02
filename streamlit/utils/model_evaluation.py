import numpy as np
import pandas as pd


def evaluate_model_performance(df_history, preds_sequence):
    """
    Evaluates a sequence of 72-hour forecasts against history.

    Args:
        df_history: DataFrame with ['time', 'nitrogen_dioxide']
        preds_sequence: A list of DataFrames (or single concatenated DF),
                        where each row is a forecast step.
                        Must contain: ['time', 'prediction', 'step']
                        ('step' is 1-72 indicating forecast horizon)
    """
    if isinstance(preds_sequence, list):
        if not preds_sequence:
            return None
        df_pred_all = pd.concat(preds_sequence)
    else:
        df_pred_all = preds_sequence.copy()

    if "step" not in df_pred_all.columns:
        df_pred_all["step"] = 1

    merged = pd.merge(
        df_pred_all,
        df_history[["time", "nitrogen_dioxide"]],
        on="time",
        how="inner",
        suffixes=("_pred", "_actual"),
    )

    if merged.empty:
        return None

    merged["error"] = merged["prediction"] - merged["nitrogen_dioxide"]
    merged["abs_error"] = merged["error"].abs()
    merged["sq_error"] = merged["error"] ** 2

    mae_global = merged["abs_error"].mean()
    rmse_global = np.sqrt(merged["sq_error"].mean())

    last_available_time = merged["time"].max()
    cutoff_time = last_available_time - pd.Timedelta(hours=24)
    df_recent = merged[merged["time"] > cutoff_time]

    if not df_recent.empty:
        mae_recent = df_recent["abs_error"].mean()
    else:
        mae_recent = None

    horizon_perf = merged.groupby("step")[["abs_error"]].mean().reset_index()
    horizon_perf.rename(columns={"abs_error": "MAE"}, inplace=True)

    return {
        "MAE_Total": mae_global,
        "RMSE_Total": rmse_global,
        "MAE_Last24h": mae_recent,
        "Count": len(merged),
        "Horizon_Perf": horizon_perf,
        "Merged_Data": merged,
    }
