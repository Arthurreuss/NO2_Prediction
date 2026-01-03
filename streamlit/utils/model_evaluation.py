import numpy as np
import pandas as pd


def calculate_smape(y_true, y_pred):
    """Calculates Symmetric Mean Absolute Percentage Error (0-100%)."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * diff


def get_horizon_metrics(df_history, preds_sequence):
    """
    Aggregates metrics by forecast step (1-72).
    """
    # 1. Prepare Prediction Data
    if isinstance(preds_sequence, list):
        if not preds_sequence:
            return None, None
        df_pred_all = pd.concat(preds_sequence)
    else:
        df_pred_all = preds_sequence.copy()

    if df_pred_all.empty:
        return None, None

    # --- FIX 1: Strict UTC & Step Calculation ---
    # Konvertiere alles strikt nach UTC
    df_pred_all["time"] = pd.to_datetime(df_pred_all["time"], utc=True)

    if "prediction_generated_at" in df_pred_all.columns:
        df_pred_all["prediction_generated_at"] = pd.to_datetime(
            df_pred_all["prediction_generated_at"], utc=True
        )

        # Calculate difference in hours
        diff = df_pred_all["time"] - df_pred_all["prediction_generated_at"]
        df_pred_all["step"] = diff.dt.total_seconds() / 3600

        # Round to nearest integer
        df_pred_all["step"] = df_pred_all["step"].round().astype(int)
    else:
        df_pred_all["step"] = 1

    # --- FIX 2: Filter invalid steps (Negative or > 72) ---
    # Das behebt das Problem mit negativen Einträgen auf der X-Achse
    df_pred_all = df_pred_all[(df_pred_all["step"] > 0) & (df_pred_all["step"] <= 72)]

    if df_pred_all.empty:
        return None, None

    # 2. Merge with Ground Truth
    df_history = df_history.copy()
    df_history["time"] = pd.to_datetime(df_history["time"], utc=True)

    merged = pd.merge(
        df_pred_all,
        df_history[["time", "nitrogen_dioxide"]],
        on="time",
        how="inner",
        suffixes=("_pred", "_actual"),
    )

    if merged.empty:
        return None, None

    # 3. Calculate Pointwise Errors
    pred_col = (
        "nitrogen_dioxide_pred"
        if "nitrogen_dioxide_pred" in merged.columns
        else "nitrogen_dioxide"
    )
    actual_col = (
        "nitrogen_dioxide_actual"
        if "nitrogen_dioxide_actual" in merged.columns
        else "nitrogen_dioxide"
    )

    merged["sq_error"] = (merged[pred_col] - merged[actual_col]) ** 2
    merged["smape"] = calculate_smape(merged[actual_col], merged[pred_col])

    # 4. Group by Horizon Step
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

    # Approx Confidence Intervals
    horizon_stats["RMSE_upper"] = np.sqrt(
        horizon_stats["MSE_mean"] + horizon_stats["MSE_std"].fillna(0)
    )
    horizon_stats["RMSE_lower"] = np.sqrt(
        (horizon_stats["MSE_mean"] - horizon_stats["MSE_std"].fillna(0)).clip(lower=0)
    )

    return horizon_stats, merged


def get_performance_over_time(df_history, preds_all):
    """
    Calculates aggregated metrics (RMSE, SMAPE) for each COMPLETE forecast run.
    Aggregates the 72h horizon into single scalar values per run.
    """
    results = []

    # Ensure history is UTC
    df_history = df_history.copy()
    df_history["time"] = pd.to_datetime(df_history["time"], utc=True)

    # Process each model
    for model_name, df_pred in preds_all.items():
        if df_pred.empty:
            continue

        # Ensure pred is UTC
        df_pred = df_pred.copy()
        df_pred["time"] = pd.to_datetime(df_pred["time"], utc=True)
        if "prediction_generated_at" in df_pred.columns:
            df_pred["prediction_generated_at"] = pd.to_datetime(
                df_pred["prediction_generated_at"], utc=True
            )
        else:
            continue

        # Merge with history
        merged = pd.merge(
            df_pred,
            df_history[["time", "nitrogen_dioxide"]],
            on="time",
            how="inner",
            suffixes=("_pred", "_actual"),
        )

        if merged.empty:
            continue

        # Calculate Errors
        pred_col = (
            "nitrogen_dioxide_pred"
            if "nitrogen_dioxide_pred" in merged.columns
            else "nitrogen_dioxide"
        )
        actual_col = (
            "nitrogen_dioxide_actual"
            if "nitrogen_dioxide_actual" in merged.columns
            else "nitrogen_dioxide"
        )

        merged["sq_error"] = (merged[pred_col] - merged[actual_col]) ** 2
        merged["smape"] = calculate_smape(merged[actual_col], merged[pred_col])

        # Group by Run (Prediction Generated At)
        grouped = merged.groupby("prediction_generated_at")

        for gen_time, group in grouped:
            # Check completeness: Only include runs that have nearly full 72h validation data
            # (Allows small missing data gaps, e.g. >= 65 steps available)
            if len(group) >= 65:
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

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("prediction_generated_at")


def evaluate_model_performance(df_history, preds_sequence):
    """
    Evaluates a sequence of forecast data against history.
    """
    # Reuse simple logic or just pass through for legacy calls
    # (Simplified for brevity as get_horizon_metrics does the heavy lifting now)
    return None
