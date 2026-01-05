import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
print(f"Project root added to sys.path: {project_root}", flush=True)

import numpy as np
import pandas as pd

from streamlit_app.utils.dataloader import load_data


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


# Konfiguration
HISTORY_PATH = "data/deployment/processed/continuous_history.parquet"
PREDS_DIR = "data/deployment/predictions"
METRICS_DIR = "data/deployment/metrics"
POLLUTANTS = ["nitrogen_dioxide", "ozone", "pm10", "pm2_5"]


def custom_serializer(obj):
    """Hilft JSON beim Speichern von Numpy/Pandas Typen"""
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def ensure_serializable(data):
    """Konvertiert rekursiv alle Daten in JSON-kompatible Formate."""
    if isinstance(data, dict):
        return {k: ensure_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [ensure_serializable(v) for v in data]
    elif isinstance(data, pd.DataFrame):
        return data.to_dict(orient="records")
    elif isinstance(data, (pd.Timestamp, np.datetime64)):
        return str(data)
    elif isinstance(data, (np.integer, int)):
        return int(data)
    elif isinstance(data, (np.floating, float)):
        return float(data)
    return data


def main():
    print("🚀 Starting Metrics Computation Job...")
    os.makedirs(METRICS_DIR, exist_ok=True)

    # 1. Daten laden (Hier ist RAM egal, der Runner hat genug)
    try:
        df_history, _, preds_all = load_data(
            HISTORY_PATH, PREDS_DIR, load_full_history=True
        )
    except FileNotFoundError:
        print("⚠️ No data found. Exiting.")
        return

    # Container für die Ergebnisse
    horizon_metrics = {}
    history_metrics = {}

    # 2. Berechnungen durchführen
    for pollutant in POLLUTANTS:
        print(f"   Processing {pollutant}...")
        horizon_metrics[pollutant] = {}

        # Performance Over Time (Verlauf der Genauigkeit)
        # Hier berechnen wir den Verlauf neu. Da 'preds_all' modular wächst,
        # wächst auch dieses Ergebnis modular mit.
        df_perf_history = get_performance_over_time(
            df_history, preds_all, pollutant=pollutant
        )

        # In Liste von Dicts umwandeln für JSON
        if not df_perf_history.empty:
            # Timestamps für JSON serialisierbar machen
            df_perf_history["prediction_generated_at"] = df_perf_history[
                "prediction_generated_at"
            ].astype(str)
            history_metrics[pollutant] = df_perf_history.to_dict(orient="records")
        else:
            history_metrics[pollutant] = []

        # Horizon Metrics (Fehler pro Stunde 1-72)
        for model_name, df_preds in preds_all.items():
            if pollutant != "nitrogen_dioxide" and model_name == "GRU":
                continue

            stats, _ = get_horizon_metrics(df_history, df_preds, pollutant)

            if stats is not None:
                # Wichtig: NaN Werte durch None/null ersetzen für valides JSON
                stats = stats.where(pd.notnull(stats), None)
                horizon_metrics[pollutant][model_name] = stats.to_dict(orient="records")

    # 3. Ergebnisse speichern (JSON)
    # Wir überschreiben die Dateien jedes Mal komplett mit dem neuesten Stand.
    # Da 'preds_all' die Historie enthält, ist das "modular" genug.

    print("💾 Saving metrics to disk...")

    with open(f"{METRICS_DIR}/horizon_metrics.json", "w") as f:
        json.dump(ensure_serializable(horizon_metrics), f, default=custom_serializer)

    with open(f"{METRICS_DIR}/history_metrics.json", "w") as f:
        json.dump(ensure_serializable(history_metrics), f, default=custom_serializer)

    print("✅ Job finished successfully.")


if __name__ == "__main__":
    main()
