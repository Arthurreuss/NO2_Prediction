import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Configuration
HISTORY_PATH = "data/deployment/processed/continuous_history.parquet"
PREDS_DIR = "data/deployment/predictions"
METRICS_DIR = "data/deployment/metrics"
POLLUTANTS = ["nitrogen_dioxide", "ozone", "pm10", "pm2_5"]


def load_data():
    """Loads history and all raw predictions."""
    if not os.path.exists(HISTORY_PATH):
        raise FileNotFoundError(f"History not found: {HISTORY_PATH}")

    # Load History
    df_hist = pd.read_parquet(HISTORY_PATH)
    df_hist["time"] = pd.to_datetime(df_hist["time"], utc=True)

    # Load Predictions
    preds_all = {}
    for f in glob.glob(os.path.join(PREDS_DIR, "*_predictions.parquet")):
        model = (
            os.path.basename(f).replace("_predictions.parquet", "").replace("_", " ")
        )
        try:
            df = pd.read_parquet(f)
            # Standardize Time
            for col in ["time", "prediction_generated_at"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], utc=True)

            if not df.empty:
                preds_all[model] = df.sort_values(["prediction_generated_at", "time"])
        except Exception as e:
            print(f"Error loading {model}: {e}")

    return df_hist, preds_all


def calculate_smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    return np.where(denominator == 0, 0.0, 100 * diff)


def process_metrics(df_hist, preds_all):
    horizon_out = {p: {} for p in POLLUTANTS}
    history_out = {p: [] for p in POLLUTANTS}

    for pollutant in POLLUTANTS:
        print(f"Processing {pollutant}...")

        for model_name, df_pred in preds_all.items():
            # Business Logic: Skip GRU for non-NO2
            if pollutant != "nitrogen_dioxide" and model_name == "GRU":
                continue

            # Merge Data (Inner Join matches timestamps)
            merged = pd.merge(
                df_pred, df_hist[["time", pollutant]], on="time", how="inner"
            ).rename(columns={pollutant: "actual"})

            if merged.empty:
                continue

            # Calc Errors
            merged["sq_error"] = (merged[f"{pollutant}_pred"] - merged["actual"]) ** 2
            merged["smape"] = calculate_smape(
                merged["actual"], merged[f"{pollutant}_pred"]
            )

            # --- 1. Horizon Metrics (Aggregated by step 1-72) ---
            merged["step"] = (
                (
                    (
                        merged["time"] - merged["prediction_generated_at"]
                    ).dt.total_seconds()
                    / 3600
                )
                .round()
                .astype(int)
            )

            stats = (
                merged.groupby("step")
                .agg(
                    MSE_mean=("sq_error", "mean"),
                    MSE_std=("sq_error", "std"),
                    SMAPE_mean=("smape", "mean"),
                )
                .reset_index()
            )

            # Calc RMSE stats
            stats["RMSE_mean"] = np.sqrt(stats["MSE_mean"])
            stats["RMSE_upper"] = np.sqrt(
                stats["MSE_mean"] + stats["MSE_std"].fillna(0)
            )
            stats["RMSE_lower"] = np.sqrt(
                (stats["MSE_mean"] - stats["MSE_std"].fillna(0)).clip(lower=0)
            )

            # Clean for JSON (Fill NaNs)
            horizon_out[pollutant][model_name] = stats.where(
                pd.notnull(stats), None
            ).to_dict(orient="records")

            # --- 2. History Metrics (Aggregated by generation time) ---
            # Filter for complete forecasts (72 hours)
            valid_runs = merged.groupby("prediction_generated_at").filter(
                lambda x: len(x) == 72
            )

            if not valid_runs.empty:
                perf = (
                    valid_runs.groupby("prediction_generated_at")
                    .agg(
                        RMSE=("sq_error", lambda x: np.sqrt(x.mean())),
                        SMAPE=("smape", "mean"),
                        Count=("sq_error", "count"),
                    )
                    .reset_index()
                )

                perf["Model"] = model_name
                # Convert timestamp to string for JSON
                perf["prediction_generated_at"] = perf[
                    "prediction_generated_at"
                ].astype(str)
                history_out[pollutant].extend(perf.to_dict(orient="records"))

        # Sort history by time after combining models
        history_out[pollutant].sort(key=lambda x: x["prediction_generated_at"])

    return horizon_out, history_out


def main():
    print("🚀 Starting Metrics Computation...")
    os.makedirs(METRICS_DIR, exist_ok=True)

    try:
        df_history, preds_all = load_data()
    except FileNotFoundError:
        print("⚠️ History file missing.")
        return

    horizon_metrics, history_metrics = process_metrics(df_history, preds_all)

    print("💾 Saving metrics...")
    with open(f"{METRICS_DIR}/horizon_metrics.json", "w") as f:
        json.dump(horizon_metrics, f)  # Standard dump works now

    with open(f"{METRICS_DIR}/history_metrics.json", "w") as f:
        json.dump(history_metrics, f)

    print("✅ Done.")


if __name__ == "__main__":
    main()
