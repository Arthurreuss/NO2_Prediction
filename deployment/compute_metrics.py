import glob
import json
import os

import numpy as np
import pandas as pd

from src.utils.cfg import load_config


def load_data(cfg):
    """Loads history and all raw predictions."""
    history_path = cfg["deployment"]["history_path"]
    preds_dir = cfg["deployment"]["predictions_dir"]
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History not found: {history_path}")

    df_hist = pd.read_parquet(history_path)
    df_hist["time"] = pd.to_datetime(df_hist["time"], utc=True)

    preds_all = {}
    for f in glob.glob(os.path.join(preds_dir, "*_predictions.parquet")):
        model = (
            os.path.basename(f).replace("_predictions.parquet", "").replace("_", " ")
        )
        try:
            df = pd.read_parquet(f)
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


def process_metrics(df_hist, preds_all, cfg):
    pollutants = cfg["data"]["target_cols"]
    horizon_out = {p: {} for p in pollutants}
    history_out = {p: [] for p in pollutants}

    for pollutant in pollutants:
        print(f"Processing {pollutant}...")

        for model_name, df_pred in preds_all.items():
            if pollutant != "nitrogen_dioxide" and model_name == "GRU":
                continue

            hist_slice = df_hist[["time", pollutant]].rename(
                columns={pollutant: "actual"}
            )

            merged = pd.merge(df_pred, hist_slice, on="time", how="inner")

            if merged.empty:
                continue

            merged["sq_error"] = (merged[pollutant] - merged["actual"]) ** 2
            merged["smape"] = calculate_smape(merged["actual"], merged[pollutant])

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

            stats["RMSE_mean"] = np.sqrt(stats["MSE_mean"])
            stats["RMSE_upper"] = np.sqrt(
                stats["MSE_mean"] + stats["MSE_std"].fillna(0)
            )
            stats["RMSE_lower"] = np.sqrt(
                (stats["MSE_mean"] - stats["MSE_std"].fillna(0)).clip(lower=0)
            )

            horizon_out[pollutant][model_name] = stats.where(
                pd.notnull(stats), None
            ).to_dict(orient="records")

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
                perf["prediction_generated_at"] = perf[
                    "prediction_generated_at"
                ].astype(str)
                history_out[pollutant].extend(perf.to_dict(orient="records"))

        if history_out[pollutant]:
            history_out[pollutant].sort(key=lambda x: x["prediction_generated_at"])

    return horizon_out, history_out


def main():
    cfg = load_config("configs/config_deployment.yaml")
    metrics_dir = cfg["deployment"]["metrics_dir"]

    print("Starting Metrics Computation...")
    os.makedirs(metrics_dir, exist_ok=True)

    try:
        df_history, preds_all = load_data()
    except FileNotFoundError:
        print("⚠️ History file missing.")
        return

    horizon_metrics, history_metrics = process_metrics(df_history, preds_all, cfg)

    print("Saving metrics...")
    with open(f"{metrics_dir}/horizon_metrics.json", "w") as f:
        json.dump(horizon_metrics, f)

    with open(f"{metrics_dir}/history_metrics.json", "w") as f:
        json.dump(history_metrics, f)

    print("Done.")


if __name__ == "__main__":
    main()
