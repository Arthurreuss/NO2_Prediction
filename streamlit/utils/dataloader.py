import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

import streamlit as st


@st.cache_data(ttl=300)
def load_data(history_path, predictions_dir):
    if os.path.exists(history_path):
        df_history = pd.read_parquet(history_path)
        if df_history["time"].dt.tz is None:
            df_history["time"] = df_history["time"].dt.tz_localize("Europe/Amsterdam")
        else:
            df_history["time"] = df_history["time"].dt.tz_convert("Europe/Amsterdam")
    else:
        df_history = pd.DataFrame(columns=["time", "nitrogen_dioxide"])

    preds = {}
    preds_all = {}
    pred_files = glob.glob(os.path.join(predictions_dir, "*_predictions.parquet"))
    for f in pred_files:
        model_name = (
            os.path.basename(f).replace("_predictions.parquet", "").replace("_", " ")
        )
        try:
            df = pd.read_parquet(f)

            for col in ["time", "prediction_generated_at"]:
                if col in df.columns:
                    if df[col].dt.tz is None:
                        df[col] = df[col].dt.tz_localize("Europe/Amsterdam")
                    else:
                        df[col] = df[col].dt.tz_convert("Europe/Amsterdam")

            latest_gen_time = df["prediction_generated_at"].max()

            latest_batch = df[df["prediction_generated_at"] == latest_gen_time]

            if len(latest_batch) < 72:
                print(
                    f"ALERT: Latest forecast for {model_name} is incomplete! Expected 72, got {len(latest_batch)}."
                )

            df = df.sort_values("prediction_generated_at", ascending=True)
            preds_all[model_name] = df.sort_values("time")
            df_stitched = df.drop_duplicates(subset=["time"], keep="last")
            preds[model_name] = df_stitched.sort_values("time")

        except Exception as e:
            print(f"Error loading {f}: {e}")

    return df_history, preds, preds_all


def compute_metrics(df_history, df_pred, target_col="nitrogen_dioxide"):
    merged = pd.merge(
        df_history[["time", target_col]],
        df_pred[["time", target_col]],
        on="time",
        how="inner",
        suffixes=("_actual", "_pred"),
    )

    if len(merged) == 0:
        return {"MAE": 0.0, "RMSE": 0.0, "Count": 0}

    y_true = merged[f"{target_col}_actual"]
    y_pred = merged[f"{target_col}_pred"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {"MAE": mae, "RMSE": rmse, "Count": len(merged)}
