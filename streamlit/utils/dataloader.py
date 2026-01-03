import glob
import os

import pandas as pd

import streamlit as st


@st.cache_data(ttl=300)
def load_data(history_path, predictions_dir):
    if os.path.exists(history_path):
        df_history = pd.read_parquet(history_path)
        df_history["time"] = df_history["time"].dt.tz_convert("Europe/Amsterdam")
    else:
        raise FileNotFoundError(f"History file not found: {history_path}")

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
                    df[col] = df[col].dt.tz_convert("Europe/Amsterdam")

            # temporary delete later
            if "prediction_generated_at" in df.columns:
                mask_30_12 = (df["prediction_generated_at"].dt.month == 12) & (
                    df["prediction_generated_at"].dt.day == 30
                )
                df = df[~mask_30_12]
                mask_31_12_04 = (
                    (df["prediction_generated_at"].dt.month == 12)
                    & (df["prediction_generated_at"].dt.day == 31)
                    & (df["prediction_generated_at"].dt.hour < 7)
                )
                df = df[~mask_31_12_04]
                mask_0_0 = df["prediction_generated_at"] == df["time"]
                df = df[~mask_0_0]
                df = df.drop_duplicates(
                    subset=["prediction_generated_at", "time"], keep="first"
                )
            ######

            if not df.empty:
                latest_gen_time = df["prediction_generated_at"].max()
                latest_batch = df[df["prediction_generated_at"] == latest_gen_time]
                if len(latest_batch) != 72:
                    print(
                        f"ALERT: Latest forecast for {model_name} is incorrect! Expected 72, got {len(latest_batch)}."
                    )

            df = df.sort_values(
                by=["prediction_generated_at", "time"], ascending=[True, True]
            )

            preds_all[model_name] = df.copy()
            df_stitched = df.drop_duplicates(subset=["time"], keep="last")
            preds[model_name] = df_stitched.sort_values("time")

        except Exception as e:
            print(f"Error loading {f}: {e}")

    return df_history, preds, preds_all
