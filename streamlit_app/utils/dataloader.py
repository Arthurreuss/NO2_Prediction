import glob
import os

import pandas as pd
import streamlit as st


@st.cache_data(ttl=3600, max_entries=1)
def load_data(
    history_path: str, predictions_dir: str
) -> tuple[pd.DataFrame, dict, dict]:
    """Loads and preprocesses historical and prediction data from disk.

    Reads the history Parquet file and converts timestamps to the 'Europe/Amsterdam'
    timezone. Iterates through prediction files in the specified directory,
    processes timestamps, validates batch sizes, and organizes them into raw
    and stitched (latest prediction per timestamp) dictionaries.

    Args:
        history_path: File path to the historical data Parquet file.
        predictions_dir: Directory path containing prediction Parquet files
            (matching the pattern *_predictions.parquet).

    Returns:
        A tuple containing three elements:
            1. The historical data DataFrame.
            2. A dictionary mapping model names to stitched (most recent) prediction DataFrames.
            3. A dictionary mapping model names to all raw prediction DataFrames.

    Raises:
        FileNotFoundError: If the history_path does not exist.
    """
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
