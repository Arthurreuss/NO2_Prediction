import glob
import json
import os

import pandas as pd
import streamlit as st


@st.cache_data(ttl=3600)
def load_data(history_path: str, predictions_dir: str, metrics_dir: str) -> tuple:
    """
    Loads all data required for the app:
    1. Historical Observations (Parquet)
    2. Stitched Forecasts (Parquet -> Dict)
    3. Pre-computed Metrics (JSON -> Dict)
    """
    # --- 1. Load History ---
    if os.path.exists(history_path):
        df_history = pd.read_parquet(history_path)
        # Convert to UTC immediately to avoid PyArrow/Streamlit timezone issues
        df_history["time"] = pd.to_datetime(df_history["time"], utc=True)
    else:
        df_history = pd.DataFrame()

    # --- 2. Load Predictions (Stitched Only) ---
    preds = {}
    pred_files = glob.glob(os.path.join(predictions_dir, "*_predictions.parquet"))

    for f in pred_files:
        model_name = (
            os.path.basename(f).replace("_predictions.parquet", "").replace("_", " ")
        )
        try:
            df = pd.read_parquet(f)

            # Standardize Timestamps
            for col in ["time", "prediction_generated_at"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], utc=True)

            if not df.empty:
                # Stitching: Sort by generation time, keep last (newest) for each target time
                df = df.sort_values("prediction_generated_at")
                df_stitched = df.drop_duplicates(
                    subset=["time"], keep="last"
                ).sort_values("time")
                preds[model_name] = df_stitched

        except Exception as e:
            print(f"Error loading {model_name}: {e}")

    # --- 3. Load Metrics (JSON) ---
    horizon_metrics = {}
    history_metrics = {}

    h_path = os.path.join(metrics_dir, "horizon_metrics.json")
    p_path = os.path.join(metrics_dir, "history_metrics.json")

    if os.path.exists(h_path):
        with open(h_path, "r") as f:
            horizon_metrics = json.load(f)

    if os.path.exists(p_path):
        with open(p_path, "r") as f:
            history_metrics = json.load(f)

    return df_history, preds, horizon_metrics, history_metrics
