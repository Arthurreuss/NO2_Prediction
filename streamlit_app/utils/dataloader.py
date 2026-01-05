import glob
import json
import os
from typing import Any, Dict, Tuple

import pandas as pd
import streamlit as st


@st.cache_data(ttl=3600)
def load_data(
    history_path: str, predictions_dir: str, metrics_dir: str
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Any], Dict[str, Any]]:
    """Loads all data required for the app, including history, forecasts, and metrics.

    Args:
        history_path (str): The file path to the historical observations Parquet file.
        predictions_dir (str): The directory path containing prediction Parquet files.
        metrics_dir (str): The directory path containing pre-computed metrics JSON files.

    Returns:
        Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - df_history: DataFrame of historical observations with UTC timestamps.
            - preds: Dictionary mapping model names to stitched forecast DataFrames.
            - horizon_metrics: Dictionary of metrics per forecast horizon.
            - history_metrics: Dictionary of metrics based on historical performance.
    """
    if os.path.exists(history_path):
        df_history = pd.read_parquet(history_path)
        df_history["time"] = pd.to_datetime(df_history["time"], utc=True)
    else:
        df_history = pd.DataFrame()

    preds = {}
    pred_files = glob.glob(os.path.join(predictions_dir, "*_predictions.parquet"))

    for f in pred_files:
        model_name = (
            os.path.basename(f).replace("_predictions.parquet", "").replace("_", " ")
        )
        try:
            df = pd.read_parquet(f)

            for col in ["time", "prediction_generated_at"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], utc=True)

            if not df.empty:
                df = df.sort_values("prediction_generated_at")
                df_stitched = df.drop_duplicates(
                    subset=["time"], keep="last"
                ).sort_values("time")
                preds[model_name] = df_stitched

        except Exception as e:
            print(f"Error loading {model_name}: {e}")

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
