import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
print(f"Project root added to sys.path: {project_root}", flush=True)

from streamlit_app.utils.dataloader import load_data
from streamlit_app.utils.model_evaluation import (
    get_horizon_metrics,
    get_performance_over_time,
)

# Konfiguration
HISTORY_PATH = "data/deployment/history.parquet"
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
