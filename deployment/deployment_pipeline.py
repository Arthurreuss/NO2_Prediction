import os
import traceback
from datetime import datetime, timedelta

import mlflow
import numpy as np
import pandas as pd
import torch

from src.features.preprocess import PreProcessingPipeline
from src.models.gru import GRUForecast
from src.models.hgru import HierarchicalGRUForecast
from src.models.multi_gru import MultiGRUForecast
from src.utils import cfg
from src.utils.cfg import load_config

# new torch version needs this to allow custom classes during load
torch.serialization.add_safe_globals(
    [MultiGRUForecast, HierarchicalGRUForecast, GRUForecast]
)
# Store original loader globally to prevent recursion errors during patching
_ORIGINAL_TORCH_LOAD = torch.load


def unsafe_torch_load(*args, **kwargs):
    """Helper to force weights_only=False during model loading."""
    kwargs["weights_only"] = False
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


class DeploymentPipeline:
    def __init__(self, config_path="config_deployment.yaml"):
        self.cfg = load_config(config_path)
        self.feature_cols = self.cfg["data"]["feature_cols"]
        self.target_cols = self.cfg["data"]["target_cols"]
        self.history_file = self.cfg["deployment"]["history_path"]
        self.output_dir = self.cfg["deployment"]["predictions_dir"]
        self.models_map = self.cfg["deployment"]["models"]
        mlflow.set_tracking_uri(
            "https://dagshub.com/Arthurreuss/NO2_Forecasting.mlflow"
        )
        mlflow.set_registry_uri(
            "https://dagshub.com/Arthurreuss/NO2_Forecasting.mlflow"
        )

    def setup_time_window(self):
        """Configures the API request window (last 8 days)."""
        today = datetime.now()
        start_dt = today - timedelta(days=8)
        end_dt = today

        self.cfg["api_requests"].setdefault("time", {})
        self.cfg["api_requests"]["time"].update(
            {
                "start_date": start_dt.strftime("%Y-%m-%d"),
                "end_date": end_dt.strftime("%Y-%m-%d"),
            }
        )

    def fetch_and_process_data(self):
        """Runs the preprocessing pipeline and returns the observed data."""
        print("Fetching data...")
        pipeline = PreProcessingPipeline(cfg=self.cfg)
        df, _, _ = pipeline.preprocess(deployment=True)

        current_hour_utc = pd.Timestamp.now(tz="UTC").floor("h")
        df["time"] = pd.to_datetime(df["time"], utc=True)

        df_observed = df[df["time"] <= current_hour_utc].copy()
        return df_observed

    def update_history(self, df_observed: pd.DataFrame):
        """Updates the local parquet file with new observations."""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)

        if os.path.exists(self.history_file):
            df_history = pd.read_parquet(self.history_file)
            df_combined = pd.concat([df_history, df_observed])
            df_combined = df_combined.drop_duplicates(subset=["time"], keep="last")
            # df_combined = df_combined.sort_values("time").reset_index(drop=True)
            df_combined.to_parquet(self.history_file)
        else:
            df_observed.to_parquet(self.history_file)

    def _load_model_and_scaler(self, model_name: str, stage: str = "production"):
        """
        Loads the PyFunc model (forcing CPU) and extracts the scaler.
        Includes monkey-patching for 'weights_only=False'.
        """
        model_uri = f"models:/{model_name}@{stage}"
        print(f"Loading {model_name}...")

        # Monkey-patch torch.load
        torch.load = unsafe_torch_load

        loaded_model = mlflow.pyfunc.load_model(model_uri)
        torch.load = _ORIGINAL_TORCH_LOAD

        custom_model = loaded_model.unwrap_python_model()

        if hasattr(custom_model, "model"):
            device = torch.device("cpu")
            custom_model.model.to(device)

        if not hasattr(custom_model, "scaler"):
            raise AttributeError(f"Model {model_name} missing 'scaler'.")

        return loaded_model, custom_model.scaler

    def _scale_input(self, input_values: pd.DataFrame, scaler) -> pd.DataFrame:
        """
        Handles the complexity of scaling:
        1. Checks what cols the scaler expects (feature_names_in_)
        2. Creates dummy columns for missing features
        3. Scales and returns the updated dataframe
        """

        scaler_cols = scaler.feature_names_in_
        df_for_scaling = input_values.copy()
        missing_cols = set(scaler_cols) - set(df_for_scaling.columns)
        for c in missing_cols:
            df_for_scaling[c] = 0.0

        df_for_scaling = df_for_scaling[scaler_cols]
        df_for_scaling = df_for_scaling.astype(float)

        scaled_values = scaler.transform(df_for_scaling)

        df_for_scaling[:] = scaled_values
        common_cols = [c for c in scaler_cols if c in input_values.columns]
        input_values[common_cols] = df_for_scaling[common_cols]

        return input_values

    def _denormalize(self, prediction: np.ndarray, scaler) -> np.ndarray:
        """
        Robust denormalization.
        Handles cases where the model predicts fewer columns (e.g., 1)
        than the scaler expects (e.g., 4).
        """
        scaler_cols = list(scaler.feature_names_in_)
        n_scaler_features = len(scaler_cols)
        total_elements = prediction.size

        horizon = self.cfg["data"].get("forecast_horizon", 72)

        if total_elements == horizon:
            n_model_targets = 1
        else:
            n_model_targets = total_elements // horizon

        prediction = prediction.reshape(-1, n_model_targets)
        n_rows = prediction.shape[0]

        dummy = np.zeros((n_rows, n_scaler_features))

        actual_target_names = self.target_cols[:n_model_targets]

        scaler_target_indices = []
        for t in actual_target_names:
            try:
                idx = scaler_cols.index(t)
                scaler_target_indices.append(idx)
            except ValueError:
                print(f"Warning: Target '{t}' not found in scaler features.")
                continue

        if len(scaler_target_indices) == prediction.shape[1]:
            dummy[:, scaler_target_indices] = prediction
        else:
            print(
                f"Dimensionality mismatch in denormalizer. Skipping {n_model_targets} targets."
            )
            return prediction

        rescaled_dummy = scaler.inverse_transform(dummy)

        rescaled_prediction = rescaled_dummy[:, scaler_target_indices]
        return rescaled_prediction

    def run_inference(self, df_input_raw: pd.DataFrame) -> dict:
        """Iterates over all defined models and generates forecasts."""
        forecasts = {}

        for name, model_name in self.models_map.items():
            print(f"Processing {name}...")
            try:
                # 1. Load Model & Scaler
                model, scaler = self._load_model_and_scaler(model_name)

                # 2. Scale
                input_df = df_input_raw[self.feature_cols].copy()
                input_df = self._scale_input(input_df, scaler)

                # 3. Predict
                raw_preds = model.predict(input_df)

                # 4. Denormalize
                preds_real = self._denormalize(raw_preds, scaler)

                # 5. Format Output
                num_output_cols = preds_real.shape[1]
                current_targets = self.target_cols[:num_output_cols]

                forecasts[name] = {
                    col_name: preds_real[:, i].tolist()
                    for i, col_name in enumerate(current_targets)
                }

                print(
                    f"{name}: Generated {len(preds_real)} predictions for {current_targets}"
                )

            except Exception as e:
                print(f"Error on {name}: {e}")
                traceback.print_exc()
                forecasts[name] = "Error"

        return forecasts

    def save_results(self, df_input_raw: pd.DataFrame, forecasts: dict):
        """
        Saves forecasts to individual parquet files per model.
        Updates existing files by overwriting overlapping timestamps with the latest prediction.
        """
        last_observed_time = df_input_raw["time"].iloc[-1]

        pred_len = self.cfg["data"]["forecast_horizon"]

        future_times = pd.date_range(
            start=last_observed_time + pd.Timedelta(hours=1), periods=pred_len, freq="h"
        )

        os.makedirs(self.output_dir, exist_ok=True)

        for model_name, new_preds in forecasts.items():
            df_new_preds = pd.DataFrame(new_preds)
            df_new_preds["time"] = future_times

            cols = ["time"] + [c for c in df_new_preds.columns if c != "time"]
            df_new_preds = df_new_preds[cols]

            safe_model_name = model_name.replace(" ", "_")
            model_file_path = os.path.join(
                self.output_dir, f"{safe_model_name}_predictions.parquet"
            )

            if os.path.exists(model_file_path):

                df_history = pd.read_parquet(model_file_path)
                df_combined = pd.concat([df_history, df_new_preds], ignore_index=True)
            else:
                df_combined = df_new_preds

            df_combined.to_parquet(model_file_path, index=False)

    def run(self):
        self.setup_time_window()
        df_observed = self.fetch_and_process_data()
        self.update_history(df_observed)

        df_model_input = df_observed.tail(self.cfg["data"]["input_length"]).reset_index(
            drop=True
        )
        forecasts = self.run_inference(df_model_input)
        self.save_results(df_model_input, forecasts)


if __name__ == "__main__":
    pipeline = DeploymentPipeline()
    pipeline.run()
