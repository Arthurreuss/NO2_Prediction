import os
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple

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


def unsafe_torch_load(*args: Any, **kwargs: Any) -> Any:
    """Force `weights_only=False` during torch model loading.

    This helper wraps the original `torch.load` and injects
    `weights_only=False` into the keyword arguments.

    Args:
        *args: Positional arguments forwarded to `torch.load`.
        **kwargs: Keyword arguments forwarded to `torch.load`.

    Returns:
        The deserialized object returned by the original `torch.load`.
    """
    kwargs["weights_only"] = False
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


class DeploymentPipeline:
    """End-to-end deployment pipeline for fetching data, forecasting, and saving results.

    The pipeline:
      - configures an API request window
      - fetches and preprocesses recent observations
      - updates a local history store
      - loads registered MLflow models and associated scalers
      - scales inputs, runs inference, denormalizes outputs
      - persists predictions to parquet files
    """

    def __init__(self, config_path: str = "config_deployment.yaml") -> None:
        """Initialize the deployment pipeline from a configuration file.

        Args:
            config_path: Path to the deployment configuration YAML.
        """
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

    def setup_time_window(self) -> None:
        """Configure the API request window (last 8 days).

        Updates `self.cfg["api_requests"]["time"]` with `start_date` and `end_date`
        formatted as YYYY-MM-DD.
        """
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

    def fetch_and_process_data(self) -> pd.DataFrame:
        """Run preprocessing and return the observed portion of the dataset.

        This method:
          - runs the preprocessing pipeline
          - converts time to UTC
          - filters out any rows beyond the current UTC hour

        Returns:
            A DataFrame containing observed data up to the current UTC hour.
        """
        print("Fetching data...")
        pipeline = PreProcessingPipeline(cfg=self.cfg)
        df, _, _ = pipeline.preprocess(deployment=True)

        current_hour_utc = pd.Timestamp.now(tz="UTC").floor("h")
        df["time"] = pd.to_datetime(df["time"], utc=True)

        df_observed = df[df["time"] <= current_hour_utc].copy()
        return df_observed

    def update_history(self, df_observed: pd.DataFrame) -> None:
        """Update the local history parquet file with new observations.

        If the history file exists, new observations are appended and duplicates
        are dropped based on the "time" column, keeping the latest occurrence.

        Args:
            df_observed: DataFrame of newly observed rows to add to history.
        """
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)

        if os.path.exists(self.history_file):
            df_history = pd.read_parquet(self.history_file)
            df_combined = pd.concat([df_history, df_observed])
            df_combined = df_combined.drop_duplicates(subset=["time"], keep="last")
            # df_combined = df_combined.sort_values("time").reset_index(drop=True)
            df_combined.to_parquet(self.history_file)
        else:
            df_observed.to_parquet(self.history_file)

    def _load_model_and_scaler(
        self, model_name: str, stage: str = "production"
    ) -> Tuple[Any, Any]:
        """Load an MLflow PyFunc model and extract its scaler.

        This method:
          - loads the model from the MLflow registry (forcing CPU)
          - temporarily monkey-patches `torch.load` to force `weights_only=False`
          - unwraps the underlying Python model to access `model` and `scaler`
          - moves the underlying torch model to CPU if present

        Args:
            model_name: Registered model name in MLflow.
            stage: Model stage or alias to load (e.g., "production").

        Returns:
            A tuple `(loaded_model, scaler)` where:
              - loaded_model is the MLflow PyFunc model object
              - scaler is the extracted scaler attached to the unwrapped model

        Raises:
            AttributeError: If the unwrapped model does not provide a `scaler`.
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

    def _scale_input(self, input_values: pd.DataFrame, scaler: Any) -> pd.DataFrame:
        """Scale input features using the provided scaler.

        This method:
          1) reads the scaler's expected column ordering from `feature_names_in_`
          2) creates dummy columns with zeros for any missing expected features
          3) transforms values and writes scaled values back into the original
             `input_values` for columns that overlap

        Args:
            input_values: DataFrame containing input feature columns to scale.
            scaler: Fitted scaler object with `feature_names_in_` and `transform`.

        Returns:
            The updated `input_values` DataFrame with scaled values for the
            overlapping columns.
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

    def _denormalize(self, prediction: np.ndarray, scaler: Any) -> np.ndarray:
        """Denormalize model predictions into original units.

        This method supports cases where the model outputs fewer target columns
        than the scaler expects. It reconstructs a dummy feature matrix sized
        to the scaler's expected features and inserts the predicted target
        columns into their respective indices before applying `inverse_transform`.

        Args:
            prediction: Array of model predictions. The total number of elements
                determines whether the output is interpreted as single-target
                (size == horizon) or multi-target (size divisible by horizon).
            scaler: Fitted scaler object with `feature_names_in_` and
                `inverse_transform`.

        Returns:
            A denormalized numpy array containing rescaled predictions for the
            predicted target columns.
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

    def run_inference(self, df_input_raw: pd.DataFrame) -> Dict[str, Any]:
        """Run inference for all configured models and return forecasts.

        For each model in `self.models_map`, this method:
          1) loads the model and scaler
          2) scales the input feature DataFrame
          3) generates predictions via the MLflow PyFunc model
          4) denormalizes predictions to real units
          5) formats output as a dict mapping target names to lists of values

        Args:
            df_input_raw: Input DataFrame containing at least the configured
                feature columns.

        Returns:
            A dictionary mapping model display names to either:
              - a dict of {target_name: list_of_predictions}, or
              - the string "Error" if inference failed for that model.
        """
        forecasts: Dict[str, Any] = {}

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

    def save_results(
        self, df_input_raw: pd.DataFrame, forecasts: Dict[str, Any]
    ) -> None:
        """Save forecasts to parquet files per model.

        Forecasts are saved to individual parquet files under `self.output_dir`.
        If a model's prediction file already exists, the new predictions are
        appended (with overlap handling described in the docstring).

        Args:
            df_input_raw: DataFrame containing the latest observed time in its "time" column.
            forecasts: Dictionary of forecasts produced by `run_inference`.
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

    def run(self) -> None:
        """Execute the full deployment workflow."""
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
