import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.data.api_requests import fetch_air_quality_data, fetch_weather_data
from src.features.schema import run_full_integrity_check


class PreProcessingPipeline:
    """End-to-end preprocessing pipeline for weather + air-quality time series.

    This pipeline:
      1) Fetches raw weather and air-quality data via API helpers.
      2) Joins the datasets per location and concatenates them into one DataFrame.
      3) Creates cyclic (sin/cos) time features (and wind direction, if available).
      4) Runs an integrity check on the resulting DataFrame.
      5) Splits into train/val/test by calendar year.
      6) Normalizes numeric columns using training statistics only.
      7) Saves the resulting splits and the fitted scaler.

    Attributes:
        cfg: Configuration dictionary.
        dfs: Per-location joined DataFrames (collected before concatenation).
        df: The concatenated full DataFrame across locations.
        train_df: Training split DataFrame.
        val_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        norm_method: Normalization method: "standard" or "minmax".
        scaler: Fitted scaler instance used to normalize numeric columns.
    """

    def __init__(self, cfg: Dict[str, Any], norm_method: str = "standard") -> None:
        """Initialize the preprocessing pipeline.

        Args:
            cfg: Configuration dictionary.
            norm_method: Normalization method to use:
                - "standard" for sklearn.preprocessing.StandardScaler
                - "minmax" for sklearn.preprocessing.MinMaxScaler
        """
        self.cfg = cfg
        self.dfs: List[pd.DataFrame] = []
        self.df: Optional[pd.DataFrame] = None

        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

        self.norm_method = norm_method
        self.scaler: Optional[object] = None

    def _fetch_data(self) -> None:
        """Fetch raw weather and air-quality data via the API helper functions."""
        fetch_weather_data(self.cfg)
        fetch_air_quality_data(self.cfg)

    def _join_data(self) -> None:
        """Join weather and air-quality data per location and concatenate.

        Reads per-location CSVs written by the API fetch functions, merges them
        on ["time", "location"], sorts chronologically, and concatenates all
        locations into a single DataFrame stored in `self.df`.
        """
        start_date = self.cfg["api_requests"]["time"]["start_date"]
        end_date = self.cfg["api_requests"]["time"]["end_date"]
        raw_dir = self.cfg["api_requests"]["raw_dir"]

        for loc_name in self.cfg["api_requests"]["locations"].keys():
            aq = pd.read_csv(
                f"{raw_dir}/air_quality/{loc_name}_air_quality_{start_date}_to_{end_date}.csv",
                parse_dates=["time"],
            )
            wx = pd.read_csv(
                f"{raw_dir}/weather/{loc_name}_weather_{start_date}_to_{end_date}.csv",
                parse_dates=["time"],
            )

            df_loc = pd.merge(wx, aq, on=["time", "location"], how="inner")
            df_loc = df_loc.sort_values("time").reset_index(drop=True)
            self.dfs.append(df_loc)

        self.df = (
            pd.concat(self.dfs, ignore_index=True)
            .sort_values("time")
            .reset_index(drop=True)
        )

    def _encode_cyclic_feature(self, name: str, col: str, max_val: int) -> None:
        """Encode a cyclical feature using sin/cos.

        Adds two columns to `self.df`:
          - f"{name}_sin"
          - f"{name}_cos"

        Args:
            name: Base name for the encoded feature (used as prefix).
            col: Column name in `self.df` containing integer-like values.
            max_val: Period of the cycle (e.g., 24 for hour, 7 for dayofweek).
        """
        self.df[f"{name}_sin"] = np.sin(2 * np.pi * self.df[col] / max_val)
        self.df[f"{name}_cos"] = np.cos(2 * np.pi * self.df[col] / max_val)

    def _drop_columns(self, cols_to_drop: List[str]) -> None:
        """Drop columns from `self.df` if they exist.

        Args:
            cols_to_drop: Column names to drop when present in `self.df`.
        """
        self.df.drop(
            columns=[c for c in cols_to_drop if c in self.df.columns], inplace=True
        )

    def _create_cyclic_features(self) -> None:
        """Create cyclic time features and (optionally) wind direction encoding.

        Creates integer time components from `time`:
          - hour, dayofweek, month

        Then encodes each cyclically via sin/cos. If "wind_direction_10m" exists,
        it is also encoded cyclically. Intermediate integer columns are dropped.
        """
        if not np.issubdtype(self.df["time"].dtype, np.datetime64):
            self.df["time"] = pd.to_datetime(self.df["time"])

        self.df["hour"] = self.df["time"].dt.hour
        self.df["dayofweek"] = self.df["time"].dt.dayofweek
        self.df["month"] = self.df["time"].dt.month

        self._encode_cyclic_feature("hour", "hour", 24)
        self._encode_cyclic_feature("dayofweek", "dayofweek", 7)
        self._encode_cyclic_feature("month", "month", 12)

        if "wind_direction_10m" in self.df.columns:
            self._encode_cyclic_feature("wind_dir_10m", "wind_direction_10m", 360)

        self._drop_columns(["hour", "dayofweek", "month", "wind_direction_10m"])

    def split_data(self) -> None:
        """Split the full DataFrame into train/val/test by calendar year.

        Split logic:
          - test: last (max) year in the dataset
          - val:  second last year
          - train: all years before val year

        The resulting splits are stored in:
          - self.train_df
          - self.val_df
          - self.test_df
        """
        years = self.df["time"].dt.year
        max_year = years.max()
        test_year = max_year
        val_year = max_year - 1

        train_mask = years < val_year
        val_mask = years == val_year
        test_mask = years == test_year

        self.train_df = self.df.loc[train_mask].reset_index(drop=True)
        self.val_df = self.df.loc[val_mask].reset_index(drop=True)
        self.test_df = self.df.loc[test_mask].reset_index(drop=True)

        print(
            f"Train years: <= {val_year-1}, rows={len(self.train_df)}\n"
            f"Val year: {val_year}, rows={len(self.val_df)}\n"
            f"Test year: {test_year}, rows={len(self.test_df)}"
        )

    def normalize_data(self) -> None:
        """Normalize numeric columns using training statistics only.

        Fits the scaler on numeric columns from `self.train_df`, then applies the
        same transform to train/val/test. The fitted scaler is stored in `self.scaler`.

        Raises:
            ValueError: If `norm_method` is not one of {"standard", "minmax"}.
        """
        all_numeric_cols = self.train_df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        numeric_cols = [
            col
            for col in all_numeric_cols
            if not (col.endswith("_sin") or col.endswith("_cos"))
        ]

        if self.norm_method == "standard":
            scaler_cls = StandardScaler
        elif self.norm_method == "minmax":
            scaler_cls = MinMaxScaler
        else:
            raise ValueError(f"Unknown norm_method: {self.norm_method}")

        self.scaler = scaler_cls()
        self.scaler.fit(self.train_df[numeric_cols])

        for split_name in ["train_df", "val_df", "test_df"]:
            df_split = getattr(self, split_name)
            df_split[numeric_cols] = df_split[numeric_cols].astype(float)
            df_split.loc[:, numeric_cols] = self.scaler.transform(
                df_split[numeric_cols]
            )
            setattr(self, split_name, df_split)

        print(
            f"Normalized {len(numeric_cols)} numeric columns using {self.norm_method} scaler."
        )

    def save_splits_and_scaler(self) -> None:
        """Save train/val/test splits and the fitted scaler to disk.

        Writes:
          - train.parquet
          - val.parquet
          - test.parquet
          - scaler.pkl

        Output directory:
          - "/processed/multi" if multiple locations are configured
          - otherwise "/processed/<single_location>"
        """
        data_dir = Path(self.cfg["api_requests"]["raw_dir"]).parent / "processed"
        if len(self.cfg["api_requests"]["locations"].keys()) > 1:
            out_dir = data_dir / "multi"
        else:
            out_dir = data_dir / list(self.cfg["api_requests"]["locations"].keys())[0]
        os.makedirs(out_dir, exist_ok=True)

        self.train_df.to_parquet(out_dir / "train.parquet", index=False)
        self.val_df.to_parquet(out_dir / "val.parquet", index=False)
        self.test_df.to_parquet(out_dir / "test.parquet", index=False)

        joblib.dump(self.scaler, out_dir / "scaler.pkl")

        print(f"Saved train/val/test splits to {out_dir}/")

    def preprocess(
        self, deployment: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run the full preprocessing pipeline.

        Steps:
          1) Fetch raw data.
          2) Join data.
          3) Create cyclic features.
          4) Run schema/integrity checks.
          5) Split into train/val/test.
          6) Normalize numeric columns.
          7) Save outputs.

        Returns:
            A tuple of (train_df, val_df, test_df).
        """
        self._fetch_data()
        self._join_data()
        self._create_cyclic_features()

        run_full_integrity_check(self.df)

        if deployment:
            return self.df, None, None

        self.split_data()
        self.normalize_data()
        self.save_splits_and_scaler()

        return self.train_df, self.val_df, self.test_df
