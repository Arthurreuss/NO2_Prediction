import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.data.api_requests import fetch_air_quality_data, fetch_weather_data
from src.features.schema import run_full_integrity_check


class PreProcessingPipeline:
    def __init__(self, cfg: Dict, norm_method: str = "standard"):
        """
        cfg: configuration dictionary (from config.yaml)
        norm_method: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
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
        """Call the API request functions."""
        fetch_weather_data(self.cfg)
        fetch_air_quality_data(self.cfg)

    def _join_data(self) -> None:
        """Join weather and air-quality data for each location and concatenate."""
        start_date = self.cfg["api_requests"]["time"]["start_date"]
        end_date = self.cfg["api_requests"]["time"]["end_date"]

        for loc_name in self.cfg["api_requests"]["locations"].keys():
            aq = pd.read_csv(
                f"data/raw/air_quality/{loc_name}_air_quality_{start_date}_to_{end_date}.csv",
                parse_dates=["time"],
            )
            wx = pd.read_csv(
                f"data/raw/weather/{loc_name}_weather_{start_date}_to_{end_date}.csv",
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
        """Encode a cyclical feature using sin/cos on an integer column."""
        self.df[f"{name}_sin"] = np.sin(2 * np.pi * self.df[col] / max_val)
        self.df[f"{name}_cos"] = np.cos(2 * np.pi * self.df[col] / max_val)

    def _drop_columns(self, cols_to_drop: List[str]) -> None:
        self.df.drop(
            columns=[c for c in cols_to_drop if c in self.df.columns], inplace=True
        )

    def _create_cyclic_features(self) -> None:
        """Create cyclic time features and wind direction encoding."""
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
        """
        Split into train/val/test by calendar year.

        - test: last full year
        - val:  second last full year
        - train: everything before that
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
        """
        Normalize numeric columns using only training data statistics.
        Applies the same scaler to train/val/test.
        """
        numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns.tolist()

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

    def save_preprocessed_data(self) -> None:
        """
        Save train/val/test splits as parquet files.
        out_dir: directory where files will be written.
        """
        if len(self.cfg["api_requests"]["locations"].keys()) > 1:
            out_dir = "data/processed/multi"
        else:
            out_dir = os.path.join(
                "data/processed", list(self.cfg["api_requests"]["locations"].keys())[0]
            )
        os.makedirs(out_dir, exist_ok=True)

        self.train_df.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
        self.val_df.to_parquet(os.path.join(out_dir, "val.parquet"), index=False)
        self.test_df.to_parquet(os.path.join(out_dir, "test.parquet"), index=False)

        joblib.dump(self.scaler, os.path.join(out_dir, "scaler.pkl"))

        print(f"Saved train/val/test splits to {out_dir}/")

    def preprocess(self):
        # self._fetch_data()
        self._join_data()
        self._create_cyclic_features()
        self._drop_columns(
            [
                # redundant temps / diagnostics
                "apparent_temperature",
                "dew_point_2m",
                "et0_fao_evapotranspiration",
                "vapour_pressure_deficit",
                # precip / snow related
                "rain",
                "snowfall",
                "snow_depth",
                "weather_code",
                # redundant pressures
                "surface_pressure",
                # clouds
                "cloud_cover",
                "cloud_cover_low",
                "cloud_cover_mid",
                "cloud_cover_high",
                # upper-level wind / gusts
                "wind_speed_100m",
                "wind_direction_100m",
                "wind_gusts_10m",
                # soil temperatures / moistures
                "soil_temperature_0_to_7cm",
                "soil_temperature_7_to_28cm",
                "soil_temperature_28_to_100cm",
                "soil_temperature_100_to_255cm",
                "soil_moisture_0_to_7cm",
                "soil_moisture_7_to_28cm",
                "soil_moisture_28_to_100cm",
                "soil_moisture_100_to_255cm",
                # additional gases and diagnostics we don't model / use
                "carbon_dioxide",
                "carbon_monoxide",
                "methane",
                "sulphur_dioxide",
                "aerosol_optical_depth",
                "ammonia",
                # radiation / UV diagnostics
                "uv_index_clear_sky",
                "uv_index",
            ]
        )

        print("Remaining columns after feature selection:")
        print(self.df.columns.tolist())

        run_full_integrity_check(self.df)

        self.split_data()

        self.normalize_data()
        self.save_preprocessed_data()

        return self.train_df, self.val_df, self.test_df
