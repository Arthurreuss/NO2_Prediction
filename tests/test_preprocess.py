import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts.preprocess import PreProcessingPipeline
from src.features.schema import run_full_integrity_check


@pytest.fixture
def basic_cfg() -> Dict:
    """Minimal config; not actually used by tests (we set df manually)."""
    return {
        "api_requests": {
            "time": {
                "start_date": "2020-01-01",
                "end_date": "2022-12-31",
            },
            "locations": {
                "utrecht": {
                    "latitude": 52.0908,
                    "longitude": 5.1222,
                }
            },
        }
    }


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Synthetic dataframe with 3 years of hourly data (small subset),
    including all columns needed by the pipeline BEFORE cyclic feature creation.
    """
    # 3 years: 2020 (train), 2021 (val), 2022 (test)
    times = pd.date_range("2020-01-01", "2022-12-31 23:00", freq="6h")
    n = len(times)

    df = pd.DataFrame(
        {
            "time": times,
            "location": ["utrecht"] * n,
            # meteorology
            "temperature_2m": np.linspace(0, 20, n),
            "relative_humidity_2m": np.linspace(40, 90, n),
            "dew_point_2m": np.linspace(-5, 15, n),
            "precipitation": np.clip(np.random.gamma(0.5, 0.5, size=n), 0, 10),
            "pressure_msl": np.linspace(990, 1020, n),
            "cloud_cover_high": np.linspace(0, 100, n),
            "et0_fao_evapotranspiration": np.clip(
                np.random.gamma(0.8, 0.2, size=n), 0, 10
            ),
            "vapour_pressure_deficit": np.linspace(0.1, 2.0, n),
            "wind_speed_10m": np.linspace(0, 40, n),
            "wind_gusts_10m": np.linspace(0, 60, n),
            "wind_direction_10m": np.linspace(0, 359, n) % 360,
            # pollutants
            "pm10": np.linspace(5, 80, n),
            "pm2_5": np.linspace(3, 40, n),
            "carbon_monoxide": np.linspace(100, 800, n),
            "nitrogen_dioxide": np.linspace(5, 80, n),
            "sulphur_dioxide": np.linspace(1, 30, n),
            "ozone": np.linspace(10, 120, n),
            # columns that will be dropped by pipeline
            "apparent_temperature": np.linspace(0, 20, n),
            "rain": np.clip(np.random.gamma(0.5, 0.5, size=n), 0, 10),
            "snowfall": np.zeros(n),
            "snow_depth": np.zeros(n),
            "weather_code": np.zeros(n),
            "surface_pressure": np.linspace(990, 1020, n),
            "cloud_cover": np.linspace(0, 100, n),
            "cloud_cover_low": np.linspace(0, 100, n),
            "cloud_cover_mid": np.linspace(0, 100, n),
            "wind_speed_100m": np.linspace(0, 40, n),
            "wind_direction_100m": np.linspace(0, 359, n) % 360,
            "soil_temperature_0_to_7cm": np.linspace(0, 20, n),
            "soil_temperature_7_to_28cm": np.linspace(0, 20, n),
            "soil_temperature_28_to_100cm": np.linspace(0, 20, n),
            "soil_temperature_100_to_255cm": np.linspace(0, 20, n),
            "soil_moisture_0_to_7cm": np.linspace(0.1, 0.4, n),
            "soil_moisture_7_to_28cm": np.linspace(0.1, 0.4, n),
            "soil_moisture_28_to_100cm": np.linspace(0.1, 0.4, n),
            "soil_moisture_100_to_255cm": np.linspace(0.1, 0.4, n),
            "carbon_dioxide": np.zeros(n),
            "methane": np.zeros(n),
            "uv_index_clear_sky": np.zeros(n),
            "uv_index": np.zeros(n),
            "aerosol_optical_depth": np.zeros(n),
            "ammonia": np.zeros(n),
        }
    )

    return df


def test_create_cyclic_features(basic_cfg, sample_df):
    pipeline = PreProcessingPipeline(cfg=basic_cfg)
    pipeline.df = sample_df.copy()

    pipeline._create_cyclic_features()

    # check new cyclic columns exist
    for col in [
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "month_sin",
        "month_cos",
        "wind_dir_10m_sin",
        "wind_dir_10m_cos",
    ]:
        assert col in pipeline.df.columns

    # check old helper columns are dropped
    for col in ["hour", "dayofweek", "month", "wind_direction_10m"]:
        assert col not in pipeline.df.columns

    # check sin/cos are within [-1, 1] before any scaling
    cyclic_cols = [
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "month_sin",
        "month_cos",
        "wind_dir_10m_sin",
        "wind_dir_10m_cos",
    ]
    for col in cyclic_cols:
        assert pipeline.df[col].between(-1.0001, 1.0001).all()


def test_split_data_by_year(basic_cfg, sample_df):
    pipeline = PreProcessingPipeline(cfg=basic_cfg)
    pipeline.df = sample_df.copy()

    pipeline._create_cyclic_features()
    pipeline.split_data()

    # train = all years before val_year (2020),
    # val = 2021, test = 2022
    assert pipeline.train_df["time"].dt.year.max() == 2020
    assert pipeline.val_df["time"].dt.year.unique().tolist() == [2021]
    assert pipeline.test_df["time"].dt.year.unique().tolist() == [2022]

    # ensure all rows are accounted for
    total_rows = len(pipeline.train_df) + len(pipeline.val_df) + len(pipeline.test_df)
    assert total_rows == len(pipeline.df)


def test_normalize_data_standard_scaler(basic_cfg, sample_df):
    pipeline = PreProcessingPipeline(cfg=basic_cfg, norm_method="standard")
    pipeline.df = sample_df.copy()

    pipeline._create_cyclic_features()
    # drop columns exactly as in pipeline.preprocess
    pipeline._drop_columns(
        [
            "apparent_temperature",
            "rain",
            "snowfall",
            "snow_depth",
            "weather_code",
            "surface_pressure",
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "wind_speed_100m",
            "wind_direction_100m",
            "soil_temperature_0_to_7cm",
            "soil_temperature_7_to_28cm",
            "soil_temperature_28_to_100cm",
            "soil_temperature_100_to_255cm",
            "soil_moisture_0_to_7cm",
            "soil_moisture_7_to_28cm",
            "soil_moisture_28_to_100cm",
            "soil_moisture_100_to_255cm",
            "carbon_dioxide",
            "methane",
            "uv_index_clear_sky",
            "uv_index",
            "aerosol_optical_depth",
            "ammonia",
        ]
    )

    pipeline.split_data()
    pipeline.normalize_data()

    # all numeric columns should have mean ~0 and std ~1 on the train set
    numeric_cols = pipeline.train_df.select_dtypes(include=[np.number]).columns
    train_means = pipeline.train_df[numeric_cols].mean()
    train_stds = pipeline.train_df[numeric_cols].std(ddof=0)

    assert (train_means.abs() < 1e-6).all()
    assert ((train_stds - 1.0).abs() < 1e-3).all()


def test_save_preprocessed_data_creates_files(tmp_path, basic_cfg, sample_df):
    pipeline = PreProcessingPipeline(cfg=basic_cfg, norm_method="standard")
    pipeline.df = sample_df.copy()

    pipeline._create_cyclic_features()
    pipeline._drop_columns(
        [
            "apparent_temperature",
            "rain",
            "snowfall",
            "snow_depth",
            "weather_code",
            "surface_pressure",
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "wind_speed_100m",
            "wind_direction_100m",
            "soil_temperature_0_to_7cm",
            "soil_temperature_7_to_28cm",
            "soil_temperature_28_to_100cm",
            "soil_temperature_100_to_255cm",
            "soil_moisture_0_to_7cm",
            "soil_moisture_7_to_28cm",
            "soil_moisture_28_to_100cm",
            "soil_moisture_100_to_255cm",
            "carbon_dioxide",
            "methane",
            "uv_index_clear_sky",
            "uv_index",
            "aerosol_optical_depth",
            "ammonia",
        ]
    )

    pipeline.split_data()
    pipeline.normalize_data()

    out_dir = tmp_path / "processed"
    pipeline.save_preprocessed_data(str(out_dir))

    # check that files exist
    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "val.parquet").exists()
    assert (out_dir / "test.parquet").exists()
    assert (out_dir / "scaler.pkl").exists()

    # check scaler can be loaded and used
    scaler = joblib.load(out_dir / "scaler.pkl")
    assert hasattr(scaler, "transform")


def test_run_full_integrity_check_on_raw_data(basic_cfg, sample_df):
    """
    Ensure that the integrity check passes on raw data after feature engineering
    and dropping the columns that are not used.
    """
    pipeline = PreProcessingPipeline(cfg=basic_cfg)
    pipeline.df = sample_df.copy()

    pipeline._create_cyclic_features()
    pipeline._drop_columns(
        [
            "apparent_temperature",
            "rain",
            "snowfall",
            "snow_depth",
            "weather_code",
            "surface_pressure",
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "wind_speed_100m",
            "wind_direction_100m",
            "soil_temperature_0_to_7cm",
            "soil_temperature_7_to_28cm",
            "soil_temperature_28_to_100cm",
            "soil_temperature_100_to_255cm",
            "soil_moisture_0_to_7cm",
            "soil_moisture_7_to_28cm",
            "soil_moisture_28_to_100cm",
            "soil_moisture_100_to_255cm",
            "carbon_dioxide",
            "methane",
            "uv_index_clear_sky",
            "uv_index",
            "aerosol_optical_depth",
            "ammonia",
        ]
    )

    # should not raise
    run_full_integrity_check(pipeline.df)
