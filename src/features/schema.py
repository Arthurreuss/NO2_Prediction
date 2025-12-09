from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

FINAL_COLUMNS: List[str] = [
    "time",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "pressure_msl",
    "cloud_cover_high",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
    "wind_speed_10m",
    "wind_gusts_10m",
    "location",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "month_sin",
    "month_cos",
    "wind_dir_10m_sin",
    "wind_dir_10m_cos",
]

TARGET_COLUMN = "nitrogen_dioxide"


EXPECTED_DTYPES: Dict[str, str] = {
    "time": "datetime64[ns]",
    "location": "str",
    # numeric (float) columns
    "temperature_2m": "float",
    "relative_humidity_2m": "int",
    "dew_point_2m": "float",
    "precipitation": "float",
    "pressure_msl": "float",
    "cloud_cover_high": "int",
    "et0_fao_evapotranspiration": "float",
    "vapour_pressure_deficit": "float",
    "wind_speed_10m": "float",
    "wind_gusts_10m": "float",
    "pm10": "float",
    "pm2_5": "float",
    "carbon_monoxide": "float",
    "nitrogen_dioxide": "float",
    "sulphur_dioxide": "float",
    "ozone": "float",
    "hour_sin": "float",
    "hour_cos": "float",
    "dayofweek_sin": "float",
    "dayofweek_cos": "float",
    "month_sin": "float",
    "month_cos": "float",
    "wind_dir_10m_sin": "float",
    "wind_dir_10m_cos": "float",
}


VALUE_RANGES: Dict[str, Tuple[float, float]] = {
    # meteorology
    "temperature_2m": (-40.0, 50.0),
    "relative_humidity_2m": (0.0, 100.0),
    "dew_point_2m": (-40.0, 30.0),
    "precipitation": (0.0, 100.0),
    "pressure_msl": (900.0, 1050.0),
    "cloud_cover_high": (0.0, 100.0),
    "et0_fao_evapotranspiration": (0.0, 20.0),
    "vapour_pressure_deficit": (0.0, 5.0),
    "wind_speed_10m": (0.0, 200.0),
    "wind_gusts_10m": (0.0, 250.0),
    # pollutants
    "pm10": (0.0, 1000.0),
    "pm2_5": (0.0, 500.0),
    "carbon_monoxide": (0.0, 5000.0),
    "nitrogen_dioxide": (0.0, 500.0),
    "sulphur_dioxide": (0.0, 1000.0),
    "ozone": (0.0, 500.0),
    # cyclic features (must lie in [-1, 1])
    "hour_sin": (-1.0001, 1.0001),
    "hour_cos": (-1.0001, 1.0001),
    "dayofweek_sin": (-1.0001, 1.0001),
    "dayofweek_cos": (-1.0001, 1.0001),
    "month_sin": (-1.0001, 1.0001),
    "month_cos": (-1.0001, 1.0001),
    "wind_dir_10m_sin": (-1.0001, 1.0001),
    "wind_dir_10m_cos": (-1.0001, 1.0001),
}


ALLOWED_LOCATIONS = ["utrecht"]


def check_columns(df: pd.DataFrame) -> None:
    """Ensure all expected columns are present and no unexpected columns appear."""
    missing = set(FINAL_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(FINAL_COLUMNS)

    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected columns present: {sorted(extra)}")


def check_dtypes(df: pd.DataFrame) -> None:
    """Check that columns have the expected dtypes."""
    # time
    if not np.issubdtype(df["time"].dtype, np.datetime64):
        raise TypeError(f"'time' must be datetime64, got {df['time'].dtype}")

    # location
    if df["location"].dtype.kind not in ("O", "U", "S"):
        raise TypeError(f"'location' must be string-like, got {df['location'].dtype}")

    # numeric
    for col, expected in EXPECTED_DTYPES.items():
        if col in ("time", "location"):
            continue
        if expected == "float":
            if not np.issubdtype(df[col].dtype, np.floating):
                raise TypeError(f"Column '{col}' must be float, got {df[col].dtype}")


def check_no_missing(df: pd.DataFrame) -> None:
    """Ensure there are no missing values in any column."""
    missing_counts = df.isna().sum()
    total_missing = int(missing_counts.sum())
    if total_missing > 0:
        raise ValueError(
            f"Found {total_missing} missing values. Per column:\n{missing_counts[missing_counts > 0]}"
        )


def check_value_ranges(df: pd.DataFrame) -> None:
    """Check that numeric values lie within physically plausible ranges."""
    violations = {}
    for col, (vmin, vmax) in VALUE_RANGES.items():
        if col not in df.columns:
            continue
        col_min = df[col].min()
        col_max = df[col].max()
        if (col_min < vmin) or (col_max > vmax):
            violations[col] = (float(col_min), float(col_max), vmin, vmax)

    if violations:
        msg_lines = ["Value range violations detected:"]
        for col, (cmin, cmax, vmin, vmax) in violations.items():
            msg_lines.append(
                f"  - {col}: observed [{cmin:.3f}, {cmax:.3f}], "
                f"expected within [{vmin:.3f}, {vmax:.3f}]"
            )
        raise ValueError("\n".join(msg_lines))


def check_location_values(df: pd.DataFrame) -> None:
    """Ensure location field only contains allowed values."""
    bad_locations = set(df["location"].unique()) - set(ALLOWED_LOCATIONS)
    if bad_locations:
        raise ValueError(f"Unexpected locations found: {sorted(bad_locations)}")


def check_time_index(df: pd.DataFrame) -> None:
    """
    Basic time-integrity checks:
    - no duplicate (time, location) pairs
    - monotonicity per location (optional but recommended)
    """
    # duplicates
    dup_mask = df.duplicated(subset=["time", "location"])
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        raise ValueError(f"Found {n_dup} duplicate (time, location) rows.")

    # optional: per-location monotonicity
    for loc, df_loc in df.groupby("location"):
        if not df_loc["time"].is_monotonic_increasing:
            raise ValueError(
                f"'time' is not monotonic increasing for location '{loc}'."
            )


def run_full_integrity_check(df: pd.DataFrame) -> None:
    """
    Run all integrity checks.
    Call this before training and before using data for inference.
    """
    check_columns(df)
    check_dtypes(df)
    check_no_missing(df)
    check_value_ranges(df)
    check_location_values(df)
    check_time_index(df)
    print("Data integrity check passed.")
