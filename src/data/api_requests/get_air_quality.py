from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests


def fetch_air_quality_data(cfg: Dict[str, Any]) -> None:
    """Fetch historical air-quality data from the Open-Meteo API and save as CSV files.

    This function iterates over all locations defined in
    `cfg["api_requests"]["locations"]`, fetches hourly air-quality data from
    the Open-Meteo Air Quality API for the specified date range, and stores
    the results as CSV files under `data/raw/air_quality/`.

    The output CSV files are named following the pattern:
        "<location>_air_quality_<start_date>_to_<end_date>.csv"

    Args:
        cfg: Configuration dictionary containing API settings.

    Returns:
        None. The function writes CSV files to disk as a side effect.

    Raises:
        requests.HTTPError: If the API request returns a non-success status code.
    """
    api_cfg = cfg["api_requests"]
    base_url = api_cfg["air_quality_api_base_url"]
    params_cfg = api_cfg["air_quality_params"]
    locations = api_cfg["locations"]
    time_cfg = api_cfg["time"]

    hourly_vars = params_cfg["hourly_variables"]
    timezone = params_cfg.get("timezone", "Europe/Amsterdam")

    start_date = time_cfg["start_date"]
    end_date = time_cfg["end_date"]

    out_dir = Path(api_cfg["raw_dir"]) / "air_quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    for loc_name, loc_cfg in locations.items():
        lat = loc_cfg["latitude"]
        lon = loc_cfg["longitude"]

        print(
            f"[AIR] Fetching {loc_name} ({lat}, {lon}) from {start_date} to {end_date}"
        )

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(hourly_vars),
            "timezone": timezone,
        }

        resp = requests.get(base_url, params=params)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly")

        df = pd.DataFrame(hourly)

        df["time"] = pd.to_datetime(df["time"])
        df["location"] = loc_name

        df = df.sort_values("time").reset_index(drop=True)

        filename = f"{loc_name}_air_quality_{start_date}_to_{end_date}.csv"
        out_path = out_dir / filename
        df.to_csv(out_path, index=False)

        print(f"[AIR] Saved {len(df)} rows to {out_path}")
