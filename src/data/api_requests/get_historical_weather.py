from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests


def fetch_weather_data(cfg: Dict[str, Any]) -> None:
    """Fetch historical weather data from the Open-Meteo API and save as CSV files.

    This function iterates over all locations defined in
    `cfg["api_requests"]["locations"]`, fetches hourly weather data from
    the Open-Meteo Weather API for the specified date range, and stores
    the results as CSV files under `data/raw/weather/`.

    The output CSV files are named following the pattern:
        "<location>_weather_<start_date>_to_<end_date>.csv"

    Args:
        cfg: Configuration dictionary containing API settings. Expected
            structure includes:
            - cfg["api_requests"]["weather_api_base_url"]
            - cfg["api_requests"]["weather_params"]
            - cfg["api_requests"]["locations"]
            - cfg["api_requests"]["time"]["start_date"]
            - cfg["api_requests"]["time"]["end_date"]

    Returns:
        None. The function writes CSV files to disk as a side effect.

    Raises:
        requests.HTTPError: If the API request returns a non-success status code.
    """
    api_cfg = cfg["api_requests"]
    base_url = api_cfg["weather_api_base_url"]
    params_cfg = api_cfg["weather_params"]
    locations = api_cfg["locations"]
    time_cfg = api_cfg["time"]

    hourly_vars = params_cfg["hourly_variables"]
    timezone = params_cfg.get("timezone", "UTC")
    temperature_unit = params_cfg.get("temperature_unit", "celsius")
    wind_speed_unit = params_cfg.get("wind_speed_unit", "kmh")
    precipitation_unit = params_cfg.get("precipitation_unit", "mm")

    start_date = time_cfg["start_date"]
    end_date = time_cfg["end_date"]

    out_dir = Path("data/raw/weather")
    out_dir.mkdir(parents=True, exist_ok=True)

    for loc_name, loc_cfg in locations.items():
        lat = loc_cfg["latitude"]
        lon = loc_cfg["longitude"]

        print(
            f"[WEATHER] Fetching {loc_name} ({lat}, {lon}) from {start_date} to {end_date}"
        )

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(hourly_vars),
            "timezone": timezone,
            "temperature_unit": temperature_unit,
            "wind_speed_unit": wind_speed_unit,
            "precipitation_unit": precipitation_unit,
        }

        resp = requests.get(base_url, params=params)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly")

        df = pd.DataFrame(hourly)

        df["time"] = pd.to_datetime(df["time"])
        df["location"] = loc_name

        df = df.sort_values("time").reset_index(drop=True)

        filename = f"{loc_name}_weather_{start_date}_to_{end_date}.csv"
        out_path = out_dir / filename
        df.to_csv(out_path, index=False)

        print(f"[WEATHER] Saved {len(df)} rows to {out_path}")
