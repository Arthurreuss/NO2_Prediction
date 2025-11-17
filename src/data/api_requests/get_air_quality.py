from pathlib import Path

import pandas as pd
import requests


def fetch_air_quality_data(cfg: dict) -> None:
    """
    Fetch historical air quality data from Open-Meteo for all locations
    defined in cfg['api_requests']['locations'] and save as CSV.
    """
    api_cfg = cfg["api_requests"]
    base_url = api_cfg["air_quality_api_base_url"]
    params_cfg = api_cfg["air_quality_params"]
    locations = api_cfg["locations"]
    time_cfg = api_cfg["time"]

    hourly_vars = params_cfg["hourly_variables"]
    timezone = params_cfg.get("timezone", "UTC")

    start_date = time_cfg["start_date"]  # e.g. "2023-01-01"
    end_date = time_cfg["end_date"]  # e.g. "2024-01-01"

    out_dir = Path("data/raw/air_quality")
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
