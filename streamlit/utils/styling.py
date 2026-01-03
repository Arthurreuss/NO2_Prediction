def get_aqi_category(
    value: float | int, pollutant: str = "nitrogen_dioxide"
) -> tuple[str, str, str]:
    """Determines the Air Quality Index (AQI) category, color code, and description.

    Based on the official European Air Quality Index (EEA) for hourly concentrations.
    Ref: https://airindex.eea.europa.eu/

    Args:
        value: The pollutant concentration value (e.g., µg/m³).
        pollutant: The specific pollutant identifier (e.g., "no2", "pm10", "so2").
            Defaults to "nitrogen_dioxide".

    Returns:
        A tuple containing three strings:
            1. The AQI Category (e.g., "Good", "Extremely Poor").
            2. The Hex color code (Official EEA colors).
            3. A short text description/advisory (Based on health messages).
    """
    p = pollutant.lower()

    c_good = "#50F0E6"
    c_fair = "#50CCAA"
    c_moderate = "#F0E641"
    c_poor = "#FF5050"
    c_very_poor = "#960032"
    c_extremely_poor = "#7D2181"

    if p in ["no2", "nitrogen_dioxide"]:
        if value <= 10:
            return "Good", c_good, "Enjoy usual activities"
        if value <= 25:
            return "Fair", c_fair, "Enjoy usual activities"
        if value <= 60:
            return (
                "Moderate",
                c_moderate,
                "Sensitive: Consider reducing intense activity",
            )
        if value <= 100:
            return "Poor", c_poor, "Sensitive: Reduce intense activity"
        if value <= 150:
            return "Very Poor", c_very_poor, "General: Reduce intense activity"
        return "Extremely Poor", c_extremely_poor, "General: Avoid outdoor activity"

    elif p in ["o3", "ozone"]:
        if value <= 60:
            return "Good", c_good, "Enjoy usual activities"
        if value <= 100:
            return "Fair", c_fair, "Enjoy usual activities"
        if value <= 120:
            return (
                "Moderate",
                c_moderate,
                "Sensitive: Consider reducing intense activity",
            )
        if value <= 160:
            return "Poor", c_poor, "Sensitive: Reduce intense activity"
        if value <= 180:
            return "Very Poor", c_very_poor, "General: Reduce intense activity"
        return "Extremely Poor", c_extremely_poor, "General: Avoid outdoor activity"

    elif p in ["pm10"]:
        if value <= 15:
            return "Good", c_good, "Enjoy usual activities"
        if value <= 45:
            return "Fair", c_fair, "Enjoy usual activities"
        if value <= 120:
            return (
                "Moderate",
                c_moderate,
                "Sensitive: Consider reducing intense activity",
            )
        if value <= 195:
            return "Poor", c_poor, "Sensitive: Reduce intense activity"
        if value <= 270:
            return "Very Poor", c_very_poor, "General: Reduce intense activity"
        return "Extremely Poor", c_extremely_poor, "General: Avoid outdoor activity"

    elif p in ["pm2_5", "pm2.5"]:
        if value <= 5:
            return "Good", c_good, "Enjoy usual activities"
        if value <= 15:
            return "Fair", c_fair, "Enjoy usual activities"
        if value <= 50:
            return (
                "Moderate",
                c_moderate,
                "Sensitive: Consider reducing intense activity",
            )
        if value <= 90:
            return "Poor", c_poor, "Sensitive: Reduce intense activity"
        if value <= 140:
            return "Very Poor", c_very_poor, "General: Reduce intense activity"
        return "Extremely Poor", c_extremely_poor, "General: Avoid outdoor activity"

    elif p in ["so2", "sulphur_dioxide", "sulfur_dioxide"]:
        if value <= 20:
            return "Good", c_good, "Enjoy usual activities"
        if value <= 40:
            return "Fair", c_fair, "Enjoy usual activities"
        if value <= 125:
            return (
                "Moderate",
                c_moderate,
                "Sensitive: Consider reducing intense activity",
            )
        if value <= 190:
            return "Poor", c_poor, "Sensitive: Reduce intense activity"
        if value <= 275:
            return "Very Poor", c_very_poor, "General: Reduce intense activity"
        return "Extremely Poor", c_extremely_poor, "General: Avoid outdoor activity"

    return "Unknown", "#808080", "No standard available"


def get_aqi_thresholds(pollutant: str) -> list[int]:
    """Retrieves the numeric boundary values for AQI categories.

    Used to draw background color bands on charts.

    Args:
        pollutant: The specific pollutant identifier.

    Returns:
        A list of FIVE integers representing the upper bounds for:
        "Good", "Fair", "Moderate", "Poor", and "Very Poor".
        Anything above the last value is "Extremely Poor".
    """
    p = pollutant.lower()
    if p in ["no2", "nitrogen_dioxide"]:
        return [10, 25, 60, 100, 150]
    if p in ["o3", "ozone"]:
        return [60, 100, 120, 160, 180]
    if p in ["pm10"]:
        return [15, 45, 120, 195, 270]
    if p in ["pm2_5", "pm2.5"]:
        return [5, 15, 50, 90, 140]
    if p in ["so2", "sulphur_dioxide", "sulfur_dioxide"]:
        return [20, 40, 125, 190, 275]

    return [10, 25, 60, 100, 150]
