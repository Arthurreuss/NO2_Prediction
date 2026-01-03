def get_aqi_category(
    value: float | int, pollutant: str = "nitrogen_dioxide"
) -> tuple[str, str, str]:
    """Determines the Air Quality Index (AQI) category, color code, and description.

    Based on the official European Air Quality Index (EEA) standards as shown in
    the provided documentation (Index levels & Health messages).

    Args:
        value: The pollutant concentration value (e.g., µg/m³).
        pollutant: The specific pollutant identifier (e.g., "no2", "pm10").
            Defaults to "nitrogen_dioxide".

    Returns:
        A tuple containing three strings:
            1. The AQI Category (e.g., "Good", "Extremely poor").
            2. The Hex color code (matching the EEA official styling).
            3. A combined text description containing advice for both
               General and Sensitive populations.
    """
    p = pollutant.lower()

    c_good = "#50F0E6"  # Turquoise
    c_fair = "#50CCAA"  # Greenish Teal
    c_moderate = "#F0E641"  # Yellow
    c_poor = "#FF5050"  # Red
    c_very_poor = "#960032"  # Dark Red
    c_extremely_poor = "#7D2181"  # Purple

    msg_good = (
        "Good",
        c_good,
        "General: The air quality is good. Enjoy your usual outdoor activities. | "
        "Sensitive: The air quality is good. Enjoy your usual outdoor activities.",
    )
    msg_fair = (
        "Fair",
        c_fair,
        "General: Enjoy your usual outdoor activities. | "
        "Sensitive: Enjoy your usual outdoor activities.",
    )
    msg_moderate = (
        "Moderate",
        c_moderate,
        "General: Enjoy your usual outdoor activities. | "
        "Sensitive: Consider reducing intense outdoor activities, if you experience symptoms.",
    )
    msg_poor = (
        "Poor",
        c_poor,
        "General: Consider reducing intense activities outdoors, if you experience symptoms "
        "such as sore eyes, a cough or sore throat. | "
        "Sensitive: Consider reducing physical activities, particularly outdoors, "
        "especially if you experience symptoms.",
    )
    msg_very_poor = (
        "Very poor",
        c_very_poor,
        "General: Consider reducing intense activities outdoors, if you experience symptoms "
        "such as sore eyes, a cough or sore throat. | "
        "Sensitive: Reduce physical activities, particularly outdoors, "
        "especially if you experience symptoms.",
    )
    msg_extremely_poor = (
        "Extremely poor",
        c_extremely_poor,
        "General: Reduce physical activities outdoors. | "
        "Sensitive: Avoid physical activities outdoors.",
    )

    def select_category(val, thresholds):
        if val <= thresholds[0]:
            return msg_good
        if val <= thresholds[1]:
            return msg_fair
        if val <= thresholds[2]:
            return msg_moderate
        if val <= thresholds[3]:
            return msg_poor
        if val <= thresholds[4]:
            return msg_very_poor
        return msg_extremely_poor

    if p in ["no2", "nitrogen_dioxide"]:
        return select_category(value, [10, 25, 60, 100, 150])

    elif p in ["o3", "ozone"]:
        return select_category(value, [60, 100, 120, 160, 180])

    elif p in ["pm10"]:
        return select_category(value, [15, 45, 120, 195, 270])

    elif p in ["pm2_5", "pm2.5"]:
        return select_category(value, [5, 15, 50, 90, 140])

    elif p in ["so2", "sulphur_dioxide", "sulfur_dioxide"]:
        return select_category(value, [20, 40, 125, 190, 275])

    return "Unknown", "#808080", "No standard available"


def get_aqi_thresholds(pollutant: str) -> list[int]:
    """Retrieves the numeric upper bound values for AQI categories.

    Used to draw background color bands on charts.

    Args:
        pollutant: The specific pollutant identifier.

    Returns:
        A list of five integers representing the upper bounds for:
        Good, Fair, Moderate, Poor, and Very poor.
        (Anything above the last value is Extremely poor).
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
