def get_aqi_category(value, pollutant="nitrogen_dioxide"):
    """
    Returns (Category, Color, Description) based on EEA (European Environment Agency) standards.
    Thresholds are approximate hourly/daily visual guides.
    """
    p = pollutant.lower()

    if p in ["no2", "nitrogen_dioxide"]:
        if value <= 50:
            return "Good", "#009966", "Satisfactory"
        if value <= 100:
            return "Fair", "#FFDE33", "Acceptable"
        if value <= 200:
            return "Moderate", "#FF9933", "Caution for sensitive groups"
        if value <= 400:
            return "Poor", "#CC0033", "Health warning"
        return "Very Poor", "#7E0023", "Emergency conditions"

    elif p in ["o3", "ozone"]:
        if value <= 50:
            return "Good", "#009966", "Satisfactory"
        if value <= 100:
            return "Fair", "#FFDE33", "Acceptable"
        if value <= 130:
            return "Moderate", "#FF9933", "Caution for sensitive groups"
        if value <= 240:
            return "Poor", "#CC0033", "Health warning"
        return "Very Poor", "#7E0023", "Emergency conditions"

    elif p in ["pm10"]:
        if value <= 20:
            return "Good", "#009966", "Satisfactory"
        if value <= 40:
            return "Fair", "#FFDE33", "Acceptable"
        if value <= 50:
            return "Moderate", "#FF9933", "Caution for sensitive groups"
        if value <= 100:
            return "Poor", "#CC0033", "Health warning"
        return "Very Poor", "#7E0023", "Emergency conditions"

    elif p in ["pm2_5", "pm2.5"]:
        if value <= 10:
            return "Good", "#009966", "Satisfactory"
        if value <= 20:
            return "Fair", "#FFDE33", "Acceptable"
        if value <= 25:
            return "Moderate", "#FF9933", "Caution for sensitive groups"
        if value <= 50:
            return "Poor", "#CC0033", "Health warning"
        return "Very Poor", "#7E0023", "Emergency conditions"

    return "Unknown", "#808080", "No standard available"


def get_aqi_thresholds(pollutant):
    """Returns the boundary values for background coloring"""
    p = pollutant.lower()
    if p in ["no2", "nitrogen_dioxide"]:
        return [50, 100, 200, 400]
    if p in ["o3", "ozone"]:
        return [50, 100, 130, 240]
    if p in ["pm10"]:
        return [20, 40, 50, 100]
    if p in ["pm2_5", "pm2.5"]:
        return [10, 20, 25, 50]
    return [50, 100, 200, 400]
