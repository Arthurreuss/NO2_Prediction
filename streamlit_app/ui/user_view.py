import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.styling import get_aqi_category, get_aqi_thresholds


def render_user_dashboard(df_history: pd.DataFrame, preds: dict[str, pd.DataFrame]):
    col_sel, col_mod = st.columns([1, 2])

    map_pol = {
        "Nitrogen Dioxide (NO₂)": "nitrogen_dioxide",
        "Ozone (O₃)": "ozone",
        "PM10": "pm10",
        "PM2.5": "pm2_5",
    }
    label = col_sel.selectbox("Select Pollutant", list(map_pol.keys()))
    col_name = map_pol[label]

    models = col_mod.multiselect(
        "Models", list(preds.keys()), default=list(preds.keys())
    )

    # --- 1. Current Status Card ---
    if not df_history.empty:
        latest = df_history.dropna(subset=[col_name]).iloc[-1]
        val = latest[col_name]
        cat, color, desc = get_aqi_category(val, col_name)

        st.markdown(
            f"""
            <div style="background-color:{color};padding:20px;border-radius:10px;color:white;text-align:center;margin-bottom:20px">
                <h3 style="margin:0">Current: {cat}</h3>
                <h1 style="margin:0;font-size:3em">{val:.1f} µg/m³</h1>
                <p style="margin:0">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- 2. Main Forecast Plot ---
    fig = go.Figure()
    curr_time = pd.Timestamp.now(tz="UTC")

    # Plot History (Last 7 days)
    hist_view = df_history[df_history["time"] > (curr_time - pd.Timedelta(days=7))]
    fig.add_trace(
        go.Scatter(
            x=hist_view["time"],
            y=hist_view[col_name],
            name="Observed",
            line=dict(color="white", width=2),
        )
    )

    # Plot Forecasts
    colors = ["#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]
    for i, m in enumerate(models):
        if col_name in preds[m].columns:
            # Show forecast starting from yesterday (continuity)
            df_p = preds[m][preds[m]["time"] > (curr_time - pd.Timedelta(days=1))]
            if not df_p.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df_p["time"],
                        y=df_p[col_name],
                        name=m,
                        line=dict(color=colors[i % 4], width=3),
                    )
                )

    # NOW Line
    fig.add_vline(
        x=curr_time.timestamp() * 1000, line_dash="dot", annotation_text="NOW"
    )

    # AQI Bands
    th = get_aqi_thresholds(col_name)
    bg_colors = ["#50F0E6", "#50CCAA", "#F0E641", "#FF5050", "#960032", "#7D2181"]
    prev = 0
    for i, limit in enumerate(th):
        fig.add_hrect(
            y0=prev,
            y1=limit,
            fillcolor=bg_colors[i],
            opacity=0.15,
            layer="below",
            line_width=0,
        )
        prev = limit
    # Extremes band
    fig.add_hrect(
        y0=th[-1],
        y1=th[-1] * 5,
        fillcolor=bg_colors[-1],
        opacity=0.15,
        layer="below",
        line_width=0,
    )

    fig.update_layout(
        height=500,
        xaxis_title="Time",
        yaxis_title="µg/m³",
        hovermode="x unified",
        margin=dict(t=10, l=10, r=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
