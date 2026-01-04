import os

import pandas as pd
import plotly.graph_objects as go

import streamlit as st
from utils.styling import get_aqi_category, get_aqi_thresholds


def check_secrets_debug():
    """Safe check to see if environment variables are loaded."""
    st.write("### 🔐 Secrets Diagnostic Check")

    # List of secrets your email function expects
    required_secrets = [
        "EMAIL_USER",
        "EMAIL_PASSWORD",
        "EMAIL_TO",
        "SMTP_SERVER",
        "SMTP_PORT",
    ]

    status_data = []

    for secret in required_secrets:
        value = os.environ.get(secret)

        if value:
            # Show first 2 chars to verify it's not just an empty string
            # e.g. "my..." or "sm..."
            masked = f"{value[:2]}..." + "*" * 4
            status = "✅ Loaded"
        else:
            masked = "MISSING"
            status = "❌ Not Found"

        status_data.append(
            {"Secret Name": secret, "Status": status, "Value Preview": masked}
        )

    st.table(status_data)

    # Extra Check: Print ALL keys (filtered) to see if there's a typo
    # e.g. Did you name it 'MAIL_USER' instead of 'EMAIL_USER'?
    st.write("**All Available Env Vars (Key Names Only):**")
    all_keys = [
        k for k in os.environ.keys() if "EMAIL" in k or "SMTP" in k or "HF" in k
    ]
    st.write(all_keys)


def render_user_dashboard(
    df_history: pd.DataFrame, preds: dict[str, pd.DataFrame]
) -> None:
    """Renders the main user dashboard for air quality monitoring and forecasting.

    Displays an interactive interface allowing users to select specific pollutants
    and forecasting models. Visualizes historical data alongside model predictions
    using Plotly, with background color bands representing EEA Air Quality Index
    (AQI) thresholds. Also displays a 'Current Status' card based on the most
    recent historical observation.

    Args:
        df_history: A DataFrame containing historical air quality data with
            timestamp and pollutant columns.
        preds: A dictionary mapping model names (e.g., 'XGBoost', 'LSTM') to
            DataFrames containing their respective prediction data.
    """
    col_sel, col_models = st.columns([1, 2])

    with col_sel:
        pollutant_map = {
            "Nitrogen Dioxide (NO₂)": "nitrogen_dioxide",
            "Ozone (O₃)": "ozone",
            "Particulate Matter (PM10)": "pm10",
            "Fine Particles (PM2.5)": "pm2_5",
        }
        selected_label = st.selectbox("Select Pollutant", list(pollutant_map.keys()))
        target_col = pollutant_map[selected_label]

    available_models = list(preds.keys())

    with col_models:
        selected_models = st.multiselect(
            "Select Forecasting Models",
            options=available_models,
            default=available_models,
        )

    st.markdown(
        f"Monitor and forecast **{selected_label}** levels to plan your outdoor activities."
    )

    if not df_history.empty and target_col in df_history.columns:
        valid_history = df_history.dropna(subset=[target_col])
        if not valid_history.empty:
            latest = valid_history.iloc[-1]
            val = latest[target_col]
            cat, color, desc = get_aqi_category(val, target_col)

            st.markdown(
                f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 25px;">
                <h3 style="margin:0;">Current Status: {cat}</h3>
                <h1 style="margin:0; font-size: 3em;">{val:.1f} µg/m³</h1>
                <p style="margin:0;">{desc}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.warning(f"No recent historical data available for {selected_label}")
    else:
        st.info("Waiting for historical data...")

    st.subheader(f"{selected_label} Forecast vs. History")
    fig = go.Figure()
    max_data_value = 0
    current_time = pd.Timestamp.now(tz="UTC")

    if not df_history.empty and target_col in df_history.columns:
        cutoff_date = current_time - pd.Timedelta(days=7)
        hist_plot = df_history[df_history["time"] > cutoff_date].copy()

        if not hist_plot.empty:
            hist_plot = hist_plot.dropna(subset=[target_col])
            hist_plot = hist_plot.sort_values("time")

            if not hist_plot.empty:
                current_max = hist_plot[target_col].max()
                if pd.notna(current_max):
                    max_data_value = max(max_data_value, current_max)

                fig.add_trace(
                    go.Scatter(
                        x=hist_plot["time"],
                        y=hist_plot[target_col],
                        name="Observed History",
                        line=dict(color="white", width=2),
                        legendgroup="data",
                    )
                )

    colors = ["#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692"]
    valid_models_found = 0

    for i, (model_name, df_p) in enumerate(preds.items()):
        if model_name not in selected_models:
            continue

        if target_col not in df_p.columns:
            continue

        if not df_p.empty:
            start_window = current_time - pd.Timedelta(days=7)
            df_view = df_p[df_p["time"] > start_window].copy()
            df_view = df_view.sort_values("time")

            if not df_view.empty:
                current_max = df_view[target_col].max()
                if pd.notna(current_max):
                    max_data_value = max(max_data_value, current_max)

                fig.add_trace(
                    go.Scatter(
                        x=df_view["time"],
                        y=df_view[target_col],
                        name=f"Forecast ({model_name})",
                        line=dict(color=colors[i % len(colors)], width=3),
                        legendgroup="data",
                    )
                )
                valid_models_found += 1

    if valid_models_found == 0 and len(selected_models) > 0:
        st.caption(f"No valid data found for the selected models.")
    elif len(selected_models) == 0:
        st.caption("Select at least one model to see the forecast.")

    current_time_ms = current_time.timestamp() * 1000

    fig.add_vline(
        x=current_time_ms,
        line_width=2,
        line_dash="dot",
        line_color="gray",
        annotation_text="NOW",
        annotation_position="top left",
    )

    th = get_aqi_thresholds(target_col)
    OPACITY = 0.15

    fig.add_hrect(
        y0=0, y1=th[0], fillcolor="green", opacity=OPACITY, line_width=0, layer="below"
    )
    fig.add_hrect(
        y0=th[0],
        y1=th[1],
        fillcolor="yellow",
        opacity=OPACITY,
        line_width=0,
        layer="below",
    )
    fig.add_hrect(
        y0=th[1],
        y1=th[2],
        fillcolor="orange",
        opacity=OPACITY,
        line_width=0,
        layer="below",
    )
    fig.add_hrect(
        y0=th[2],
        y1=th[3],
        fillcolor="red",
        opacity=OPACITY,
        line_width=0,
        layer="below",
    )

    aqi_legend = [
        (f"Good (<{th[0]})", "green"),
        (f"Fair ({th[0]}-{th[1]})", "#FFD700"),
        (f"Moderate ({th[1]}-{th[2]})", "orange"),
        (f"Poor (>{th[2]})", "red"),
    ]
    for label, color in aqi_legend:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color, symbol="square"),
                name=label,
                legendgroup="aqi",
                showlegend=True,
            )
        )

    calculated_top = max_data_value * 1.1
    minimum_view = th[0] * 1.2
    final_top_limit = max(calculated_top, minimum_view)

    zoom_start = current_time
    zoom_end = current_time + pd.Timedelta(hours=48)

    fig.update_layout(
        height=600,
        xaxis_title="Time (UTC)",
        yaxis_title=f"{selected_label} (µg/m³)",
        yaxis_range=[0, final_top_limit],
        xaxis=dict(
            type="date", rangeslider=dict(visible=True), range=[zoom_start, zoom_end]
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        dragmode="pan",
    )

    st.plotly_chart(fig, width="stretch")
    st.info(
        f"ℹ️ Guidelines based on European Environment Agency (EEA) standards for {selected_label}."
    )
    check_secrets_debug()
