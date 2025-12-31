import pandas as pd
import plotly.graph_objects as go

import streamlit as st
from utils.styling import get_aqi_category, get_aqi_thresholds


def render_user_dashboard(df_history, preds):
    col_sel, col_empty = st.columns([1, 3])
    with col_sel:
        pollutant_map = {
            "Nitrogen Dioxide (NO₂)": "nitrogen_dioxide",
            "Ozone (O₃)": "ozone",
            "Particulate Matter (PM10)": "pm10",
            "Fine Particles (PM2.5)": "pm2_5",
        }
        selected_label = st.selectbox("Select Pollutant", list(pollutant_map.keys()))
        target_col = pollutant_map[selected_label]

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

    if valid_models_found == 0:
        st.caption(f"No models currently available for {selected_label}.")

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

    zoom_start = current_time  # - pd.Timedelta(hours=24)
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

    st.plotly_chart(fig, use_container_width=True)
    st.info(
        f"ℹ️ Guidelines based on European Environment Agency (EEA) standards for {selected_label}."
    )
