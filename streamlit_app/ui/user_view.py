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

    if not df_history.empty:
        valid_hist = df_history.dropna(subset=[col_name])
        if not valid_hist.empty:
            latest = valid_hist.iloc[-1]
            val = latest[col_name]
            cat, color, desc = get_aqi_category(val, col_name)

            st.markdown(
                f"""
                <div style="background-color:{color};padding:20px;border-radius:10px;color:white;text-align:center;margin-bottom:25px">
                    <h3 style="margin:0">Current Status: {cat}</h3>
                    <h1 style="margin:0;font-size:3em">{val:.1f} µg/m³</h1>
                    <p style="margin:0">{desc}</p>
                </div>
            """,
                unsafe_allow_html=True,
            )

    fig = go.Figure()
    current_time = pd.Timestamp.now(tz="UTC")
    cutoff_date = current_time - pd.Timedelta(days=7)
    max_y_val = 0

    if not df_history.empty:
        hist_plot = df_history[df_history["time"] > cutoff_date].copy()
        if not hist_plot.empty and col_name in hist_plot.columns:
            hist_plot = hist_plot.dropna(subset=[col_name]).sort_values("time")
            if not hist_plot.empty:
                max_y_val = max(max_y_val, hist_plot[col_name].max())

                fig.add_trace(
                    go.Scatter(
                        x=hist_plot["time"],
                        y=hist_plot[col_name],
                        name="Observed History",
                        line=dict(color="white", width=2),
                        showlegend=True,
                    )
                )

    colors = ["#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692"]
    for i, model_name in enumerate(models):
        df_p = preds.get(model_name)
        if df_p is None or df_p.empty or col_name not in df_p.columns:
            continue

        df_view = df_p[df_p["time"] > cutoff_date].copy().sort_values("time")
        if not df_view.empty:
            max_y_val = max(max_y_val, df_view[col_name].max())

            fig.add_trace(
                go.Scatter(
                    x=df_view["time"],
                    y=df_view[col_name],
                    name=model_name,
                    line=dict(color=colors[i % len(colors)], width=3),
                    showlegend=True,
                )
            )

    th = get_aqi_thresholds(col_name)
    bg_colors = ["#50F0E6", "#50CCAA", "#F0E641", "#FF5050", "#960032", "#7D2181"]

    final_top_limit = max(max_y_val * 1.1, th[2])

    prev = 0
    for i in range(5):
        if prev < final_top_limit:
            fig.add_hrect(
                y0=prev,
                y1=th[i],
                fillcolor=bg_colors[i],
                opacity=0.15,
                layer="below",
                line_width=0,
            )
        prev = th[i]
    if prev < final_top_limit:
        fig.add_hrect(
            y0=th[4],
            y1=max(th[4] * 2, final_top_limit),
            fillcolor=bg_colors[5],
            opacity=0.15,
            layer="below",
            line_width=0,
        )

    aqi_labels = [
        (f"Good (<{th[0]})", bg_colors[0]),
        (f"Fair ({th[0]}-{th[1]})", bg_colors[1]),
        (f"Moderate ({th[1]}-{th[2]})", bg_colors[2]),
        (f"Poor ({th[2]}-{th[3]})", bg_colors[3]),
        (f"Very Poor ({th[3]}-{th[4]})", bg_colors[4]),
        (f"Extremely Poor (>{th[4]})", bg_colors[5]),
    ]

    for label, color in aqi_labels:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color, symbol="square"),
                name=label,
                showlegend=True,
                hoverinfo="none",
            )
        )

    fig.add_vline(
        x=current_time.timestamp() * 1000,
        line_dash="dot",
        annotation_text="NOW",
        line_color="gray",
    )

    fig.update_layout(
        height=600,
        title=f"{label} Forecast",
        xaxis_title="Time (UTC)",
        yaxis_title="µg/m³",
        yaxis_range=[0, final_top_limit],
        xaxis=dict(
            type="date",
            rangeslider=dict(visible=True),
            range=[current_time, current_time + pd.Timedelta(hours=48)],
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.info(
        f"ℹ️ Guidelines based on European Environment Agency (EEA) standards for {label}."
    )
