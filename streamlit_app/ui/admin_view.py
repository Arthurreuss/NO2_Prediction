import json
import os
import shutil

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import streamlit as st


def show_system_stats(sys_stats_path: str):
    """Displays real-time container health and latest pipeline status."""
    with st.expander("System & Pipeline Stats", expanded=True):

        # --- Container Health ---
        with st.container(border=True):
            st.markdown("### ☁️ Container Health")

            # RAM (Current Process)
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / 1024 / 1024

            # Disk Usage
            total, used, free = shutil.disk_usage(".")

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "App RAM",
                f"{int(mem_mb)} MB",
                help="RAM used by this Streamlit process",
            )
            c2.metric("Disk Free", f"{free / (1024**3):.1f} GB")
            c3.metric("CPU Load", f"{psutil.cpu_percent()}%")

        # --- Pipeline Status ---
        with st.container(border=True):
            st.markdown("### 🚀 Data Pipeline Status")

            if os.path.exists(sys_stats_path):
                with open(sys_stats_path, "r") as f:
                    stats = json.load(f)

                k1, k2, k3 = st.columns(3)
                status = stats.get("status", "UNKNOWN")

                k1.metric(
                    "Status",
                    status.upper(),
                    delta_color="normal" if status == "success" else "inverse",
                )
                k2.metric("Duration", f"{stats.get('duration_seconds', 0)}s")
                k3.metric(
                    "Last Update", stats.get("last_run", "N/A").split(" ")[-1]
                )  # Show time only
            else:
                st.warning("⚠️ No pipeline statistics found.")


def render_admin_dashboard(df_history, preds, horizon_metrics, history_metrics, cfg):
    st.title("Admin Dashboard")

    # 1. System Stats
    show_system_stats(cfg["deployment"]["system_usage_path"])

    # 2. Check Data
    if not horizon_metrics:
        st.error("⚠️ Pre-computed metrics missing. Please run the backend pipeline.")
        return

    # 3. Visuals per Pollutant
    pollutants = {
        "NO₂": "nitrogen_dioxide",
        "O₃": "ozone",
        "PM10": "pm10",
        "PM2.5": "pm2_5",
    }

    for label, col in pollutants.items():
        st.markdown(f"--- \n### {label}")

        # --- Horizon Analysis ---
        if col in horizon_metrics:
            tab1, tab2 = st.tabs(["RMSE (Error)", "SMAPE (%)"])

            def plot_horizon(metric_key, title):
                fig = go.Figure()
                for m, data in horizon_metrics[col].items():
                    if data:
                        df = pd.DataFrame(data)
                        fig.add_trace(
                            go.Scatter(
                                x=df["step"], y=df[metric_key], mode="lines", name=m
                            )
                        )
                fig.update_layout(
                    title=title,
                    xaxis_title="Horizon (Hours)",
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                return fig

            with tab1:
                st.plotly_chart(
                    plot_horizon("RMSE_mean", f"RMSE vs Horizon ({label})"),
                    use_container_width=True,
                )
            with tab2:
                st.plotly_chart(
                    plot_horizon("SMAPE_mean", f"SMAPE vs Horizon ({label})"),
                    use_container_width=True,
                )

        # --- Stability Over Time ---
        if col in history_metrics and history_metrics[col]:
            df_perf = pd.DataFrame(history_metrics[col])
            fig = px.line(
                df_perf,
                x="prediction_generated_at",
                y="RMSE",
                color="Model",
                markers=True,
                title=f"Model Stability ({label})",
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    # 4. Data Inspector
    with st.expander("Inspect Raw Data"):
        st.subheader("History (Last 5)")
        st.dataframe(df_history.tail(5).astype(str))

        st.subheader("Latest Forecasts (First 5)")
        for m, df in preds.items():
            st.write(f"**{m}**")
            st.dataframe(df.head(5).astype(str))
