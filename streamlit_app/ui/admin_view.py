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

            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / 1024 / 1024
            total, used, free = shutil.disk_usage(".")

            c1, c2, c3 = st.columns(3)
            c1.metric("App RAM", f"{int(mem_mb)} MB")
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
                k3.metric("Last Update", stats.get("last_run", "N/A").split(" ")[-1])
            else:
                st.warning("⚠️ No pipeline statistics found.")


def render_admin_dashboard(df_history, preds, horizon_metrics, history_metrics, cfg):
    st.title("Admin Dashboard")
    show_system_stats(cfg["deployment"]["system_usage_path"])

    if not horizon_metrics:
        st.error("⚠️ Metrics missing. Run backend pipeline.")
        return

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

            with tab1:
                # Restoration: CI Toggle
                show_ci = st.toggle(
                    f"Show Confidence Intervals ({col})", value=False, key=f"ci_{col}"
                )

                fig = go.Figure()
                colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

                for i, (m, data) in enumerate(horizon_metrics[col].items()):
                    if not data:
                        continue
                    df = pd.DataFrame(data)
                    color = colors[i % len(colors)]

                    # Main Line
                    fig.add_trace(
                        go.Scatter(
                            x=df["step"],
                            y=df["RMSE_mean"],
                            mode="lines",
                            name=m,
                            line=dict(color=color),
                        )
                    )

                    # Restoration: Shaded Confidence Interval
                    if show_ci and "RMSE_upper" in df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=pd.concat([df["step"], df["step"][::-1]]),
                                y=pd.concat([df["RMSE_upper"], df["RMSE_lower"][::-1]]),
                                fill="toself",
                                fillcolor=color,
                                opacity=0.15,
                                line=dict(color="rgba(255,255,255,0)"),
                                hoverinfo="skip",
                                showlegend=False,
                            )
                        )

                fig.update_layout(
                    title=f"RMSE vs Horizon ({label})",
                    xaxis_title="Horizon (Hours)",
                    height=400,
                    hovermode="x unified",
                )
                st.plotly_chart(fig, width="stretch")

            with tab2:
                # SMAPE usually doesn't need CI in this context, keeping it simple
                fig = go.Figure()
                for m, data in horizon_metrics[col].items():
                    if data:
                        df = pd.DataFrame(data)
                        fig.add_trace(
                            go.Scatter(x=df["step"], y=df["SMAPE_mean"], name=m)
                        )
                fig.update_layout(
                    title=f"SMAPE vs Horizon ({label})",
                    xaxis_title="Horizon (Hours)",
                    height=400,
                    hovermode="x unified",
                )
                st.plotly_chart(fig, width="stretch")

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
            st.plotly_chart(fig, width="stretch")

    with st.expander("Inspect Raw Data"):
        st.subheader("History (Last 5)")
        st.dataframe(df_history.tail(5).astype(str))
        st.subheader("Latest Forecasts")
        for m, df in preds.items():
            st.write(f"**{m}**")
            st.dataframe(df.head(5).astype(str))
