import json
import os
import shutil

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import streamlit as st


def get_dir_size(start_path="."):
    """Recursively calculates total size of the application directory in MB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / 1024 / 1024  # Convert bytes to MB


def show_system_stats(sys_stats_path: str):
    """Displays real-time container health and pipeline stats with CORRECT Free Tier limits."""
    with st.expander("System & Pipeline Stats", expanded=True):

        # --- LEFT: Hugging Face Container (Real-Time) ---
        with st.container(border=True):
            st.markdown("### ☁️ HF Space (Free Tier)")

            # 1. RAM: Limit is 16GB
            process = psutil.Process(os.getpid())
            mem_used_mb = process.memory_info().rss / 1024 / 1024
            hf_ram_limit = 16 * 1024  # 16 GB Hard Limit
            ram_percent = (mem_used_mb / hf_ram_limit) * 100

            # 2. Disk: Limit is 50GB
            app_size_mb = get_dir_size(".")
            disk_percent = (app_size_mb / (50 * 1024)) * 100

            # 3. CPU: Limit is 2 vCPUs (Standard Free Tier)
            # psutil.cpu_percent() gives us usage relative to the allocated quota in containers
            cpu_usage = psutil.cpu_percent()

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "RAM Usage", f"{int(mem_used_mb)} MB", f"{ram_percent:.1f}% of 16 GB"
            )
            c2.metric(
                "Disk Usage", f"{app_size_mb:.0f} MB", f"{disk_percent:.1f}% of 50 GB"
            )
            c3.metric("CPU Load", f"{cpu_usage}%", f"of 2 vCPUs")

        # --- RIGHT: GitHub Pipeline (Job Metrics) ---
        with st.container(border=True):
            st.markdown("### 🚀 GitHub Action (Standard Runner)")

            if os.path.exists(sys_stats_path):
                with open(sys_stats_path, "r") as f:
                    stats = json.load(f)
                peaks = stats.get("peak_metrics", {})

                k1, k2, k3 = st.columns(3)

                # 1. RAM (Standard Runner is ~7GB)
                max_ram = peaks.get("max_ram_mb", 0)
                gh_ram_limit = 7000  # ~7 GB
                ram_pct = (max_ram / gh_ram_limit) * 100

                k1.metric("Peak RAM", f"{max_ram:.0f} MB", f"{ram_pct:.1f}% of 7 GB")

                # 2. CPU (Standard Runner is 2 vCPUs)
                avg_cpu = peaks.get("cpu_percent", 0)
                # If avg_cpu is > 100%, it means we used >1 core.
                # Max possible is 200% (2 cores).
                k2.metric("Avg CPU", f"{avg_cpu}%", "of 2 vCPUs")

                # 3. Data
                k3.metric("New Data", f"{peaks.get('data_size_mb', 0)} MB", "Generated")

                st.caption(
                    f"Status: **{stats.get('status', 'UNKNOWN').upper()}** | Duration: **{stats.get('duration_seconds', 0)}s** | Last Run: **{stats.get('last_run', 'N/A').split(' ')[-1]}**"
                )
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
