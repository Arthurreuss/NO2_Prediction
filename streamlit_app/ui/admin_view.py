import json
import os
import shutil

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import streamlit as st


def show_system_stats(sys_stats_path: str):
    """Displays real-time container health and peak pipeline stats."""
    with st.expander("System & Pipeline Stats", expanded=True):

        # --- LEFT: Real-Time Hugging Face Container ---
        with st.container(border=True):
            st.markdown("### ☁️ HF Container (Real-Time)")

            # 1. RAM (Current Process vs 16GB Limit)
            process = psutil.Process(os.getpid())
            mem_used_mb = process.memory_info().rss / 1024 / 1024

            # Auto-detect cgroup limit or default to 16GB
            limit_mb = 16 * 1024
            try:
                for path in [
                    "/sys/fs/cgroup/memory.max",
                    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                ]:
                    if os.path.exists(path):
                        with open(path, "r") as f:
                            val = int(f.read().strip())
                            if val < 10**15:
                                limit_mb = val / 1024 / 1024
                                break
            except:
                pass

            ram_percent = (mem_used_mb / limit_mb) * 100

            # 2. Disk
            total, used, free = shutil.disk_usage(".")
            used_gb = used / (1024**3)
            disk_percent = (used / total) * 100

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "App RAM",
                f"{int(mem_used_mb)} MB",
                f"{ram_percent:.1f}% of {int(limit_mb/1024)}GB",
            )
            c2.metric("Disk Used", f"{used_gb:.1f} GB", f"{disk_percent:.1f}%")
            c3.metric("CPU Load", f"{psutil.cpu_percent()}%", "Instant")

        # --- RIGHT: GitHub Pipeline Peak Stats ---
        with st.container(border=True):
            st.markdown("### 🚀 GitHub Pipeline (Peak Metrics)")

            if os.path.exists(sys_stats_path):
                with open(sys_stats_path, "r") as f:
                    stats = json.load(f)

                # Get the nested metrics safely
                peaks = stats.get("peak_metrics", {})

                k1, k2, k3 = st.columns(3)

                # 1. Max RAM used during inference
                max_ram = peaks.get("max_ram_mb", 0)
                k1.metric("Peak RAM", f"{max_ram:.0f} MB", "Max Usage")

                # 2. Max CPU Load during inference
                cpu_load = peaks.get("cpu_percent", "N/A")
                k2.metric("Peak CPU", f"{cpu_load}", "% allocated")

                # 3. Disk Space Consumed by Data Folder
                data_size = peaks.get("data_size_mb", 0)
                k3.metric("Data Size", f"{data_size} MB", "Generated")

                # Context Info (Bottom Line)
                st.caption(
                    f"Status: **{stats.get('status', 'UNKNOWN').upper()}** | Duration: **{stats.get('duration_seconds', 0)}s** | Updated: **{stats.get('last_run', 'N/A').split(' ')[-1]}**"
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
