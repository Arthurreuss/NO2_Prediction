import json
import os
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import streamlit as st


def get_dir_size(start_path: str = ".") -> float:
    """Recursively calculates total size of the application directory in MB.

    Args:
        start_path (str): The root directory to start calculating size from.
            Defaults to current directory.

    Returns:
        float: The total size of the directory in Megabytes (MB).
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / 1024 / 1024


def show_system_stats(sys_stats_path: str) -> None:
    """Displays real-time container health and pipeline stats with Free Tier limits.

    Calculates memory usage based on the current process (container) rather than
    the total system memory, ensuring accurate monitoring for Hugging Face Spaces.

    Args:
        sys_stats_path (str): Path to the JSON file containing GitHub Action
            pipeline statistics.
    """
    with st.expander("System & Pipeline Stats", expanded=True):

        with st.container(border=True):
            st.markdown("### HF Space (Free Tier)")

            # 1. RAM: Limit is 16GB
            process = psutil.Process(os.getpid())
            mem_used_mb = process.memory_info().rss / 1024 / 1024
            hf_ram_limit = 16 * 1024  # 16 GB Hard Limit
            ram_percent = (mem_used_mb / hf_ram_limit) * 100

            # 2. Disk: Limit is 50GB
            app_size_mb = get_dir_size(".")
            disk_percent = (app_size_mb / (50 * 1024)) * 100

            # 3. CPU: Limit is 2 vCPUs (Standard Free Tier)
            cpu_usage = psutil.cpu_percent()

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "RAM Usage", f"{int(mem_used_mb)} MB", f"{ram_percent:.1f}% of 16 GB"
            )
            c2.metric(
                "Disk Usage", f"{app_size_mb:.0f} MB", f"{disk_percent:.1f}% of 50 GB"
            )
            c3.metric("CPU Load", f"{cpu_usage}%", f"of 2 vCPUs")

        with st.container(border=True):
            st.markdown("### GitHub Action (Standard Runner)")

            if os.path.exists(sys_stats_path):
                with open(sys_stats_path, "r") as f:
                    stats = json.load(f)
                peaks = stats.get("peak_metrics", {})

                k1, k2, k3 = st.columns(3)

                # 1. RAM (Standard Runner is ~7GB)
                max_ram = peaks.get("max_ram_mb", 0)
                gh_ram_limit = 7000
                ram_pct = (max_ram / gh_ram_limit) * 100

                k1.metric("Peak RAM", f"{max_ram:.0f} MB", f"{ram_pct:.1f}% of 7 GB")

                # 2. CPU (Standard Runner is 2 vCPUs)
                avg_cpu = peaks.get("cpu_percent", 0)
                k2.metric("Avg CPU", f"{avg_cpu}", "of 2 vCPUs")

                # 3. Data
                k3.metric("New Data", f"{peaks.get('data_size_mb', 0)} MB", "Generated")

                st.caption(
                    f"Status: **{stats.get('status', 'UNKNOWN').upper()}** | Duration: **{stats.get('duration_seconds', 0)}s** | Last Run: **{stats.get('last_run', 'N/A').split(' ')[-1]}**"
                )
            else:
                st.warning("⚠️ No pipeline statistics found.")


def render_admin_dashboard(
    df_history: pd.DataFrame,
    preds: Dict[str, pd.DataFrame],
    horizon_metrics: Dict[str, Any],
    history_metrics: Dict[str, Any],
    cfg: Dict[str, Any],
) -> None:
    """Renders the complete Admin Dashboard, including system stats and model metrics.

    Args:
        df_history (pd.DataFrame): Historical observation data.
        preds (Dict[str, pd.DataFrame]): Dictionary mapping model names to forecast DataFrames.
        horizon_metrics (Dict[str, Any]): Dictionary containing RMSE/SMAPE metrics per horizon step.
        history_metrics (Dict[str, Any]): Dictionary containing historical model performance metrics.
        cfg (Dict[str, Any]): Configuration dictionary containing file paths and settings.
    """
    st.title("Admin Dashboard")
    show_system_stats(cfg["deployment"]["system_usage_path"])

    if not horizon_metrics:
        st.error("Metrics missing. Run backend pipeline.")
        return

    pollutants = {
        "NO₂": "nitrogen_dioxide",
        "O₃": "ozone",
        "PM10": "pm10",
        "PM2.5": "pm2_5",
    }

    for label, col in pollutants.items():
        st.markdown(f"--- \n### {label}")

        if col in horizon_metrics:
            st.caption("Horizon Analysis (Average Error per Step)")
            tab1, tab2 = st.tabs(["RMSE (Error)", "SMAPE (%)"])

            with tab1:
                models = list(horizon_metrics[col].keys())
                if models:
                    cols = st.columns(len(models))
                    for idx, m in enumerate(models):
                        data = horizon_metrics[col][m]
                        if data:
                            df_m = pd.DataFrame(data)
                            df_m = df_m[df_m["step"] > 0]
                            avg_rmse = df_m["RMSE_mean"].mean()
                            cols[idx].metric(f"{m} Avg", f"{avg_rmse:.2f}")

                show_ci = st.toggle(
                    f"Show Confidence Intervals ({col})", value=False, key=f"ci_{col}"
                )

                fig = go.Figure()
                colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

                for i, (m, data) in enumerate(horizon_metrics[col].items()):
                    if not data:
                        continue
                    df = pd.DataFrame(data)
                    df = df[df["step"] > 0]

                    color = colors[i % len(colors)]

                    fig.add_trace(
                        go.Scatter(
                            x=df["step"],
                            y=df["RMSE_mean"],
                            mode="lines",
                            name=m,
                            line=dict(color=color),
                        )
                    )

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
                models = list(horizon_metrics[col].keys())
                if models:
                    cols = st.columns(len(models))
                    for idx, m in enumerate(models):
                        data = horizon_metrics[col][m]
                        if data:
                            df_m = pd.DataFrame(data)
                            df_m = df_m[df_m["step"] > 0]
                            avg_smape = df_m["SMAPE_mean"].mean()
                            cols[idx].metric(f"{m} Avg", f"{avg_smape:.1f}%")

                fig = go.Figure()
                for m, data in horizon_metrics[col].items():
                    if data:
                        df = pd.DataFrame(data)
                        df = df[df["step"] > 0]
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

        if col in history_metrics and history_metrics[col]:
            st.caption("Model Stability (Performance over Time)")
            df_perf = pd.DataFrame(history_metrics[col])

            stab_tab1, stab_tab2 = st.tabs(["RMSE (Error)", "SMAPE (%)"])

            with stab_tab1:
                fig = px.line(
                    df_perf,
                    x="prediction_generated_at",
                    y="RMSE",
                    color="Model",
                    markers=True,
                    title=f"Stability: RMSE over Time ({label})",
                )
                st.plotly_chart(fig, width="stretch")

            with stab_tab2:
                if "SMAPE" in df_perf.columns:
                    fig = px.line(
                        df_perf,
                        x="prediction_generated_at",
                        y="SMAPE",
                        color="Model",
                        markers=True,
                        title=f"Stability: SMAPE over Time ({label})",
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("SMAPE history not available.")

    with st.expander("Inspect Raw Data"):
        st.subheader("History (Last 5)")
        st.dataframe(df_history.tail(5).astype(str))
        st.subheader("Latest Forecasts")
        for m, df in preds.items():
            st.write(f"**{m}**")
            st.dataframe(df.head(5).astype(str))
