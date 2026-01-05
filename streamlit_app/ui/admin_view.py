import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil

import streamlit_app as st
from utils.alerts import check_and_alert_health
from utils.model_evaluation import get_horizon_metrics, get_performance_over_time


def safe_display_df(df: pd.DataFrame, limit: int = 5) -> None:
    """Converts datetime columns to string and displays dataframe in Streamlit.

    Prevents PyArrow/Streamlit crashes associated with specific timezones (e.g.,
    'Europe/Amsterdam') by converting time-based columns to strings before rendering.

    Args:
        df: The DataFrame to display.
        limit: Maximum number of rows to display. Defaults to 500.
    """
    if df is None or df.empty:
        st.write("Empty DataFrame")
        return

    display_df = df.head(limit).copy()

    for col in display_df.select_dtypes(include=["datetime", "datetimetz"]).columns:
        display_df[col] = display_df[col].astype(str)

    for col in ["time", "prediction_generated_at"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].astype(str)

    st.dataframe(display_df)


def plot_horizon_metric(
    data_dict: dict, metric_col: str, title: str, y_label: str, show_ci: bool = False
) -> go.Figure:
    """Generates a Plotly line chart for horizon metrics across different models.

    Args:
        data_dict: Dictionary mapping model names to their performance DataFrames.
        metric_col: The column name in the DataFrames to plot (e.g., 'RMSE_mean').
        title: The title of the plot.
        y_label: The label for the Y-axis.
        show_ci: Whether to show the confidence interval shading (RMSE only).
            Defaults to False.

    Returns:
        go.Figure: A Plotly graph object containing the aggregated step view.
    """
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, (model_name, df) in enumerate(data_dict.items()):
        if df is None or df.empty:
            continue
        color = colors[i % len(colors)]

        df = df.sort_values("step")

        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df[metric_col],
                mode="lines",
                name=model_name,
                line=dict(color=color, width=2),
            )
        )

        if show_ci and "RMSE" in metric_col:
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
        title=title,
        xaxis_title="Horizon (Hours Ahead)",
        yaxis_title=y_label,
        hovermode="x unified",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def get_dir_size_mb(start_path: str) -> float:
    """Recursively calculates the size of a directory in MB.

    Args:
        start_path (str): The root directory path to begin scanning.

    Returns:
        float: The total size of the directory in megabytes.
    """
    total_size = 0
    for dirpath, _, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / 1024 / 1024


def show_system_and_pipeline_stats(sys_stats_path: str) -> None:
    """Displays system resource usage and pipeline statistics in a Streamlit expander.

    This function creates two main visual cards:
    1. Hugging Face Space Status: Shows real-time RAM, CPU, and Disk usage of the container.
    2. GitHub Actions Status: Shows the results of the latest automated pipeline run (status, duration, runner stats).

    Args:
        sys_stats_path (str): The file path to the JSON file containing statistics
            from the latest GitHub Actions run.
    """
    with st.expander("System & Pipeline Stats", expanded=False):
        with st.container(border=True):
            st.markdown("Hugging Face Space Status")

            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            curr_ram = mem_info.rss / 1024 / 1024
            sys_mem = psutil.virtual_memory()

            project_root = os.getcwd()
            repo_size_mb = get_dir_size_mb(project_root)
            repo_size_gb = repo_size_mb / 1024
            hf_limit_gb = 50.0

            c1, c2, c3 = st.columns(3)
            c1.metric("RAM Usage", f"{sys_mem.percent}%", f"{int(curr_ram)} MB (App)")
            c2.metric("CPU Usage", f"{psutil.cpu_percent(interval=0.1)}%")
            c3.metric(
                "Disk Usage",
                f"{repo_size_mb:.0f} MB",
                f"{repo_size_gb:.2f} / {hf_limit_gb} GB",
            )

            st.caption(f"Environment ID: {os.environ.get('SPACE_ID', 'Local/Unknown')}")

        with st.container(border=True):
            st.markdown("GitHub Actions Status")

            if os.path.exists(sys_stats_path):
                with open(sys_stats_path, "r") as f:
                    stats = json.load(f)

                k1, k2, k3 = st.columns(3)

                status = stats.get("status", "UNKNOWN")
                if status.lower() == "success":
                    k1.success(f"**{status.upper()}**")
                else:
                    k1.error(f"**{status.upper()}**")

                k2.metric("Duration", f"{stats.get('duration_seconds', 0)}s")
                k3.metric(
                    "Last Run",
                    stats.get("last_run", "N/A").split(" ")[1][:5] + " (CET)",
                )

                st.divider()

                m1, m2, m3 = st.columns(3)
                sys_metrics = stats.get("system_metrics", {})

                m1.metric("Runner OS", stats.get("runner_os", "Linux"))

                mem = sys_metrics.get("memory_available", "N/A")
                mem_disp = mem if len(mem) < 10 else "View JSON"
                m2.metric("Runner RAM (Avail)", mem_disp)

                disk = sys_metrics.get("disk_free", "N/A")
                disk_disp = disk if len(disk) < 10 else "View JSON"
                m3.metric("Runner Disk (Free)", disk_disp)

            else:
                st.warning("No pipeline statistics found yet.")


def render_admin_dashboard(
    df_history: pd.DataFrame, preds: dict, preds_all: dict, cfg: dict
) -> None:
    """Renders the complete Admin Dashboard Streamlit interface.

    Orchestrates health checks, system stats display, pollutant performance analysis,
    and data inspection tabs.

    Args:
        df_history: DataFrame containing historical air quality data.
        preds: Dictionary of DataFrames containing recent model predictions.
        preds_all: Dictionary of DataFrames containing all model predictions (raw).
        cfg: Configuration dictionary containing deployment paths and settings.
    """
    sys_stats_path = cfg["deployment"]["system_usage_path"]
    alert_file_path = cfg["deployment"]["alert_file"]
    st.title("Admin Dashboard")

    check_and_alert_health(df_history, sys_stats_path, alert_file_path)

    show_system_and_pipeline_stats(sys_stats_path)

    pollutants = {
        "NO₂ (Nitrogen Dioxide)": "nitrogen_dioxide",
        "O₃ (Ozone)": "ozone",
        "PM10": "pm10",
        "PM2.5": "pm2_5",
    }

    for pol_label, target_col in pollutants.items():
        st.markdown("---")
        st.header(f"{pol_label}")

        agg_metrics = {}
        for model_name, df_all_predictions in preds_all.items():
            if target_col != "nitrogen_dioxide" and model_name == "GRU":
                continue
            stats_all, _ = get_horizon_metrics(
                df_history, df_all_predictions, pollutant=target_col
            )
            if stats_all is not None:
                agg_metrics[model_name] = stats_all

        df_perf_history = get_performance_over_time(
            df_history, preds_all, pollutant=target_col
        )
        if not df_perf_history.empty:
            df_perf_history["prediction_generated_at"] = df_perf_history[
                "prediction_generated_at"
            ].dt.tz_convert("Europe/Amsterdam")

        st.subheader(f"Global Horizon Analysis ({pol_label})")
        st.caption(
            "Error vs. Forecast Horizon (1-72h). Shaded area shows approximate confidence (RMSE +/- StdDev)."
        )

        tab1, tab2 = st.tabs([f"RMSE - {pol_label}", f"SMAPE - {pol_label}"])
        with tab1:
            toggle = st.toggle("CI Toggle", key=f"ci_toggle_{target_col}")
            if agg_metrics:
                fig = plot_horizon_metric(
                    agg_metrics,
                    "RMSE_mean",
                    f"Avg RMSE per Horizon Step ({pol_label})",
                    "RMSE",
                    show_ci=toggle,
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No model data available for this metric.")

        with tab2:
            if agg_metrics:
                fig = plot_horizon_metric(
                    agg_metrics,
                    "SMAPE_mean",
                    f"Avg SMAPE per Horizon Step ({pol_label})",
                    "SMAPE (%)",
                    show_ci=False,
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No model data available for this metric.")

        st.subheader(f"Performance Evolution ({pol_label})")

        if not df_perf_history.empty:
            tab_ev1, tab_ev2 = st.tabs(["RMSE History", "SMAPE History"])

            with tab_ev1:
                fig_ev_rmse = px.line(
                    df_perf_history,
                    x="prediction_generated_at",
                    y="RMSE",
                    color="Model",
                    title=f"RMSE per Forecast Run ({pol_label})",
                    markers=True,
                )
                fig_ev_rmse.update_layout(
                    xaxis_title="Run Time", yaxis_title="Average RMSE"
                )
                st.plotly_chart(fig_ev_rmse, width="stretch")

            with tab_ev2:
                fig_ev_smape = px.line(
                    df_perf_history,
                    x="prediction_generated_at",
                    y="SMAPE",
                    color="Model",
                    title=f"SMAPE per Forecast Run ({pol_label})",
                    markers=True,
                )
                fig_ev_smape.update_layout(
                    xaxis_title="Run Time", yaxis_title="Average SMAPE (%)"
                )
                st.plotly_chart(fig_ev_smape, width="stretch")
        else:
            st.info("Not enough historical data yet (need completed 72h runs).")

    st.markdown("---")

    with st.expander("1. Inspect API Data (History)", expanded=False):
        if df_history.empty:
            st.error("API History DataFrame is empty!")
        else:
            st.write(f"Rows: {len(df_history)}")
            safe_display_df(df_history)

    with st.expander("2. Inspect Predictions", expanded=False):
        if not preds:
            st.error("No prediction models found!")
        for model_name, df_p in preds.items():
            st.subheader(f"Model: {model_name}")
            safe_display_df(df_p)

    with st.expander("3. Inspect All Predictions (Raw)", expanded=False):
        if not preds_all:
            st.error("No prediction models found!")
        for model_name, df_p in preds_all.items():
            st.subheader(f"Model: {model_name}")
            safe_display_df(df_p)
