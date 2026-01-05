import json
import os
import shutil

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import streamlit as st


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
    with st.expander("System & Pipeline Stats", expanded=True):

        # --- TEIL 1: Hugging Face Container Status (Real Data) ---
        with st.container(border=True):
            st.markdown("### ☁️ Hugging Face Container Health")

            # A. RAM: Wir messen nur DEINEN Prozess, nicht den ganzen Server
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            app_ram_mb = mem_info.rss / 1024 / 1024

            # Hugging Face Free Tier hat in der Regel 16 GB (= 16384 MB)
            # Wir nutzen das als Basis für die Prozentanzeige
            HF_RAM_LIMIT_MB = 16 * 1024
            ram_percent = (app_ram_mb / HF_RAM_LIMIT_MB) * 100

            # B. DISK: Echter Speicherplatz im Container
            total, used, free = shutil.disk_usage(".")
            used_gb = used / (1024**3)
            total_gb = total / (1024**3)
            disk_percent = (used / total) * 100

            # C. CPU
            cpu_usage = psutil.cpu_percent(interval=0.1)

            # Darstellung in Spalten
            c1, c2, c3 = st.columns(3)

            c1.metric(
                label="App RAM Usage",
                value=f"{int(app_ram_mb)} MB",
                delta=f"{ram_percent:.1f}% of 16GB Limit",
                delta_color="normal" if ram_percent < 50 else "inverse",
            )

            c2.metric(
                label="Container Disk Usage",
                value=f"{used_gb:.1f} GB",
                delta=f"{free / (1024**3):.1f} GB free",
                delta_color="normal" if disk_percent < 80 else "inverse",
            )

            c3.metric(
                label="CPU Load (Instant)",
                value=f"{cpu_usage}%",
                help="Momentane CPU-Last. Kann auf Shared-Runnern schwanken.",
            )

            # Visualisierung als Progress Bars (für schnellen Check)
            st.caption("RAM Usage (App vs Limit)")
            st.progress(min(ram_percent / 100, 1.0))

        # --- TEIL 2: GitHub Pipeline Status ---
        with st.container(border=True):
            st.markdown("### 🚀 GitHub Actions Pipeline (Data Update)")

            if os.path.exists(sys_stats_path):
                with open(sys_stats_path, "r") as f:
                    stats = json.load(f)

                k1, k2, k3 = st.columns(3)

                # Status mit Farbe
                status = stats.get("status", "UNKNOWN")
                last_run_str = stats.get("last_run", "N/A")

                if status.lower() == "success":
                    k1.success(f"Status: {status.upper()}")
                else:
                    k1.error(f"Status: {status.upper()}")

                # Dauer
                duration = stats.get("duration_seconds", 0)
                k2.metric("Build Duration", f"{duration}s")

                # Zeitpunkt
                # Wir extrahieren nur die Uhrzeit für bessere Lesbarkeit
                if " " in last_run_str:
                    date_part, time_part = last_run_str.split(" ")
                    display_time = f"{time_part[:5]} ({date_part})"
                else:
                    display_time = last_run_str

                k3.metric("Last Update (CET)", display_time)

                st.divider()
                st.caption("Github Runner Telemetry")

                sys_metrics = stats.get("system_metrics", {})
                m1, m2, m3 = st.columns(3)
                m1.markdown(f"**OS:** `{stats.get('runner_os', 'Linux')}`")
                m2.markdown(
                    f"**Runner RAM Avail:** `{sys_metrics.get('memory_available', 'N/A')}`"
                )
                m3.markdown(
                    f"**Runner Disk Free:** `{sys_metrics.get('disk_free', 'N/A')}`"
                )

            else:
                st.warning(
                    "⚠️ No pipeline statistics found yet. Waiting for first run..."
                )


def render_admin_dashboard(df_history: pd.DataFrame, preds: dict, cfg: dict) -> None:
    """Renders the complete Admin Dashboard Streamlit interface.

    Orchestrates health checks, system stats display, pollutant performance analysis,
    and data inspection tabs.

    Args:
        df_history: DataFrame containing historical air quality data.
        preds: Dictionary of DataFrames containing recent model predictions.
        cfg: Configuration dictionary containing deployment paths and settings.
    """
    sys_stats_path = cfg["deployment"]["system_usage_path"]
    metrics_dir = "data/deployment/metrics"  # Pfad zu den neuen JSONs

    st.title("Admin Dashboard")
    show_system_and_pipeline_stats(sys_stats_path)

    # Lade die vorberechneten Daten
    try:
        with open(f"{metrics_dir}/horizon_metrics.json", "r") as f:
            horizon_metrics = json.load(f)
        with open(f"{metrics_dir}/history_metrics.json", "r") as f:
            history_metrics = json.load(f)
    except FileNotFoundError:
        st.error("⚠️ Pre-computed metrics not found. Wait for the next pipeline run.")
        return

    pollutants = {
        "NO₂ (Nitrogen Dioxide)": "nitrogen_dioxide",
        "O₃ (Ozone)": "ozone",
        "PM10": "pm10",
        "PM2.5": "pm2_5",
    }

    for pol_label, target_col in pollutants.items():
        st.markdown("---")
        st.header(f"{pol_label}")

        # Daten für Horizon Plot vorbereiten (Dict -> DataFrame Rückwandlung)
        # Das geht blitzschnell und braucht kaum RAM
        agg_metrics = {}
        if target_col in horizon_metrics:
            for model_name, data_list in horizon_metrics[target_col].items():
                if data_list:
                    agg_metrics[model_name] = pd.DataFrame(data_list)

        # Daten für History Plot vorbereiten
        df_perf_history = pd.DataFrame()
        if target_col in history_metrics and history_metrics[target_col]:
            df_perf_history = pd.DataFrame(history_metrics[target_col])
            # Strings zurück zu Datetime für den Plot
            df_perf_history["prediction_generated_at"] = pd.to_datetime(
                df_perf_history["prediction_generated_at"]
            )

        st.subheader(f"Global Horizon Analysis ({pol_label})")

        # ... (Dein Tab Code bleibt fast gleich, nutzt aber jetzt agg_metrics)
        tab1, tab2 = st.tabs([f"RMSE - {pol_label}", f"SMAPE - {pol_label}"])

        with tab1:
            toggle = st.toggle("CI Toggle", key=f"ci_toggle_{target_col}")
            # Funktion plot_horizon_metric aufrufen wie vorher
            if agg_metrics:
                # Importiere plot_horizon_metric lokal oder stelle sicher, dass es verfügbar ist
                fig = plot_horizon_metric(
                    agg_metrics,
                    "RMSE_mean",
                    f"Avg RMSE ({pol_label})",
                    "RMSE",
                    show_ci=toggle,
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No data.")

        with tab2:
            if agg_metrics:
                fig = plot_horizon_metric(
                    agg_metrics,
                    "SMAPE_mean",
                    f"Avg SMAPE ({pol_label})",
                    "SMAPE (%)",
                    show_ci=False,
                )
                st.plotly_chart(fig, width="stretch")

        # ... (Der Rest für Performance Evolution bleibt gleich, nutzt df_perf_history)
        st.subheader(f"Performance Evolution ({pol_label})")
        if not df_perf_history.empty:
            # Hier dein PX Line Chart Code für fig_ev_rmse und fig_ev_smape...
            # (Code ist identisch zu vorher, nur die Quelle ist jetzt das JSON-DF)
            fig_ev_rmse = px.line(
                df_perf_history,
                x="prediction_generated_at",
                y="RMSE",
                color="Model",
                markers=True,
            )
            st.plotly_chart(fig_ev_rmse, width="stretch")
        else:
            st.info("Not enough historical data.")

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

    # with st.expander("3. Inspect All Predictions (Raw)", expanded=False):
    #     if not preds_all:
    #         st.error("No prediction models found!")
    #     for model_name, df_p in preds_all.items():
    #         st.subheader(f"Model: {model_name}")
    #         safe_display_df(df_p)
