import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import yaml

import streamlit as st
from utils.model_evaluation import evaluate_model_performance


def render_admin_dashboard(df_history, preds, preds_all):
    """
    preds: Dict of simple line predictions (legacy/plotting)
    preds_all: Dict of {model_name: [list of forecast dfs]} containing full 72h horizon data
    """
    st.title("Admin Dashboard")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    performance_summary = []
    horizon_plots = {}

    for model_name, predictions in preds_all.items():
        results = evaluate_model_performance(df_history, predictions)

        if results:
            performance_summary.append(
                {
                    "Model": model_name,
                    "Total MAE": f"{results['MAE_Total']:.2f}",
                    "Last 24h MAE": (
                        f"{results['MAE_Last24h']:.2f}"
                        if results["MAE_Last24h"]
                        else "N/A"
                    ),
                    "RMSE": f"{results['RMSE_Total']:.2f}",
                    "Evaluated Points": results["Count"],
                }
            )
            horizon_plots[model_name] = results["Horizon_Perf"]

    with col1:
        st.subheader("Model Performance")
        if performance_summary:
            st.table(pd.DataFrame(performance_summary))
        else:
            st.warning("No overlap found between history and predictions yet.")

    with col2:
        st.subheader("System Health")
        st.metric(label="Total History Records", value=len(df_history))
        st.metric(
            label="Last Ground Truth",
            value=str(df_history["time"].iloc[-1]) if not df_history.empty else "N/A",
        )

    st.markdown("---")

    st.subheader("Horizon Analysis: Accuracy vs. Forecast Distance")
    st.caption(
        "How does error increase as we predict further into the future (1h to 72h)?"
    )

    if horizon_plots:
        fig_horizon = go.Figure()
        for model_name, df_h in horizon_plots.items():
            fig_horizon.add_trace(
                go.Scatter(
                    x=df_h["step"], y=df_h["MAE"], mode="lines+markers", name=model_name
                )
            )
        fig_horizon.update_layout(
            xaxis_title="Forecast Step (Hours Ahead)",
            yaxis_title="Mean Absolute Error (MAE)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_horizon, use_container_width=True)

    st.markdown("---")

    st.subheader("Deep Dive: Prediction Overlap")
    st.markdown("Comparing the most recent forecast line against actuals.")

    if preds:
        model_choice = st.selectbox("Select Model to Inspect", list(preds.keys()))
        df_p = preds[model_choice]

        merged_viz = pd.merge(
            df_history[["time", "nitrogen_dioxide"]],
            df_p[["time", "nitrogen_dioxide"]],
            on="time",
            how="inner",
            suffixes=("_actual", "_pred"),
        )

        if not merged_viz.empty:
            fig = px.line(
                merged_viz,
                x="time",
                y=["nitrogen_dioxide_actual", "nitrogen_dioxide_pred"],
                labels={"value": "NO2 (µg/m³)", "variable": "Source"},
                color_discrete_map={
                    "nitrogen_dioxide_actual": "black",
                    "nitrogen_dioxide_pred": "blue",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No historical overlap data available for this model yet.")

    with st.expander("1. Inspect API Data", expanded=True):
        if df_history.empty:
            st.error("API History DataFrame is empty!")
        else:
            st.write(f"Rows: {len(df_history)}")
            st.write(
                "Time Range:", df_history["time"].min(), "to", df_history["time"].max()
            )
            st.write("First 5 rows:", df_history.head())
            st.write("Data Types:", df_history.dtypes)

    with st.expander("2. Inspect Predictions", expanded=True):
        if not preds:
            st.error("No prediction models found!")
        for model_name, df_p in preds.items():
            st.subheader(f"Model: {model_name}")
            if df_p.empty:
                st.write("Empty DataFrame")
            else:
                st.write(f"Predicted hours total: {len(df_p)}")
                st.write("Range:", df_p["time"].min(), "to", df_p["time"].max())
                st.write(df_p.head())

    st.markdown("---")
    st.subheader("Live App Monitor (Hugging Face)")

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024

    cpu_usage = psutil.cpu_percent(interval=None)

    col1, col2 = st.columns(2)
    col1.metric("RAM", f"{int(mem_mb)} MB")
    col2.metric("CPU", f"{cpu_usage}%")

    st.caption("Real-time metrics from HF Space")

    with open("config_deployment.yaml") as f:
        cfg = yaml.safe_load(f)

    file_path = cfg["deployment"]["system_usage_path"]
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            stats = json.load(f)

        last_run_str = stats["last_run"]
        last_run_dt = datetime.strptime(last_run_str, "%Y-%m-%d %H:%M:%S UTC")

        now = datetime.now(ZoneInfo("Europe/Amsterdam"))
        diff = now - last_run_dt
        minutes_ago = int(diff.total_seconds() / 60)

        st.markdown("---")
        st.subheader("Data Pipeline")

        if minutes_ago > 110:
            st.error(f"Stale Data (Last: {minutes_ago}m ago)")
        else:
            st.success(f"Fresh Data (Last: {minutes_ago}m ago)")

        with st.expander("Pipeline Details"):
            st.write(f"**Last Update:** {last_run_str}")
            st.write(f"**Duration:** {stats.get('duration_seconds', 'N/A')} sec")

            if "system_metrics" in stats:
                st.markdown("---")
                st.caption("🏗️ GitHub Builder Resources")
                st.text("Builder RAM:")
                st.code(stats["system_metrics"]["memory"], language="text")
    else:
        st.warning("No pipeline stats found yet.")
