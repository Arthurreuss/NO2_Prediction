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
from utils.model_evaluation import get_horizon_metrics, get_performance_over_time


def safe_display_df(df, limit=500):
    """
    Converts datetime columns to string to prevent PyArrow/Streamlit crashes
    with 'Europe/Amsterdam' timezones.
    """
    if df is None or df.empty:
        st.write("Empty DataFrame")
        return

    # Create a copy to not modify the original data used for plotting
    display_df = df.head(limit).copy()

    # Convert all datetime columns (including index if datetime) to string
    for col in display_df.select_dtypes(include=["datetime", "datetimetz"]).columns:
        display_df[col] = display_df[col].astype(str)

    # Also check specific column names just in case dtype detection missed it
    for col in ["time", "prediction_generated_at"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].astype(str)

    st.dataframe(display_df)


def plot_horizon_metric(data_dict, metric_col, title, y_label, show_ci=False):
    """Helper for Horizon Plots (Aggregated Step View)"""
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, (model_name, df) in enumerate(data_dict.items()):
        if df is None or df.empty:
            continue
        color = colors[i % len(colors)]

        # Robust sort
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


def render_admin_dashboard(df_history, preds, preds_all, sys_stats_path):
    st.title("Admin Dashboard")

    agg_metrics = {}
    recent_metrics = {}
    recent_run_time_str = "N/A"

    for model_name, df_all_predictions in preds_all.items():
        # add confidence bands
        # make plots for the other polutants
        # if model name gru or df_all_predictions dimesnsion of gru then only do for nitrogen dioxide
        stats_all, _ = get_horizon_metrics(
            df_history, df_all_predictions, "nitrogen_dioxide"
        )
        agg_metrics[model_name] = stats_all

    # B. Performance Over Time (New Feature)
    df_perf_history = get_performance_over_time(
        df_history, preds_all, "nitrogen_dioxide"
    )

    st.subheader("Global Horizon Analysis (Aggregated)")
    st.caption(
        "How does error increase as we forecast further into the future? (Averaged over all history)"
    )

    tab1, tab2 = st.tabs(["RMSE (by Step)", "SMAPE (by Step)"])
    with tab1:
        st.plotly_chart(
            plot_horizon_metric(
                agg_metrics, "RMSE_mean", "Avg RMSE per Horizon Step", "RMSE"
            ),
            width="stretch",
        )
    with tab2:
        st.plotly_chart(
            plot_horizon_metric(
                agg_metrics, "SMAPE_mean", "Avg SMAPE per Horizon Step", "SMAPE (%)"
            ),
            width="stretch",
        )

    st.markdown("---")

    st.subheader("Model Performance Evolution")
    st.caption(
        "Average error per forecast run (72h) over time. Only showing completed runs (>3 days old)."
    )

    if not df_perf_history.empty:
        tab_ev1, tab_ev2 = st.tabs(["RMSE History", "SMAPE History"])

        with tab_ev1:
            fig_ev_rmse = px.line(
                df_perf_history,
                x="prediction_generated_at",
                y="RMSE",
                color="Model",
                title="RMSE per Forecast Run (Avg over 72h)",
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
                title="SMAPE per Forecast Run (Avg over 72h)",
                markers=True,
            )
            fig_ev_smape.update_layout(
                xaxis_title="Run Time", yaxis_title="Average SMAPE (%)"
            )
            st.plotly_chart(fig_ev_smape, width="stretch")
    else:
        st.info(
            "Not enough historical data yet to show performance evolution (need runs older than 72h)."
        )

    st.markdown("---")

    # --- 5. INSPECTION & SYSTEM STATS (FIXED) ---
    with st.expander("1. Inspect API Data (History)", expanded=False):
        if df_history.empty:
            st.error("API History DataFrame is empty!")
        else:
            st.write(f"Rows: {len(df_history)}")
            # --- CRITICAL FIX: Safe Display ---
            safe_display_df(df_history)

    with st.expander("2. Inspect Predictions", expanded=False):
        if not preds:
            st.error("No prediction models found!")
        for model_name, df_p in preds.items():
            st.subheader(f"Model: {model_name}")
            # --- CRITICAL FIX: Safe Display ---
            safe_display_df(df_p)

    with st.expander("3. Inspect All Predictions (Raw)", expanded=False):
        if not preds_all:
            st.error("No prediction models found!")
        for model_name, df_p in preds_all.items():
            st.subheader(f"Model: {model_name}")
            # --- CRITICAL FIX: Safe Display ---
            safe_display_df(df_p)
    # --- 5. SYSTEM STATS (Legacy) ---
    st.markdown("---")
    with st.expander("System & Pipeline Stats"):
        process = psutil.Process(os.getpid())
        col1, col2 = st.columns(2)
        col1.metric("RAM", f"{int(process.memory_info().rss / 1024 / 1024)} MB")
        col2.metric("CPU", f"{psutil.cpu_percent()}%")

        if os.path.exists(sys_stats_path):
            with open(sys_stats_path, "r") as f:
                stats = json.load(f)
            st.write(stats)
