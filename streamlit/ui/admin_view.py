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


def render_admin_dashboard(df_history, preds, preds_all, file_path):
    st.title("Admin Dashboard")

    # --- 1. PRE-CALCULATE DATA ---
    agg_metrics = {}
    recent_metrics = {}
    recent_run_time_str = "N/A"

    # A. Horizon Analysis (Step-by-Step)
    for model_name, df_all_predictions in preds_all.items():
        # Global Aggregate
        stats_all, _ = get_horizon_metrics(df_history, df_all_predictions)
        agg_metrics[model_name] = stats_all

        # Single Recent Run (Manual Split)
        # --- FIX: Manually split DataFrame to find the latest valid run ---
        if "prediction_generated_at" in df_all_predictions.columns:
            unique_gen_times = sorted(
                df_all_predictions["prediction_generated_at"].unique(), reverse=True
            )

            best_overlap_run = None
            # Search backwards for a run with enough validation data
            for gen_time in unique_gen_times:
                single_run_df = df_all_predictions[
                    df_all_predictions["prediction_generated_at"] == gen_time
                ].copy()
                run_stats, _ = get_horizon_metrics(df_history, single_run_df)

                if (
                    run_stats is not None and run_stats["Count"].sum() >= 24
                ):  # Require at least 24h overlap
                    best_overlap_run = run_stats
                    ts = pd.to_datetime(gen_time)
                    recent_run_time_str = ts.strftime("%Y-%m-%d %H:%M UTC")
                    break

            recent_metrics[model_name] = best_overlap_run

    # B. Performance Over Time (New Feature)
    df_perf_history = get_performance_over_time(df_history, preds_all)

    # --- 2. GLOBAL HORIZON PLOTS ---
    st.subheader("🌍 Global Horizon Analysis (Aggregated)")
    st.caption(
        "How does error increase as we forecast further into the future? (Averaged over all history)"
    )

    tab1, tab2 = st.tabs(["RMSE (by Step)", "SMAPE (by Step)"])
    with tab1:
        st.plotly_chart(
            plot_horizon_metric(
                agg_metrics, "RMSE_mean", "Avg RMSE per Horizon Step", "RMSE"
            ),
            use_container_width=True,
        )
    with tab2:
        st.plotly_chart(
            plot_horizon_metric(
                agg_metrics, "SMAPE_mean", "Avg SMAPE per Horizon Step", "SMAPE (%)"
            ),
            use_container_width=True,
        )

    st.markdown("---")

    # --- 3. PERFORMANCE EVOLUTION PLOTS (NEW) ---
    st.subheader("📈 Model Performance Evolution")
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
            st.plotly_chart(fig_ev_rmse, use_container_width=True)

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
            st.plotly_chart(fig_ev_smape, use_container_width=True)
    else:
        st.info(
            "Not enough historical data yet to show performance evolution (need runs older than 72h)."
        )

    st.markdown("---")

    # --- 4. RECENT RUN ANALYSIS ---
    st.subheader("⏱️ Recent Run Analysis")
    st.caption(
        f"Performance of the most recent validatable forecast run (Run Time: {recent_run_time_str})"
    )

    tab3, tab4 = st.tabs(["RMSE (Last Run)", "SMAPE (Last Run)"])
    with tab3:
        if any(v is not None for v in recent_metrics.values()):
            st.plotly_chart(
                plot_horizon_metric(
                    recent_metrics, "RMSE_mean", "RMSE (Last Run)", "RMSE"
                ),
                use_container_width=True,
            )
        else:
            st.warning("No validated recent runs found.")
    with tab4:
        if any(v is not None for v in recent_metrics.values()):
            st.plotly_chart(
                plot_horizon_metric(
                    recent_metrics, "SMAPE_mean", "SMAPE (Last Run)", "SMAPE (%)"
                ),
                use_container_width=True,
            )
        else:
            st.warning("No validated recent runs found.")

    # --- 5. SYSTEM STATS (Legacy) ---
    st.markdown("---")
    with st.expander("System & Pipeline Stats"):
        process = psutil.Process(os.getpid())
        col1, col2 = st.columns(2)
        col1.metric("RAM", f"{int(process.memory_info().rss / 1024 / 1024)} MB")
        col2.metric("CPU", f"{psutil.cpu_percent()}%")

        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                stats = json.load(f)
            st.write(stats)
