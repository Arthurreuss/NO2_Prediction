import json
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil

import streamlit as st
from utils.model_evaluation import get_horizon_metrics, get_performance_over_time


def send_alert_email(subject, body):
    user = os.environ.get("EMAIL_USER")
    password = os.environ.get("EMAIL_PASSWORD")
    to_email = os.environ.get("EMAIL_TO")
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 465))

    if not user or not password or not to_email:
        print("⚠️ Email credentials missing. Skipping alert email.")
        return

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_email

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(user, password)
            server.send_message(msg)
        print("Alert email sent successfully.")
    except Exception as e:
        print(f"Failed to send alert email: {e}")


def check_and_alert_health(df_history, sys_stats_path):
    """
    Checks for system anomalies (High RAM, Low Disk, Stale Data)
    and sends an email if 'something is off'.
    """
    issues = []

    mem = psutil.virtual_memory()
    if mem.percent > 90:
        issues.append(f"CRITICAL: System RAM is at {mem.percent}%")

    disk = psutil.disk_usage(".")
    free_gb = disk.free / (1024**3)
    if free_gb < 0.5:
        issues.append(f"CRITICAL: Low Disk Space ({free_gb:.2f} GB remaining)")

    if df_history is None or df_history.empty:
        issues.append("CRITICAL: History DataFrame is empty/missing")

    if os.path.exists(sys_stats_path):
        try:
            with open(sys_stats_path, "r") as f:
                stats = json.load(f)

            last_run_str = stats.get("last_run")
            if last_run_str:
                last_run = pd.to_datetime(last_run_str)

                now = pd.Timestamp.now(tz="Europe/Amsterdam")
                diff_hours = (now - last_run).total_seconds() / 3600

                if diff_hours > 3:
                    issues.append(
                        f"WARNING: Pipeline Data Stale. Last run was {diff_hours:.1f} hours ago."
                    )
        except Exception as e:
            issues.append(f"ERROR: Could not parse pipeline stats: {e}")

    if issues:
        if "alert_sent_session" not in st.session_state:
            subject = f"Admin Dashboard Alert: {len(issues)} Issues Detected"
            body = (
                "The following issues were detected on your dashboard:\n\n"
                + "\n".join(issues)
            )

            send_alert_email(subject, body)

            st.session_state["alert_sent_session"] = True
            st.toast(f"Alert sent: {len(issues)} issues detected.", icon="⚠️")

        with st.expander("Active System Alerts", expanded=True):
            for issue in issues:
                st.error(issue)


def safe_display_df(df, limit=500):
    """
    Converts datetime columns to string to prevent PyArrow/Streamlit crashes
    with 'Europe/Amsterdam' timezones.
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


def plot_horizon_metric(data_dict, metric_col, title, y_label, show_ci=False):
    """Helper for Horizon Plots (Aggregated Step View)"""
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


def show_system_and_pipeline_stats(sys_stats_path):
    with st.expander("System & Pipeline Stats"):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        current_process_ram = mem_info.rss / 1024 / 1024

        sys_mem = psutil.virtual_memory()
        percent_ram = sys_mem.percent

        disk = psutil.disk_usage(".")
        free_disk = disk.free / 1024 / 1024 / 1024
        total_disk = disk.total / 1024 / 1024 / 1024

        space_id = os.environ.get("SPACE_ID", "Local/Unknown")

        st.caption(f"Hosting Environment: {space_id}")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "RAM Usage (System)",
            f"{percent_ram}%",
            f"{int(current_process_ram)}MB used by App",
        )

        col2.metric("CPU Usage", f"{psutil.cpu_percent()}%")

        col3.metric("Disk Free", f"{free_disk:.1f} GB", f"of {total_disk:.1f} GB")

        st.markdown("---")

        st.write("Github Actions Pipeline Stats:")
        if os.path.exists(sys_stats_path):
            with open(sys_stats_path, "r") as f:
                stats = json.load(f)
            st.json(stats)
        else:
            st.warning("No external pipeline stats found.")


def render_admin_dashboard(df_history, preds, preds_all, sys_stats_path):
    st.title("Admin Dashboard")

    check_and_alert_health(df_history, sys_stats_path)

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
