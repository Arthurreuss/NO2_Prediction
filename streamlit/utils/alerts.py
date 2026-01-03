import json
import os
import smtplib
from email.message import EmailMessage

import pandas as pd
import psutil

import streamlit as st


def send_alert_email(subject: str, body: str) -> None:
    """Sends an email alert using SMTP credentials from environment variables.

    Args:
        subject: The subject line of the email.
        body: The plain text body content of the email.
    """
    user = os.environ.get("EMAIL_USER")
    password = os.environ.get("EMAIL_PASSWORD")
    to_email = os.environ.get("EMAIL_TO")
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 465))

    if not user or not password or not to_email:
        print("Email credentials missing. Skipping alert email.")
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


def check_and_alert_health(
    df_history: pd.DataFrame, sys_stats_path: str, alert_file: str
) -> None:
    """Checks system resources and data pipeline health, sending alerts if necessary.

    Monitors RAM usage, disk space, and data freshness. Enforces a cooldown period
    to prevent spamming alerts.

    Args:
        df_history: The DataFrame containing historical data to check for emptiness.
        sys_stats_path: Path to the JSON file containing pipeline run statistics.
        alert_file: Path to the JSON file used to store/check alert cooldown timestamps.
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
                if last_run.tz is None:
                    last_run = last_run.tz_localize("UTC")

                now = pd.Timestamp.now(tz="UTC")
                diff_hours = (now - last_run).total_seconds() / 3600

                if diff_hours > 3:
                    issues.append(
                        f"WARNING: Pipeline Data Stale. Last run was {diff_hours:.1f} hours ago."
                    )
        except Exception as e:
            issues.append(f"ERROR: Could not parse pipeline stats: {e}")

    if issues:
        should_send = True

        if os.path.exists(alert_file):
            try:
                with open(alert_file, "r") as f:
                    cooldown_data = json.load(f)

                last_sent_str = cooldown_data.get("last_sent")
                if last_sent_str:
                    last_sent = pd.to_datetime(last_sent_str)
                    now = pd.Timestamp.now()
                    seconds_since_last = (now - last_sent).total_seconds()

                    if seconds_since_last < 3600:  # 3600 seconds = 1 hour
                        should_send = False
                        print(
                            f"Skipping email alert. Last sent {int(seconds_since_last/60)} min ago."
                        )
            except Exception:
                pass

        if should_send:
            subject = f"Admin Dashboard Alert: {len(issues)} Issues Detected"
            body = (
                "The following issues were detected on your dashboard:\n\n"
                + "\n".join(issues)
            )

            send_alert_email(subject, body)

            with open(alert_file, "w") as f:
                json.dump({"last_sent": str(pd.Timestamp.now())}, f)

            st.toast(f"Alert sent via Email ({len(issues)} issues).")

        with st.expander("Active System Alerts", expanded=True):
            for issue in issues:
                st.error(issue)
            if not should_send:
                st.caption(
                    "ℹ️ Email notification suppressed (Cooldown active: Max 1 email/hour)."
                )
