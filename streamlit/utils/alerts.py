import json
import os
import smtplib
from email.message import EmailMessage

import pandas as pd
import psutil

import streamlit as st


def send_alert_email(subject: str, body: str) -> bool:
    """Sends an email alert using SMTP credentials from environment variables.

    Args:
        subject: The subject line of the email.
        body: The plain text body content of the email.

    Returns:
        True if the email was sent successfully, False otherwise.
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
        return True
    except Exception as e:
        print(f"Failed to send alert email: {e}")
        return False


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
    with st.expander("Admin Controls", expanded=True):
        if st.button("Reset Email Cooldown"):
            if os.path.exists(alert_file):
                os.remove(alert_file)
                st.success("Cooldown reset! You can now trigger a new email.")
                st.rerun()
            else:
                st.info("No active cooldown found.")

    issues = []

    issues.append("TEST: This is a forced test alert.")

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
                last_run = last_run.tz_convert("Europe/Amsterdam")

                now_ams = pd.Timestamp.now(tz="Europe/Amsterdam")
                diff_hours = (now_ams - last_run).total_seconds() / 3600

                if diff_hours > 3:
                    issues.append(
                        f"WARNING: Pipeline Stale. Last run {diff_hours:.1f} hours ago."
                    )
        except Exception:
            pass

    if issues:
        should_send = True
        cooldown_msg = ""
        now_ams = pd.Timestamp.now(tz="Europe/Amsterdam")

        if os.path.exists(alert_file):
            try:
                with open(alert_file, "r") as f:
                    data = json.load(f)
                    last_sent = pd.to_datetime(data["last_sent"])
                    last_sent = last_sent.tz_convert("Europe/Amsterdam")
                    seconds_since = (now_ams - last_sent).total_seconds()

                    if seconds_since < 3600:
                        should_send = False
                        mins_left = int((3600 - seconds_since) / 60)
                        cooldown_msg = f" (Cooldown active: Wait {mins_left} mins)"
            except Exception:
                pass

        st.error(f"Active System Alerts {cooldown_msg}")
        for issue in issues:
            st.write(f"- {issue}")

        if should_send:
            subject = f"Dashboard Alert: {len(issues)} Issues Detected"
            body = "The following issues were detected:\n\n" + "\n".join(issues)
            success = send_alert_email(subject, body)

            if success:
                with open(alert_file, "w") as f:
                    json.dump({"last_sent": now_ams.isoformat()}, f)
                st.toast("📧 Alert sent to admin!")
        elif not should_send:
            st.caption("ℹ️ Email suppressed by cooldown.")

    else:
        st.success("System Status: Healthy")
