import json
import os

import pandas as pd
import psutil
import requests
import streamlit as st


def send_discord_alert(subject: str, body: str) -> bool:
    """Sends an alert to a Discord channel via Webhook.

    Args:
        subject: The title/subject of the alert.
        body: The detailed body content.

    Returns:
        True if the request was successful, False otherwise.
    """
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    webhook_url = webhook_url.replace("discord.com", "162.159.135.232")

    if not webhook_url:
        print("Discord Webhook URL missing. Skipping alert.", flush=True)
        return False

    payload = {
        "username": "System Health Bot",
        "embeds": [
            {
                "title": f"🚨 {subject}",
                "description": body,
                "color": 15158332,  # Red color
                "footer": {"text": "Streamlit Dashboard Monitor"},
            }
        ],
    }

    try:
        headers = {"Host": "discord.com"}
        response = requests.post(
            webhook_url, json=payload, headers=headers, verify=False
        )
        response.raise_for_status()
        print("Discord alert sent successfully.", flush=True)
        return True
    except Exception as e:
        print(f"Failed to send Discord alert: {e}", flush=True)
        return False


def check_and_alert_health(
    df_history: pd.DataFrame, sys_stats_path: str, alert_file: str
) -> None:
    """Checks system resources and data pipeline health, sending Discord alerts if necessary.

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
                last_run = last_run.tz_convert("Europe/Amsterdam")

                now_ams = pd.Timestamp.now(tz="Europe/Amsterdam")
                diff_hours = (now_ams - last_run).total_seconds() / 3600

                if diff_hours > 3:
                    issues.append(
                        f"WARNING: Pipeline Stale. Last run {diff_hours:.1f} hours ago."
                    )
        except Exception:
            pass

    issues.append("TEST: This is a forced test alert to verify Discord.")

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
            body = "**The following issues were detected:**\n" + "\n".join(
                [f"- {i}" for i in issues]
            )

            success = send_discord_alert(subject, body)

            if success:
                with open(alert_file, "w") as f:
                    json.dump({"last_sent": now_ams.isoformat()}, f)
                st.toast("Discord notification sent to admin!")
            else:
                st.error("Failed to send Discord notification.")
        elif not should_send:
            st.caption("ℹ️ Notification suppressed by cooldown.")

    else:
        st.success("System Status: Healthy")
