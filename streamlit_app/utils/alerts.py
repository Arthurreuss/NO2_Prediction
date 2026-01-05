import json
import os
from typing import Any, Dict, Optional

import pandas as pd
import psutil
import requests
from ui.admin_view import get_dir_size


def send_discord_alert(subject: str, body: str, color: int) -> bool:
    """Sends an alert to a Discord channel via Webhook.

    Note:
        This function includes a manual DNS resolution patch for specific
        deployment environments where standard DNS resolution for discord.com
        might fail.

    Args:
        subject (str): Title of the embed message.
        body (str): Main text content of the alert.
        color (int): Decimal color code for the embed sidebar
            (e.g., Red=15158332, Green=3066993).

    Returns:
        bool: True if the alert was sent successfully, False otherwise.
    """
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")

    if webhook_url:
        webhook_url = webhook_url.replace("discord.com", "162.159.135.232")
    else:
        print("Discord Webhook URL missing. Skipping alert.", flush=True)
        return False

    payload = {
        "username": "System Health Bot",
        "embeds": [
            {
                "title": f"{subject}",
                "description": body,
                "color": color,
                "footer": {"text": "Streamlit Dashboard Monitor"},
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        ],
    }

    try:
        headers = {"Host": "discord.com"}
        response = requests.post(
            webhook_url, json=payload, headers=headers, verify=False
        )
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Failed to send Discord alert: {e}", flush=True)
        return False


def check_and_alert_health(
    df_history: pd.DataFrame, sys_stats_path: str, alert_file: str
) -> None:
    """Checks system health and data freshness, sending alerts if necessary.

    Monitors container resources (RAM, Disk) and pipeline status. Manages
    alert fatigue by tracking the last alert state in a local file.

    Sends:
    - 🔴 ERROR alert if resources are critical or data is stale.
    - 🟢 SUCCESS alert if everything is healthy AND new data just arrived.

    Args:
        df_history (pd.DataFrame): The DataFrame containing historical data.
            Used to check if data is loaded.
        sys_stats_path (str): Path to the JSON file containing pipeline execution stats.
        alert_file (str): Path to the local JSON file used to persist alert state
            and prevent duplicate notifications.
    """
    issues = []

    process = psutil.Process(os.getpid())
    mem_used_mb = process.memory_info().rss / 1024 / 1024

    limit_mb = 16 * 1024
    try:
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
            val = int(f.read().strip())
            if val < 10**15:
                limit_mb = val / 1024 / 1024
    except:
        pass

    mem_percent = (mem_used_mb / limit_mb) * 100
    if mem_percent > 90:
        issues.append(
            f"**Critical RAM:** {mem_percent:.1f}% used ({int(mem_used_mb)}/{int(limit_mb)} MB)"
        )

    app_size_bytes = get_dir_size(".")
    app_size_gb = app_size_bytes / (1024**3)

    hf_quota_gb = 50.0
    effective_free_gb = hf_quota_gb - app_size_gb

    if effective_free_gb < 0.5:
        issues.append(
            f"**Low Disk Space:** Only {effective_free_gb:.2f} GB remaining of {hf_quota_gb} GB quota"
        )

    data_is_fresh = False
    last_run_ts: Optional[pd.Timestamp] = None

    if df_history is None or df_history.empty:
        issues.append("**Data Missing:** History DataFrame is empty")

    if os.path.exists(sys_stats_path):
        try:
            with open(sys_stats_path, "r") as f:
                stats: Dict[str, Any] = json.load(f)

            if stats.get("status") != "success":
                issues.append(
                    f"**Pipeline Failure:** Last run status was '{stats.get('status')}'"
                )

            last_run_str = stats.get("last_run")
            if last_run_str:
                last_run_ts = pd.to_datetime(last_run_str).tz_localize(
                    "Europe/Amsterdam"
                )

                now_ams = pd.Timestamp.now(tz="Europe/Amsterdam")
                age_hours = (now_ams - last_run_ts).total_seconds() / 3600

                if age_hours > 3:
                    issues.append(
                        f"**Stale Data:** Last update was {age_hours:.1f} hours ago"
                    )
                elif age_hours < 1:
                    data_is_fresh = True

        except Exception as e:
            issues.append(f"**Read Error:** Could not parse system stats ({str(e)})")
    else:
        issues.append("**Missing Stats:** 'latest_run.json' not found")

    prev_state: Dict[str, str] = {}
    if os.path.exists(alert_file):
        try:
            with open(alert_file, "r") as f:
                prev_state = json.load(f)
        except:
            pass

    if issues:
        last_error = pd.to_datetime(
            prev_state.get("last_error_time", "2000-01-01")
        ).tz_localize(None)
        now_naive = pd.Timestamp.now()

        if (now_naive - last_error).total_seconds() > 3600:
            subject = f"🚨 Dashboard Alert: {len(issues)} Issues"
            body = "\n".join([f"- {i}" for i in issues])

            if send_discord_alert(subject, body, color=15158332):  # Red
                prev_state["last_error_time"] = now_naive.isoformat()
                with open(alert_file, "w") as f:
                    json.dump(prev_state, f)

    elif data_is_fresh and last_run_ts:
        last_success_ts_str = prev_state.get("last_success_ts")
        current_run_str = last_run_ts.isoformat()

        if last_success_ts_str != current_run_str:
            subject = "✅ System Healthy & Data Updated"
            body = (
                f"**Pipeline:** Success\n"
                f"**Freshness:** {last_run_ts.strftime('%H:%M %d-%m')}\n"
                f"**RAM:** {mem_percent:.1f}%\n"
                f"**Disk Usage:** {app_size_gb:.1f}/{hf_quota_gb} GB ({effective_free_gb:.1f} GB Free)"
            )

            if send_discord_alert(subject, body, color=3066993):
                prev_state["last_success_ts"] = current_run_str
                prev_state["last_error_time"] = "2000-01-01"
                with open(alert_file, "w") as f:
                    json.dump(prev_state, f)
