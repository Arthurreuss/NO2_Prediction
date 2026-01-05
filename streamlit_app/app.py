import sys
from pathlib import Path

import streamlit as st
import yaml
from streamlit_autorefresh import st_autorefresh

# Path setup for local modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ui.admin_view import render_admin_dashboard
from ui.user_view import render_user_dashboard

from utils.alerts import check_and_alert_health
from utils.dataloader import load_data

# Config
st.set_page_config(page_title="Utrecht NO2 Forecast", layout="wide")
with open("configs/config_deployment.yaml") as f:
    cfg = yaml.safe_load(f)

# Auto-refresh (5 mins)
st_autorefresh(interval=300000, key="data_refresh")

# --- LOAD DATA (All at once) ---
try:
    df_history, preds, h_metrics, p_metrics = load_data(
        cfg["deployment"]["history_path"],
        cfg["deployment"]["predictions_dir"],
        "data/deployment/metrics",  # Ensure this matches your folder structure
    )
except Exception as e:
    st.error(f"Critical Data Error: {e}")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["Public Dashboard", "Admin Panel"])

    if page == "Admin Panel":
        if not st.session_state.get("logged_in"):
            st.subheader("Login")
            user = st.text_input("User")
            pw = st.text_input("Password", type="password")
            if st.button("Login"):
                if user == "admin" and pw == "admin":
                    st.session_state["logged_in"] = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        else:
            if st.button("Logout"):
                st.session_state["logged_in"] = False
                st.rerun()

# --- ROUTING ---
st.markdown("## 🇳🇱 Utrecht Air Quality Forecast")

if page == "Public Dashboard":
    render_user_dashboard(df_history, preds)
    # Background Health Check
    check_and_alert_health(
        df_history,
        cfg["deployment"]["system_usage_path"],
        cfg["deployment"]["alert_file"],
    )

elif page == "Admin Panel":
    if st.session_state.get("logged_in"):
        # Pass all loaded data including metrics
        render_admin_dashboard(df_history, preds, h_metrics, p_metrics, cfg)
    else:
        st.info("Please log in to access Admin tools.")
