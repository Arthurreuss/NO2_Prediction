import sys
from pathlib import Path

import streamlit as st
import yaml
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh
from ui.admin_view import render_admin_dashboard
from ui.user_view import render_user_dashboard

from utils.alerts import check_and_alert_health
from utils.dataloader import load_data

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
load_dotenv()

with open(str("configs/config_deployment.yaml")) as f:
    cfg: dict = yaml.safe_load(f)

st.set_page_config(page_title="Utrecht NO2 Forecasting", layout="wide")

# Automatically refresh the page every 5 minutes (300,000 milliseconds)
count: int = st_autorefresh(interval=300000, key="data_refresh")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

with st.sidebar:
    st.title("Navigation")
    page: str = st.radio("Go to", ["Public Dashboard", "Admin Panel"])

    if page == "Admin Panel":
        if not st.session_state["logged_in"]:
            st.subheader("Admin Login")
            username: str = st.text_input("Username")
            password: str = st.text_input("Password", type="password")

            if st.button("Login"):
                if username == "admin" and password == "admin":
                    st.session_state["logged_in"] = True
                    st.rerun()
                else:
                    st.error("Invalid credentials (try 'admin'/'admin')")
        else:
            st.success("Logged In")
            if st.button("Logout"):
                st.session_state["logged_in"] = False
                st.rerun()

try:
    df_history, preds, preds_all = load_data(
        cfg["deployment"]["history_path"], cfg["deployment"]["predictions_dir"]
    )
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


st.markdown("## 🇳🇱 Utrecht NO₂ Air Quality Forecast")

if page == "Public Dashboard":
    if df_history.empty and not preds:
        st.warning("No data available to display.")
    else:
        render_user_dashboard(df_history, preds)

elif page == "Admin Panel":
    sys_stats_path = cfg["deployment"]["system_usage_path"]
    alert_file_path = cfg["deployment"]["alert_file"]
    check_and_alert_health(df_history, sys_stats_path, alert_file_path)

    if st.session_state["logged_in"]:
        render_admin_dashboard(df_history, preds, preds_all, cfg)
    else:
        st.info("Please log in from the sidebar using 'admin' / 'admin'.")
