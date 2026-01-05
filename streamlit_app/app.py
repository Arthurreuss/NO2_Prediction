import sys
from pathlib import Path

import streamlit as st
import yaml
from streamlit_autorefresh import st_autorefresh

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.cfg import load_config
from streamlit_app.ui.admin_view import render_admin_dashboard
from streamlit_app.ui.user_view import render_user_dashboard
from streamlit_app.utils.alerts import check_and_alert_health
from streamlit_app.utils.dataloader import load_data

st.set_page_config(page_title="Utrecht NO2 Forecast", layout="wide")
cfg = load_config("configs/config_deployment.yaml")

# Auto-refresh (5 mins)
st_autorefresh(interval=300000, key="data_refresh")

try:
    df_history, preds, h_metrics, p_metrics = load_data(
        cfg["deployment"]["history_path"],
        cfg["deployment"]["predictions_dir"],
        cfg["deployment"]["metrics_dir"],
    )
except Exception as e:
    st.error(f"Critical Data Error: {e}")
    st.stop()

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

st.markdown("## 🇳🇱 Utrecht Air Quality Forecast")

if page == "Public Dashboard":
    render_user_dashboard(df_history, preds)
    check_and_alert_health(
        df_history,
        cfg["deployment"]["system_usage_path"],
        cfg["deployment"]["alert_file"],
    )

elif page == "Admin Panel":
    if st.session_state.get("logged_in"):
        render_admin_dashboard(df_history, preds, h_metrics, p_metrics, cfg)
    else:
        st.info("Please log in to access Admin tools.")
