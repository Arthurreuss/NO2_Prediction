from streamlit_autorefresh import st_autorefresh
from ui.admin_view import render_admin_dashboard
from ui.user_view import render_user_dashboard

import streamlit as st
from utils.dataloader import load_data

st.set_page_config(page_title="Utrecht NO2 Forecasting", layout="wide")

count = st_autorefresh(interval=300000, key="data_refresh")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["Public Dashboard", "Admin Panel"])
    st.markdown("---")

    if page == "Admin Panel":
        if not st.session_state["logged_in"]:
            st.subheader("Admin Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

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

    st.markdown("---")
    show_debug = st.checkbox("Show Debug Info", value=False)

try:
    df_history, preds = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

if show_debug:
    st.warning("Debug Mode Active")

    with st.expander("1. Inspect History Data", expanded=True):
        if df_history.empty:
            st.error("History DataFrame is empty!")
        else:
            st.write(f"Rows: {len(df_history)}")
            st.write(
                "Time Range:", df_history["time"].min(), "to", df_history["time"].max()
            )
            st.write("First 5 rows:", df_history.head())
            st.write("Data Types:", df_history.dtypes)

    with st.expander("2. Inspect Predictions", expanded=True):
        if not preds:
            st.error("No prediction models found!")
        for model_name, df_p in preds.items():
            st.subheader(f"Model: {model_name}")
            if df_p.empty:
                st.write("Empty DataFrame")
            else:
                st.write(f"Rows: {len(df_p)}")
                st.write("Range:", df_p["time"].min(), "to", df_p["time"].max())
                st.write(df_p.head())


st.markdown("## 🇳🇱 Utrecht NO₂ Air Quality Forecast")

if page == "Public Dashboard":
    if df_history.empty and not preds:
        st.warning("No data available to display.")
    else:
        render_user_dashboard(df_history, preds)

elif page == "Admin Panel":
    if st.session_state["logged_in"]:
        render_admin_dashboard(df_history, preds)
    else:
        st.info("Please log in from the sidebar using 'admin' / 'admin'.")
