import pandas as pd
import plotly.express as px

import streamlit as st
from utils.dataloader import compute_metrics


def render_admin_dashboard(df_history, preds):
    st.title("🔒 Admin Dashboard")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Model Performance")

        metrics_data = []
        for model_name, df_p in preds.items():
            m = compute_metrics(df_history, df_p)
            metrics_data.append(
                {
                    "Model": model_name,
                    "MAE": f"{m['MAE']:.2f}",
                    "RMSE": f"{m['RMSE']:.2f}",
                    "Data Points Evaluated": m["Count"],
                }
            )

        st.table(pd.DataFrame(metrics_data))

        if len(metrics_data) > 0 and int(metrics_data[0]["Data Points Evaluated"]) == 0:
            st.warning(
                "⚠️ No overlap found between history and predictions yet. Metrics cannot be calculated until time passes."
            )

    with col2:
        st.subheader("System Health")
        st.metric(label="Total History Records", value=len(df_history))
        st.metric(
            label="Last Data Update",
            value=str(df_history["time"].iloc[-1]) if not df_history.empty else "N/A",
        )

    st.markdown("---")
    st.subheader("Deep Dive: Prediction Overlap")
    st.markdown("Comparing past predictions against what actually happened.")

    if preds:
        model_choice = st.selectbox("Select Model to Inspect", list(preds.keys()))
        df_p = preds[model_choice]

        merged = pd.merge(
            df_history[["time", "nitrogen_dioxide"]],
            df_p[["time", "nitrogen_dioxide"]],
            on="time",
            how="inner",
            suffixes=("_actual", "_pred"),
        )

        if not merged.empty:
            fig = px.line(
                merged,
                x="time",
                y=["nitrogen_dioxide_actual", "nitrogen_dioxide_pred"],
                labels={"value": "NO2 (µg/m³)", "variable": "Source"},
                color_discrete_map={
                    "nitrogen_dioxide_actual": "black",
                    "nitrogen_dioxide_pred": "blue",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No historical overlap data available for this model yet.")
