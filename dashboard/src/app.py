"""
Streamlit dashboard entrypoint. Wires together data access + components;
no business logic lives here on purpose.

Run inside Docker via dashboard/Dockerfile CMD, or locally with:
    streamlit run dashboard/src/app.py
"""
import time
import streamlit as st
from data_access import load_alerts, load_entity_history, load_entity_alerts
from components import (
    inject_theme, render_summary_metrics, render_filters, render_alert_queue,
    render_alert_detail, render_entity_timeline, render_model_insights,
)

st.set_page_config(
    page_title="Behavioural Anomaly Detection", layout="wide", page_icon="🛡️",
)
inject_theme()

st.title("🛡️ AI-Powered Behavioural Anomaly Detection")
st.caption("SOC analyst dashboard — LSTM primary detector, IsolationForest fallback, SHAP-explained alerts")

REFRESH_SECONDS = 5

alerts_df = load_alerts()
filtered_df = render_filters(alerts_df)

render_summary_metrics(filtered_df)
st.divider()

tab_queue, tab_entity, tab_insights = st.tabs(
    ["🚨 Alert Queue", "🔍 Entity Investigation", "📊 Model Insights"]
)

selected_entity = None
with tab_queue:
    selected_event, selected_entity = render_alert_queue(filtered_df)
    if selected_event and selected_event != "(none)":
        st.divider()
        render_alert_detail(filtered_df, selected_event)

with tab_entity:
    entities = ["(select an entity)"] + (
        sorted(alerts_df["entity_id"].unique()) if not alerts_df.empty else []
    )
    default_idx = entities.index(selected_entity) if selected_entity in entities else 0
    picked_entity = st.selectbox("Entity", options=entities, index=default_idx)
    if picked_entity and picked_entity != "(select an entity)":
        history_df = load_entity_history(picked_entity)
        entity_alerts_df = load_entity_alerts(alerts_df, picked_entity)
        render_entity_timeline(history_df, entity_alerts_df, picked_entity)
    else:
        st.info("Pick an entity above, or select an alert in the Alert Queue tab, to inspect its activity.")

with tab_insights:
    render_model_insights(alerts_df)

st.divider()
st.caption(f"Auto-refreshing every {REFRESH_SECONDS}s")
time.sleep(REFRESH_SECONDS)
st.rerun()
