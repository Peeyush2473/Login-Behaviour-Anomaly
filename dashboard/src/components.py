"""
Reusable Streamlit render functions, kept separate from app.py's page
wiring so each piece stays small, readable, and independently testable.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

SEVERITY_COLORS = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}
MODEL_LABELS = {"lstm": "LSTM (primary)", "isolation_forest": "IsolationForest (fallback)"}
MODEL_COLORS = {"lstm": "#6366f1", "isolation_forest": "#64748b"}


def inject_theme():
    """Dark, SOC-analyst-style theme. Kept in one place so the visual
    language (colors, card styling, badges) stays consistent across
    every tab instead of each component styling itself ad hoc."""
    st.markdown(
        """
        <style>
        .stApp { background-color: #0b0f19; }
        [data-testid="stSidebar"] { background-color: #0f1424; }
        h1, h2, h3 { color: #e5e7eb !important; }
        p, span, label, li { color: #cbd5e1; }

        .kpi-card {
            background: linear-gradient(145deg, #131a2b, #0f1424);
            border: 1px solid #1f2a44;
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 6px;
        }
        .kpi-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #8ea0c4;
            margin-bottom: 6px;
        }
        .kpi-value {
            font-size: 1.9rem;
            font-weight: 700;
            color: #f1f5f9;
        }
        .kpi-sub {
            font-size: 0.78rem;
            color: #64748b;
            margin-top: 4px;
        }

        .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _kpi_card(label: str, value: str, sub: str = "", color: str = "#f1f5f9"):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{color}">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_metrics(df: pd.DataFrame):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _kpi_card("Total Alerts", f"{len(df):,}" if not df.empty else "0")
    with col2:
        avg_risk = f"{df['risk_score'].mean():.2f}" if not df.empty else "—"
        _kpi_card("Avg Risk Score", avg_risk)
    with col3:
        high_count = int((df["severity"] == "High").sum()) if not df.empty else 0
        _kpi_card("High-Severity", f"{high_count:,}", color=SEVERITY_COLORS["High"])
    with col4:
        top_type = df["predicted_type"].mode()[0] if not df.empty else "—"
        _kpi_card("Most Common Type", str(top_type))


def render_model_split(df: pd.DataFrame):
    st.markdown("**Detector used per alert**")
    if df.empty:
        st.caption("No alerts yet.")
        return
    counts = df["model_used"].map(lambda m: MODEL_LABELS.get(m, m)).value_counts()
    fig = go.Figure(
        data=[go.Pie(
            labels=counts.index, values=counts.values, hole=0.55,
            marker=dict(colors=[
                MODEL_COLORS.get(k, "#64748b")
                for k in df["model_used"].value_counts().index
            ]),
        )]
    )
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10), height=260,
        paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Sidebar filters shared by the alert queue + downstream tabs."""
    st.sidebar.header("Filters")
    if df.empty:
        return df

    min_risk = float(df["risk_score"].min())
    max_risk = float(df["risk_score"].max())
    risk_range = st.sidebar.slider(
        "Risk score range", min_value=round(min_risk, 2), max_value=round(max_risk, 2),
        value=(round(min_risk, 2), round(max_risk, 2)),
    )
    severities = st.sidebar.multiselect(
        "Severity", options=["High", "Medium", "Low"], default=["High", "Medium", "Low"]
    )
    types = st.sidebar.multiselect(
        "Predicted attack type", options=sorted(df["predicted_type"].dropna().unique()),
        default=sorted(df["predicted_type"].dropna().unique()),
    )
    models = st.sidebar.multiselect(
        "Detector used", options=sorted(df["model_used"].dropna().unique()),
        default=sorted(df["model_used"].dropna().unique()),
        format_func=lambda m: MODEL_LABELS.get(m, m),
    )

    filtered = df[
        df["risk_score"].between(risk_range[0], risk_range[1])
        & df["severity"].isin(severities)
        & df["predicted_type"].isin(types)
        & df["model_used"].isin(models)
    ]
    return filtered


def render_alert_queue(df: pd.DataFrame):
    st.subheader("Ranked Alert Queue")
    if df.empty:
        st.info("No alerts match the current filters — waiting for streaming data or try widening the filters.")
        return None, None

    display_df = df[[
        "timestamp", "entity_id", "severity", "predicted_type", "risk_score",
        "confidence", "model_used", "explanation",
    ]].copy()
    display_df["model_used"] = display_df["model_used"].map(lambda m: MODEL_LABELS.get(m, m))
    display_df = display_df.rename(columns={
        "timestamp": "Time", "entity_id": "Entity", "severity": "Severity",
        "predicted_type": "Predicted Type", "risk_score": "Risk Score",
        "confidence": "Confidence", "model_used": "Detector", "explanation": "Explanation",
    })

    # ProgressColumn min/max must not be equal — with very few alerts
    # (e.g. exactly one row) risk_score.min() == risk_score.max(), and
    # Streamlit's Arrow-based table renderer has a known native crash
    # (SIGSEGV, not a catchable Python exception) on a zero-width
    # range. Pad it slightly whenever that happens.
    risk_min = float(df["risk_score"].min())
    risk_max = float(df["risk_score"].max())
    if risk_min == risk_max:
        risk_min -= 0.5
        risk_max += 0.5

    st.dataframe(
        display_df,
        use_container_width=True,
        height=420,
        column_config={
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score", min_value=risk_min, max_value=risk_max, format="%.2f",
            ),
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="%.2f",
            ),
        },
        hide_index=True,
    )

    event_options = ["(none)"] + list(df["event_id"])
    label_map = {
        row.event_id: f"{row.timestamp} · {row.entity_id} · {row.predicted_type} "
                       f"(risk {row.risk_score:.2f})"
        for row in df.itertuples()
    }
    selected_event = st.selectbox(
        "Inspect alert:", options=event_options,
        format_func=lambda eid: "(none)" if eid == "(none)" else label_map.get(eid, eid),
    )
    selected_entity = None
    if selected_event != "(none)":
        selected_entity = df.loc[df["event_id"] == selected_event, "entity_id"].iloc[0]
    return selected_event, selected_entity


def render_alert_detail(df: pd.DataFrame, event_id: str):
    row = df.loc[df["event_id"] == event_id].iloc[0]
    st.subheader("Alert Detail")

    badge_color = SEVERITY_COLORS.get(row["severity"], "#64748b")
    st.markdown(
        f'<span class="badge" style="background:{badge_color}22;color:{badge_color};">'
        f'{row["severity"]} severity</span> &nbsp; '
        f'<span class="badge" style="background:#1f2a4433;color:#93c5fd;">'
        f'{MODEL_LABELS.get(row["model_used"], row["model_used"])}</span>',
        unsafe_allow_html=True,
    )
    st.write(f"**Entity:** {row['entity_id']}  |  **Predicted type:** {row['predicted_type']}  "
             f"|  **Confidence:** {row['confidence']:.2f}  |  **Risk score:** {row['risk_score']:.2f}")
    st.write(f"**Explanation:** {row['explanation']}")

    shap_values = row.get("shap_values")
    if shap_values:
        # Defensive: coerce to numeric and drop anything that isn't a
        # clean scalar (e.g. a stray value from an older/incompatible
        # shap version). Without this, a single bad value anywhere in
        # the dict crashes sort_values() for the whole chart.
        shap_df = pd.DataFrame(
            {"feature": list(shap_values.keys()), "impact": list(shap_values.values())}
        )
        shap_df["impact"] = pd.to_numeric(shap_df["impact"], errors="coerce")
        shap_df = shap_df.dropna(subset=["impact"]).sort_values("impact")

        if shap_df.empty:
            st.caption("SHAP attribution stored for this alert was malformed and couldn't be charted.")
        else:
            st.markdown("**Feature attribution (SHAP)**")
            fig = go.Figure(go.Bar(
                x=shap_df["impact"], y=shap_df["feature"], orientation="h",
                marker_color=["#ef4444" if v > 0 else "#22c55e" for v in shap_df["impact"]],
            ))
            fig.update_layout(
                height=280, margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1", xaxis_title="SHAP value (→ more anomalous)",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No SHAP attribution stored for this alert (rule-based explanation was used).")


def render_entity_timeline(history_df: pd.DataFrame, alerts_df: pd.DataFrame, entity_id: str):
    st.subheader(f"Activity timeline — {entity_id}")
    if history_df.empty:
        st.info("No history recorded yet for this entity.")
        return

    fig = px.scatter(
        history_df, x="timestamp", y="resource_accessed",
        title=None, opacity=0.6,
    )
    fig.update_traces(marker=dict(color="#64748b", size=7), name="Access event", showlegend=True)

    if not alerts_df.empty:
        # Mark exactly when this entity was flagged, on the same
        # timeline as its raw access events, so an analyst can see the
        # alert in context rather than as an isolated row in a table.
        for ts in alerts_df["timestamp"]:
            fig.add_vline(x=ts, line_width=1.5, line_dash="dash", line_color="#ef4444")

    fig.update_layout(
        height=350, margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

    if not alerts_df.empty:
        st.markdown("**Alerts for this entity**")
        st.dataframe(
            alerts_df[["timestamp", "severity", "predicted_type", "risk_score", "model_used", "explanation"]]
            .rename(columns={"model_used": "detector"}),
            use_container_width=True, hide_index=True,
        )
    st.markdown("**Raw access history**")
    st.dataframe(history_df, use_container_width=True, hide_index=True, height=300)


def render_model_insights(df: pd.DataFrame):
    st.subheader("Model Insights")
    if df.empty:
        st.info("No alerts yet.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Risk score distribution**")
        fig = px.histogram(df, x="risk_score", nbins=30, color_discrete_sequence=["#6366f1"])
        fig.update_layout(
            height=300, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Predicted attack type breakdown**")
        counts = df["predicted_type"].value_counts()
        fig = px.bar(x=counts.values, y=counts.index, orientation="h",
                     color_discrete_sequence=["#22c55e"])
        fig.update_layout(
            height=300, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
            xaxis_title="Alerts", yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    render_model_split(df)