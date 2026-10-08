"""
Read-only SQLite queries used by the Streamlit dashboard.
Points at the same DB file the detector writes to (shared Docker volume,
see docker-compose.yml).
"""
import json
import sqlite3
import pandas as pd
from config import DB_PATH


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def _severity(risk_score: float, high: float, medium: float) -> str:
    if risk_score >= high:
        return "High"
    if risk_score >= medium:
        return "Medium"
    return "Low"


def load_alerts(limit: int = 500) -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?", conn, params=(limit,)
        )
    except pd.errors.DatabaseError:
        df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["model_used"] = df.get("model_used", pd.Series(dtype=str)).fillna("isolation_forest")

    high_cut = df["risk_score"].quantile(0.85) if len(df) > 5 else df["risk_score"].max()
    med_cut = df["risk_score"].quantile(0.60) if len(df) > 5 else df["risk_score"].mean()
    df["severity"] = df["risk_score"].apply(lambda s: _severity(s, high_cut, med_cut))

    def _parse_shap(raw):
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None

    df["shap_values"] = df.get("shap_values", pd.Series(dtype=str)).apply(_parse_shap)
    return df


def load_entity_history(entity_id: str) -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM entity_history WHERE entity_id = ? ORDER BY timestamp",
            conn, params=(entity_id,),
        )
    except pd.errors.DatabaseError:
        df = pd.DataFrame()
    finally:
        conn.close()

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def load_entity_alerts(alerts_df: pd.DataFrame, entity_id: str) -> pd.DataFrame:
    if alerts_df.empty:
        return alerts_df
    return alerts_df[alerts_df["entity_id"] == entity_id].sort_values("timestamp")
