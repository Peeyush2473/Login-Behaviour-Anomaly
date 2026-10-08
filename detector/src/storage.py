"""
Lightweight SQLite persistence for alerts and entity event history.
Both the detector (writer) and dashboard (reader) point at the same
file, mounted as a shared Docker volume — see docker-compose.yml.
"""
import sqlite3
import json
from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    event_id TEXT PRIMARY KEY,
    entity_id TEXT,
    entity_type TEXT,
    timestamp TEXT,
    risk_score REAL,
    predicted_type TEXT,
    confidence REAL,
    explanation TEXT,
    model_used TEXT,
    shap_values TEXT,
    raw_event TEXT
);
CREATE TABLE IF NOT EXISTS entity_history (
    event_id TEXT PRIMARY KEY,
    entity_id TEXT,
    timestamp TEXT,
    resource_accessed TEXT,
    raw_event TEXT
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.executescript(SCHEMA)
    _ensure_columns(conn)
    return conn


def _ensure_columns(conn):
    """Adds columns introduced after the original schema so existing
    alerts.db files (e.g. committed for the free-tier dashboard demo)
    keep working without a manual migration step."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(alerts)")}
    if "model_used" not in existing:
        conn.execute("ALTER TABLE alerts ADD COLUMN model_used TEXT")
    if "shap_values" not in existing:
        conn.execute("ALTER TABLE alerts ADD COLUMN shap_values TEXT")
    conn.commit()


def save_alert(conn, event: dict, risk_score: float, predicted_type: str,
               confidence: float, explanation: str, model_used: str = None,
               shap_values: str = None):
    conn.execute(
        "INSERT OR REPLACE INTO alerts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            event["event_id"], event["entity_id"], event["entity_type"],
            event["timestamp"], risk_score, predicted_type, confidence,
            explanation, model_used, shap_values, json.dumps(event),
        ),
    )
    conn.commit()


def save_history(conn, event: dict):
    conn.execute(
        "INSERT OR REPLACE INTO entity_history VALUES (?, ?, ?, ?, ?)",
        (
            event["event_id"], event["entity_id"], event["timestamp"],
            event["resource_accessed"], json.dumps(event),
        ),
    )
    conn.commit()
