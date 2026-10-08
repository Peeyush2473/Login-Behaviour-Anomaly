import os

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
TOPIC_ACCESS_EVENTS = "access-events"
TOPIC_ANOMALY_ALERTS = "anomaly-alerts"
CONSUMER_GROUP = "detector-group"

DB_PATH = os.getenv("DB_PATH", "/data/alerts.db")

MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
DETECTION_MODEL_PATH = f"{MODEL_DIR}/isolation_forest.joblib"          # fallback detector
CLASSIFIER_MODEL_PATH = f"{MODEL_DIR}/attack_classifier.joblib"
SCALER_PATH = f"{MODEL_DIR}/scaler.joblib"

LSTM_MODEL_PATH = f"{MODEL_DIR}/lstm_autoencoder.keras"                # primary detector
LSTM_SCALER_PATH = f"{MODEL_DIR}/lstm_scaler.joblib"
LSTM_THRESHOLD_PATH = f"{MODEL_DIR}/lstm_threshold.joblib"

SHAP_BACKGROUND_PATH = f"{MODEL_DIR}/shap_background.joblib"

# Minimum events seen before an entity is treated as "warm" — this is
# the cold-start handling required by the assignment.
COLD_START_MIN_EVENTS = 5

# Number of consecutive per-entity feature vectors the LSTM needs to form
# a sequence window. Entities with fewer stored feature vectors than this
# (new/cold-start entities, or a freshly-restarted service) can't be
# scored by the LSTM yet, so detection falls back to IsolationForest,
# which scores a single event in isolation.
SEQUENCE_LENGTH = 10

# Top-k% of scored events flagged as alerts, matching the assignment's
# "realistic analyst alert budget (e.g., top 1% of events)" criterion.
ALERT_BUDGET_PERCENTILE = 99
