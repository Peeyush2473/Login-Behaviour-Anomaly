"""
Hybrid unsupervised detector.

Primary: LSTMDetectionModel — scores a window of an entity's recent
behaviour by reconstruction error. This is the preferred model because
it looks at the *sequence* of an entity's recent activity, not just one
event in isolation, so it can catch anomalies that only show up as a
change in pattern over several events (e.g. a slow drift, or a burst of
otherwise-individually-unremarkable actions).

Fallback: IsolationForestModel — scores a single event's feature vector
in isolation. Used whenever the LSTM can't be used yet:
  - the entity is new / cold-start and doesn't have SEQUENCE_LENGTH
    events of history yet,
  - the LSTM model failed to load (e.g. TensorFlow unavailable in a
    stripped-down deployment) — this keeps the detector usable even
    without the heavier dependency.

Train both offline first with:  python -m detector.train.train_models
"""
import numpy as np
import joblib
from .feature_engineering import FEATURE_NAMES
from .config import DETECTION_MODEL_PATH, SCALER_PATH, SEQUENCE_LENGTH


class IsolationForestModel:
    """Single-event fallback detector."""

    def __init__(self):
        self.model = joblib.load(DETECTION_MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

    def score(self, features: dict) -> float:
        vec = np.array([[features[name] for name in FEATURE_NAMES]])
        vec_scaled = self.scaler.transform(vec)
        # decision_function: lower (more negative) = more anomalous.
        # Flip sign so higher = more anomalous, matching the LSTM's
        # "higher reconstruction error = more anomalous" convention.
        raw_score = -self.model.decision_function(vec_scaled)[0]
        return float(raw_score)


class HybridDetectionModel:
    """Routes each event to the LSTM when possible, otherwise to the
    IsolationForest fallback. Exposes which model actually scored the
    event (`last_model_used`) so it can be logged/displayed downstream."""

    def __init__(self):
        self.fallback = IsolationForestModel()
        self.lstm = None
        self.lstm_available = True
        try:
            from .lstm_detector import LSTMDetectionModel
            self.lstm = LSTMDetectionModel()
        except Exception as exc:  # missing TensorFlow, missing artifact, etc.
            print(f"[detector] LSTM primary model unavailable ({exc}); "
                  f"running on IsolationForest fallback only.")
            self.lstm_available = False

        self.last_model_used = None

    def score(self, features: dict, entity_state) -> float:
        if (
            self.lstm_available
            and self.lstm is not None
            and entity_state.has_sequence(SEQUENCE_LENGTH)
        ):
            self.last_model_used = "lstm"
            return self.lstm.score(entity_state.get_sequence(SEQUENCE_LENGTH))

        self.last_model_used = "isolation_forest"
        return self.fallback.score(features)
