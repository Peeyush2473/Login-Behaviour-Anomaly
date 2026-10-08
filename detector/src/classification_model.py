"""
Loads the pre-trained supervised classifier that predicts WHICH attack
type a flagged event most resembles (assignment deliverable 4:
"not just anomalous, but which attack category it resembles").
"""
import numpy as np
import joblib
from .feature_engineering import FEATURE_NAMES
from .config import CLASSIFIER_MODEL_PATH


class ClassificationModel:
    def __init__(self):
        self.model = joblib.load(CLASSIFIER_MODEL_PATH)

    def predict(self, features: dict):
        vec = np.array([[features[name] for name in FEATURE_NAMES]])
        pred = self.model.predict(vec)[0]
        proba = max(self.model.predict_proba(vec)[0])
        return pred, float(proba)
