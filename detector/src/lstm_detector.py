"""
Primary detector: an LSTM autoencoder over sequences of an entity's
recent per-event feature vectors.

Why an autoencoder: we don't have a supervised "next anomaly" label at
inference time, so instead of predicting a class we teach the network to
reconstruct short windows of an entity's own (mostly normal) behaviour.
A window that reconstructs poorly is one the network hasn't learned to
expect from that kind of entity — i.e. anomalous. Reconstruction error
(MSE between the real window and the model's reconstruction of it) is
the anomaly score, matching the "higher = more anomalous" convention
used by the rest of the pipeline.

This needs SEQUENCE_LENGTH consecutive feature vectors for the entity,
so it is only usable once an entity has enough history — see
HybridDetectionModel in detection_model.py for the IsolationForest
fallback used before that point (cold start / just-restarted service).
"""
import numpy as np
import joblib
from .config import (
    LSTM_MODEL_PATH, LSTM_SCALER_PATH, LSTM_THRESHOLD_PATH, SEQUENCE_LENGTH,
)
from .feature_engineering import FEATURE_NAMES

N_FEATURES = len(FEATURE_NAMES)


def build_lstm_autoencoder(sequence_length: int = SEQUENCE_LENGTH, n_features: int = N_FEATURES):
    """Small encoder-decoder LSTM. Kept deliberately compact (this scores
    9-dim vectors over short windows, not raw text/audio) so it trains
    fast on a laptop and stays cheap enough to run inline per streamed
    event."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(sequence_length, n_features))
    encoded = layers.LSTM(32, activation="tanh", return_sequences=False)(inputs)
    repeated = layers.RepeatVector(sequence_length)(encoded)
    decoded = layers.LSTM(32, activation="tanh", return_sequences=True)(repeated)
    outputs = layers.TimeDistributed(layers.Dense(n_features))(decoded)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


class LSTMDetectionModel:
    """Loads the trained autoencoder + its feature scaler + the
    calibrated alerting threshold, and scores sequence windows."""

    def __init__(self):
        from tensorflow import keras
        self.model = keras.models.load_model(LSTM_MODEL_PATH)
        self.scaler = joblib.load(LSTM_SCALER_PATH)
        # Not used for alert routing (main.py keeps its own rolling
        # percentile threshold across whichever model scored each event)
        # but kept for reference / offline evaluation.
        self.reference_threshold = joblib.load(LSTM_THRESHOLD_PATH)

    def score(self, sequence: list) -> float:
        """sequence: list of SEQUENCE_LENGTH feature vectors (each a
        list of N_FEATURES floats, oldest first). Returns a scalar
        anomaly score — mean squared reconstruction error."""
        arr = np.array(sequence, dtype="float32")
        arr_scaled = self.scaler.transform(arr)
        batch = arr_scaled[np.newaxis, ...]  # (1, seq_len, n_features)
        recon = self.model.predict(batch, verbose=0)[0]
        mse = float(np.mean((arr_scaled - recon) ** 2))
        return mse

    def reconstruction_error_fn(self):
        """Returns a callable(flat_sequence_batch) -> mse_scores, used by
        the SHAP explainer to attribute the anomaly score back to the
        most recent event's features. `flat_sequence_batch` has shape
        (n_samples, seq_len * n_features)."""
        def _fn(flat_batch: np.ndarray) -> np.ndarray:
            n = flat_batch.shape[0]
            batch = flat_batch.reshape(n, SEQUENCE_LENGTH, N_FEATURES)
            batch_scaled = np.stack([self.scaler.transform(seq) for seq in batch])
            recon = self.model.predict(batch_scaled, verbose=0)
            return np.mean((batch_scaled - recon) ** 2, axis=(1, 2))
        return _fn
