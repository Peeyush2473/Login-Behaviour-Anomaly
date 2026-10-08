"""
OFFLINE training script — run this once, from the repo root, BEFORE
starting the streaming pipeline. It generates a labeled batch dataset
(reusing the producer's own generation logic so training data matches
what will later stream through Kafka), extracts features, and trains:

  1. LSTM autoencoder       — PRIMARY unsupervised detector, scored on
                               windows of an entity's recent behaviour
                               (reconstruction error = anomaly score)
  2. IsolationForest        — FALLBACK unsupervised detector, used for
                               cold-start entities without enough
                               history yet for a sequence window
  3. RandomForestClassifier — supervised attack-type classifier
  4. A small background sample for the LSTM's SHAP KernelExplainer

Saves all artifacts to detector/models/.

Run from the repo root:
    pip install -r producer/requirements.txt -r detector/requirements.txt
    python -m detector.train.train_models
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import random
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from producer.src.entity_profiles import build_entity_pool
from producer.src.event_generator import generate_normal_event
from producer.src.attack_injectors import ATTACK_FUNCTIONS
from detector.src.baseline_profile import BaselineStore
from detector.src.feature_engineering import extract_features, FEATURE_NAMES
from detector.src.config import SEQUENCE_LENGTH

N_EVENTS = 20000
ATTACK_RATE = 0.02
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
SHAP_BACKGROUND_SIZE = 60


def generate_training_batch() -> list:
    pool = build_entity_pool()
    attack_names = list(ATTACK_FUNCTIONS.keys())
    events = []
    while len(events) < N_EVENTS:
        profile = random.choice(list(pool.values()))
        if random.random() < ATTACK_RATE:
            attack_name = random.choice(attack_names)
            events.extend(ATTACK_FUNCTIONS[attack_name](profile, pool))
        else:
            events.append(generate_normal_event(profile))
    events.sort(key=lambda e: e["timestamp"])
    return events


def build_sequence_windows(events: list, store: BaselineStore):
    """Replays events through the same feature-extraction path used at
    inference time, tracking per-entity feature vectors as we go, and
    slices out SEQUENCE_LENGTH-length windows. A window's label is the
    label of its last (most recent) event — a reasonable proxy since
    attack injectors concentrate the anomalous signal there."""
    per_entity_vectors: dict[str, list] = {}
    per_entity_labels: dict[str, list] = {}
    X, y = [], []          # single-event feature rows (IsolationForest / classifier)
    windows, window_labels = [], []  # sequence windows (LSTM)

    for event in events:
        features = extract_features(event, store)
        row = [features[name] for name in FEATURE_NAMES]
        X.append(row)
        y.append(event["label"])

        eid = event["entity_id"]
        per_entity_vectors.setdefault(eid, []).append(row)
        per_entity_labels.setdefault(eid, []).append(event["label"])

        vecs = per_entity_vectors[eid]
        labels = per_entity_labels[eid]
        if len(vecs) >= SEQUENCE_LENGTH:
            windows.append(vecs[-SEQUENCE_LENGTH:])
            window_labels.append(labels[-1])

    return np.array(X), np.array(y), np.array(windows), np.array(window_labels)


def train_lstm_autoencoder(windows: np.ndarray, window_labels: np.ndarray):
    from detector.src.lstm_detector import build_lstm_autoencoder

    normal_mask = window_labels == "normal"
    normal_windows = windows[normal_mask]

    n_features = windows.shape[2]
    lstm_scaler = StandardScaler().fit(normal_windows.reshape(-1, n_features))

    def scale_windows(w):
        n = w.shape[0]
        return lstm_scaler.transform(w.reshape(-1, n_features)).reshape(n, SEQUENCE_LENGTH, n_features)

    normal_scaled = scale_windows(normal_windows)
    train_w, val_w = train_test_split(normal_scaled, test_size=0.1, random_state=42)

    print(f"[train] Training LSTM autoencoder on {len(train_w)} normal windows "
          f"(validating on {len(val_w)})...")
    autoencoder = build_lstm_autoencoder(SEQUENCE_LENGTH, n_features)
    autoencoder.fit(
        train_w, train_w,
        validation_data=(val_w, val_w),
        epochs=15, batch_size=64, verbose=2,
    )

    # Reference threshold + a quick separation check: reconstruction
    # error should be higher on attack windows than normal ones if the
    # model actually learned something useful.
    all_scaled = scale_windows(windows)
    recon = autoencoder.predict(all_scaled, verbose=0)
    errors = np.mean((all_scaled - recon) ** 2, axis=(1, 2))
    normal_errors = errors[normal_mask]
    attack_errors = errors[~normal_mask]
    reference_threshold = float(np.percentile(normal_errors, 95))
    print(f"[train] LSTM reconstruction error — normal mean={normal_errors.mean():.4f}, "
          f"attack mean={attack_errors.mean() if len(attack_errors) else float('nan'):.4f}, "
          f"reference threshold (95th pct of normal)={reference_threshold:.4f}")

    background_idx = np.random.choice(
        len(normal_scaled), size=min(SHAP_BACKGROUND_SIZE, len(normal_scaled)), replace=False
    )
    shap_background = normal_scaled[background_idx].reshape(len(background_idx), -1)

    return autoencoder, lstm_scaler, reference_threshold, shap_background


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("[train] Generating labeled synthetic batch...")
    events = generate_training_batch()

    store = BaselineStore()
    X, y, windows, window_labels = build_sequence_windows(events, store)

    # --- IsolationForest (fallback detector) ---
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    print("[train] Fitting IsolationForest (fallback detector) on (mostly normal) data...")
    iso_forest = IsolationForest(contamination=ATTACK_RATE, random_state=42)
    iso_forest.fit(X_scaled)

    # --- RandomForest attack-type classifier ---
    print("[train] Fitting RandomForest classifier on labeled attack types...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    classifier = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    classifier.fit(X_train, y_train)
    print(classification_report(y_test, classifier.predict(X_test)))

    # --- LSTM autoencoder (primary detector) ---
    autoencoder, lstm_scaler, lstm_threshold, shap_background = train_lstm_autoencoder(
        windows, window_labels
    )

    joblib.dump(iso_forest, os.path.join(MODEL_DIR, "isolation_forest.joblib"))
    joblib.dump(classifier, os.path.join(MODEL_DIR, "attack_classifier.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    autoencoder.save(os.path.join(MODEL_DIR, "lstm_autoencoder.keras"))
    joblib.dump(lstm_scaler, os.path.join(MODEL_DIR, "lstm_scaler.joblib"))
    joblib.dump(lstm_threshold, os.path.join(MODEL_DIR, "lstm_threshold.joblib"))
    joblib.dump(shap_background, os.path.join(MODEL_DIR, "shap_background.joblib"))

    print(f"[train] Models saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
