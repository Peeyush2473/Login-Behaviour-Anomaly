"""
SHAP-based explainability.

Every alert gets a per-feature attribution (SHAP values) explaining why
IT, specifically, was flagged / classified the way it was — not just a
static "high hour_deviation triggers an alert" rule. This replaces the
rule-based approach in explainability.py as the primary explanation
path; that module is kept as an emergency fallback if SHAP itself
throws (e.g. incompatible shap/sklearn/tensorflow versions at deploy
time) so the pipeline never breaks on the explanation step.

Three SHAP explainers are used, one per model that can produce a score
an analyst needs explained:
  - RandomForestClassifier (attack-type prediction) -> shap.TreeExplainer,
    exact and fast for tree ensembles.
  - IsolationForest (fallback detector)              -> shap.TreeExplainer,
    which has native support for sklearn's IsolationForest.
  - LSTM autoencoder (primary detector)               -> shap.KernelExplainer
    over the flattened sequence window, using reconstruction error as the
    model output. Model-agnostic but slower, so nsamples/background are
    kept small — this runs once per alert (top ~1% of events), not per
    event, so the cost is acceptable.

IMPORTANT: different shap/sklearn/tensorflow version combinations return
shap_values() in different shapes (plain array, list-of-arrays per
class, an (n, features, classes) array, or a shap.Explanation object).
_extract_row() below normalises ALL of those down to a flat 1-D array
of plain Python floats — one float per FEATURE_NAMES entry — because
anything else (a stray nested array/Explanation object surviving into
the stored dict) breaks JSON serialisation downstream and, worse, can
silently round-trip into the dashboard as a non-numeric value that
crashes sort_values(). Every public explain_* method below goes through
_to_float_dict() as the single choke point that guarantees this.
"""
import numpy as np
import joblib
from .feature_engineering import FEATURE_NAMES
from .config import SHAP_BACKGROUND_PATH, SEQUENCE_LENGTH

N_FEATURES = len(FEATURE_NAMES)
TOP_K = 3  # how many features to name in the human-readable summary


def _top_features_text(shap_row: np.ndarray, feature_names=FEATURE_NAMES) -> str:
    order = np.argsort(-np.abs(shap_row))[:TOP_K]
    parts = []
    for idx in order:
        val = shap_row[idx]
        direction = "pushed toward anomalous" if val > 0 else "pushed toward normal"
        parts.append(f"{feature_names[idx]} ({direction}, impact {val:+.2f})")
    return "Top contributing factors: " + "; ".join(parts)


def _to_float_dict(row, feature_names=FEATURE_NAMES) -> dict:
    """Coerces `row` (whatever numeric-ish shape it came in as) into a
    flat dict of {feature_name: plain_python_float}, one entry per
    feature. Any value that can't be coerced to a scalar float is
    dropped rather than stored — a missing bar in the SHAP chart is
    harmless, a non-numeric value stored in the DB is not."""
    flat = np.asarray(row, dtype=object).reshape(-1)
    out = {}
    for name, val in zip(feature_names, flat):
        try:
            out[name] = float(np.asarray(val).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            continue
    return out


class ShapExplainer:
    def __init__(self, classifier_model, iso_forest_model, lstm_model=None):
        """Each of the three explainers is built independently and
        wrapped in its own try/except. This matters: shap/sklearn
        version combinations, or a not-yet-trained LSTM, can make any
        one of these throw — and originally a single failure here
        (e.g. only the LSTM's) took down SHAP for every alert, because
        the whole constructor failing meant build_shap_explainer() in
        main.py discarded the entire ShapExplainer and fell back to
        rule-based explanations across the board. Now a failure in one
        explainer only disables that one; the others keep working."""
        import shap

        self._shap = shap

        self.classifier_explainer = None
        try:
            self.classifier_explainer = shap.TreeExplainer(classifier_model)
        except Exception as exc:
            print(f"[detector] SHAP classifier explainer unavailable ({exc}); "
                  f"attack-type SHAP attribution will be unavailable.")

        self.iso_forest_explainer = None
        try:
            self.iso_forest_explainer = shap.TreeExplainer(iso_forest_model)
        except Exception as exc:
            print(f"[detector] SHAP IsolationForest explainer unavailable ({exc}); "
                  f"IsolationForest-scored alerts will fall back to rule-based explanation.")

        self.lstm_explainer = None
        if lstm_model is not None:
            try:
                background = joblib.load(SHAP_BACKGROUND_PATH)  # (n, seq_len*n_features)
                self.lstm_explainer = shap.KernelExplainer(
                    lstm_model.reconstruction_error_fn(), background
                )
            except Exception as exc:
                print(f"[detector] SHAP LSTM explainer unavailable ({exc}); "
                      f"LSTM-scored alerts will fall back to rule-based explanation.")

        print(f"[detector] SHAP explainers ready: "
              f"classifier={'yes' if self.classifier_explainer else 'no'}, "
              f"isolation_forest={'yes' if self.iso_forest_explainer else 'no'}, "
              f"lstm={'yes' if self.lstm_explainer else 'no'}")

    def explain_classification(self, features: dict, predicted_label: str, classifier_classes):
        """Why the RandomForest predicted the attack type it did.
        `classifier_classes` is the fitted model's `.classes_`.
        Returns (text, {feature_name: float})."""
        vec = np.array([[features[name] for name in FEATURE_NAMES]])
        sv = self.classifier_explainer.shap_values(vec)
        class_idx = int(np.where(np.array(classifier_classes) == predicted_label)[0][0])
        row = _extract_row(sv, class_idx)
        shap_dict = _to_float_dict(row)
        return _top_features_text(row), shap_dict

    def explain_isolation_forest(self, features: dict):
        if self.iso_forest_explainer is None:
            raise RuntimeError("IsolationForest SHAP explainer not initialised")
        vec = np.array([[features[name] for name in FEATURE_NAMES]])
        sv = self.iso_forest_explainer.shap_values(vec)
        row = _extract_row(sv, 0)
        shap_dict = _to_float_dict(row)
        return _top_features_text(row), shap_dict

    def explain_lstm(self, feature_sequence: list):
        """feature_sequence: SEQUENCE_LENGTH feature vectors, oldest
        first. Attribution is summed per underlying feature across the
        window so it lines up with the same FEATURE_NAMES used
        everywhere else in the UI."""
        if self.lstm_explainer is None:
            raise RuntimeError("LSTM SHAP explainer not initialised")
        flat = np.array(feature_sequence, dtype="float32").reshape(1, -1)
        sv = self.lstm_explainer.shap_values(flat, nsamples=100)
        arr = np.asarray(sv[0] if isinstance(sv, list) else sv, dtype="float64")
        row = arr.reshape(SEQUENCE_LENGTH, N_FEATURES)
        per_feature = row.sum(axis=0)  # collapse the time axis
        shap_dict = _to_float_dict(per_feature)
        return _top_features_text(per_feature), shap_dict


def _extract_row(shap_values, class_idx: int) -> np.ndarray:
    """Normalises the many shapes shap_values() can come back as (across
    shap/sklearn versions) down to a single flat (n_features,) array of
    floats for one sample / one class."""
    if isinstance(shap_values, list):
        # list of per-class arrays, each (n_samples, n_features)
        arr = np.asarray(shap_values[class_idx], dtype="float64")
        return arr.reshape(-1, N_FEATURES)[0]

    arr = np.asarray(shap_values, dtype="float64")
    if arr.ndim == 3:  # (n_samples, n_features, n_classes)
        return arr[0, :, class_idx]
    if arr.ndim == 2:  # (n_samples, n_features)
        return arr[0]
    return arr.reshape(-1)[:N_FEATURES]