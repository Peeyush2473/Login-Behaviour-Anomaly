"""
Detector entrypoint: consumes raw events, scores them, classifies flagged
ones, explains them, persists + republishes alerts.

Requires models to already exist in detector/models/ — run
detector/train/train_models.py once before starting this service.
"""
import json
import numpy as np
from .kafka_consumer import get_consumer, get_alert_producer, publish_alert
from .baseline_profile import BaselineStore
from .feature_engineering import extract_features
from .detection_model import HybridDetectionModel
from .classification_model import ClassificationModel
from .explainability import explain as rule_based_explain
from .storage import get_connection, save_alert, save_history
from .config import ALERT_BUDGET_PERCENTILE, SEQUENCE_LENGTH

RECENT_SCORES_WINDOW = 500


def build_shap_explainer(detection_model, classification_model):
    """Best-effort SHAP explainer. Any failure (missing shap, version
    mismatch, missing background artifact) falls back to the rule-based
    explainer for the whole run rather than crashing the detector."""
    try:
        from .shap_explainability import ShapExplainer
        return ShapExplainer(
            classifier_model=classification_model.model,
            iso_forest_model=detection_model.fallback.model,
            lstm_model=detection_model.lstm if detection_model.lstm_available else None,
        )
    except Exception as exc:
        print(f"[detector] SHAP explainability unavailable ({exc}); "
              f"using rule-based explanations instead.")
        return None


def explain_alert(shap_explainer, features, entity_state, model_used):
    """Returns (explanation_text, shap_values_dict_or_None)."""
    if shap_explainer is not None:
        try:
            if model_used == "lstm" and shap_explainer.lstm_explainer is not None:
                return shap_explainer.explain_lstm(entity_state.get_sequence(SEQUENCE_LENGTH))
            if model_used == "isolation_forest":
                return shap_explainer.explain_isolation_forest(features)
        except Exception as exc:
            print(f"[detector] SHAP explanation failed for this alert ({exc}); "
                  f"falling back to rule-based explanation.")
    return rule_based_explain(features), None


def main():
    print("[detector] Loading models...")
    detection_model = HybridDetectionModel()
    classification_model = ClassificationModel()
    shap_explainer = build_shap_explainer(detection_model, classification_model)
    store = BaselineStore()
    conn = get_connection()

    consumer = get_consumer()
    alert_producer = get_alert_producer()

    recent_scores = []

    print("[detector] Listening for events...")
    for message in consumer:
        event = message.value
        save_history(conn, event)

        entity_state = store.get_or_create(event["entity_id"], event["entity_type"])
        features = extract_features(event, store)
        risk_score = detection_model.score(features, entity_state)
        model_used = detection_model.last_model_used

        # Maintain a rolling score window so the alert threshold adapts
        # over time instead of being a fixed magic number — this is the
        # "top 1% alert budget" mechanism from the evaluation criteria.
        # LSTM and IsolationForest scores are on different scales, but
        # since the threshold is a live percentile over whichever model
        # scored each recent event, it adapts to the current mix rather
        # than assuming one fixed scale.
        recent_scores.append(risk_score)
        if len(recent_scores) > RECENT_SCORES_WINDOW:
            recent_scores.pop(0)
        threshold = (
            np.percentile(recent_scores, ALERT_BUDGET_PERCENTILE)
            if len(recent_scores) > 20 else 1e9
        )

        if risk_score >= threshold:
            predicted_type, confidence = classification_model.predict(features)
            explanation, shap_values = explain_alert(
                shap_explainer, features, entity_state, model_used
            )
            save_alert(
                conn, event, risk_score, predicted_type, confidence, explanation,
                model_used=model_used,
                shap_values=json.dumps(shap_values) if shap_values else None,
            )
            publish_alert(alert_producer, {
                "event_id": event["event_id"],
                "entity_id": event["entity_id"],
                "risk_score": risk_score,
                "predicted_type": predicted_type,
                "confidence": confidence,
                "explanation": explanation,
                "model_used": model_used,
            })
            print(f"[detector] ALERT {event['entity_id']} -> {predicted_type} "
                  f"(score={risk_score:.2f}, model={model_used}) {explanation}")


if __name__ == "__main__":
    main()
