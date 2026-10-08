"""
Turns a feature vector into a short, analyst-readable explanation string
— e.g. "Flagged due to geo-velocity anomaly + new/mismatched device
fingerprint", matching the example format given in the assignment.

Uses simple rule-based attribution (fast, dependency-light) rather than
full SHAP so the detector stays lightweight for streaming use. SHAP can
be layered on top of the offline classifier for a deeper attribution
section in the written report if desired.
"""

THRESHOLDS = {
    "hour_deviation": 4.0,
    "geo_velocity_kmh": 900.0,   # faster than sustained commercial flight speed
    "is_new_device": 0.5,
    "is_new_resource": 0.5,
    "is_new_ip": 0.5,
    "auth_failed": 0.5,
}

LABELS = {
    "hour_deviation": "off-hours access",
    "geo_velocity_kmh": "geo-velocity anomaly",
    "is_new_device": "new/mismatched device fingerprint",
    "is_new_resource": "access to unfamiliar resource",
    "is_new_ip": "new source IP",
    "auth_failed": "authentication failure",
    "is_cold_start": "no prior history (cold start)",
}


def explain(features: dict) -> str:
    reasons = []
    for key, threshold in THRESHOLDS.items():
        if features.get(key, 0) >= threshold:
            reasons.append(LABELS[key])
    if features.get("is_cold_start"):
        reasons.append(LABELS["is_cold_start"])
    if not reasons:
        reasons.append("aggregate deviation from baseline profile")
    return "Flagged due to " + " + ".join(reasons)
