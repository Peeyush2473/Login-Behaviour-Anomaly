"""
Converts a raw event + its entity's rolling baseline state into a fixed
numeric feature vector for the detection & classification models.
Every feature here is also referenced by name in explainability.py so
that alerts can cite exactly which factor drove the flag — this is the
link between detection and the assignment's explainability requirement.
"""
import numpy as np
from geopy.distance import geodesic
from .baseline_profile import BaselineStore
from .config import COLD_START_MIN_EVENTS

FEATURE_NAMES = [
    "hour_deviation",
    "is_new_resource",
    "is_new_device",
    "is_new_ip",
    "geo_velocity_kmh",
    "auth_failed",
    "session_duration_zscore",
    "resource_novelty_ratio",
    "is_cold_start",
]


def _geo_velocity(prev_event: dict, curr_event: dict) -> float:
    """Implied travel speed (km/h) between an entity's last known location
    and its current one — the core signal for impossible travel."""
    if not prev_event:
        return 0.0
    try:
        dist_km = geodesic(
            (prev_event["geo"]["lat"], prev_event["geo"]["lon"]),
            (curr_event["geo"]["lat"], curr_event["geo"]["lon"]),
        ).km
        t1 = np.datetime64(prev_event["timestamp"])
        t2 = np.datetime64(curr_event["timestamp"])
        hours = max((t2 - t1) / np.timedelta64(1, "h"), 1e-6)
        return float(dist_km / hours)
    except Exception:
        return 0.0


def extract_features(event: dict, store: BaselineStore) -> dict:
    state = store.get_or_create(event["entity_id"], event["entity_type"])
    prev_event = state.events[-1] if state.events else None
    cold_start = state.is_cold_start(COLD_START_MIN_EVENTS)

    hour = int(event["timestamp"][11:13])
    baseline_hour = (
        store.population_mean_hour(event["entity_type"])
        if cold_start else state.mean_login_hour
    )

    features = {
        "hour_deviation": abs(hour - baseline_hour),
        "is_new_resource": float(event["resource_accessed"] not in state.known_resources),
        "is_new_device": float(event["device_fingerprint"]["mac"] not in state.known_devices),
        "is_new_ip": float(event["source_ip"] not in state.known_ips),
        "geo_velocity_kmh": _geo_velocity(prev_event, event),
        "auth_failed": float(not event.get("auth_success", True)),
        "session_duration_zscore": 0.0,  # extend with rolling std if desired
        "resource_novelty_ratio": float(event["resource_accessed"] not in state.known_resources),
        "is_cold_start": float(cold_start),
    }

    # Update state AFTER extracting features so "is_new_*" reflects
    # history strictly before this event.
    store.record(event)
    state.push_features([features[name] for name in FEATURE_NAMES])
    return features
