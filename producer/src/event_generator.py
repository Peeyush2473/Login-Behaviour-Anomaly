"""
Generates a single NORMAL (benign) access event for a given entity,
sampled from that entity's habitual profile with added noise — the
"Normal baseline" row of the assignment's Behaviours-to-Simulate table.
"""
import random
import uuid
from datetime import datetime, timezone
import numpy as np


def generate_normal_event(profile: dict) -> dict:
    hour = float(np.clip(
        np.random.normal(profile["typical_login_hour_mean"], profile["typical_login_hour_std"]),
        0, 23.99
    ))
    now = datetime.now(timezone.utc).replace(
        hour=int(hour), minute=random.randint(0, 59), second=random.randint(0, 59)
    )
    return {
        "event_id": str(uuid.uuid4()),
        "entity_id": profile["entity_id"],
        "entity_type": profile["entity_type"],
        "timestamp": now.isoformat(),
        "source_ip": profile["source_ip"],
        "geo": profile["home_geo"],
        "resource_accessed": random.choice(profile["usual_resources"]),
        "auth_method": profile["usual_auth_method"],
        "auth_success": True,
        "session_duration_sec": max(5, int(np.random.normal(600, 200))),
        "command_sequence": [],
        "device_fingerprint": profile["usual_device"],
        "label": "normal",
    }
