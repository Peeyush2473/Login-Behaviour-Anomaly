"""
Builds the per-entity 'habitual' profile used to sample normal behaviour.
Each entity gets a stable home geo-location, typical login-hour window,
usual resource set, usual auth method, and a device fingerprint — this
is the "per-entity habitual pattern" referenced in the assignment's
Behaviours-to-Simulate table.
"""
import random
from faker import Faker
from .config import NUM_ENTITIES, ENTITY_TYPES, ENTITY_TYPE_WEIGHTS, AUTH_METHODS

fake = Faker()

RESOURCE_POOL = [f"/api/resource/{i}" for i in range(1, 60)]


def _random_geo() -> dict:
    return {"lat": float(fake.latitude()), "lon": float(fake.longitude())}


def _device_fingerprint() -> dict:
    return {
        "os": random.choice(["Windows11", "macOS14", "Ubuntu22", "iOS17", "LinuxEdge"]),
        "mac": fake.mac_address(),
        "protocol": random.choice(["HTTPS", "SSH", "MQTT", "RDP"]),
    }


def build_entity_profile(entity_id: str) -> dict:
    """Returns one entity's stable behavioural baseline."""
    entity_type = random.choices(ENTITY_TYPES, weights=ENTITY_TYPE_WEIGHTS)[0]
    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "home_geo": _random_geo(),
        "typical_login_hour_mean": random.uniform(7, 19),
        "typical_login_hour_std": random.uniform(0.5, 2.0),
        "usual_resources": random.sample(RESOURCE_POOL, k=random.randint(3, 8)),
        "usual_auth_method": random.choice(AUTH_METHODS),
        "usual_device": _device_fingerprint(),
        "source_ip": fake.ipv4_public(),
    }


def build_entity_pool() -> dict:
    """Builds NUM_ENTITIES profiles keyed by entity_id."""
    profiles = {}
    for i in range(NUM_ENTITIES):
        entity_id = f"ent_{i:05d}"
        profiles[entity_id] = build_entity_profile(entity_id)
    return profiles
