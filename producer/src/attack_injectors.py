"""
One function per attack pattern from the assignment's Behaviours-to-Simulate
table. Each function returns a LIST of events (some attacks span multiple
events, e.g. brute force = many rapid attempts). Every event carries
label = <attack_name> as ground truth, kept separate from what the
detector sees at inference time.
"""
import random
import uuid
from datetime import datetime, timedelta, timezone
from faker import Faker

fake = Faker()


def _base_event(profile: dict, label: str, **overrides) -> dict:
    now = datetime.now(timezone.utc)
    event = {
        "event_id": str(uuid.uuid4()),
        "entity_id": profile["entity_id"],
        "entity_type": profile["entity_type"],
        "timestamp": now.isoformat(),
        "source_ip": profile["source_ip"],
        "geo": profile["home_geo"],
        "resource_accessed": random.choice(profile["usual_resources"]),
        "auth_method": profile["usual_auth_method"],
        "auth_success": True,
        "session_duration_sec": 60,
        "command_sequence": [],
        "device_fingerprint": profile["usual_device"],
        "label": label,
    }
    event.update(overrides)
    return event


def brute_force(profile: dict) -> list:
    """Rapid repeated failed-auth attempts from one source in a short window."""
    attacker_ip = fake.ipv4_public()
    now = datetime.now(timezone.utc)
    events = []
    for i in range(random.randint(8, 25)):
        events.append(_base_event(
            profile, "brute_force",
            timestamp=(now + timedelta(seconds=i * random.uniform(0.5, 3))).isoformat(),
            source_ip=attacker_ip,
            auth_success=False,
            session_duration_sec=2,
        ))
    return events


def impossible_travel(profile: dict) -> list:
    """Same entity_id logging in from geographically distant locations
    within an implausible time gap."""
    now = datetime.now(timezone.utc)
    far_geo = {"lat": float(fake.latitude()), "lon": float(fake.longitude())}
    e1 = _base_event(profile, "impossible_travel", timestamp=now.isoformat())
    e2 = _base_event(
        profile, "impossible_travel",
        timestamp=(now + timedelta(minutes=random.randint(2, 10))).isoformat(),
        geo=far_geo,
        source_ip=fake.ipv4_public(),
    )
    return [e1, e2]


def credential_stuffing(profile: dict, entity_pool: dict) -> list:
    """Many entity_ids, few source_ips, high failure rate."""
    attacker_ip = fake.ipv4_public()
    victims = random.sample(list(entity_pool.values()), k=min(15, len(entity_pool)))
    now = datetime.now(timezone.utc)
    events = []
    for i, victim in enumerate(victims):
        events.append(_base_event(
            victim, "credential_stuffing",
            timestamp=(now + timedelta(seconds=i * random.uniform(0.2, 1.5))).isoformat(),
            source_ip=attacker_ip,
            auth_success=random.random() < 0.1,
            session_duration_sec=1,
        ))
    return events


def lateral_movement(profile: dict) -> list:
    """A compromised entity accessing an unusual sequence/breadth of
    resources it never touched before."""
    unseen = [f"/api/resource/{random.randint(60, 99)}" for _ in range(random.randint(4, 10))]
    now = datetime.now(timezone.utc)
    events = []
    for i, res in enumerate(unseen):
        events.append(_base_event(
            profile, "lateral_movement",
            timestamp=(now + timedelta(seconds=i * 20)).isoformat(),
            resource_accessed=res,
        ))
    return events


def device_spoofing(profile: dict) -> list:
    """A device_id reappearing with a mismatched fingerprint."""
    spoofed_device = {
        "os": random.choice(["Android12", "WindowsXP", "UnknownOS"]),
        "mac": fake.mac_address(),
        "protocol": random.choice(["Telnet", "FTP"]),
    }
    return [_base_event(profile, "device_spoofing", device_fingerprint=spoofed_device)]


def low_and_slow_exfiltration(profile: dict) -> list:
    """Gradual, small, off-hours resource access building up over days/weeks."""
    now = datetime.now(timezone.utc)
    events = []
    for day in range(random.randint(5, 14)):
        events.append(_base_event(
            profile, "low_and_slow_exfiltration",
            timestamp=(now + timedelta(days=day, hours=random.uniform(1, 4))).isoformat(),
            resource_accessed=random.choice(profile["usual_resources"]),
            session_duration_sec=random.randint(30, 90),
        ))
    return events


def insider_drift(profile: dict) -> list:
    """Legitimate entity slowly expanding privilege/resource footprint —
    ambiguous edge case, useful for false-positive tuning."""
    now = datetime.now(timezone.utc)
    new_resources = [f"/api/resource/{random.randint(60, 99)}" for _ in range(random.randint(2, 5))]
    events = []
    for i, res in enumerate(new_resources):
        events.append(_base_event(
            profile, "insider_drift",
            timestamp=(now + timedelta(days=i * 3)).isoformat(),
            resource_accessed=res,
        ))
    return events


ATTACK_FUNCTIONS = {
    "brute_force": lambda p, pool: brute_force(p),
    "impossible_travel": lambda p, pool: impossible_travel(p),
    "credential_stuffing": lambda p, pool: credential_stuffing(p, pool),
    "lateral_movement": lambda p, pool: lateral_movement(p),
    "device_spoofing": lambda p, pool: device_spoofing(p),
    "low_and_slow_exfiltration": lambda p, pool: low_and_slow_exfiltration(p),
    "insider_drift": lambda p, pool: insider_drift(p),
}
