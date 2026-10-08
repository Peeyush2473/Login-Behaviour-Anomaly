"""
Central configuration for the producer service.
Edit these values (or set as env vars in docker-compose.yml) to control
simulation scale and attack rates.
"""
import os

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
TOPIC_ACCESS_EVENTS = "access-events"

NUM_ENTITIES = int(os.getenv("NUM_ENTITIES", 300))
EVENTS_PER_SECOND = float(os.getenv("EVENTS_PER_SECOND", 5))

# Fraction of total events that should be injected attacks (0.5% - 3%,
# per the assignment's suggested injection rate).
ATTACK_INJECTION_RATE = float(os.getenv("ATTACK_INJECTION_RATE", 0.02))

# Relative weight of each attack type when an attack is injected.
ATTACK_TYPE_WEIGHTS = {
    "brute_force": 0.20,
    "impossible_travel": 0.15,
    "credential_stuffing": 0.15,
    "lateral_movement": 0.15,
    "device_spoofing": 0.15,
    "low_and_slow_exfiltration": 0.10,
    "insider_drift": 0.10,
}

ENTITY_TYPES = ["user", "service_account", "edge_device"]
ENTITY_TYPE_WEIGHTS = [0.7, 0.15, 0.15]

AUTH_METHODS = ["password", "token", "certificate", "biometric"]
