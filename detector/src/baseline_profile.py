"""
Maintains a live per-entity behavioural baseline as events stream in.
Handles the cold-start problem: brand-new entities fall back to a
population-level profile (grouped by entity_type) until they've built
up enough history of their own — directly addresses the assignment's
"how to score a brand-new user or device with no history" requirement.
"""
from collections import defaultdict, deque

MAX_HISTORY = 200  # events kept per entity for rolling stats


class EntityState:
    def __init__(self, entity_type: str):
        self.entity_type = entity_type
        self.events = deque(maxlen=MAX_HISTORY)
        self.known_resources = set()
        self.known_devices = set()
        self.known_ips = set()
        self.login_hours = deque(maxlen=MAX_HISTORY)
        # Rolling history of extracted feature vectors (dicts), in
        # chronological order. This is what feeds the LSTM sequence
        # window — the detector needs SEQUENCE_LENGTH consecutive
        # vectors for an entity before it can be scored by the LSTM.
        self.feature_history = deque(maxlen=MAX_HISTORY)

    def update(self, event: dict):
        self.events.append(event)
        self.known_resources.add(event["resource_accessed"])
        self.known_devices.add(event["device_fingerprint"]["mac"])
        self.known_ips.add(event["source_ip"])
        self.login_hours.append(int(event["timestamp"][11:13]))

    def push_features(self, feature_vector: list):
        self.feature_history.append(feature_vector)

    def has_sequence(self, sequence_length: int) -> bool:
        return len(self.feature_history) >= sequence_length

    def get_sequence(self, sequence_length: int):
        """Last `sequence_length` feature vectors, oldest first."""
        return list(self.feature_history)[-sequence_length:]

    def is_cold_start(self, min_events: int) -> bool:
        return len(self.events) < min_events

    @property
    def mean_login_hour(self) -> float:
        return sum(self.login_hours) / len(self.login_hours) if self.login_hours else 12.0


class BaselineStore:
    """In-memory store of all entity states, plus population-level fallback
    profiles per entity_type for cold-start scoring."""

    def __init__(self):
        self._entities: dict[str, EntityState] = {}
        self._population_hours: dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))

    def get_or_create(self, entity_id: str, entity_type: str) -> EntityState:
        if entity_id not in self._entities:
            self._entities[entity_id] = EntityState(entity_type)
        return self._entities[entity_id]

    def record(self, event: dict):
        state = self.get_or_create(event["entity_id"], event["entity_type"])
        state.update(event)
        self._population_hours[event["entity_type"]].append(int(event["timestamp"][11:13]))

    def population_mean_hour(self, entity_type: str) -> float:
        hours = self._population_hours[entity_type]
        return sum(hours) / len(hours) if hours else 12.0
