"""
Producer entrypoint.
Builds the entity pool once, then loops forever emitting a mix of
normal events and injected-attack events at the configured rate.

Run inside Docker via the producer/Dockerfile CMD, or locally with:
    python -m src.main   (from inside the producer/ folder)
"""
import random
import time
from .config import EVENTS_PER_SECOND, ATTACK_INJECTION_RATE, ATTACK_TYPE_WEIGHTS
from .entity_profiles import build_entity_pool
from .event_generator import generate_normal_event
from .attack_injectors import ATTACK_FUNCTIONS
from .kafka_producer import get_producer, send_event


def main():
    print("[producer] Building entity pool...")
    entity_pool = build_entity_pool()
    producer = get_producer()
    print(f"[producer] Streaming ~{EVENTS_PER_SECOND} events/sec, "
          f"attack rate {ATTACK_INJECTION_RATE * 100:.1f}%")

    attack_names = list(ATTACK_TYPE_WEIGHTS.keys())
    attack_weights = list(ATTACK_TYPE_WEIGHTS.values())

    while True:
        profile = random.choice(list(entity_pool.values()))

        if random.random() < ATTACK_INJECTION_RATE:
            attack_name = random.choices(attack_names, weights=attack_weights)[0]
            events = ATTACK_FUNCTIONS[attack_name](profile, entity_pool)
        else:
            events = [generate_normal_event(profile)]

        for event in events:
            send_event(producer, event)

        producer.flush()
        time.sleep(1.0 / EVENTS_PER_SECOND)


if __name__ == "__main__":
    main()
