"""
OFFLINE batch export — generates a labeled synthetic dataset and writes
it to disk as CSV files, separate from the live Kafka streaming path in
main.py. This gives you an inspectable artifact for submission (assignment
deliverable 1) without touching the streaming pipeline.

Writes two files to the data/ directory:
  - synthetic_access_logs.csv   (all fields except label — what a model
                                  would see at inference time)
  - ground_truth_labels.csv     (event_id, label — kept separate, per
                                  the assignment's schema requirement)

Run from inside producer/:
    python -m src.export_dataset
"""
import os
import csv
import random
from .entity_profiles import build_entity_pool
from .event_generator import generate_normal_event
from .attack_injectors import ATTACK_FUNCTIONS
from .config import NUM_ENTITIES, ATTACK_INJECTION_RATE, ATTACK_TYPE_WEIGHTS

N_EVENTS = int(os.getenv("EXPORT_N_EVENTS", 20000))
OUTPUT_DIR = os.getenv("EXPORT_OUTPUT_DIR", "../data")

FIELDNAMES = [
    "event_id", "entity_id", "entity_type", "timestamp",
    "source_ip", "geo_lat", "geo_lon", "resource_accessed",
    "auth_method", "auth_success", "session_duration_sec",
    "command_sequence", "device_os", "device_mac", "device_protocol",
]


def _flatten(event: dict) -> dict:
    """Flattens nested geo/device_fingerprint dicts into flat CSV columns."""
    return {
        "event_id": event["event_id"],
        "entity_id": event["entity_id"],
        "entity_type": event["entity_type"],
        "timestamp": event["timestamp"],
        "source_ip": event["source_ip"],
        "geo_lat": event["geo"]["lat"],
        "geo_lon": event["geo"]["lon"],
        "resource_accessed": event["resource_accessed"],
        "auth_method": event["auth_method"],
        "auth_success": event["auth_success"],
        "session_duration_sec": event["session_duration_sec"],
        "command_sequence": "|".join(event.get("command_sequence", [])),
        "device_os": event["device_fingerprint"]["os"],
        "device_mac": event["device_fingerprint"]["mac"],
        "device_protocol": event["device_fingerprint"]["protocol"],
    }


def generate_batch() -> list:
    print(f"[export] Building {NUM_ENTITIES} entity profiles...")
    pool = build_entity_pool()
    attack_names = list(ATTACK_TYPE_WEIGHTS.keys())
    attack_weights = list(ATTACK_TYPE_WEIGHTS.values())

    events = []
    while len(events) < N_EVENTS:
        profile = random.choice(list(pool.values()))
        if random.random() < ATTACK_INJECTION_RATE:
            attack_name = random.choices(attack_names, weights=attack_weights)[0]
            events.extend(ATTACK_FUNCTIONS[attack_name](profile, pool))
        else:
            events.append(generate_normal_event(profile))

    events.sort(key=lambda e: e["timestamp"])
    return events[:N_EVENTS]


def write_csvs(events: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logs_path = os.path.join(output_dir, "synthetic_access_logs.csv")
    labels_path = os.path.join(output_dir, "ground_truth_labels.csv")

    with open(logs_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for event in events:
            writer.writerow(_flatten(event))

    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["event_id", "label"])
        writer.writeheader()
        for event in events:
            writer.writerow({"event_id": event["event_id"], "label": event["label"]})

    print(f"[export] Wrote {len(events)} events to {logs_path}")
    print(f"[export] Wrote {len(events)} ground-truth labels to {labels_path}")

    label_counts = {}
    for event in events:
        label_counts[event["label"]] = label_counts.get(event["label"], 0) + 1
    print("[export] Label distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    {label:<28} {count:>6}  ({100 * count / len(events):.2f}%)")


def main():
    events = generate_batch()
    write_csvs(events, OUTPUT_DIR)


if __name__ == "__main__":
    main()