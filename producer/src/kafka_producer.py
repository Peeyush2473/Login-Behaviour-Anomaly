"""
Thin wrapper around kafka-python's KafkaProducer.
Keeps Kafka-specific code isolated from event-generation logic so the
generator modules stay easy to unit-test without a broker running.
"""
import json
import time
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
from .config import KAFKA_BROKER, TOPIC_ACCESS_EVENTS


def get_producer(retries: int = 10, delay: int = 5) -> KafkaProducer:
    """Retries connecting to Kafka since the broker container may still
    be starting up when this service launches."""
    for attempt in range(retries):
        try:
            return KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
        except NoBrokersAvailable:
            print(f"[producer] Kafka not ready, retry {attempt + 1}/{retries}")
            time.sleep(delay)
    raise RuntimeError("Could not connect to Kafka broker")


def send_event(producer: KafkaProducer, event: dict):
    producer.send(TOPIC_ACCESS_EVENTS, value=event)
