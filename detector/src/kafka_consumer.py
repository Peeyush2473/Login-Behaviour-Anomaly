"""
Thin wrapper around kafka-python's consumer/producer for the detector
service. Consumes raw events, republishes only the flagged ones as alerts.
"""
import json
import time
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable
from .config import KAFKA_BROKER, TOPIC_ACCESS_EVENTS, CONSUMER_GROUP


def get_consumer(retries: int = 10, delay: int = 5) -> KafkaConsumer:
    for attempt in range(retries):
        try:
            return KafkaConsumer(
                TOPIC_ACCESS_EVENTS,
                bootstrap_servers=KAFKA_BROKER,
                group_id=CONSUMER_GROUP,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="latest",
            )
        except NoBrokersAvailable:
            print(f"[detector] Kafka not ready, retry {attempt + 1}/{retries}")
            time.sleep(delay)
    raise RuntimeError("Could not connect to Kafka broker")


def get_alert_producer(retries: int = 10, delay: int = 5) -> KafkaProducer:
    for attempt in range(retries):
        try:
            return KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
        except NoBrokersAvailable:
            time.sleep(delay)
    raise RuntimeError("Could not connect to Kafka broker")


def publish_alert(producer: KafkaProducer, alert: dict):
    from .config import TOPIC_ANOMALY_ALERTS
    producer.send(TOPIC_ANOMALY_ALERTS, value=alert)
