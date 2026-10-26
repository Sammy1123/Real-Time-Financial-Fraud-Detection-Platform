"""
Kafka consumer — reads from 'transactions' topic, scores each transaction
via the scoring API, publishes fraud alerts to 'fraud-alerts' topic,
and appends flagged transactions to streaming/fraud_alerts.jsonl.
"""
import json
import os
import time
import requests
from pathlib import Path
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaConnectionError
from dotenv import load_dotenv

load_dotenv()

BROKER        = os.getenv("KAFKA_BROKER", "localhost:9092")
IN_TOPIC      = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "transactions")
OUT_TOPIC     = os.getenv("KAFKA_TOPIC_ALERTS", "fraud-alerts")
GROUP_ID      = os.getenv("KAFKA_GROUP_ID", "fraud-consumer-group")
API_URL       = os.getenv("SCORING_API_URL", "http://localhost:8000/score")
ALERTS_FILE   = Path("streaming/fraud_alerts.jsonl")

ALERTS_FILE.parent.mkdir(exist_ok=True)


def make_consumer() -> KafkaConsumer:
    return KafkaConsumer(
        IN_TOPIC,
        bootstrap_servers=BROKER,
        group_id=GROUP_ID,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )


def make_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks=1,
    )


def score_transaction(tx: dict) -> dict | None:
    """POST transaction to scoring API. Returns response dict or None on error."""
    try:
        resp = requests.post(API_URL, json=tx, timeout=2)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        print("  WARNING: Scoring API not reachable. Is it running?")
        return None
    except Exception as e:
        print(f"  WARNING: Scoring error — {e}")
        return None


def main():
    print(f"Connecting to Kafka broker at {BROKER}...")
    try:
        consumer = make_consumer()
        producer = make_producer()
    except KafkaConnectionError:
        print("ERROR: No Kafka broker available. Start Kafka first.")
        print("  Quick start: docker run -p 9092:9092 apache/kafka:3.7.0")
        return

    print(f"Listening on '{IN_TOPIC}' → scoring → publishing alerts to '{OUT_TOPIC}'")
    print(f"Fraud alerts logged to: {ALERTS_FILE}\n")

    processed = flagged = errors = 0
    start = time.time()

    try:
        for msg in consumer:
            tx     = msg.value
            result = score_transaction(tx)

            if result is None:
                errors += 1
                continue

            processed += 1
            alert = {
                "TransactionID":    tx.get("TransactionID"),
                "TransactionAmt":   tx.get("TransactionAmt"),
                "fraud_probability": result["fraud_probability"],
                "is_fraud":          result["is_fraud"],
                "risk_level":        result["risk_level"],
                "threshold_used":    result["threshold_used"],
                "timestamp":         time.time(),
            }

            if result["is_fraud"]:
                flagged += 1
                producer.send(OUT_TOPIC, value=alert)
                with open(ALERTS_FILE, "a") as f:
                    f.write(json.dumps(alert) + "\n")
                print(f"  🚨 FRAUD  | ID {alert['TransactionID']} "
                      f"| ${alert['TransactionAmt']:.2f} "
                      f"| prob {alert['fraud_probability']:.4f}")

            if processed % 200 == 0:
                elapsed = time.time() - start
                rate    = processed / elapsed
                print(f"  [{processed:,} processed | {flagged} flagged "
                      f"| {rate:.1f} tx/s | errors: {errors}]")

    except KeyboardInterrupt:
        print(f"\nStopped. Processed: {processed:,}  Flagged: {flagged}  Errors: {errors}")
    finally:
        producer.flush()
        producer.close()
        consumer.close()
        print("Consumer closed.")


if __name__ == "__main__":
    main()
