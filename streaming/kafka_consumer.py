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

BROKER: str        = os.getenv("KAFKA_BROKER", "localhost:9092")
IN_TOPIC: str      = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "transactions")
OUT_TOPIC: str     = os.getenv("KAFKA_TOPIC_ALERTS", "fraud-alerts")
DLQ_TOPIC: str     = os.getenv("KAFKA_TOPIC_DLQ", "fraud-dlq")
GROUP_ID: str      = os.getenv("KAFKA_GROUP_ID", "fraud-consumer-group")
API_URL: str       = os.getenv("SCORING_API_URL", "http://localhost:8000/score")
ALERTS_FILE: Path  = Path("streaming/fraud_alerts.jsonl")

MAX_RETRIES: int   = 3
RETRY_BASE_S: float = 1.0   # doubles each attempt: 1s, 2s, 4s

ALERTS_FILE.parent.mkdir(exist_ok=True)


def make_consumer() -> KafkaConsumer:
    """Subscribe to IN_TOPIC at the earliest committed offset."""
    return KafkaConsumer(
        IN_TOPIC,
        bootstrap_servers=BROKER,
        group_id=GROUP_ID,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )


def make_producer() -> KafkaProducer:
    """Create a JSON producer for alerts and dead-letter messages."""
    return KafkaProducer(
        bootstrap_servers=BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks=1,
    )


def score_transaction(tx: dict) -> dict | None:
    """POST to the scoring API with exponential backoff.

    Retries up to MAX_RETRIES on connection errors and non-2xx responses,
    doubling the wait between attempts. Returns the parsed response dict,
    or None if every attempt fails (caller should dead-letter).
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(API_URL, json=tx, timeout=2)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            print(f"  WARNING: Scoring API not reachable (attempt {attempt}/{MAX_RETRIES})")
        except Exception as e:
            print(f"  WARNING: Scoring error (attempt {attempt}/{MAX_RETRIES}) — {e}")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BASE_S * (2 ** (attempt - 1)))
    return None


def main() -> None:
    """Consume transactions, score them, publish alerts.

    Flow per message: score via API → on success publish an alert to
    OUT_TOPIC when is_fraud and append it to ALERTS_FILE; on failure after
    retries dead-letter the transaction to DLQ_TOPIC. Progress prints every
    200 messages with throughput and error counters.
    """
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

    processed = flagged = errors = dead_lettered = 0
    start = time.time()

    try:
        for msg in consumer:
            tx     = msg.value
            result = score_transaction(tx)

            if result is None:
                errors += 1
                dead_lettered += 1
                producer.send(DLQ_TOPIC, value={"tx": tx, "reason": "scoring_failed", "timestamp": time.time()})
                print(f"  DLQ: transaction {tx.get('TransactionID')} sent to '{DLQ_TOPIC}' after {MAX_RETRIES} retries")
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
                      f"| {rate:.1f} tx/s | errors: {errors} | dlq: {dead_lettered}]")

    except KeyboardInterrupt:
        print(f"\nStopped. Processed: {processed:,}  Flagged: {flagged}  Errors: {errors}  DLQ: {dead_lettered}")
    finally:
        producer.flush()
        producer.close()
        consumer.close()
        print("Consumer closed.")


if __name__ == "__main__":
    main()
