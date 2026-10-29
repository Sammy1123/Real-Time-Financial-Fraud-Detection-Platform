"""
Kafka producer — streams transactions from data/processed/test.parquet
to the 'transactions' topic, simulating real-time transaction flow.
Each message is validated against the transaction schema before publishing;
invalid messages are dropped and counted (optionally sent to the DLQ topic).
"""
import json
import time
import os
import pandas as pd
import numpy as np
from pathlib import Path
from kafka import KafkaProducer
from kafka.errors import KafkaConnectionError
from dotenv import load_dotenv

load_dotenv()

BROKER     = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC      = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "transactions")
DLQ_TOPIC  = os.getenv("KAFKA_TOPIC_DLQ", "fraud-dlq")
DELAY_MS   = int(os.getenv("PRODUCER_DELAY_MS", 100)) / 1000
PROC       = Path("data/processed")

# Raw test data for the original feature columns the API expects
RAW_COLS = [
    "TransactionID", "TransactionDT", "TransactionAmt",
    "card1", "card2", "card4", "card6",
    "addr1", "P_emaildomain", "R_emaildomain", "ProductCD",
    "dist1", "dist2",
    "M1","M2","M3","M4","M5","M6","M7","M8","M9",
    "V40","V44","V45","V51","V52","V86","V87",
    "V147","V148","V149","V154","V155","V156","V157","V158",
    "V170","V171","V188","V189","V199","V200","V201",
    "V228","V230","V242","V243","V244","V246","V257","V258",
]


def make_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=3,
    )


def load_transactions() -> pd.DataFrame:
    raw_tx   = pd.read_csv(Path("data/raw") / "test_transaction.csv")
    raw_id   = pd.read_csv(Path("data/raw") / "test_identity.csv")
    df       = raw_tx.merge(raw_id, on="TransactionID", how="left")
    cols     = [c for c in RAW_COLS if c in df.columns]
    return df[cols].where(pd.notnull(df[cols]), None)  # NaN → None for JSON


def serialize(row: dict) -> dict:
    """Convert numpy types to Python natives for JSON serialization."""
    return {
        k: (None if v is None or (isinstance(v, float) and np.isnan(v))
            else int(v) if isinstance(v, (np.integer,))
            else float(v) if isinstance(v, (np.floating,))
            else v)
        for k, v in row.items()
    }


# ── Schema validation ──────────────────────────────────────────────────────
REQUIRED_FIELDS = {
    "TransactionID":   int,
    "TransactionDT":   int,
    "TransactionAmt":  (int, float),
}

NUMERIC_FIELDS = {"TransactionAmt", "card1", "card2", "addr1", "dist1", "dist2"}


def validate_message(msg: dict) -> tuple[bool, str]:
    """Validate a serialized transaction against the schema.

    Returns (ok, reason). A valid message must have required fields with the
    right types, numeric fields numeric-or-None, and a positive amount.
    """
    if not isinstance(msg, dict):
        return False, "message is not a JSON object"

    for field, expected in REQUIRED_FIELDS.items():
        if field not in msg:
            return False, f"missing required field '{field}'"
        if msg[field] is not None and not isinstance(msg[field], expected):
            return False, f"field '{field}' has wrong type: {type(msg[field]).__name__}"

    for field in NUMERIC_FIELDS:
        val = msg.get(field)
        if val is not None and not isinstance(val, (int, float)):
            return False, f"numeric field '{field}' is {type(val).__name__}"

    amt = msg.get("TransactionAmt")
    if amt is not None and amt <= 0:
        return False, f"TransactionAmt must be positive, got {amt}"

    if msg.get("TransactionID") is not None and msg["TransactionID"] < 0:
        return False, "TransactionID must be non-negative"

    return True, ""


def main():
    print(f"Connecting to Kafka broker at {BROKER}...")
    try:
        producer = make_producer()
    except KafkaConnectionError:
        print("ERROR: No Kafka broker available. Start Kafka first.")
        print("  Quick start: docker run -p 9092:9092 apache/kafka:3.7.0")
        return

    df = load_transactions()
    print(f"Loaded {len(df):,} test transactions. Publishing to '{TOPIC}'...")
    print(f"Delay between messages: {DELAY_MS * 1000:.0f}ms  (Ctrl+C to stop)\n")

    sent = invalid = 0
    try:
        for _, row in df.iterrows():
            msg = serialize(row.to_dict())

            ok, reason = validate_message(msg)
            if not ok:
                invalid += 1
                producer.send(DLQ_TOPIC, value={"error": reason, "payload": msg})
                continue

            producer.send(TOPIC, value=msg)
            sent += 1

            if sent % 500 == 0:
                producer.flush()
                print(f"  Sent {sent:,} transactions ({invalid:,} rejected)...")

            time.sleep(DELAY_MS)

    except KeyboardInterrupt:
        print(f"\nStopped. Sent {sent:,} transactions, rejected {invalid:,} invalid.")
    finally:
        producer.flush()
        producer.close()
        print("Producer closed.")


if __name__ == "__main__":
    main()
