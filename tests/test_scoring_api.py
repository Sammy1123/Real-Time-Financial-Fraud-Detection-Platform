"""
Integration tests for the fraud scoring API.
Requires the API to be running at http://localhost:8000.
Run: pytest tests/ -v
"""
import pytest
import requests


# ── /health ─────────────────────────────────────────────────────────────────
def test_health_status(base_url):
    r = requests.get(f"{base_url}/health")
    assert r.status_code == 200

def test_health_fields(base_url):
    data = requests.get(f"{base_url}/health").json()
    assert data["status"] == "ok"
    assert data["model"] == "xgboost"
    assert data["features"] == 62
    assert "cache" in data


# ── /score ───────────────────────────────────────────────────────────────────
def test_score_returns_200(base_url, legit_tx):
    r = requests.post(f"{base_url}/score", json=legit_tx)
    assert r.status_code == 200

def test_score_response_schema(base_url, legit_tx):
    data = requests.post(f"{base_url}/score", json=legit_tx).json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "threshold_used" in data
    assert "risk_level" in data
    assert "cache_hit" in data

def test_score_probability_range(base_url, legit_tx):
    data = requests.post(f"{base_url}/score", json=legit_tx).json()
    assert 0.0 <= data["fraud_probability"] <= 1.0

def test_score_risk_level_valid(base_url, legit_tx):
    data = requests.post(f"{base_url}/score", json=legit_tx).json()
    assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")

def test_score_legit_tx_low_risk(base_url, legit_tx):
    data = requests.post(f"{base_url}/score", json=legit_tx).json()
    assert data["is_fraud"] is False
    assert data["fraud_probability"] < 0.5

def test_score_high_risk_tx_higher_prob(base_url, legit_tx, high_risk_tx):
    legit_prob = requests.post(f"{base_url}/score", json=legit_tx).json()["fraud_probability"]
    risky_prob = requests.post(f"{base_url}/score", json=high_risk_tx).json()["fraud_probability"]
    assert risky_prob > legit_prob

def test_score_repeat_same_probability(base_url, legit_tx):
    prob1 = requests.post(f"{base_url}/score", json=legit_tx).json()["fraud_probability"]
    prob2 = requests.post(f"{base_url}/score", json=legit_tx).json()["fraud_probability"]
    assert prob1 == prob2

def test_score_missing_optional_fields(base_url):
    """Only required fields — all optional fields omitted."""
    minimal = {"TransactionAmt": 100.0, "TransactionDT": 86400}
    r = requests.post(f"{base_url}/score", json=minimal)
    assert r.status_code == 200
    assert "fraud_probability" in r.json()

def test_score_missing_required_field(base_url):
    """Missing TransactionAmt — should return 422."""
    r = requests.post(f"{base_url}/score", json={"TransactionDT": 86400})
    assert r.status_code == 422

def test_score_threshold_used_matches_meta(base_url, legit_tx):
    import json
    from pathlib import Path
    meta = json.load(open(Path("models/model_meta.json")))
    data = requests.post(f"{base_url}/score", json=legit_tx).json()
    assert abs(data["threshold_used"] - meta["threshold"]) < 1e-6


# ── /score/batch ─────────────────────────────────────────────────────────────
def test_batch_returns_200(base_url, legit_tx):
    r = requests.post(f"{base_url}/score/batch", json={"transactions": [legit_tx, legit_tx]})
    assert r.status_code == 200

def test_batch_response_schema(base_url, legit_tx):
    data = requests.post(f"{base_url}/score/batch", json={"transactions": [legit_tx]}).json()
    assert "results" in data
    assert "total" in data
    assert "fraud_count" in data
    assert "avg_prob" in data

def test_batch_total_matches_input(base_url, legit_tx):
    txs = [legit_tx] * 5
    data = requests.post(f"{base_url}/score/batch", json={"transactions": txs}).json()
    assert data["total"] == 5

def test_batch_fraud_count_lte_total(base_url, legit_tx, high_risk_tx):
    txs = [legit_tx, high_risk_tx, legit_tx]
    data = requests.post(f"{base_url}/score/batch", json={"transactions": txs}).json()
    assert data["fraud_count"] <= data["total"]

def test_batch_avg_prob_in_range(base_url, legit_tx):
    data = requests.post(f"{base_url}/score/batch", json={"transactions": [legit_tx] * 3}).json()
    assert 0.0 <= data["avg_prob"] <= 1.0

def test_batch_empty_returns_422(base_url):
    r = requests.post(f"{base_url}/score/batch", json={"transactions": []})
    assert r.status_code == 422

def test_batch_oversized_returns_422(base_url, legit_tx):
    r = requests.post(f"{base_url}/score/batch", json={"transactions": [legit_tx] * 501})
    assert r.status_code == 422

def test_batch_single_tx_matches_single_score(base_url, legit_tx):
    single = requests.post(f"{base_url}/score", json=legit_tx).json()["fraud_probability"]
    batch  = requests.post(f"{base_url}/score/batch", json={"transactions": [legit_tx]}).json()
    assert batch["results"][0]["fraud_probability"] == single
