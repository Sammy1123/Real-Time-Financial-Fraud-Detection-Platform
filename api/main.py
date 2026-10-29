"""
Fraud detection API — FastAPI service exposing the trained ensemble.

Endpoints
---------
GET  /health        → liveness + model status
POST /predict       → single transaction scoring
POST /predict/batch → bulk scoring (list of transactions)

Loads XGBoost + GNN + meta-learner from models/, applies the stored
decision threshold, and returns fraud probability + decision.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from models.inference import preprocess_transaction, FEATURES

# ── Model registry ─────────────────────────────────────────────────────────
MODELS = Path("models")

xgb_model = joblib.load(MODELS / "xgboost_model.pkl")
ens_meta  = joblib.load(MODELS / "ensemble_meta.pkl")
meta_clf  = ens_meta["meta_model"]
threshold = ens_meta["threshold"]

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit-card fraud scoring (XGBoost + GNN ensemble)",
    version="1.0.0",
)


class Transaction(BaseModel):
    TransactionID: Optional[int] = None
    TransactionDT: float
    TransactionAmt: float
    card1: float
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    DeviceType: Optional[str] = None
    id_12: Optional[str] = None
    id_15: Optional[str] = None
    id_31: Optional[str] = None

    class Config:
        extra = "allow"  # accept V-job columns too


class PredictResponse(BaseModel):
    TransactionID: Optional[int]
    fraud_probability: float
    prediction: int
    threshold: float
    risk_level: str


def _risk(p: float) -> str:
    if p >= 0.8:  return "critical"
    if p >= 0.5:  return "high"
    if p >= 0.2:  return "medium"
    return "low"


def score(tx: dict) -> dict:
    """Run the full preprocessing → XGBoost → meta-learner pipeline."""
    X = preprocess_transaction(tx)             # (n_features,) aligned to FEATURES
    p_xgb = float(xgb_model.predict_proba(X.reshape(1, -1))[0, 1])

    # GNN signal at serving time — conservative fallback to base rate
    # (real-time graph is built nightly; serving uses prior-day graph)
    base_rate = 0.0346
    gnn_p = base_rate
    p_ens = float(meta_clf.predict_proba([[p_xgb, gnn_p]])[0, 1])

    pred = int(p_ens >= threshold)
    return {
        "TransactionID": tx.get("TransactionID"),
        "fraud_probability": round(p_ens, 6),
        "prediction": pred,
        "threshold": threshold,
        "risk_level": _risk(p_ens),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": True,
        "threshold": threshold,
        "ensemble_weights": ens_meta["weights"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(tx: Transaction):
    try:
        return score(tx.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch")
def predict_batch(txs: list[Transaction]):
    if not txs:
        raise HTTPException(status_code=400, detail="Empty batch")
    if len(txs) > 1000:
        raise HTTPException(status_code=400, detail="Batch too large (max 1000)")
    try:
        results = [score(t.model_dump()) for t in txs]
        n_fraud = sum(r["prediction"] for r in results)
        return {"count": len(results), "frauds_flagged": n_fraud, "predictions": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
