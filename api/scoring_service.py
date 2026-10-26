"""
Real-time fraud scoring API.
POST /score  → fraud probability + decision
GET  /health → liveness check
"""
import json
import joblib
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

MODELS = Path("models")
FS     = Path("features/feature_store_data")

# ── Load at module level (fast — no CSV reading) ───────────────────────────
meta      = json.load(open(MODELS / "model_meta.json"))
model     = joblib.load(MODELS / "xgboost_model.pkl")
FEATURES  = meta["features"]
THRESHOLD = meta["threshold"]

_FREQ_MAPS: dict = {}
_CAT_MAPS:  dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading lookup maps...")
    _FREQ_MAPS.update(joblib.load(FS / "freq_maps.pkl"))
    _CAT_MAPS.update(joblib.load(FS  / "cat_maps.pkl"))
    print(f"API ready — {len(FEATURES)} features, threshold {THRESHOLD:.2f}")
    yield


app = FastAPI(title="Fraud Detection API", version="1.0", lifespan=lifespan)


# ── Schemas ────────────────────────────────────────────────────────────────
class Transaction(BaseModel):
    TransactionAmt:   float
    TransactionDT:    int
    card1:            Optional[float] = None
    card2:            Optional[float] = None
    card4:            Optional[str]   = None
    card6:            Optional[str]   = None
    addr1:            Optional[float] = None
    P_emaildomain:    Optional[str]   = None
    R_emaildomain:    Optional[str]   = None
    ProductCD:        Optional[str]   = None
    dist1:            Optional[float] = None
    dist2:            Optional[float] = None
    M1: Optional[str] = None
    M2: Optional[str] = None
    M3: Optional[str] = None
    M4: Optional[str] = None
    M5: Optional[str] = None
    M6: Optional[str] = None
    M7: Optional[str] = None
    M8: Optional[str] = None
    M9: Optional[str] = None
    V40:  Optional[float] = None
    V44:  Optional[float] = None
    V45:  Optional[float] = None
    V51:  Optional[float] = None
    V52:  Optional[float] = None
    V86:  Optional[float] = None
    V87:  Optional[float] = None
    V147: Optional[float] = None
    V148: Optional[float] = None
    V149: Optional[float] = None
    V154: Optional[float] = None
    V155: Optional[float] = None
    V156: Optional[float] = None
    V157: Optional[float] = None
    V158: Optional[float] = None
    V170: Optional[float] = None
    V171: Optional[float] = None
    V188: Optional[float] = None
    V189: Optional[float] = None
    V199: Optional[float] = None
    V200: Optional[float] = None
    V201: Optional[float] = None
    V228: Optional[float] = None
    V230: Optional[float] = None
    V242: Optional[float] = None
    V243: Optional[float] = None
    V244: Optional[float] = None
    V246: Optional[float] = None
    V257: Optional[float] = None
    V258: Optional[float] = None


class ScoreResponse(BaseModel):
    fraud_probability: float
    is_fraud:          bool
    threshold_used:    float
    risk_level:        str


# ── Feature engineering ────────────────────────────────────────────────────
def engineer(t: Transaction) -> np.ndarray:
    row = {}

    row["log_amt"]        = np.log1p(t.TransactionAmt)
    row["TransactionAmt"] = t.TransactionAmt
    row["hour"]           = (t.TransactionDT // 3600) % 24
    row["day"]            = (t.TransactionDT // 86400) % 7

    c1mean = _FREQ_MAPS["card1_amt_mean"].get(t.card1, 0) or 0
    c1std  = _FREQ_MAPS["card1_amt_std"].get(t.card1, 0)  or 0
    row["card1_amt_mean"]    = c1mean
    row["card1_amt_std"]     = c1std
    row["amt_vs_card1_mean"] = t.TransactionAmt / (c1mean + 1)

    row["card1_freq"]         = _FREQ_MAPS["card1"].get(t.card1, 0)
    row["card2_freq"]         = _FREQ_MAPS["card2"].get(t.card2, 0)
    row["addr1_freq"]         = _FREQ_MAPS["addr1"].get(t.addr1, 0)
    row["P_emaildomain_freq"] = _FREQ_MAPS["P_emaildomain"].get(t.P_emaildomain, 0)
    row["R_emaildomain_freq"] = _FREQ_MAPS["R_emaildomain"].get(t.R_emaildomain, 0)

    for col in ["ProductCD","card4","card6","M1","M2","M3","M4","M5","M6","M7","M8","M9"]:
        row[col] = _CAT_MAPS.get(col, {}).get(getattr(t, col, None), -1)

    for f in FEATURES:
        if f.startswith("V"):
            row[f] = getattr(t, f, None) or -999

    return np.array([[row.get(f, -999) for f in FEATURES]], dtype=np.float32)


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "xgboost", "features": len(FEATURES)}


@app.post("/score", response_model=ScoreResponse)
def score(transaction: Transaction):
    try:
        prob = float(model.predict_proba(engineer(transaction))[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    risk = "HIGH" if prob >= 0.8 else "MEDIUM" if prob >= 0.4 else "LOW"
    return ScoreResponse(
        fraud_probability=round(prob, 4),
        is_fraud=prob >= THRESHOLD,
        threshold_used=THRESHOLD,
        risk_level=risk,
    )
