"""
Real-time fraud scoring API.
POST /score        -> fraud probability + decision (single transaction)
POST /score/batch  -> batch scoring up to 500 transactions
GET  /health       -> liveness check
"""
import hashlib
import json
import logging
import os
import time
import joblib
import numpy as np
from logging.handlers import RotatingFileHandler
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

try:
    import redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

# ── Logging setup ──────────────────────────────────────────────────────────
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

_fmt     = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
_fh      = RotatingFileHandler(LOGS_DIR / "api.log", maxBytes=5_000_000, backupCount=3)
_fh.setFormatter(_fmt)
_ch      = logging.StreamHandler()
_ch.setFormatter(_fmt)

logger = logging.getLogger("fraud_api")
logger.setLevel(logging.INFO)
logger.addHandler(_fh)
logger.addHandler(_ch)

MODELS         = Path("models")
FS             = Path("features/feature_store_data")
REDIS_URL      = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL      = int(os.getenv("REDIS_CACHE_TTL", "60"))

# ── Load at module level (fast — no CSV reading) ───────────────────────────
logger.info("Loading model and metadata...")
meta      = json.load(open(MODELS / "model_meta.json"))
model     = joblib.load(MODELS / "xgboost_model.pkl")
FEATURES  = meta["features"]
THRESHOLD = meta["threshold"]
logger.info(f"Model loaded — {len(FEATURES)} features, threshold {THRESHOLD:.2f}")

_FREQ_MAPS: dict = {}
_CAT_MAPS:  dict = {}
_redis_client     = None


def _get_redis():
    """Return Redis client if available, else None."""
    if not _REDIS_AVAILABLE or _redis_client is None:
        return None
    return _redis_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis_client
    logger.info("Loading lookup maps...")
    _FREQ_MAPS.update(joblib.load(FS / "freq_maps.pkl"))
    _CAT_MAPS.update(joblib.load(FS  / "cat_maps.pkl"))
    logger.info("Lookup maps loaded")
    if _REDIS_AVAILABLE:
        try:
            _redis_client = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
            _redis_client.ping()
            logger.info(f"Redis cache connected ({REDIS_URL}) — TTL {CACHE_TTL}s")
        except Exception as e:
            _redis_client = None
            logger.warning(f"Redis not reachable ({e}) — caching disabled")
    else:
        logger.warning("redis-py not installed — caching disabled")
    logger.info(f"API ready — {len(FEATURES)} features, threshold {THRESHOLD:.2f}")
    yield
    if _redis_client:
        _redis_client.close()
        logger.info("Redis connection closed")


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
    cache_hit:         bool = False


class BatchRequest(BaseModel):
    transactions: List[Transaction]

    class Config:
        json_schema_extra = {"example": {"transactions": []}}


class BatchScoreResponse(BaseModel):
    results:      List[ScoreResponse]
    total:        int
    fraud_count:  int
    avg_prob:     float


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


def _cache_key(vec: np.ndarray) -> str:
    return "fraud:score:" + hashlib.md5(vec.tobytes()).hexdigest()


def _cache_get(key: str) -> Optional[dict]:
    rc = _get_redis()
    if rc is None:
        return None
    try:
        val = rc.get(key)
        return json.loads(val) if val else None
    except Exception as e:
        logger.warning(f"Redis GET failed: {e}")
        return None


def _cache_set(key: str, value: dict) -> None:
    rc = _get_redis()
    if rc is None:
        return
    try:
        rc.setex(key, CACHE_TTL, json.dumps(value))
    except Exception as e:
        logger.warning(f"Redis SET failed: {e}")


def _to_risk(prob: float) -> str:
    return "HIGH" if prob >= 0.8 else "MEDIUM" if prob >= 0.4 else "LOW"


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    rc           = _get_redis()
    cache_status = "connected" if rc else "disabled"
    logger.info(f"Health check — cache: {cache_status}")
    return {"status": "ok", "model": "xgboost", "features": len(FEATURES), "cache": cache_status}


@app.post("/score", response_model=ScoreResponse)
def score(transaction: Transaction):
    t0 = time.perf_counter()
    try:
        vec = engineer(transaction)
    except Exception as e:
        logger.exception(f"Feature engineering failed for TransactionID={transaction.TransactionDT}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    key    = _cache_key(vec)
    cached = _cache_get(key)
    if cached:
        ms = (time.perf_counter() - t0) * 1000
        logger.info(f"SCORE | id={transaction.TransactionDT} | prob={cached['fraud_probability']} "
                    f"| risk={cached['risk_level']} | cache=HIT | {ms:.1f}ms")
        return ScoreResponse(**cached, cache_hit=True)

    try:
        prob = float(model.predict_proba(vec)[0, 1])
    except Exception as e:
        logger.exception(f"Model inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    result = dict(
        fraud_probability=round(prob, 4),
        is_fraud=prob >= THRESHOLD,
        threshold_used=THRESHOLD,
        risk_level=_to_risk(prob),
    )
    _cache_set(key, result)
    ms = (time.perf_counter() - t0) * 1000
    logger.info(f"SCORE | id={transaction.TransactionDT} | prob={result['fraud_probability']} "
                f"| risk={result['risk_level']} | fraud={result['is_fraud']} | cache=MISS | {ms:.1f}ms")
    return ScoreResponse(**result, cache_hit=False)


@app.post("/score/batch", response_model=BatchScoreResponse)
def score_batch(batch: BatchRequest):
    if not batch.transactions:
        raise HTTPException(status_code=422, detail="transactions list is empty")
    if len(batch.transactions) > 500:
        raise HTTPException(status_code=422, detail="batch size exceeds limit of 500")

    t0 = time.perf_counter()
    try:
        X     = np.vstack([engineer(t) for t in batch.transactions])
        probs = model.predict_proba(X)[:, 1]
    except Exception as e:
        logger.exception(f"Batch scoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    results = [
        ScoreResponse(
            fraud_probability=round(float(p), 4),
            is_fraud=float(p) >= THRESHOLD,
            threshold_used=THRESHOLD,
            risk_level=_to_risk(float(p)),
        )
        for p in probs
    ]
    fraud_count = sum(r.is_fraud for r in results)
    ms = (time.perf_counter() - t0) * 1000
    logger.info(f"BATCH | n={len(results)} | fraud={fraud_count} | avg_prob={probs.mean():.4f} | {ms:.1f}ms")
    return BatchScoreResponse(
        results=results,
        total=len(results),
        fraud_count=fraud_count,
        avg_prob=round(float(probs.mean()), 4),
    )
