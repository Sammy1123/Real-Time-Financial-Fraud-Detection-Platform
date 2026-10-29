"""
Shared inference preprocessing — used by API, consumer, and ensemble.

Loads feature metadata + frequency/categorical maps produced by
features/feature_engineering.py, and turns a raw transaction dict into
the exact feature vector the models were trained on.
"""
import json
import joblib
import numpy as np
from pathlib import Path

FS = Path("features/feature_store_data")

FEATURES      = json.load(open(FS / "features.json"))
_freq_maps    = joblib.load(FS / "freq_maps.pkl")
_cat_maps     = joblib.load(FS / "cat_maps.pkl")


def _load_graph_cache() -> dict:
    """Return the nightly graph cache (or {} if not built yet).

    Provides per-card/addr/email precomputed GNN fraud probabilities plus a
    global base rate. Refreshed by api/build_graph_cache.py.
    """
    cache_path = FS / "graph_cache.pkl"
    if not cache_path.exists():
        return {}
    try:
        return joblib.load(cache_path)
    except Exception:
        return {}


_graph_cache = _load_graph_cache()


def gnn_signal(tx: dict) -> float:
    """GNN fraud probability for a live transaction from the graph cache.

    Looks up the transaction's card1 / addr1 / P_emaildomain against the
    nightly-built cache and averages the hits. Unknown identifiers fall
    back to the global base rate.
    """
    if not _graph_cache:
        return 0.0346  # global fraud base rate

    hits = []
    for key, field in (("card_probs", "card1"),
                       ("addr_probs", "addr1"),
                       ("mail_probs", "P_emaildomain")):
        val = tx.get(field)
        if val is None:
            continue
        p = _graph_cache.get(key, {}).get(val)
        if p is not None:
            hits.append(p)

    if hits:
        return float(np.mean(hits))
    return _graph_cache.get("base_rate", 0.0346)


def preprocess_transaction(tx: dict) -> np.ndarray:
    """Return a feature vector of len(FEATURES) matching training schema."""
    row = dict(tx)

    # Engineered scalars (same logic as feature_engineering.py)
    amt = row.get("TransactionAmt") or 0.0
    row["log_amt"] = float(np.log1p(max(amt, 0)))

    dt = row.get("TransactionDT") or 0
    row["hour"] = float((dt // 3600) % 24)
    row["day"]  = float((dt // 86400) % 7)

    card1 = row.get("card1")
    if card1 is not None:
        row.setdefault("card1_amt_mean", amt)
        row.setdefault("card1_amt_std", 0.0)
        row.setdefault("amt_vs_card1_mean", 1.0)
        row.setdefault("card1_tx_count_1h", 1.0)
        row.setdefault("card1_tx_amount_1h", amt)
        row.setdefault("card1_tx_count_24h", 1.0)
        row.setdefault("card1_tx_amount_24h", amt)
        row.setdefault("card1_addr1_freq", _freq_maps.get("card1", {}).get(card1, 0))

    # Frequency encodings
    for col, mapping in _freq_maps.items():
        val = row.get(col)
        row.setdefault(f"{col}_freq",
                       mapping.get(val, 0) if val is not None else 0)

    # Target / categorical encodings
    for col, mapping in _cat_maps.items():
        val = row.get(col)
        row.setdefault(f"{col}_target",
                       mapping.get(val, mapping.get("__default__", 0))
                       if val is not None else 0)

    out = np.full(len(FEATURES), np.nan, dtype=np.float32)
    for i, feat in enumerate(FEATURES):
        v = row.get(feat)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            try:
                out[i] = float(v)
            except (TypeError, ValueError):
                pass
    return np.nan_to_num(out, nan=-999.0)
