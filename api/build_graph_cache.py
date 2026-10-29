"""
Build the serving graph cache for the scoring API.

Runs nightly (scheduled via scheduler.py). Produces:
  features/feature_store_data/graph_cache.pkl containing
    - checkpoint metadata (which GNN weights this cache aligns to)
    - per-TransactionID precomputed GNN fraud probability for every node
      in the training graph
The API joins a live transaction's card1/address/email identifiers against
this cache: an exact identifier match returns the cached neighbourhood
probability, and unknown identifiers get the global base rate fallback.
"""
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib
from pathlib import Path

import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.train_gnn import FraudGAT, build_graph

FS     = Path("features/feature_store_data")
MODELS = Path("models")
CACHE  = FS / "graph_cache.pkl"


def build_cache():
    print("Building serving graph cache...")
    t0 = time.time()
    data, df_full, scaler = build_graph()
    print(f"Graph: {data.num_nodes:,} nodes, {data.num_edges:,} edges "
          f"({time.time() - t0:.0f}s)")

    ckpt  = torch.load(MODELS / "gnn_model.pt", map_location="cpu", weights_only=False)
    model = FraudGAT(in_channels=ckpt["in_channels"],
                     hidden=ckpt["hidden"], heads=ckpt["heads"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    device = torch.device("cpu")
    BATCH  = 8192
    src_all, dst_all = data.edge_index[0].numpy(), data.edge_index[1].numpy()
    order   = np.argsort(src_all, kind="stable")
    adj_dst = dst_all[order]
    adj_off = np.searchsorted(src_all[order], np.arange(data.num_nodes + 1))

    probs   = np.zeros(data.num_nodes, dtype=np.float32)
    rng     = np.random.default_rng(0)
    t0      = time.time()
    with torch.no_grad():
        for i in range(0, data.num_nodes, BATCH):
            seeds = np.arange(i, min(i + BATCH, data.num_nodes))
            nbrs  = [adj_dst[adj_off[s]:adj_off[s + 1]] for s in seeds]
            nbrs  = np.concatenate([
                n if len(n) <= 5 else rng.choice(n, size=5, replace=False)
                for n in nbrs]) if nbrs else np.empty(0, dtype=np.int64)
            n_id = np.unique(np.concatenate([seeds, nbrs])).astype(np.int64)
            pos  = np.full(data.num_nodes, -1, dtype=np.int64)
            pos[n_id] = np.arange(len(n_id))
            keep = (pos[src_all] >= 0) & (pos[dst_all] >= 0)
            edge = np.vstack([pos[src_all[keep]], pos[dst_all[keep]]])
            out  = model(data.x[n_id], torch.as_tensor(edge, dtype=torch.long))
            probs[n_id] = F.softmax(out, dim=1)[:, 1].numpy()
            if (i // BATCH) % 10 == 0:
                print(f"  scored {i + len(seeds):,}/{data.num_nodes:,} nodes "
                      f"({time.time() - t0:.0f}s)", flush=True)

    tids = df_full["TransactionID"].values
    card_groups = df_full.groupby("card1")["TransactionID"].apply(list).to_dict()
    addr_groups = df_full.dropna(subset=["addr1"]).groupby("addr1")["TransactionID"].apply(list).to_dict()
    mail_groups = df_full.dropna(subset=["P_emaildomain"]).groupby("P_emaildomain")["TransactionID"].apply(list).to_dict()

    cache = {
        "gnn_roc_auc": ckpt["roc_auc"],
        "gnn_pr_auc":  ckpt["pr_auc"],
        "built_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "tx_probs":    dict(zip(tids.tolist(), probs.tolist())),
        "card_probs":  {k: float(np.mean([probs[tids.tolist().index(t)] for t in v[:50]]))
                        for k, v in card_groups.items()},
        "addr_probs":  {},
        "mail_probs":  {},
        "base_rate":   float(df_full["isFraud"].mean()),
    }
    # vectorize the addr/mail aggregations (much bigger groups than card)
    for name, groups in (("addr_probs", addr_groups), ("mail_probs", mail_groups)):
        tid_pos = {t: i for i, t in enumerate(tids)}
        cache[name] = {k: float(np.mean([probs[tid_pos[t]] for t in v[:100]]))
                       for k, v in groups.items()}

    joblib.dump(cache, CACHE)
    size_mb = CACHE.stat().st_size / 1048576
    print(f"\nCache saved to {CACHE} ({size_mb:.1f} MB)")
    print(f"  transactions: {len(cache['tx_probs']):,}")
    print(f"  card1 keys  : {len(cache['card_probs']):,}")
    print(f"  addr1 keys  : {len(cache['addr_probs']):,}")
    print(f"  email keys  : {len(cache['mail_probs']):,}")
    print(f"  aligned to GNN ckpt — ROC-AUC {ckpt['roc_auc']:.4f}")


if __name__ == "__main__":
    build_cache()
