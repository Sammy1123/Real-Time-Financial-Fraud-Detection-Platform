"""
Build the serving graph cache for the scoring API.

Runs nightly (scheduled via scheduler.py). Produces:
  features/feature_store_data/graph_cache.pkl containing
    - checkpoint metadata (which GNN weights this cache aligns to)
    - per-TransactionID precomputed GNN fraud probability for every node
      in the graph (TRAIN + streaming TEST transactions)
    - per-card1 / addr1 / P_emaildomain aggregated probabilities covering
      both train and test identifiers
The API joins a live transaction's card1/address/email identifiers against
this cache: an exact identifier match returns the cached neighbourhood
probability, and unknown identifiers get the global base rate fallback.

Inference matches the training receptive field: 2-hop sampled subgraphs
(k=10 at hop 1, k=5 at hop 2) with the GAT's two conv layers.
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

HOP1_K, HOP2_K = 10, 5   # match training sample_subgraph(k=10, k2=5)


def sample_subgraph(adj_dst, adj_off, num_nodes, seeds, src_all, dst_all,
                    rng, k1=HOP1_K, k2=HOP2_K):
    """2-hop sampled subgraph (seeds + neighbours) — same receptive field
    the GAT sees at training time. Returns (n_id, local_edge_index)."""
    nbrs1 = [adj_dst[adj_off[s]:adj_off[s + 1]] for s in seeds]
    nbrs1 = np.concatenate([
        n if len(n) <= k1 else rng.choice(n, size=k1, replace=False)
        for n in nbrs1]) if nbrs1 else np.empty(0, dtype=np.int64)
    nbrs2 = [adj_dst[adj_off[n]:adj_off[n + 1]] for n in nbrs1]
    nbrs2 = np.concatenate([
        m if len(m) <= k2 else rng.choice(m, size=k2, replace=False)
        for m in nbrs2]) if nbrs2 else np.empty(0, dtype=np.int64)
    n_id = np.unique(np.concatenate([seeds, nbrs1, nbrs2])).astype(np.int64)
    pos  = np.full(num_nodes, -1, dtype=np.int64)
    pos[n_id] = np.arange(len(n_id))
    keep = (pos[src_all] >= 0) & (pos[dst_all] >= 0)
    edge = np.vstack([pos[src_all[keep]], pos[dst_all[keep]]])
    return n_id, edge


def build_cache():
    print("Building serving graph cache (train + streaming test nodes)...")
    t0 = time.time()
    data, df_full, scaler = build_graph(include_test=True)
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

    probs = np.zeros(data.num_nodes, dtype=np.float32)
    rng   = np.random.default_rng(0)
    t0    = time.time()
    with torch.no_grad():
        for i in range(0, data.num_nodes, BATCH):
            seeds = np.arange(i, min(i + BATCH, data.num_nodes))
            n_id, edge = sample_subgraph(adj_dst, adj_off, data.num_nodes,
                                         seeds, src_all, dst_all, rng)
            out = model(data.x[n_id].to(device),
                        torch.as_tensor(edge, dtype=torch.long, device=device))
            probs[n_id] = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            if (i // BATCH) % 20 == 0:
                print(f"  scored {i + len(seeds):,}/{data.num_nodes:,} nodes "
                      f"({time.time() - t0:.0f}s)", flush=True)

    tids = df_full["TransactionID"].values

    # Aggregate identifier -> mean GNN prob across train AND test members
    def group_means(col, cap):
        groups = df_full.dropna(subset=[col]).groupby(col)["TransactionID"].apply(list)
        tid_pos = pd.Series(np.arange(len(tids)), index=tids)
        out = {}
        for k, v in groups.items():
            positions = tid_pos.reindex(pd.Index(v[:cap])).dropna().values.astype(np.int64)
            if len(positions):
                out[k] = float(np.mean(probs[positions]))
        return out

    card_probs = group_means("card1", 50)
    addr_probs = group_means("addr1", 100)
    mail_probs = group_means("P_emaildomain", 100)

    cache = {
        "gnn_roc_auc": ckpt["roc_auc"],
        "gnn_pr_auc":  ckpt["pr_auc"],
        "gnn_sampling": ckpt.get("sampling", "2-hop"),
        "includes_test": True,
        "built_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "tx_probs":    dict(zip(tids.tolist(), probs.tolist())),
        "card_probs":  card_probs,
        "addr_probs":  addr_probs,
        "mail_probs":  mail_probs,
        "base_rate":   float(df_full["isFraud"].fillna(0).mean()),
    }

    joblib.dump(cache, CACHE)
    size_mb = CACHE.stat().st_size / 1048576
    print(f"\nCache saved to {CACHE} ({size_mb:.1f} MB)")
    print(f"  transactions: {len(cache['tx_probs']):,} (train + streaming test)")
    print(f"  card1 keys  : {len(cache['card_probs']):,}")
    print(f"  addr1 keys  : {len(cache['addr_probs']):,}")
    print(f"  email keys  : {len(cache['mail_probs']):,}")
    print(f"  aligned to GNN ckpt — ROC-AUC {ckpt['roc_auc']:.4f} "
          f"(sampling: {ckpt.get('sampling', 'unknown')})")


if __name__ == "__main__":
    build_cache()
