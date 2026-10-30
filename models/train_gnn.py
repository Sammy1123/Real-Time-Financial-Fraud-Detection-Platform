"""
GNN fraud detection model using GAT (Graph Attention Network).
Graph: nodes = ALL transactions, edges = shared card1 / addr1 / P_emaildomain.
Node features scaled with StandardScaler to fix high-loss issue.
Trains on the full dataset with manual 2-hop subgraph batching (CPU-friendly).
Saves model to models/gnn_model.pt and logs to MLflow.
"""
import json
import time
import warnings as _warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import mlflow
from pathlib import Path
from torch import nn

# torch_geometric's internals call torch.jit.script at import time, emitting
# a FutureWarning. Under pytest -W error::FutureWarning that aborts the import
# entirely, so suppress warnings around the import and restore state after.
with _warnings.catch_warnings():
    _warnings.simplefilter("ignore", FutureWarning)
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

PROC   = Path("data/processed")
FS     = Path("features/feature_store_data")
MODELS = Path("models")

MAX_LINKS = 4          # edges per node per edge-type (chain linking)
BATCH     = 2048
EPOCHS    = 30


def chain_edges(codes: np.ndarray, max_links: int = MAX_LINKS):
    """Vectorized chain linking: within each group (same code), connect each
    node to up to max_links following nodes in sorted order. Code -1 (NaN) is
    skipped. Returns bidirectional (src, dst) int64 arrays — O(n*max_links)."""
    order        = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    srcs, dsts   = [], []
    for off in range(1, max_links + 1):
        same  = sorted_codes[:-off] == sorted_codes[off:]
        valid = same & (sorted_codes[:-off] >= 0)
        srcs.append(order[:-off][valid])
        dsts.append(order[off:][valid])
    src = np.concatenate(srcs)
    dst = np.concatenate(dsts)
    return (np.concatenate([src, dst]).astype(np.int64),
            np.concatenate([dst, src]).astype(np.int64))


def build_graph(include_test: bool = False):
    """Build the full-dataset graph (scaled features, multi-edge).

    Deterministic. Returns (data, df, scaler). df is the parquet in graph
    node order (row i == node i).

    include_test=True appends the streaming test transactions as unlabeled
    nodes (edges formed over shared identifiers), so serving caches can
    cover them. Scaler is always fit on the TRAIN rows only.
    """
    df       = pd.read_parquet(PROC / "train.parquet")
    FEATURES = json.load(open(FS / "features.json"))

    # Scale node features (fixes high loss) — fit on train only
    X_train  = df[FEATURES].fillna(-999).values.astype(np.float32)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    if include_test:
        test_df = pd.read_parquet(PROC / "test.parquet")
        df = pd.concat([df, test_df], ignore_index=True)
        X_scaled = np.vstack([
            X_scaled,
            scaler.transform(test_df[FEATURES].fillna(-999).values.astype(np.float32)),
        ]).astype(np.float32)

    X = torch.tensor(X_scaled, dtype=torch.float)
    y_val = df["isFraud"].values if "isFraud" in df.columns else np.zeros(len(df))
    y = torch.tensor(np.nan_to_num(y_val, nan=0.0), dtype=torch.long)
    X = torch.tensor(X_scaled, dtype=torch.float)
    y = torch.tensor(df["isFraud"].values, dtype=torch.long)

    # Multi-edge graph: card1 + addr1 + P_emaildomain
    cols  = ["TransactionID", "card1", "addr1", "P_emaildomain"]
    raw   = pd.read_csv(Path("data/raw") / "train_transaction.csv", usecols=cols)
    raw_t = pd.read_csv(Path("data/raw") / "test_transaction.csv", usecols=cols)
    raw   = pd.concat([raw, raw_t], ignore_index=True)
    df    = df.merge(raw, on="TransactionID", how="left")

    srcs, dsts = [], []
    for col in ("card1", "addr1", "P_emaildomain"):
        codes = pd.factorize(df[col], use_na_sentinel=True)[0]
        s, d  = chain_edges(codes)
        srcs.append(s)
        dsts.append(d)

    src = np.concatenate(srcs)
    dst = np.concatenate(dsts)
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    data = Data(x=X, edge_index=edge_index, y=y)
    return data, df, scaler


# Backwards-compatible alias (ensemble.py imports this name)
build_graph_sample = build_graph


class FraudGAT(nn.Module):
    def __init__(self, in_channels, hidden=64, heads=4, out=2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden, heads=heads, dropout=0.3)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, concat=False, dropout=0.3)
        self.head  = nn.Linear(hidden, out)
        self.drop  = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.drop(x)
        x = F.elu(self.conv2(x, edge_index))
        return self.head(x)


def main():
    # ── 1. Load data + build full graph ────────────────────────────────
    print("Building full-dataset graph...")
    t0 = time.time()
    data, df, scaler = build_graph()
    FEATURES = json.load(open(FS / "features.json"))
    print(f"Nodes: {data.num_nodes:,}  Edges: {data.num_edges:,}  "
          f"Fraud: {df.isFraud.mean():.4%}  ({time.time() - t0:.0f}s)")

    # ── 2. Train / val masks ───────────────────────────────────────────
    idx = np.arange(len(df))
    y   = data.y.numpy()
    tr_idx, val_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)

    train_mask = torch.zeros(len(df), dtype=torch.bool)
    val_mask   = torch.zeros(len(df), dtype=torch.bool)
    train_mask[tr_idx] = True
    val_mask[val_idx]  = True

    data.train_mask = train_mask
    data.val_mask   = val_mask

    # NeighborLoader/NeighborSampler need torch-sparse/pyg-lib (not installed,
    # no C++ build tools on this box) — fall back to manual subgraph batching:
    # each step samples seed nodes, takes their k-hop edges from the full graph,
    # and trains only on that subgraph. Same idea, no extra deps.
    src_all, dst_all = data.edge_index[0].numpy(), data.edge_index[1].numpy()

    # CSR-style adjacency: sort edges by src, keep offsets
    order    = np.argsort(src_all, kind="stable")
    adj_dst  = dst_all[order]
    adj_off  = np.searchsorted(src_all[order], np.arange(data.num_nodes + 1))

    def sample_subgraph(seeds: np.ndarray, k: int = 10, k2: int = 5):
        """Return (n_id, edge_index_local) for seeds + sampled 2-hop neighbours.

        GATConv has two layers, so each seed needs neighbours at 1-hop and
        2-hop for the receptive field. Sampling k per 1-hop node and k2 per
        resulting 2-hop node keeps the subgraph size bounded.
        """
        rng  = np.random.default_rng()
        # Hop 1: neighbours of the seed nodes
        nbrs1 = [adj_dst[adj_off[s]:adj_off[s + 1]] for s in seeds]
        nbrs1 = np.concatenate([
            n if len(n) <= k else rng.choice(n, size=k, replace=False)
            for n in nbrs1]) if nbrs1 else np.empty(0, dtype=np.int64)
        # Hop 2: neighbours of hop-1 nodes (without re-visiting seeds)
        nbrs2 = [adj_dst[adj_off[n]:adj_off[n + 1]] for n in nbrs1]
        nbrs2 = np.concatenate([
            m if len(m) <= k2 else rng.choice(m, size=k2, replace=False)
            for m in nbrs2]) if nbrs2 else np.empty(0, dtype=np.int64)
        n_id = np.unique(np.concatenate([seeds, nbrs1, nbrs2])).astype(np.int64)
        pos  = np.full(data.num_nodes, -1, dtype=np.int64)
        pos[n_id] = np.arange(len(n_id))
        s_mask = pos[src_all] >= 0
        d_mask = pos[dst_all] >= 0
        keep   = s_mask & d_mask
        edge   = np.vstack([pos[src_all[keep]], pos[dst_all[keep]]])
        return n_id, edge

    def make_batches(seed_nodes: np.ndarray, batch_size: int, shuffle: bool):
        seeds = seed_nodes.copy()
        if shuffle:
            np.random.shuffle(seeds)
        for i in range(0, len(seeds), batch_size):
            yield sample_subgraph(seeds[i:i + batch_size])

    # ── 3. Model / optimizer ───────────────────────────────────────────
    device = torch.device("cpu")
    model  = FraudGAT(in_channels=len(FEATURES)).to(device)

    neg, pos      = (y == 0).sum().item(), (y == 1).sum().item()
    class_weights = torch.tensor([1.0, neg / pos], dtype=torch.float).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ── 4. Evaluation (mini-batch inference over val nodes) ────────────
    def evaluate():
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for n_id, edge in make_batches(val_idx, 4 * BATCH, shuffle=False):
                x_sub = data.x[n_id].to(device)
                out   = model(x_sub, torch.as_tensor(edge, device=device))
                preds.append(F.softmax(out, dim=1)[:, 1].cpu().numpy())
                ys.append(data.y[n_id].cpu().numpy())
        val_labels = np.concatenate(ys)
        val_proba  = np.concatenate(preds)
        return (
            roc_auc_score(val_labels, val_proba),
            average_precision_score(val_labels, val_proba),
        )

    mlflow.set_experiment("fraud-detection-gnn")

    best_prauc = 0.0
    best_state = None

    with mlflow.start_run():
        mlflow.log_params({
            "model": "GAT", "hidden": 64, "heads": 4, "epochs": EPOCHS,
            "lr": 1e-3, "dropout": 0.3, "nodes": data.num_nodes,
            "batch": BATCH, "neighbors": "10,5 (2-hop)", "sampling": "2-hop",
            "n_features": len(FEATURES), "scaled": True,
            "edge_types": "card1+addr1+P_emaildomain",
        })

        print(f"\nTraining GAT on full dataset ({EPOCHS} epochs, subgraph batching)...")
        for epoch in range(1, EPOCHS + 1):
            model.train()
            epoch_loss, n_batches = 0.0, 0
            for n_id, edge in make_batches(tr_idx, BATCH, shuffle=True):
                optimizer.zero_grad()
                x_sub = data.x[n_id].to(device)
                out   = model(x_sub, torch.as_tensor(edge, device=device))
                loss  = F.cross_entropy(out,
                                        data.y[n_id].to(device),
                                        weight=class_weights)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()

            if epoch % 2 == 0:
                roc, pr = evaluate()
                print(f"Epoch {epoch:3d} | Loss {epoch_loss / n_batches:.4f} | "
                      f"ROC-AUC {roc:.4f} | PR-AUC {pr:.4f}", flush=True)
                mlflow.log_metrics({"loss": epoch_loss / n_batches,
                                    "roc_auc": roc, "pr_auc": pr}, step=epoch)
                if pr > best_prauc:
                    best_prauc = pr
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # ── 5. Final eval & save ───────────────────────────────────────
        model.load_state_dict(best_state)
        roc, pr = evaluate()
        print(f"\nBest model — ROC-AUC: {roc:.4f}  PR-AUC: {pr:.4f}")

        mlflow.log_metrics({"best_roc_auc": roc, "best_pr_auc": pr})

        torch.save({
            "model_state": best_state,
            "in_channels":  len(FEATURES),
            "hidden":       64,
            "heads":        4,
            "pr_auc":       pr,
            "roc_auc":      roc,
            "features":     FEATURES,
            "training":     "full-graph manual 2-hop subgraph batching (10,5)",
            "sampling":     "2-hop",
            "scaler_mean":  scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        }, MODELS / "gnn_model.pt")

        mlflow.log_artifact(str(MODELS / "gnn_model.pt"))
        print(f"GNN model saved to {MODELS / 'gnn_model.pt'}")
        print("GNN training complete.")


if __name__ == "__main__":
    main()
