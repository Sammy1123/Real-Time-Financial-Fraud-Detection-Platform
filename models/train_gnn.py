"""
GNN fraud detection model using GAT (Graph Attention Network).
Graph: nodes = transactions, edges = shared card1 / addr1 / P_emaildomain.
Node features scaled with StandardScaler to fix high-loss issue.
Trains on a stratified 67k subsample (CPU-friendly).
Saves model to models/gnn_model.pt and logs to MLflow.
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import mlflow
from pathlib import Path
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

PROC   = Path("data/processed")
FS     = Path("features/feature_store_data")
MODELS = Path("models")


def build_graph_sample(random_state: int = 42):
    """Build the stratified subsample graph (scaled features, multi-edge).

    Deterministic — same subsample and node ordering as the training run,
    so saved checkpoint node indices align. Returns (data, df_sample, scaler).
    """
    df       = pd.read_parquet(PROC / "train.parquet")
    FEATURES = json.load(open(FS / "features.json"))

    # Stratified subsample — 67k keeps it CPU-trainable
    df_sample = (
        pd.concat([
            df[df.isFraud == 1],
            df[df.isFraud == 0].sample(
                n=min(64000, (df.isFraud == 0).sum()), random_state=random_state)
        ])
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    # Scale node features (fixes high loss)
    X_raw    = df_sample[FEATURES].fillna(-999).values.astype(np.float32)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X = torch.tensor(X_scaled, dtype=torch.float)
    y = torch.tensor(df_sample["isFraud"].values, dtype=torch.long)

    # Multi-edge graph: card1 + addr1 + P_emaildomain
    raw = pd.read_csv(
        Path("data/raw") / "train_transaction.csv",
        usecols=["TransactionID", "card1", "addr1", "P_emaildomain"]
    )
    df_sample = df_sample.merge(raw, on="TransactionID", how="left")

    src, dst = [], []

    def add_edges(groups):
        nonlocal src, dst
        for nodes in groups:
            if len(nodes) < 2:
                continue
            for i in range(len(nodes)):
                for j in range(i + 1, min(i + 6, len(nodes))):
                    src += [nodes[i], nodes[j]]
                    dst += [nodes[j], nodes[i]]

    card1_groups = df_sample.groupby("card1").apply(
        lambda g: g.index.tolist(), include_groups=False)
    add_edges(card1_groups)

    addr1_groups = df_sample.dropna(subset=["addr1"]).groupby("addr1").apply(
        lambda g: g.index.tolist(), include_groups=False)
    add_edges(addr1_groups)

    email_groups = df_sample.dropna(subset=["P_emaildomain"]).groupby("P_emaildomain").apply(
        lambda g: g.index.tolist(), include_groups=False)
    add_edges(email_groups)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    data = Data(x=X, edge_index=edge_index, y=y)
    return data, df_sample, scaler


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
    # ── 1. Load data ───────────────────────────────────────────────────────
    print("Loading data...")
    df       = pd.read_parquet(PROC / "train.parquet")
    FEATURES = json.load(open(FS / "features.json"))

    # Stratified subsample — 67k keeps it CPU-trainable
    df_sample = (
        pd.concat([
            df[df.isFraud == 1],
            df[df.isFraud == 0].sample(
                n=min(64000, (df.isFraud == 0).sum()), random_state=42)
        ])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    print(f"Subsample: {len(df_sample)}  Fraud: {df_sample.isFraud.mean():.4%}")

    # ── 2. Scale node features (fixes high loss) ──────────────────────
    print("Scaling node features...")
    X_raw   = df_sample[FEATURES].fillna(-999).values.astype(np.float32)
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X = torch.tensor(X_scaled, dtype=torch.float)
    y = torch.tensor(df_sample["isFraud"].values, dtype=torch.long)

    # ── 3. Build multi-edge graph ──────────────────────────────────────
    print("Building graph edges (card1 + addr1 + P_emaildomain)...")

    raw = pd.read_csv(
        Path("data/raw") / "train_transaction.csv",
        usecols=["TransactionID", "card1", "addr1", "P_emaildomain"]
    )
    df_sample = df_sample.merge(raw, on="TransactionID", how="left")

    src, dst = [], []

    def add_edges(groups, src, dst):
        for nodes in groups:
            if len(nodes) < 2:
                continue
            for i in range(len(nodes)):
                for j in range(i + 1, min(i + 6, len(nodes))):
                    src += [nodes[i], nodes[j]]
                    dst += [nodes[j], nodes[i]]

    # card1 edges
    card1_groups = df_sample.groupby("card1").apply(
        lambda g: g.index.tolist(), include_groups=False)
    add_edges(card1_groups, src, dst)

    # addr1 edges (skip NaN)
    addr1_groups = df_sample.dropna(subset=["addr1"]).groupby("addr1").apply(
        lambda g: g.index.tolist(), include_groups=False)
    add_edges(addr1_groups, src, dst)

    # P_emaildomain edges (skip NaN)
    email_groups = df_sample.dropna(subset=["P_emaildomain"]).groupby("P_emaildomain").apply(
        lambda g: g.index.tolist(), include_groups=False)
    add_edges(email_groups, src, dst)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    print(f"Nodes: {len(df_sample)}  Edges: {edge_index.shape[1]:,}")

    # ── 4. Train / val masks ───────────────────────────────────────────────
    idx = np.arange(len(df_sample))
    tr_idx, val_idx = train_test_split(idx, test_size=0.2, stratify=y.numpy(), random_state=42)

    train_mask = torch.zeros(len(df_sample), dtype=torch.bool)
    val_mask   = torch.zeros(len(df_sample), dtype=torch.bool)
    train_mask[tr_idx] = True
    val_mask[val_idx]  = True

    data = Data(x=X, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask)

    # ── 5. GAT model (FraudGAT defined at module level above) ─────────────
    device = torch.device("cpu")
    model  = FraudGAT(in_channels=len(FEATURES)).to(device)
    data   = data.to(device)

    neg, pos      = (y == 0).sum().item(), (y == 1).sum().item()
    class_weights = torch.tensor([1.0, neg / pos], dtype=torch.float).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ── 6. Training loop ─────────────────────────────────────────────
    def evaluate():
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            proba  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            labels = data.y.cpu().numpy()
        val_proba  = proba[val_mask.cpu().numpy()]
        val_labels = labels[val_mask.cpu().numpy()]
        return (
            roc_auc_score(val_labels, val_proba),
            average_precision_score(val_labels, val_proba),
        )

    mlflow.set_experiment("fraud-detection-gnn")

    EPOCHS     = 30
    best_prauc = 0.0
    best_state = None

    with mlflow.start_run():
        mlflow.log_params({
            "model": "GAT", "hidden": 64, "heads": 4, "epochs": EPOCHS,
            "lr": 1e-3, "dropout": 0.3, "subsample": len(df_sample),
            "n_features": len(FEATURES), "scaled": True,
            "edge_types": "card1+addr1+P_emaildomain",
        })

        print(f"\nTraining GAT for {EPOCHS} epochs...")
        for epoch in range(1, EPOCHS + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            loss   = F.cross_entropy(logits[train_mask], data.y[train_mask],
                                     weight=class_weights)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 5 == 0:
                roc, pr = evaluate()
                print(f"Epoch {epoch:3d} | Loss {loss.item():.4f} | ROC-AUC {roc:.4f} | PR-AUC {pr:.4f}")
                mlflow.log_metrics({"loss": loss.item(), "roc_auc": roc, "pr_auc": pr}, step=epoch)
                if pr > best_prauc:
                    best_prauc = pr
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # ── 7. Final eval & save ──────────────────────────────────────
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
            "scaler_mean":  scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        }, MODELS / "gnn_model.pt")

        mlflow.log_artifact(str(MODELS / "gnn_model.pt"))
        print(f"GNN model saved to {MODELS / 'gnn_model.pt'}")
        print("GNN training complete.")


if __name__ == "__main__":
    main()
