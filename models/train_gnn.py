"""
GNN fraud detection model using GraphSAGE.
Graph: nodes = transactions, edges = shared card1 (same card → potential fraud ring).
Trains on a stratified 50k subsample (CPU-friendly).
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
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

PROC   = Path("data/processed")
FS     = Path("features/feature_store_data")
MODELS = Path("models")

# ── 1. Load data ───────────────────────────────────────────────────────────
print("Loading data...")
df       = pd.read_parquet(PROC / "train.parquet")
FEATURES = json.load(open(FS / "features.json"))

# Stratified subsample — 50k keeps it CPU-trainable
df_sample = (
    pd.concat([
        df[df.isFraud == 1],                                        # all fraud
        df[df.isFraud == 0].sample(
            n=min(47000, (df.isFraud == 0).sum()), random_state=42) # legit to ~50k total
    ])
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)
print(f"Subsample: {len(df_sample)}  Fraud: {df_sample.isFraud.mean():.4%}")

X = torch.tensor(df_sample[FEATURES].fillna(-999).values, dtype=torch.float)
y = torch.tensor(df_sample["isFraud"].values, dtype=torch.long)

# ── 2. Build graph (edges = shared card1) ─────────────────────────────────
print("Building graph edges...")
# Read card1 from raw train to align with subsample TransactionIDs
raw_card1 = pd.read_csv(
    Path("data/raw") / "train_transaction.csv",
    usecols=["TransactionID", "card1"]
)
df_sample = df_sample.merge(
    raw_card1.rename(columns={"card1": "_card1"}),
    on="TransactionID", how="left"
)

# Group node indices by card1 value
card1_groups = df_sample.groupby("_card1").apply(lambda g: g.index.tolist(), include_groups=False)

src, dst = [], []
for nodes in card1_groups:
    if len(nodes) < 2:
        continue
    # Connect every pair within the group (bidirectional)
    for i in range(len(nodes)):
        for j in range(i + 1, min(i + 6, len(nodes))):  # cap at 5 neighbours per node
            src += [nodes[i], nodes[j]]
            dst += [nodes[j], nodes[i]]

edge_index = torch.tensor([src, dst], dtype=torch.long)
print(f"Nodes: {len(df_sample)}  Edges: {edge_index.shape[1]}")

# ── 3. Train / val masks ───────────────────────────────────────────────────
idx       = np.arange(len(df_sample))
tr_idx, val_idx = train_test_split(idx, test_size=0.2, stratify=y.numpy(), random_state=42)

train_mask = torch.zeros(len(df_sample), dtype=torch.bool)
val_mask   = torch.zeros(len(df_sample), dtype=torch.bool)
train_mask[tr_idx] = True
val_mask[val_idx]  = True

data = Data(x=X, edge_index=edge_index, y=y,
            train_mask=train_mask, val_mask=val_mask)

# ── 4. Model ───────────────────────────────────────────────────────────────
class FraudSAGE(nn.Module):
    def __init__(self, in_channels, hidden=128, out=2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden // 2)
        self.head  = nn.Linear(hidden // 2, out)
        self.drop  = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.drop(x)
        x = F.relu(self.conv2(x, edge_index))
        return self.head(x)


device = torch.device("cpu")
model  = FraudSAGE(in_channels=len(FEATURES)).to(device)
data   = data.to(device)

# Class weights for imbalance
neg, pos      = (y == 0).sum().item(), (y == 1).sum().item()
class_weights = torch.tensor([1.0, neg / pos], dtype=torch.float).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ── 5. Training loop ───────────────────────────────────────────────────────
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
        "model": "GraphSAGE", "hidden": 128, "epochs": EPOCHS,
        "lr": 1e-3, "dropout": 0.3, "subsample": len(df_sample),
        "n_features": len(FEATURES),
    })

    print(f"\nTraining GraphSAGE for {EPOCHS} epochs...")
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

    # ── 6. Final eval & save ──────────────────────────────────────────────
    model.load_state_dict(best_state)
    roc, pr = evaluate()
    print(f"\nBest model — ROC-AUC: {roc:.4f}  PR-AUC: {pr:.4f}")

    mlflow.log_metrics({"best_roc_auc": roc, "best_pr_auc": pr})

    torch.save({
        "model_state": best_state,
        "in_channels": len(FEATURES),
        "hidden": 128,
        "pr_auc": pr,
        "roc_auc": roc,
        "features": FEATURES,
    }, MODELS / "gnn_model.pt")

    mlflow.log_artifact(str(MODELS / "gnn_model.pt"))
    print(f"GNN model saved to {MODELS / 'gnn_model.pt'}")
    print("GNN training complete.")
