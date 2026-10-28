"""
Ensemble meta-learner — combines XGBoost and GNN fraud probabilities.

Loads out-of-fold predictions from the tuned XGBoost and the trained GAT,
stacks them with a Logistic Regression meta-learner, and reports the uplift
over each standalone model. Saves the meta-learner + blend weights to
models/ensemble_meta.pkl and logs metrics to MLflow.
"""
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

PROC   = Path("data/processed")
MODELS = Path("models")

# Allow running as a plain script (python models/ensemble.py)
import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── 1. Load validation data + base models ─────────────────────────────────
print("Loading data and base models...")
meta     = json.load(open(MODELS / "model_meta.json"))
FEATURES = meta["features"]
df       = pd.read_parquet(PROC / "train.parquet")
_, val_df = train_test_split(df, test_size=0.2, stratify=df["isFraud"], random_state=42)
y_val = val_df["isFraud"].values

xgb = joblib.load(MODELS / "xgboost_model.pkl")
X_xgb = val_df[FEATURES].values.astype(np.float32)
p_xgb = xgb.predict_proba(X_xgb)[:, 1]
print(f"XGBoost  — ROC-AUC {roc_auc_score(y_val, p_xgb):.4f}  PR-AUC {average_precision_score(y_val, p_xgb):.4f}")

# ── 2. GNN probabilities on the same validation nodes ─────────────────────
gnn_ckpt = torch.load(MODELS / "gnn_model.pt", map_location="cpu", weights_only=False)
print("Importing GNN architecture from train_gnn...")
from models.train_gnn import FraudGAT  # reuse class definition

# Rebuild the sample graph exactly as training did so node indices align
from models.train_gnn import build_graph_sample  # helper extracted from training script

data, df_sample, scaler_info = build_graph_sample()
gnn = FraudGAT(in_channels=gnn_ckpt["in_channels"], hidden=gnn_ckpt["hidden"], heads=gnn_ckpt["heads"])
gnn.load_state_dict(gnn_ckpt["model_state"])
gnn.eval()

import torch.nn.functional as F
with torch.no_grad():
    logits = gnn(data.x, data.edge_index)
    gnn_proba = F.softmax(logits, dim=1)[:, 1].numpy()

# Map validation rows to their position in the sample graph
pos_map = {tid: i for i, tid in enumerate(df_sample["TransactionID"].values)}
mask    = val_df["TransactionID"].isin(pos_map).values
p_gnn   = np.full(len(val_df), 0.5, dtype=np.float32)
idxs    = [pos_map[tid] for tid in val_df.loc[mask, "TransactionID"]]
p_gnn[mask] = gnn_proba[idxs]
covered = mask.mean()
print(f"GNN      — ROC-AUC {roc_auc_score(y_val[mask], p_gnn[mask]):.4f}  PR-AUC {average_precision_score(y_val[mask], p_gnn[mask]):.4f}  (covers {covered:.0%} of val rows)")

# ── 3. Meta-learner ────────────────────────────────────────────────────────
S = np.column_stack([p_xgb, p_gnn])
meta_clf = LogisticRegression(C=1.0, max_iter=1000)
meta_clf.fit(S, y_val)

p_ens   = meta_clf.predict_proba(S)[:, 1]
w        = meta_clf.coef_[0]
thresholds = np.arange(0.05, 0.95, 0.01)
best_t   = float(thresholds[np.argmax([f1_score(y_val, p_ens >= t) for t in thresholds])])

roc_e  = roc_auc_score(y_val, p_ens)
pr_e   = average_precision_score(y_val, p_ens)
f1_e   = f1_score(y_val, p_ens >= best_t)
roc_x  = roc_auc_score(y_val, p_xgb)
pr_x   = average_precision_score(y_val, p_xgb)

print(f"\nEnsemble — ROC-AUC {roc_e:.4f}  PR-AUC {pr_e:.4f}  F1 {f1_e:.4f} @ t={best_t:.2f}")
print(f"Weights  — xgb {w[0]:+.4f}  gnn {w[1]:+.4f}  intercept {meta_clf.intercept_[0]:+.4f}")
print(f"Uplift   — ROC {(roc_e - roc_x)*100:+.2f}pp  PR {(pr_e - pr_x)*100:+.2f}pp")

# ── 4. Save + MLflow ───────────────────────────────────────────────────────
joblib.dump({
    "meta_model":      meta_clf,
    "weights":         {"xgboost": float(w[0]), "gnn": float(w[1])},
    "threshold":       best_t,
    "val_roc_auc":     roc_e,
    "val_pr_auc":      pr_e,
    "val_f1":          f1_e,
    "xgb_roc_auc":     roc_x,
    "xgb_pr_auc":      pr_x,
}, MODELS / "ensemble_meta.pkl")
print(f"\nSaved to {MODELS / 'ensemble_meta.pkl'}")

mlflow.set_experiment("fraud-detection-ensemble")
with mlflow.start_run(run_name="xgb_gnn_meta_lr"):
    mlflow.log_params({"meta_model": "LogisticRegression", "base_models": "xgboost+gat"})
    mlflow.log_metrics({
        "ensemble_roc_auc": roc_e, "ensemble_pr_auc": pr_e, "ensemble_f1": f1_e,
        "threshold": best_t, "weight_xgboost": float(w[0]), "weight_gnn": float(w[1]),
        "roc_uplift_pp": (roc_e - roc_x) * 100, "pr_uplift_pp": (pr_e - pr_x) * 100,
    })
    mlflow.log_artifact(str(MODELS / "ensemble_meta.pkl"))

print("Ensemble training complete.")
