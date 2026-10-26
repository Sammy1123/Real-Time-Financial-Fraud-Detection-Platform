"""
Threshold tuning for XGBoost fraud model.
Plots PR curve + F1/Precision/Recall vs threshold.
Optionally re-saves model_meta.json with a new threshold.

Usage:
    python models/tune_threshold.py                    # show plots, keep current threshold
    python models/tune_threshold.py --target-recall 0.8  # set threshold at 80% recall
    python models/tune_threshold.py --target-recall 0.8 --save  # also update model_meta.json
"""
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, roc_curve,
    roc_auc_score, average_precision_score,
    f1_score, confusion_matrix, classification_report
)

PROC   = Path("data/processed")
MODELS = Path("models")
OUT    = MODELS / "threshold_plots"
OUT.mkdir(exist_ok=True)

# ── Args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--target-recall", type=float, default=None,
                    help="Pick threshold that achieves this recall (e.g. 0.8)")
parser.add_argument("--save", action="store_true",
                    help="Update model_meta.json with the new threshold")
args = parser.parse_args()

# ── Load ───────────────────────────────────────────────────────────────────
print("Loading model and data...")
meta     = json.load(open(MODELS / "model_meta.json"))
model    = joblib.load(MODELS / "xgboost_model.pkl")
FEATURES = meta["features"]
current_thresh = meta["threshold"]

df = pd.read_parquet(PROC / "train.parquet")
_, val = train_test_split(df, test_size=0.2, stratify=df["isFraud"], random_state=42)

X_val = val[FEATURES].values
y_val = val["isFraud"].values
proba = model.predict_proba(X_val)[:, 1]

print(f"Validation set: {len(y_val):,}  Fraud: {y_val.mean():.4%}")
print(f"Current threshold: {current_thresh:.2f}")

# ── Curves ─────────────────────────────────────────────────────────────────
precision, recall, pr_thresh = precision_recall_curve(y_val, proba)
fpr, tpr, roc_thresh         = roc_curve(y_val, proba)

thresh_range = np.arange(0.05, 0.95, 0.01)
f1s   = [f1_score(y_val, proba >= t, zero_division=0) for t in thresh_range]
precs = [precision[np.argmin(np.abs(pr_thresh - t))] if t <= pr_thresh.max() else 0
         for t in thresh_range]
recs  = [recall[np.argmin(np.abs(pr_thresh - t))] if t <= pr_thresh.max() else 0
         for t in thresh_range]

best_f1_thresh = float(thresh_range[np.argmax(f1s)])
best_f1        = max(f1s)

# ── Target recall threshold ────────────────────────────────────────────────
new_thresh = current_thresh
if args.target_recall:
    # Find threshold where recall >= target
    idx = np.where(recall >= args.target_recall)[0]
    if len(idx):
        # highest threshold (most precise) that still hits target recall
        new_thresh = float(pr_thresh[idx[-1]])
        preds      = (proba >= new_thresh).astype(int)
        print(f"\nTarget recall: {args.target_recall:.0%}")
        print(f"New threshold: {new_thresh:.4f}")
        print(f"Achieved recall   : {recall[idx[-1]]:.4%}")
        print(f"Achieved precision: {precision[idx[-1]]:.4%}")
        print(f"F1 at new threshold: {f1_score(y_val, preds):.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_val, preds, target_names=["Legit", "Fraud"]))
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, preds))
    else:
        print(f"WARNING: Cannot achieve {args.target_recall:.0%} recall — model max recall is {recall.max():.4%}")

# ── Plot 1: PR Curve ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(recall, precision, color="steelblue", lw=2,
        label=f"PR Curve (AUC={average_precision_score(y_val, proba):.3f})")
ax.axvline(recall[np.argmin(np.abs(pr_thresh - current_thresh))],
           color="orange", linestyle="--", label=f"Current ({current_thresh:.2f})")
if args.target_recall and new_thresh != current_thresh:
    ax.axvline(args.target_recall, color="crimson", linestyle="--",
               label=f"Target recall ({args.target_recall:.0%}) → thresh {new_thresh:.2f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "pr_curve.png", dpi=150)
plt.close()

# ── Plot 2: F1 / Precision / Recall vs Threshold ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresh_range, f1s,   color="steelblue", lw=2, label="F1")
ax.plot(thresh_range, precs, color="green",     lw=2, label="Precision")
ax.plot(thresh_range, recs,  color="crimson",   lw=2, label="Recall")
ax.axvline(current_thresh, color="orange", linestyle="--",
           label=f"Current ({current_thresh:.2f})  F1={best_f1:.3f}")
ax.axvline(best_f1_thresh, color="purple", linestyle=":",
           label=f"Best F1 ({best_f1_thresh:.2f})")
if args.target_recall and new_thresh != current_thresh:
    ax.axvline(new_thresh, color="crimson", linestyle="--",
               label=f"Target recall threshold ({new_thresh:.2f})")
ax.set_xlabel("Threshold")
ax.set_ylabel("Score")
ax.set_title("F1 / Precision / Recall vs Threshold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "threshold_tradeoff.png", dpi=150)
plt.close()

# ── Plot 3: Cost matrix analysis ──────────────────────────────────────────
# Assume FN (missed fraud) costs 10x more than FP (blocked legit tx)
FN_COST, FP_COST = 10, 1
costs = []
for t in thresh_range:
    preds = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
    costs.append(fn * FN_COST + fp * FP_COST)

best_cost_thresh = float(thresh_range[np.argmin(costs)])
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresh_range, costs, color="crimson", lw=2)
ax.axvline(best_cost_thresh, color="steelblue", linestyle="--",
           label=f"Min cost threshold ({best_cost_thresh:.2f})")
ax.axvline(current_thresh, color="orange", linestyle="--",
           label=f"Current ({current_thresh:.2f})")
ax.set_xlabel("Threshold")
ax.set_ylabel(f"Total Cost (FN={FN_COST}x, FP={FP_COST}x)")
ax.set_title("Business Cost vs Threshold  (FN costs 10x more than FP)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "cost_analysis.png", dpi=150)
plt.close()

print(f"\nSummary:")
print(f"  Current threshold : {current_thresh:.2f}")
print(f"  Best F1 threshold : {best_f1_thresh:.2f}  (F1={best_f1:.4f})")
print(f"  Min cost threshold: {best_cost_thresh:.2f}  (FN=10x, FP=1x)")
if args.target_recall:
    print(f"  Target recall ({args.target_recall:.0%}) : {new_thresh:.2f}")
print(f"\nPlots saved to {OUT}/")

# ── Save ───────────────────────────────────────────────────────────────────
if args.save and new_thresh != current_thresh:
    meta["threshold"]          = new_thresh
    meta["threshold_strategy"] = f"target_recall_{args.target_recall}"
    json.dump(meta, open(MODELS / "model_meta.json", "w"), indent=2)
    print(f"\nmodel_meta.json updated: threshold {current_thresh:.2f} → {new_thresh:.2f}")
elif args.save:
    print("\n--save passed but threshold unchanged.")
