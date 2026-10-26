"""
Train XGBoost fraud detection model.
Reads data/processed/train.parquet, trains with early stopping,
evaluates on a held-out val set, and saves model + metadata to models/.
Logs everything to MLflow.
"""
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)

PROC     = Path("data/processed")
FS       = Path("features/feature_store_data")
MODELS   = Path("models")
MODELS.mkdir(exist_ok=True)

# ── 1. Load ────────────────────────────────────────────────────────────────
print("Loading processed data...")
df       = pd.read_parquet(PROC / "train.parquet")
FEATURES = json.load(open(FS / "features.json"))

X = df[FEATURES].values
y = df["isFraud"].values
print(f"Dataset: {X.shape}  Fraud rate: {y.mean():.4%}")

# ── 2. Train / val split ───────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

neg, pos        = np.bincount(y_train)
scale_pos_weight = neg / pos
print(f"scale_pos_weight: {scale_pos_weight:.1f}")

# ── 3. Model ───────────────────────────────────────────────────────────────
params = dict(
    n_estimators      = 1000,
    learning_rate     = 0.05,
    max_depth         = 6,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = scale_pos_weight,
    eval_metric       = "aucpr",
    early_stopping_rounds = 50,
    random_state      = 42,
    n_jobs            = -1,
)

mlflow.set_experiment("fraud-detection-xgboost")

with mlflow.start_run():
    mlflow.log_params({k: v for k, v in params.items() if k != "early_stopping_rounds"})
    mlflow.log_param("early_stopping_rounds", 50)
    mlflow.log_param("n_features", len(FEATURES))
    mlflow.log_param("train_size", len(X_train))

    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    # ── 4. Evaluate ───────────────────────────────────────────────────────
    proba = model.predict_proba(X_val)[:, 1]

    roc_auc = roc_auc_score(y_val, proba)
    pr_auc  = average_precision_score(y_val, proba)
    print(f"\nROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")

    # best threshold by F1
    from sklearn.metrics import f1_score
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1s        = [f1_score(y_val, proba >= t) for t in thresholds]
    best_thresh = float(thresholds[np.argmax(f1s)])
    preds       = (proba >= best_thresh).astype(int)

    print(f"Best threshold: {best_thresh:.2f}  F1: {max(f1s):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, preds, target_names=["Legit", "Fraud"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, preds))

    mlflow.log_metrics({
        "roc_auc": roc_auc,
        "pr_auc":  pr_auc,
        "best_f1": max(f1s),
        "best_threshold": best_thresh,
    })

    # ── 5. Save ───────────────────────────────────────────────────────────
    joblib.dump(model, MODELS / "xgboost_model.pkl")
    meta = {"features": FEATURES, "threshold": best_thresh,
            "roc_auc": roc_auc, "pr_auc": pr_auc}
    json.dump(meta, open(MODELS / "model_meta.json", "w"), indent=2)

    mlflow.xgboost.log_model(model, "xgboost_model")
    mlflow.log_artifact(str(MODELS / "model_meta.json"))

    print(f"\nModel saved to {MODELS / 'xgboost_model.pkl'}")
    print(f"Meta  saved to {MODELS / 'model_meta.json'}")
    print("Training complete.")
