"""
Optuna hyperparameter tuning for XGBoost fraud detection model.
Optimizes PR-AUC over N trials, then retrains final model with best params
and saves it over xgboost_model.pkl + model_meta.json.

Usage:
    python models/tune_xgboost_optuna.py [--trials 50]
"""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import optuna
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROC   = Path("data/processed")
FS     = Path("features/feature_store_data")
MODELS = Path("models")

# ── Load data once ──────────────────────────────────────────────────────────
print("Loading processed data...")
df       = pd.read_parquet(PROC / "train.parquet")
FEATURES = json.load(open(FS / "features.json"))

X = df[FEATURES].values
y = df["isFraud"].values
print(f"Dataset: {X.shape}  Fraud rate: {y.mean():.4%}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

neg, pos         = np.bincount(y_train)
scale_pos_weight = neg / pos
print(f"scale_pos_weight: {scale_pos_weight:.1f}\n")


# ── Objective ───────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    params = dict(
        n_estimators          = 1000,
        learning_rate         = trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        max_depth             = trial.suggest_int("max_depth", 4, 8),
        subsample             = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree      = trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight      = trial.suggest_int("min_child_weight", 1, 10),
        gamma                 = trial.suggest_float("gamma", 0.0, 1.0),
        reg_alpha             = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda            = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        scale_pos_weight      = scale_pos_weight,
        eval_metric           = "aucpr",
        early_stopping_rounds = 50,
        random_state          = 42,
        n_jobs                = -1,
    )
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    proba = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, proba)


# ── Run study ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    args = parser.parse_args()

    print(f"Starting Optuna study — {args.trials} trials (optimizing PR-AUC)...\n")

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\nBest trial #{best.number}  PR-AUC: {best.value:.4f}")
    print("Best params:", json.dumps(best.params, indent=2))

    # ── Retrain final model with best params ─────────────────────────────
    print("\nRetraining final model with best params...")
    final_params = dict(
        n_estimators          = 1000,
        scale_pos_weight      = scale_pos_weight,
        eval_metric           = "aucpr",
        early_stopping_rounds = 50,
        random_state          = 42,
        n_jobs                = -1,
        **best.params,
    )

    mlflow.set_experiment("fraud-detection-xgboost")
    with mlflow.start_run(run_name="optuna-best"):
        mlflow.log_params({k: v for k, v in final_params.items()
                           if k not in ("early_stopping_rounds",)})
        mlflow.log_param("early_stopping_rounds", 50)
        mlflow.log_param("optuna_trials", args.trials)
        mlflow.log_param("n_features", len(FEATURES))

        model = XGBClassifier(**final_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

        proba   = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, proba)
        pr_auc  = average_precision_score(y_val, proba)

        thresholds  = np.arange(0.1, 0.9, 0.01)
        f1s         = [f1_score(y_val, proba >= t) for t in thresholds]
        best_thresh = float(thresholds[np.argmax(f1s)])

        print(f"\nROC-AUC : {roc_auc:.4f}")
        print(f"PR-AUC  : {pr_auc:.4f}")
        print(f"Best F1 : {max(f1s):.4f}  @ threshold {best_thresh:.2f}")

        mlflow.log_metrics({
            "roc_auc":        roc_auc,
            "pr_auc":         pr_auc,
            "best_f1":        max(f1s),
            "best_threshold": best_thresh,
        })

        joblib.dump(model, MODELS / "xgboost_model.pkl")
        meta = {"features": FEATURES, "threshold": best_thresh,
                "roc_auc": roc_auc, "pr_auc": pr_auc}
        json.dump(meta, open(MODELS / "model_meta.json", "w"), indent=2)

        mlflow.xgboost.log_model(model, "xgboost_model")
        mlflow.log_artifact(str(MODELS / "model_meta.json"))

        print(f"\nModel saved: {MODELS / 'xgboost_model.pkl'}")
        print(f"Meta  saved: {MODELS / 'model_meta.json'}")
        print("Tuning complete.")
