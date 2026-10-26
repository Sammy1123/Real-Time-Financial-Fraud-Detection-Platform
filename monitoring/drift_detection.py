"""
Drift Detection — monitors data drift (PSI per feature) and model/concept drift
(fraud score distribution shift) between a reference window and current window.
Outputs: monitoring/drift_report.json + monitoring/drift_plots/
Logs to MLflow experiment 'fraud-detection-monitoring'.
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path
from datetime import datetime
from scipy.stats import ks_2samp

PROC    = Path("data/processed")
MODELS  = Path("models")
MON     = Path("monitoring")
PLOTS   = MON / "drift_plots"
MON.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

PSI_WARN     = 0.1   # slight drift
PSI_ALERT    = 0.2   # significant drift
KS_ALPHA     = 0.05  # p-value threshold for KS test

# ── 1. Load data ───────────────────────────────────────────────────────────
print("Loading data...")
meta     = json.load(open(MODELS / "model_meta.json"))
model    = joblib.load(MODELS / "xgboost_model.pkl")
FEATURES = meta["features"]

df = pd.read_parquet(PROC / "train.parquet")

# Reference = first 70% (chronologically), current = last 30%
split     = int(len(df) * 0.7)
reference = df.iloc[:split][FEATURES].fillna(-999)
current   = df.iloc[split:][FEATURES].fillna(-999)

print(f"Reference: {len(reference):,} rows  |  Current: {len(current):,} rows")

# ── 2. PSI per feature ─────────────────────────────────────────────────────
def compute_psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index — measures distribution shift."""
    breakpoints = np.percentile(ref, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    ref_counts = np.histogram(ref, bins=breakpoints)[0]
    cur_counts = np.histogram(cur, bins=breakpoints)[0]

    ref_pct = (ref_counts / len(ref)).clip(1e-6)
    cur_pct = (cur_counts / len(cur)).clip(1e-6)

    # Align lengths after np.unique may reduce bins
    min_len = min(len(ref_pct), len(cur_pct))
    ref_pct, cur_pct = ref_pct[:min_len], cur_pct[:min_len]

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


print("Computing PSI for all features...")
psi_results = {}
for feat in FEATURES:
    psi_results[feat] = compute_psi(reference[feat].values, current[feat].values)

psi_series = pd.Series(psi_results).sort_values(ascending=False)
drifted    = psi_series[psi_series >= PSI_ALERT]
warned     = psi_series[(psi_series >= PSI_WARN) & (psi_series < PSI_ALERT)]

print(f"\nPSI Results:")
print(f"  Features with significant drift (PSI >= {PSI_ALERT}): {len(drifted)}")
print(f"  Features with slight drift     (PSI >= {PSI_WARN}):  {len(warned)}")
if len(drifted):
    print(f"  Top drifted: {drifted.head(5).to_dict()}")

# ── 3. KS test per feature ─────────────────────────────────────────────────
print("\nRunning KS tests...")
ks_results = {}
for feat in FEATURES:
    stat, pval = ks_2samp(reference[feat].values, current[feat].values)
    ks_results[feat] = {"statistic": round(float(stat), 4), "p_value": round(float(pval), 4)}

ks_drifted = {f: v for f, v in ks_results.items() if v["p_value"] < KS_ALPHA}
print(f"  Features failing KS test (p < {KS_ALPHA}): {len(ks_drifted)}")

# ── 4. Model score drift ───────────────────────────────────────────────────
print("\nComputing model score distributions...")
ref_scores = model.predict_proba(reference.values)[:, 1]
cur_scores = model.predict_proba(current.values)[:, 1]

score_ks_stat, score_ks_pval = ks_2samp(ref_scores, cur_scores)
score_psi = compute_psi(ref_scores, cur_scores, bins=20)

ref_fraud_rate = (ref_scores >= meta["threshold"]).mean()
cur_fraud_rate = (cur_scores >= meta["threshold"]).mean()
fraud_rate_delta = float(cur_fraud_rate - ref_fraud_rate)

print(f"  Score PSI          : {score_psi:.4f}")
print(f"  Score KS p-value   : {score_ks_pval:.4f}")
print(f"  Ref fraud rate     : {ref_fraud_rate:.4%}")
print(f"  Current fraud rate : {cur_fraud_rate:.4%}")
print(f"  Delta              : {fraud_rate_delta:+.4%}")

# ── 5. Plots ───────────────────────────────────────────────────────────────
# PSI bar chart
top_n = min(20, len(psi_series))
fig, ax = plt.subplots(figsize=(10, 5))
colors  = ["crimson" if v >= PSI_ALERT else "orange" if v >= PSI_WARN else "steelblue"
           for v in psi_series.head(top_n).values]
psi_series.head(top_n).sort_values().plot(kind="barh", ax=ax, color=colors[::-1])
ax.axvline(PSI_WARN,  color="orange", linestyle="--", label=f"Warn ({PSI_WARN})")
ax.axvline(PSI_ALERT, color="crimson", linestyle="--", label=f"Alert ({PSI_ALERT})")
ax.set_title(f"Top {top_n} Feature PSI Scores")
ax.set_xlabel("PSI")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS / "psi_scores.png", dpi=150)
plt.close()

# Score distribution shift
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(ref_scores, bins=50, alpha=0.6, color="steelblue", label="Reference", density=True)
ax.hist(cur_scores, bins=50, alpha=0.6, color="crimson",   label="Current",   density=True)
ax.axvline(meta["threshold"], color="black", linestyle="--", label=f"Threshold ({meta['threshold']:.2f})")
ax.set_title(f"Fraud Score Distribution  (PSI={score_psi:.3f}, KS p={score_ks_pval:.3f})")
ax.set_xlabel("Fraud Probability")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS / "score_distribution.png", dpi=150)
plt.close()

print(f"\nPlots saved to {PLOTS}/")

# ── 6. Report ──────────────────────────────────────────────────────────────
report = {
    "run_timestamp": datetime.utcnow().isoformat(),
    "reference_size": len(reference),
    "current_size":   len(current),
    "data_drift": {
        "psi_threshold_warn":  PSI_WARN,
        "psi_threshold_alert": PSI_ALERT,
        "features_drifted":    len(drifted),
        "features_warned":     len(warned),
        "top_drifted_features": drifted.head(10).to_dict(),
        "all_psi": psi_series.to_dict(),
    },
    "ks_test": {
        "alpha": KS_ALPHA,
        "features_failing": len(ks_drifted),
        "results": ks_results,
    },
    "model_drift": {
        "score_psi":          round(score_psi, 4),
        "score_ks_statistic": round(float(score_ks_stat), 4),
        "score_ks_pvalue":    round(float(score_ks_pval), 4),
        "ref_fraud_rate":     round(float(ref_fraud_rate), 6),
        "cur_fraud_rate":     round(float(cur_fraud_rate), 6),
        "fraud_rate_delta":   round(fraud_rate_delta, 6),
    },
    "alerts": {
        "data_drift_alert":  len(drifted) > 0,
        "model_drift_alert": score_psi >= PSI_ALERT or abs(fraud_rate_delta) > 0.01,
    },
}

report_path = MON / "drift_report.json"
json.dump(report, open(report_path, "w"), indent=2)
print(f"Report saved to {report_path}")

# ── 7. MLflow logging ──────────────────────────────────────────────────────
mlflow.set_experiment("fraud-detection-monitoring")
with mlflow.start_run(run_name=f"drift_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"):
    mlflow.log_metrics({
        "features_drifted":  len(drifted),
        "features_warned":   len(warned),
        "score_psi":         score_psi,
        "score_ks_pvalue":   score_ks_pval,
        "ref_fraud_rate":    float(ref_fraud_rate),
        "cur_fraud_rate":    float(cur_fraud_rate),
        "fraud_rate_delta":  fraud_rate_delta,
    })
    mlflow.log_artifact(str(report_path))
    mlflow.log_artifact(str(PLOTS / "psi_scores.png"))
    mlflow.log_artifact(str(PLOTS / "score_distribution.png"))

# ── 8. Summary ─────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("DRIFT DETECTION SUMMARY")
print("="*50)
print(f"  Data drift alert   : {'YES' if report['alerts']['data_drift_alert'] else 'NO'}")
print(f"  Model drift alert  : {'YES' if report['alerts']['model_drift_alert'] else 'NO'}")
print(f"  Features drifted   : {len(drifted)}")
print(f"  Score PSI          : {score_psi:.4f}")
print(f"  Fraud rate delta   : {fraud_rate_delta:+.4%}")
print("="*50)
print("Drift detection complete.")
