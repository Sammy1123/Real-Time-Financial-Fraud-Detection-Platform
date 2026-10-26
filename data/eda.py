"""
EDA for IEEE-CIS Fraud Detection dataset.
Outputs plots and a summary report to data/eda_outputs/.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/eda_outputs")
OUT.mkdir(exist_ok=True)

sns.set_theme(style="darkgrid")

# ── 1. Load ────────────────────────────────────────────────────────────────
print("Loading data...")
trans = pd.read_csv(RAW / "train_transaction.csv")
ident = pd.read_csv(RAW / "train_identity.csv")
df = trans.merge(ident, on="TransactionID", how="left")
print(f"Merged shape: {df.shape}")

# ── 2. Basic stats ─────────────────────────────────────────────────────────
fraud_rate = df["isFraud"].mean()
print(f"\nFraud rate: {fraud_rate:.4%}")
print(f"Fraud cases: {df['isFraud'].sum():,} / {len(df):,}")

# ── 3. Class imbalance ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["isFraud"].value_counts()
ax.bar(["Legit", "Fraud"], counts.values, color=["steelblue", "crimson"])
for i, v in enumerate(counts.values):
    ax.text(i, v + 500, f"{v:,}\n({v/len(df):.2%})", ha="center", fontsize=9)
ax.set_title("Class Distribution")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(OUT / "01_class_distribution.png", dpi=150)
plt.close()

# ── 4. Transaction amount distribution ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, log in zip(axes, [False, True]):
    for label, color in [(0, "steelblue"), (1, "crimson")]:
        vals = df[df["isFraud"] == label]["TransactionAmt"]
        vals = np.log1p(vals) if log else vals
        ax.hist(vals, bins=80, alpha=0.6, color=color,
                label=["Legit", "Fraud"][label], density=True)
    ax.set_title(f"TransactionAmt {'(log1p)' if log else '(raw)'}")
    ax.legend()
plt.tight_layout()
plt.savefig(OUT / "02_transaction_amount.png", dpi=150)
plt.close()

# ── 5. Missing values ──────────────────────────────────────────────────────
miss = df.isnull().mean().sort_values(ascending=False)
miss_high = miss[miss > 0.3]
fig, ax = plt.subplots(figsize=(10, max(4, len(miss_high) * 0.18)))
miss_high.plot(kind="barh", ax=ax, color="salmon")
ax.set_title(f"Features with >30% Missing ({len(miss_high)} cols)")
ax.set_xlabel("Missing Fraction")
plt.tight_layout()
plt.savefig(OUT / "03_missing_values.png", dpi=150)
plt.close()
print(f"\nFeatures >30% missing: {len(miss_high)}")
print(f"Features >90% missing: {(miss > 0.9).sum()}")

# ── 6. Fraud rate by ProductCD ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
prod = df.groupby("ProductCD")["isFraud"].mean().sort_values(ascending=False)
prod.plot(kind="bar", ax=ax, color="steelblue")
ax.set_title("Fraud Rate by ProductCD")
ax.set_ylabel("Fraud Rate")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(OUT / "04_fraud_by_productcd.png", dpi=150)
plt.close()

# ── 7. Fraud rate by card4 (card network) & card6 (card type) ─────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col in zip(axes, ["card4", "card6"]):
    rate = df.groupby(col)["isFraud"].mean().sort_values(ascending=False)
    rate.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(f"Fraud Rate by {col}")
    ax.set_ylabel("Fraud Rate")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
plt.tight_layout()
plt.savefig(OUT / "05_fraud_by_card_type.png", dpi=150)
plt.close()

# ── 8. Transaction hour & day patterns ────────────────────────────────────
df["hour"] = (df["TransactionDT"] // 3600) % 24
df["day"]  = (df["TransactionDT"] // 86400) % 7

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, col, label in zip(axes, ["hour", "day"], ["Hour of Day", "Day of Week"]):
    hourly = df.groupby(col)["isFraud"].mean()
    ax.plot(hourly.index, hourly.values, marker="o", color="crimson")
    ax.set_title(f"Fraud Rate by {label}")
    ax.set_xlabel(label)
    ax.set_ylabel("Fraud Rate")
plt.tight_layout()
plt.savefig(OUT / "06_time_patterns.png", dpi=150)
plt.close()

# ── 9. Top V-feature correlations with isFraud ────────────────────────────
v_cols = [c for c in df.columns if c.startswith("V")]
v_corr = df[v_cols + ["isFraud"]].corr()["isFraud"].drop("isFraud")
top_v = v_corr.abs().nlargest(20).index
fig, ax = plt.subplots(figsize=(8, 5))
v_corr[top_v].sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Top 20 V-Features Correlated with isFraud")
ax.set_xlabel("Pearson Correlation")
plt.tight_layout()
plt.savefig(OUT / "07_top_v_features.png", dpi=150)
plt.close()

# ── 10. Correlation heatmap (key numeric features) ────────────────────────
key_cols = ["TransactionAmt", "card1", "card2", "card3", "card5",
            "addr1", "addr2", "dist1", "dist2", "isFraud"]
key_cols = [c for c in key_cols if c in df.columns]
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(df[key_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, linewidths=0.5)
ax.set_title("Correlation Heatmap (Key Features)")
plt.tight_layout()
plt.savefig(OUT / "08_correlation_heatmap.png", dpi=150)
plt.close()

# ── 11. Summary report ────────────────────────────────────────────────────
report = f"""
IEEE-CIS Fraud Detection — EDA Summary
=======================================
Total transactions : {len(df):,}
Fraud cases        : {df['isFraud'].sum():,} ({fraud_rate:.4%})
Total features     : {df.shape[1]}

Missing data
  >30% missing     : {len(miss_high)} features
  >90% missing     : {(miss > 0.9).sum()} features

TransactionAmt
  Mean (legit)     : ${df[df.isFraud==0]['TransactionAmt'].mean():.2f}
  Mean (fraud)     : ${df[df.isFraud==1]['TransactionAmt'].mean():.2f}
  Median (legit)   : ${df[df.isFraud==0]['TransactionAmt'].median():.2f}
  Median (fraud)   : ${df[df.isFraud==1]['TransactionAmt'].median():.2f}

Top ProductCD fraud rates:
{prod.to_string()}

Outputs saved to: {OUT.resolve()}
"""
print(report)
(OUT / "eda_summary.txt").write_text(report)
print("EDA complete. All outputs saved to data/eda_outputs/")
