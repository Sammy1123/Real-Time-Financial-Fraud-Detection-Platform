"""
Feature engineering for IEEE-CIS Fraud Detection.
Outputs processed train/test parquets to data/processed/
and saves the feature list to features/feature_store_data/features.json
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

RAW  = Path("data/raw")
PROC = Path("data/processed")
FS   = Path("features/feature_store_data")
PROC.mkdir(exist_ok=True)
FS.mkdir(exist_ok=True)

# ── 1. Load & merge ────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv(RAW / "train_transaction.csv").merge(
    pd.read_csv(RAW / "train_identity.csv"), on="TransactionID", how="left")
test  = pd.read_csv(RAW / "test_transaction.csv").merge(
    pd.read_csv(RAW / "test_identity.csv"),  on="TransactionID", how="left")

print(f"Train: {train.shape}  Test: {test.shape}")

# ── 2. Time features ───────────────────────────────────────────────────────
for df in [train, test]:
    df["hour"]    = (df["TransactionDT"] // 3600) % 24
    df["day"]     = (df["TransactionDT"] // 86400) % 7
    df["log_amt"] = np.log1p(df["TransactionAmt"])

# ── 3. Frequency encoding ──────────────────────────────────────────────────
FREQ_COLS = ["card1", "card2", "addr1", "P_emaildomain", "R_emaildomain"]

for col in FREQ_COLS:
    freq = train[col].value_counts(normalize=True)
    train[f"{col}_freq"] = train[col].map(freq).fillna(0)
    test[f"{col}_freq"]  = test[col].map(freq).fillna(0)

# ── 4. Aggregated stats per card1 ──────────────────────────────────────────
agg = train.groupby("card1")["TransactionAmt"].agg(
    card1_amt_mean="mean", card1_amt_std="std").reset_index()
train = train.merge(agg, on="card1", how="left")
test  = test.merge(agg,  on="card1", how="left")

for df in [train, test]:
    df["card1_amt_std"] = df["card1_amt_std"].fillna(0)
    df["amt_vs_card1_mean"] = df["TransactionAmt"] / (df["card1_amt_mean"] + 1)

# ── 5. Velocity features (tx count + amount per card1 in last 1h / 24h) ────
print("Computing velocity features...")
train = train.sort_values("TransactionDT").reset_index(drop=True)
test  = test.sort_values("TransactionDT").reset_index(drop=True)

def add_velocity(df, ref_df, windows={"1h": 3600, "24h": 86400}):
    """Ultra-fast velocity using numpy searchsorted per card1 group."""
    ref = ref_df[["TransactionDT", "card1", "TransactionAmt"]].sort_values(["card1", "TransactionDT"])
    df  = df.sort_values("TransactionDT").copy()

    for label, secs in windows.items():
        counts  = np.zeros(len(df), dtype=np.int32)
        amounts = np.zeros(len(df), dtype=np.float32)

        for card, grp in ref.groupby("card1", sort=False):
            times = grp["TransactionDT"].values
            amts  = grp["TransactionAmt"].values
            cumamt = np.concatenate([[0], np.cumsum(amts)])

            # rows in df for this card
            df_mask = df["card1"].values == card
            df_idx  = np.where(df_mask)[0]
            if len(df_idx) == 0:
                continue

            df_times = df["TransactionDT"].values[df_idx]
            hi = np.searchsorted(times, df_times, side="left")       # exclude current tx
            lo = np.searchsorted(times, df_times - secs, side="left")
            counts[df_idx]  = hi - lo
            amounts[df_idx] = cumamt[hi] - cumamt[lo]

        df[f"card1_tx_count_{label}"]  = counts
        df[f"card1_tx_amount_{label}"] = amounts
    return df

# Use train as reference for both (test looks up against train history)
train = add_velocity(train, train)
test  = add_velocity(test,  train)

# ── 6. card1 + addr1 pair frequency encoding ──────────────────────────────
train["card1_addr1"] = train["card1"].astype(str) + "_" + train["addr1"].astype(str)
test["card1_addr1"]  = test["card1"].astype(str)  + "_" + test["addr1"].astype(str)
pair_freq = train["card1_addr1"].value_counts(normalize=True)
train["card1_addr1_freq"] = train["card1_addr1"].map(pair_freq).fillna(0)
test["card1_addr1_freq"]  = test["card1_addr1"].map(pair_freq).fillna(0)

# ── 7. Target encoding for high-cardinality categoricals ──────────────────
for col in ["ProductCD", "card4", "card6"]:
    target_mean = train.groupby(col)["isFraud"].mean()
    train[f"{col}_target"] = train[col].map(target_mean).fillna(train["isFraud"].mean())
    test[f"{col}_target"]  = test[col].map(target_mean).fillna(train["isFraud"].mean())

# ── 8. Top V-features (by correlation with isFraud on train) ──────────────
v_cols = [c for c in train.columns if c.startswith("V")]
v_corr = train[v_cols + ["isFraud"]].corr()["isFraud"].drop("isFraud")
TOP_V  = v_corr.abs().nlargest(30).index.tolist()

# ── 9. Categorical label encoding ─────────────────────────────────────────
CAT_COLS = ["ProductCD", "card4", "card6", "M1", "M2", "M3",
            "M4", "M5", "M6", "M7", "M8", "M9"]
CAT_COLS = [c for c in CAT_COLS if c in train.columns]

for col in CAT_COLS:
    mapping = {v: i for i, v in enumerate(train[col].dropna().unique())}
    train[col] = train[col].map(mapping).fillna(-1).astype(int)
    test[col]  = test[col].map(mapping).fillna(-1).astype(int)

# ── 10. Assemble final feature list ───────────────────────────────────────
BASE = ["log_amt", "TransactionAmt", "hour", "day",
        "card1_amt_mean", "card1_amt_std", "amt_vs_card1_mean",
        "card1_tx_count_1h", "card1_tx_amount_1h",
        "card1_tx_count_24h", "card1_tx_amount_24h",
        "card1_addr1_freq",
        "ProductCD_target", "card4_target", "card6_target"]
FREQ_FEATS = [f"{c}_freq" for c in FREQ_COLS]
FEATURES   = BASE + FREQ_FEATS + CAT_COLS + TOP_V
FEATURES   = [f for f in FEATURES if f in train.columns]

print(f"Total features: {len(FEATURES)}")

# ── 11. Fill remaining NaNs & save ─────────────────────────────────────────
train[FEATURES] = train[FEATURES].fillna(-999)
test[FEATURES]  = test[FEATURES].fillna(-999)

train[FEATURES + ["isFraud", "TransactionID"]].to_parquet(PROC / "train.parquet", index=False)
test[FEATURES  + ["TransactionID"]].to_parquet(PROC / "test.parquet",  index=False)

json.dump(FEATURES, open(FS / "features.json", "w"), indent=2)

# ── 12. Serialize lookup maps for API fast startup ──────────────────────────
import joblib

freq_maps = {}
for col in FREQ_COLS:
    freq_maps[col] = train[col].value_counts(normalize=True).to_dict()
freq_maps["card1_amt_mean"]    = train.groupby("card1")["TransactionAmt"].mean().to_dict()
freq_maps["card1_amt_std"]     = train.groupby("card1")["TransactionAmt"].std().fillna(0).to_dict()
freq_maps["card1_addr1_freq"]  = pair_freq.to_dict()
freq_maps["ProductCD_target"]  = train.groupby("ProductCD")["isFraud"].mean().to_dict()
freq_maps["card4_target"]      = train.groupby("card4")["isFraud"].mean().to_dict()
freq_maps["card6_target"]      = train.groupby("card6")["isFraud"].mean().to_dict()
freq_maps["global_fraud_mean"] = float(train["isFraud"].mean())

cat_maps = {}
for col in CAT_COLS:
    cat_maps[col] = {v: i for i, v in enumerate(train[col].dropna().unique())}

joblib.dump(freq_maps, FS / "freq_maps.pkl")
joblib.dump(cat_maps,  FS / "cat_maps.pkl")

print("Saved:")
print(f"  {PROC / 'train.parquet'}")
print(f"  {PROC / 'test.parquet'}")
print(f"  {FS / 'features.json'}")
print(f"  {FS / 'freq_maps.pkl'}")
print(f"  {FS / 'cat_maps.pkl'}")
print("Feature engineering complete.")
