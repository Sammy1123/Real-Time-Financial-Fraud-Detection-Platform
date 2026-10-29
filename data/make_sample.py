"""Create a committable sample parquet so the dashboard runs without Kaggle data.

Keeps every fraud row and subsamples legitimate transactions to a fixed total,
preserving the same columns as the full training set. Written to
data/processed/train_sample.parquet (git-tracked).
"""
import json
import os
from pathlib import Path

import pandas as pd

PROC = Path("data/processed")
TARGET_ROWS = 150_000

df = pd.read_parquet(PROC / "train.parquet")
fraud = df[df.isFraud == 1]
n_legit = TARGET_ROWS - len(fraud)
legit = df[df.isFraud == 0].sample(n=n_legit, random_state=42)
sample = (
    pd.concat([fraud, legit])
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)
sample.to_parquet(PROC / "train_sample.parquet", index=False)

size_mb = (PROC / "train_sample.parquet").stat().st_size / 1048576
print(f"sample: {len(sample):,} rows, fraud rate {sample.isFraud.mean():.3%}, "
      f"{size_mb:.1f} MB")

meta = json.load(open("models/model_meta.json"))
print(f"model_meta.json keys: {list(meta.keys())[:4]}...")
