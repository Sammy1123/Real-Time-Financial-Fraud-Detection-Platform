"""
Downloads the IEEE-CIS Fraud Detection dataset from Kaggle
and copies CSVs into data/raw/.
"""
import shutil
from pathlib import Path
import kagglehub

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(exist_ok=True)

def main():
    print("Downloading IEEE-CIS Fraud Detection dataset...")
    path = Path(kagglehub.competition_download("ieee-fraud-detection"))
    print(f"Downloaded to: {path}")

    csv_files = list(path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path}")

    for src in csv_files:
        dst = RAW_DIR / src.name
        shutil.copy2(src, dst)
        print(f"  Copied {src.name} -> {dst}")

    print(f"\nAll files saved to {RAW_DIR}")

if __name__ == "__main__":
    main()
