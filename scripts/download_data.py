from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "creditcard.csv"
DEST.parent.mkdir(parents=True, exist_ok=True)

def main():
    if DEST.exists():
        print(f"Dataset already exists: {DEST}")
        return
    try:
        from sklearn.datasets import fetch_openml
        import pandas as pd
        print("Downloading ULB Credit Card Fraud dataset from OpenML data_id=1597...")
        # OpenML mirrors the standard ULB dataset. Some versions omit Time; the
        # training pipeline supports both canonical and mirrored feature schemas.
        bunch = fetch_openml(data_id=1597, as_frame=True, parser="auto")
        X = bunch.data
        y = bunch.target
        df = X.copy()
        df["Class"] = pd.to_numeric(y, errors="coerce")
        if df["Class"].isna().any():
            df["Class"] = y.astype(str).str.strip().map({"0": 0, "1": 1})
        if df["Class"].isna().any():
            raise ValueError("Could not normalize OpenML target labels.")
        df["Class"] = df["Class"].astype(int)
        # OpenML may name the target differently or omit Time. Verify that the
        # columns are compatible with app.ml.data before writing.
        required = {f"V{i}" for i in range(1, 29)} | {"Amount", "Class"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"OpenML columns don't match expected schema. Got: {list(df.columns)}")
        df.to_csv(DEST, index=False)
        print(f"Saved {len(df):,} rows to {DEST}")
        print("Note: OpenML mirrors may omit Time; this pipeline handles that.")
    except Exception as exc:
        print(f"Automatic OpenML download failed: {exc}", file=sys.stderr)
        print("\nFallback:")
        print("1. Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"2. Copy it to: {DEST}")
        sys.exit(1)

if __name__ == "__main__":
    main()
