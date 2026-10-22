from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Class"
EXPECTED_PCA_FEATURES = [f"V{i}" for i in range(1, 29)]

def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python scripts/download_data.py` "
            "or place Kaggle's creditcard.csv at data/raw/creditcard.csv."
        )
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    # Standard ULB dataset has Class; tolerate lowercase names.
    rename = {c: c.capitalize() for c in df.columns if c.lower() == "class"}
    df = df.rename(columns=rename)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found. Columns: {list(df.columns)}")
    if not set(df[TARGET].dropna().unique()).issubset({0, 1, 0.0, 1.0, "0", "1"}):
        raise ValueError("Class target must contain binary labels 0 and 1.")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    if df[TARGET].nunique() != 2:
        raise ValueError("Dataset must contain both legitimate and fraud classes.")
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df

def feature_columns(df: pd.DataFrame) -> list[str]:
    # The canonical Kaggle CSV has Time, V1..V28, Amount, Class.
    # OpenML mirrors can omit Time, so use only available known input columns.
    cols = (["Time"] if "Time" in df.columns else []) + EXPECTED_PCA_FEATURES + ["Amount"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected features: {missing}")
    return cols

def split_data(df: pd.DataFrame, features: list[str]):
    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET]
    if X.isna().any().any() or not X.notna().all().all():
        raise ValueError("Input features contain missing or non-numeric values.")
    if not (X.dtypes.apply(lambda d: pd.api.types.is_numeric_dtype(d))).all():
        raise ValueError("All model features must be numeric.")
    # Stratification keeps the very rare positive class represented in each split.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
