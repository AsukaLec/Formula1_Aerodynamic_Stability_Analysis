import pandas as pd
import numpy as np
import os

from src.utils.config import RAW_DATA, PROCESSED_DIR, DTYPE, RANDOM_STATE

np.random.seed(RANDOM_STATE)


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA)


def detect_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    report = report[report["missing_count"] > 0].sort_values("missing_pct", ascending=False)
    return report


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        missing_pct = df[col].isnull().mean() * 100
        if missing_pct == 0:
            continue
        if missing_pct < 5:
            df = df.dropna(subset=[col])
        elif missing_pct <= 20:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
        else:
            df = df.drop(columns=[col])
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    if n_before != n_after:
        print(f"  Dedup: {n_before} -> {n_after} (removed {n_before - n_after} rows)")
    return df


def detect_outliers_iqr(df: pd.DataFrame, cols: list, factor: float = 3.0) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        col_mask = (df[col] >= lower) & (df[col] <= upper)
        mask = mask & col_mask
    return mask


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col == "drs_active":
            df[col] = df[col].astype(np.int8)
        elif df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].astype(np.float32)
    return df


def run_cleaning(df: pd.DataFrame = None) -> pd.DataFrame:
    print("=" * 60)
    print("  1.1 Data Cleaning")

    if df is None:
        print("  Loading raw data...")
        df = load_raw()
    print(f"  Raw samples: {len(df)}")
    print(f"  Raw columns: {len(df.columns)}")

    print("\n  [1] Missing value detection")
    missing_report = detect_missing(df)
    if len(missing_report) == 0:
        print("    No missing values")
    else:
        print(missing_report.to_string())
        df = handle_missing(df)
        print(f"    Samples after handling: {len(df)}")

    print("\n  [2] Duplicate detection")
    df = remove_duplicates(df)

    num_cols = ["speed_kmh", "wing_angle_deg", "downforce_n", "drag_n"]
    outlier_mask = detect_outliers_iqr(df, num_cols, factor=3.0)
    n_outliers = (~outlier_mask).sum()
    print(f"\n  [3] Outlier detection (IQR, factor=3.0)")
    print(f"    Outlier count: {n_outliers} ({n_outliers/len(df)*100:.2f}%)")
    df = df[outlier_mask].reset_index(drop=True)

    print("\n  [4] Type conversion")
    df = convert_dtypes(df)
    print(f"    Unified to {DTYPE} / int8 (drs_active)")

    out_path = os.path.join(PROCESSED_DIR, "clean_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Cleaned dataset saved: {out_path}")
    print(f"  Samples: {len(df)}, Columns: {len(df.columns)}")
    return df
