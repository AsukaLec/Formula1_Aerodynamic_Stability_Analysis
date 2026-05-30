import numpy as np
import pandas as pd
import os

from src.utils.config import PROCESSED_DIR, STABILITY_BINS, STABILITY_LABELS


def assign_stability_bin(y: pd.Series) -> pd.Series:
    return pd.cut(y, bins=STABILITY_BINS, labels=STABILITY_LABELS, include_lowest=True, right=False)


def compute_correlations(df: pd.DataFrame, cols: list) -> tuple:
    pearson = df[cols].corr(method="pearson")
    spearman = df[cols].corr(method="spearman")
    return pearson, spearman


def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["force_ratio"] = df["downforce_n"] / df["drag_n"].replace(0, np.nan)
    df["force_ratio"] = df["force_ratio"].fillna(0).astype("float32")
    df["speed_wing"] = df["speed_kmh"] * df["wing_angle_deg"]
    df["speed_wing"] = df["speed_wing"].astype("float32")
    return df


def add_quantile_bins(df: pd.DataFrame, cols: list, n_bins: int = 5) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[f"{col}_qbin"] = pd.qcut(df[col], q=n_bins, labels=False, duplicates="drop")
        df[f"{col}_qbin"] = df[f"{col}_qbin"].astype("int8")
    return df


def add_polynomial_features(df: pd.DataFrame, cols: list, degree: int = 2) -> pd.DataFrame:
    df = df.copy()
    for i, c1 in enumerate(cols):
        for c2 in cols[i:]:
            if c1 == c2:
                name = f"{c1}_sq"
                df[name] = (df[c1] ** 2).astype("float32")
            else:
                name = f"{c1}_{c2}_inter"
                df[name] = (df[c1] * df[c2]).astype("float32")
    return df


def run_feature_engineering(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  1.4 Feature Engineering")

    num_cols = ["speed_kmh", "wing_angle_deg", "downforce_n", "drag_n", "stability_index"]

    print("\n  [1] Pearson/Spearman Correlation Matrix")
    pearson, spearman = compute_correlations(df, num_cols)

    print("\n  [2] 4-bin Stability Labels")

    print("\n  [3] Physical Composite Features")
    df = add_composite_features(df)
    print(f"    Added: force_ratio (downforce_n / drag_n)")
    print(f"    Added: speed_wing (speed_kmh * wing_angle_deg)")
    df["porpoising_flag"] = (df["stability_index"] < 95).astype("int8")
    print(f"    Added: porpoising_flag (stability_index < 95)")

    print("\n  [4] Quantile Binning (n_bins=5)")
    bin_cols = ["speed_kmh", "wing_angle_deg", "downforce_n", "drag_n"]
    df = add_quantile_bins(df, bin_cols, n_bins=5)
    print(f"    Added: {', '.join(f'{c}_qbin' for c in bin_cols)}")

    df.to_csv(os.path.join(PROCESSED_DIR, "features_dataset.csv"), index=False)
    print(f"\n  Feature-engineered dataset saved: {os.path.join(PROCESSED_DIR, 'features_dataset.csv')}")
    print(f"  Total columns: {len(df.columns)}")
    return df, pearson, spearman
