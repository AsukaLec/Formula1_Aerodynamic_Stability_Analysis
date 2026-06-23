#!/usr/bin/env python
"""
T01: Data Preprocessing & Feature Engineering — End-to-end pipeline
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import (
    PROCESSED_DIR, REPORTS_DIR, RANDOM_STATE,
    STABILITY_BINS, STABILITY_LABELS,
)
from src.preprocessing.cleaner import run_cleaning
from src.preprocessing.normalizer import run_normalization, normalize_splits
from src.preprocessing.feature_engineering import run_feature_engineering


def stratified_split(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  1.3 Data Split (70/15/15 stratified)")
    print("=" * 60)

    df["stability_label"] = pd.cut(
        df["stability_index"],
        bins=STABILITY_BINS,
        labels=STABILITY_LABELS,
        include_lowest=True,
        right=False,
    )

    train, temp = train_test_split(
        df, test_size=0.30, random_state=RANDOM_STATE,
        stratify=df["stability_label"],
    )
    valid, test = train_test_split(
        temp, test_size=0.50, random_state=RANDOM_STATE,
        stratify=temp["stability_label"],
    )

    splits = {"train": train, "valid": valid, "test": test}
    for name, split in splits.items():
        split.to_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"), index=False)
        print(f"  {name}: {len(split)} samples")

    print("\n  Bin distribution check:")
    for name, split in splits.items():
        counts = split["stability_label"].value_counts(normalize=True).sort_index()
        summary = {str(k): round(v, 4) for k, v in counts.items()}
        print(f"    {name}: {summary}")

    return train, valid, test


def generate_imbalance_report(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  1.5 Data Imbalance Report")
    print("=" * 60)

    df = df.copy()
    df["stability_label"] = pd.cut(
        df["stability_index"],
        bins=STABILITY_BINS,
        labels=STABILITY_LABELS,
        include_lowest=True,
        right=False,
    )

    counts = df["stability_label"].value_counts().sort_index()
    total = len(df)
    severe_count = counts.get("severe", 0)
    stable_count = counts.get("stable", 0)
    unstable_total = total - stable_count

    weights = {}
    for label in STABILITY_LABELS:
        if counts.get(label, 0) > 0:
            weights[label] = total / (len(STABILITY_LABELS) * counts[label])
        else:
            weights[label] = 1.0

    sample_weights = df["stability_label"].map(weights).values
    np.save(os.path.join(PROCESSED_DIR, "sample_weights.npy"), sample_weights.astype("float32"))

    bin_desc = ", ".join(f"{STABILITY_LABELS[i]}: [{STABILITY_BINS[i]}, {STABILITY_BINS[i+1]})" for i in range(len(STABILITY_LABELS)))

    report_path = os.path.join(REPORTS_DIR, "imbalance_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Data Imbalance Report\n\n")
        f.write("## Overview\n\n")
        f.write(f"- Total samples: {total}\n")
        f.write(f"- Stability bins: {bin_desc}\n")
        f.write(f"- Unstable (stability < 95): {unstable_total} ({unstable_total/total*100:.2f}%)\n\n")

        f.write("## Bin Distribution\n\n")
        f.write("| Bin | Count | Ratio | Weight |\n")
        f.write("|------|--------|------|------|\n")
        for label in STABILITY_LABELS:
            c = counts.get(label, 0)
            pct = c / total * 100
            w = weights[label]
            f.write(f"| {label} | {c} | {pct:.2f}% | {w:.4f} |\n")

        f.write("\n## Skew Analysis\n\n")
        imbalance_ratio = stable_count / max(severe_count, 1)
        f.write(f"- stable / severe ratio: {imbalance_ratio:.2f}:1\n")
        if imbalance_ratio > 3:
            f.write(f"- **Severely skewed**: stable samples dominate (ratio > 3:1)\n")
        elif imbalance_ratio > 1.5:
            f.write(f"- **Moderately skewed**: stable samples somewhat over-represented (ratio > 1.5:1)\n")
        else:
            f.write(f"- **Mildly skewed**: distribution is relatively balanced\n")

        f.write("\n## Weight Strategy\n\n")
        f.write(f"- Class-balanced weighting: `weight_i = N / (K * n_i)`\n")
        for label in STABILITY_LABELS:
            f.write(f"- {label} weight: {weights.get(label, 0):.4f}\n")
        f.write(f"- Weights saved to `data/processed/sample_weights.npy`\n")

    print(f"  Report saved: {report_path}")
    for label in STABILITY_LABELS:
        c = counts.get(label, 0)
        print(f"    {label}: {c} ({c/total*100:.2f}%), weight={weights[label]:.4f}")


def main():
    print("=" * 60)
    print("  T01: Data Preprocessing & Feature Engineering")
    print("=" * 60)

    df_clean = run_cleaning()

    train, valid, test = stratified_split(df_clean)

    full_feature_cols = ["speed_kmh", "wing_angle_deg", "drs_active", "downforce_n", "drag_n"]
    _, _ = run_normalization(df_clean)

    print("\n  Applying normalization to train/valid/test splits...")
    normalize_splits(train, valid, test, full_feature_cols, scaler_type="mm")
    normalize_splits(train, valid, test, full_feature_cols, scaler_type="ss")

    df_feat, pearson, spearman = run_feature_engineering(df_clean)

    generate_imbalance_report(df_clean)

    print("\n" + "=" * 60)
    print("  T01 Complete!")
    print("=" * 60)
    print("  Output files:")
    print("    data/processed/clean_dataset.csv")
    print("    data/processed/train.csv, valid.csv, test.csv")
    print("    data/processed/X_train_mm.npy, y_train.npy etc.")
    print("    data/processed/scaler_mm.pkl, scaler_ss.pkl")
    print("    data/processed/sample_weights.npy")
    print("    data/processed/features_dataset.csv")
    print("    reports/imbalance_report.md")


if __name__ == "__main__":
    main()
