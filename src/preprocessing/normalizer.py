import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.utils.config import PROCESSED_DIR, RANDOM_STATE


def fit_save_scalers(df: pd.DataFrame, feature_cols: list):
    X = df[feature_cols].values

    mm = MinMaxScaler()
    mm.fit(X)

    ss = StandardScaler()
    ss.fit(X)

    mm_path = os.path.join(PROCESSED_DIR, "scaler_mm.pkl")
    ss_path = os.path.join(PROCESSED_DIR, "scaler_ss.pkl")
    with open(mm_path, "wb") as f:
        pickle.dump(mm, f)
    with open(ss_path, "wb") as f:
        pickle.dump(ss, f)

    print(f"  MinMax scaler saved: {mm_path}")
    print(f"  Standard scaler saved: {ss_path}")
    print(f"  MinMax range: [{mm.data_min_}, {mm.data_max_}]")
    print(f"  Standard mean: {ss.mean_}, std: {ss.scale_}")

    return mm, ss


def load_scaler(scaler_type: str = "mm"):
    path = os.path.join(PROCESSED_DIR, f"scaler_{scaler_type}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_splits(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list,
    scaler_type: str = "mm",
):
    scaler = load_scaler(scaler_type)

    def _norm(subset):
        X = scaler.transform(subset[feature_cols].values)
        y = subset["stability_index"].values
        return X.astype("float32"), y.astype("float32")

    Xt, yt = _norm(train)
    Xv, yv = _norm(valid)
    Xs, ys = _norm(test)

    prefix = f"{scaler_type}"
    np.save(os.path.join(PROCESSED_DIR, f"X_train_{prefix}.npy"), Xt)
    np.save(os.path.join(PROCESSED_DIR, f"y_train.npy"), yt)
    np.save(os.path.join(PROCESSED_DIR, f"X_valid_{prefix}.npy"), Xv)
    np.save(os.path.join(PROCESSED_DIR, f"y_valid.npy"), yv)
    np.save(os.path.join(PROCESSED_DIR, f"X_test_{prefix}.npy"), Xs)
    np.save(os.path.join(PROCESSED_DIR, f"y_test.npy"), ys)

    print(f"  [{scaler_type}] Normalization done: X_train={Xt.shape}, y_train={yt.shape}")
    return (Xt, yt), (Xv, yv), (Xs, ys)


def run_normalization(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  1.2 Data Normalization")
    print("=" * 60)

    feature_cols = ["speed_kmh", "wing_angle_deg", "drs_active", "downforce_n", "drag_n"]
    mm, ss = fit_save_scalers(df, feature_cols)
    return mm, ss
