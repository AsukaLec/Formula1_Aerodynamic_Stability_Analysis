#!/usr/bin/env python
"""
T03: Surrogate Model Comparison — Train & evaluate 6 models
"""
import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import (
    PROCESSED_DIR, MODELS_OUTPUT_DIR, DEEP_ENSEMBLE_DIR,
    MODELS_COMPARISON_CSV, MODELS_FIGURES_DIR, REPORTS_DIR,
    FEATURE_COLS, STABILITY_BINS, STABILITY_LABELS,
    RANDOM_STATE, INFERENCE_WARMUP, INFERENCE_REPEATS,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LR,
    NN_BATCH_SIZE, NN_MAX_EPOCHS, NN_EARLY_STOP_PATIENCE,
    NN_LR, NN_WEIGHT_DECAY, MLP_HIDDEN_UNITS, MLP_DROPOUT,
    DEEP_ENSEMBLE_M, TABNET_MAX_EPOCHS,
)
from src.utils.metrics import compute_metrics, compute_regional_metrics, measure_inference_latency
from src.models.baseline import RidgeModel, RandomForestModel, XGBoostModel
from src.models.mlp import MLPTrainer
from src.models.tabnet_model import TabNetModel
from src.models.deep_ensemble import DeepEnsemble
from src.models.evaluate import (
    evaluate_model, compare_models, plot_predicted_vs_actual, plot_predicted_vs_actual_stratified, plot_training_curves,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_FIGURES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODELS_COMPARISON_CSV), exist_ok=True)


def load_data():
    X_train_mm = np.load(os.path.join(PROCESSED_DIR, "X_train_mm.npy")).astype(np.float32)
    X_valid_mm = np.load(os.path.join(PROCESSED_DIR, "X_valid_mm.npy")).astype(np.float32)
    X_test_mm = np.load(os.path.join(PROCESSED_DIR, "X_test_mm.npy")).astype(np.float32)
    X_train_ss = np.load(os.path.join(PROCESSED_DIR, "X_train_ss.npy")).astype(np.float32)
    X_valid_ss = np.load(os.path.join(PROCESSED_DIR, "X_valid_ss.npy")).astype(np.float32)
    X_test_ss = np.load(os.path.join(PROCESSED_DIR, "X_test_ss.npy")).astype(np.float32)
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy")).astype(np.float32)
    y_valid = np.load(os.path.join(PROCESSED_DIR, "y_valid.npy")).astype(np.float32)
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy")).astype(np.float32)

    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    counts = train_df["stability_label"].value_counts()
    N = len(train_df)
    K = len(STABILITY_LABELS)
    weight_map = {label: N / (K * counts.get(label, 1)) for label in STABILITY_LABELS}
    sample_weight = train_df["stability_label"].map(weight_map).values.astype(np.float32)

    return {
        "X_train_mm": X_train_mm, "X_valid_mm": X_valid_mm, "X_test_mm": X_test_mm,
        "X_train_ss": X_train_ss, "X_valid_ss": X_valid_ss, "X_test_ss": X_test_ss,
        "y_train": y_train, "y_valid": y_valid, "y_test": y_test,
        "sample_weight": sample_weight,
    }


def load_scalers():
    with open(os.path.join(PROCESSED_DIR, "scaler_mm.pkl"), "rb") as f:
        scaler_mm = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "scaler_ss.pkl"), "rb") as f:
        scaler_ss = pickle.load(f)
    return scaler_mm, scaler_ss


def identity_scaler(y):
    return y


def y_div_100_scaler(y):
    return y * 100.0


def run_all(data):
    X_train_mm, X_valid_mm, X_test_mm = data["X_train_mm"], data["X_valid_mm"], data["X_test_mm"]
    X_train_ss, X_valid_ss, X_test_ss = data["X_train_ss"], data["X_valid_ss"], data["X_test_ss"]
    y_train, y_valid, y_test = data["y_train"], data["y_valid"], data["y_test"]
    s_weight = data["sample_weight"]

    y_train_scaled = y_train / 100.0
    y_valid_scaled = y_valid / 100.0
    y_test_scaled = y_test / 100.0

    all_results = []
    best_model_info = {"R2": -np.inf, "model": None, "name": None}
    pred_dict = {}

    # ============ M1: Ridge ============
    print("\n" + "=" * 60)
    print("  M1: Ridge Regression (StandardScaler)")
    print("=" * 60)
    m1 = RidgeModel()
    result, y_pred_m1 = evaluate_model(
        m1, X_train_ss, y_train, X_test_ss, y_test,
        model_name="M1_Ridge", y_scaler=identity_scaler, sample_weight=s_weight,
    )
    all_results.append(result)
    pred_dict["M1_Ridge"] = y_pred_m1
    with open(os.path.join(MODELS_OUTPUT_DIR, "M1_Ridge.pkl"), "wb") as f:
        pickle.dump(m1.model, f)
    print(f"  R2={result['R2']}, MSE={result['MSE']}")
    if result["R2"] > best_model_info["R2"]:
        best_model_info = {"R2": result["R2"], "model": m1, "name": "M1_Ridge", "file": "M1_Ridge.pkl"}

    # ============ M2: Random Forest ============
    print("\n" + "=" * 60)
    print("  M2: Random Forest (StandardScaler)")
    print("=" * 60)
    m2 = RandomForestModel()
    result, y_pred_m2 = evaluate_model(
        m2, X_train_ss, y_train, X_test_ss, y_test,
        model_name="M2_RandomForest", y_scaler=identity_scaler, sample_weight=s_weight,
    )
    all_results.append(result)
    pred_dict["M2_RandomForest"] = y_pred_m2
    with open(os.path.join(MODELS_OUTPUT_DIR, "M2_RandomForest.pkl"), "wb") as f:
        pickle.dump(m2.model, f)
    print(f"  R2={result['R2']}, MSE={result['MSE']}")
    if result["R2"] > best_model_info["R2"]:
        best_model_info = {"R2": result["R2"], "model": m2, "name": "M2_RandomForest", "file": "M2_RandomForest.pkl"}

    # ============ M3: XGBoost ============
    print("\n" + "=" * 60)
    print("  M3: XGBoost (StandardScaler)")
    print("=" * 60)
    m3 = XGBoostModel()
    result, y_pred_m3 = evaluate_model(
        m3, X_train_ss, y_train, X_test_ss, y_test,
        model_name="M3_XGBoost", y_scaler=identity_scaler, sample_weight=s_weight,
    )
    all_results.append(result)
    pred_dict["M3_XGBoost"] = y_pred_m3
    with open(os.path.join(MODELS_OUTPUT_DIR, "M3_XGBoost.pkl"), "wb") as f:
        pickle.dump(m3.model, f)
    print(f"  R2={result['R2']}, MSE={result['MSE']}")
    if result["R2"] > best_model_info["R2"]:
        best_model_info = {"R2": result["R2"], "model": m3, "name": "M3_XGBoost", "file": "M3_XGBoost.pkl"}

    # ============ M4: MLP ============
    print("\n" + "=" * 60)
    print("  M4: MLP (MinMax, y/100)")
    print("=" * 60)
    m4 = MLPTrainer(
        input_dim=5, hidden_units=MLP_HIDDEN_UNITS, dropout=MLP_DROPOUT,
        lr=NN_LR, weight_decay=NN_WEIGHT_DECAY, batch_size=NN_BATCH_SIZE,
        max_epochs=NN_MAX_EPOCHS, patience=NN_EARLY_STOP_PATIENCE, seed=RANDOM_STATE,
    )
    t0 = time.time()
    m4.train(X_train_mm, y_train_scaled, X_valid_mm, y_valid_scaled, sample_weight=s_weight)
    train_time = time.time() - t0
    y_pred_m4 = m4.predict(X_test_mm) * 100.0
    metrics_m4 = compute_metrics(y_test, y_pred_m4)
    regional_m4 = compute_regional_metrics(y_test, y_pred_m4, threshold=95.0)
    X_sample = X_test_mm[:1]
    try:
        latency_ms = measure_inference_latency(m4, X_sample, INFERENCE_WARMUP, INFERENCE_REPEATS)["latency_ms"]
    except Exception:
        latency_ms = float("nan")
    result = {
        "Model": "M4_MLP",
        "R2": round(metrics_m4["R2"], 4), "MSE": round(metrics_m4["MSE"], 4),
        "MAE": round(metrics_m4["MAE"], 4), "RMSE": round(metrics_m4["RMSE"], 4),
        "R2_low": round(regional_m4["R2_low"], 4), "MSE_low": round(regional_m4["MSE_low"], 4),
        "R2_high": round(regional_m4["R2_high"], 4), "MSE_high": round(regional_m4["MSE_high"], 4),
        "Latency_ms": round(latency_ms, 3), "Train_Time_s": round(train_time, 2),
    }
    all_results.append(result)
    pred_dict["M4_MLP"] = y_pred_m4
    torch.save(m4.model.state_dict(), os.path.join(MODELS_OUTPUT_DIR, "M4_MLP.pt"))
    print(f"  R2={result['R2']}, MSE={result['MSE']}")
    if result["R2"] > best_model_info["R2"]:
        best_model_info = {"R2": result["R2"], "model": m4, "name": "M4_MLP", "file": "M4_MLP.pt"}

    # ============ M5: TabNet ============
    print("\n" + "=" * 60)
    print("  M5: TabNet (MinMax, y/100)")
    print("=" * 60)
    try:
        m5 = TabNetModel(max_epochs=TABNET_MAX_EPOCHS)
        t0 = time.time()
        m5.train(X_train_mm, y_train_scaled, X_valid_mm, y_valid_scaled, sample_weight=s_weight)
        train_time = time.time() - t0
        y_pred_m5 = m5.predict(X_test_mm) * 100.0
        metrics_m5 = compute_metrics(y_test, y_pred_m5)
        regional_m5 = compute_regional_metrics(y_test, y_pred_m5, threshold=95.0)
        X_sample = X_test_mm[:1]
        try:
            latency = measure_inference_latency(m5, X_sample, INFERENCE_WARMUP, INFERENCE_REPEATS)
            latency_ms = latency["latency_ms"]
        except Exception:
            latency_ms = float("nan")
        result = {
            "Model": "M5_TabNet",
            "R2": round(metrics_m5["R2"], 4), "MSE": round(metrics_m5["MSE"], 4),
            "MAE": round(metrics_m5["MAE"], 4), "RMSE": round(metrics_m5["RMSE"], 4),
            "R2_low": round(regional_m5["R2_low"], 4), "MSE_low": round(regional_m5["MSE_low"], 4),
            "R2_high": round(regional_m5["R2_high"], 4), "MSE_high": round(regional_m5["MSE_high"], 4),
            "Latency_ms": round(latency_ms, 3), "Train_Time_s": round(train_time, 2),
        }
        all_results.append(result)
        pred_dict["M5_TabNet"] = y_pred_m5
        m5.model.save_model(os.path.join(MODELS_OUTPUT_DIR, "M5_TabNet"))
        print(f"  R2={result['R2']}, MSE={result['MSE']}")
        if result["R2"] > best_model_info["R2"]:
            best_model_info = {"R2": result["R2"], "model": m5, "name": "M5_TabNet", "file": "M5_TabNet"}
    except Exception as e:
        print(f"  TabNet failed: {e}")
        all_results.append({
            "Model": "M5_TabNet", "R2": float("nan"), "MSE": float("nan"),
            "MAE": float("nan"), "RMSE": float("nan"),
            "R2_low": float("nan"), "MSE_low": float("nan"),
            "R2_high": float("nan"), "MSE_high": float("nan"),
            "Latency_ms": float("nan"), "Train_Time_s": float("nan"),
        })

    # ============ M6: DeepEnsemble ============
    print("\n" + "=" * 60)
    print("  M6: DeepEnsemble (M={}, MinMax, y/100)".format(DEEP_ENSEMBLE_M))
    print("=" * 60)
    de = DeepEnsemble(
        input_dim=5, hidden_units=MLP_HIDDEN_UNITS, M=DEEP_ENSEMBLE_M,
        lr=NN_LR, weight_decay=NN_WEIGHT_DECAY,
        batch_size=NN_BATCH_SIZE, max_epochs=NN_MAX_EPOCHS,
        patience=NN_EARLY_STOP_PATIENCE,
    )
    t0 = time.time()
    de.train(X_train_mm, y_train_scaled, X_valid_mm, y_valid_scaled, sample_weight=s_weight)
    train_time = time.time() - t0
    y_pred_de = de.predict(X_test_mm) * 100.0
    metrics_de = compute_metrics(y_test, y_pred_de)
    regional_de = compute_regional_metrics(y_test, y_pred_de, threshold=95.0)
    X_sample = X_test_mm[:1]
    try:
        latency = measure_inference_latency(de, X_sample, INFERENCE_WARMUP, INFERENCE_REPEATS)
        latency_ms = latency["latency_ms"]
    except Exception:
        latency_ms = float("nan")
    result = {
        "Model": "M6_DeepEnsemble",
        "R2": round(metrics_de["R2"], 4), "MSE": round(metrics_de["MSE"], 4),
        "MAE": round(metrics_de["MAE"], 4), "RMSE": round(metrics_de["RMSE"], 4),
        "R2_low": round(regional_de["R2_low"], 4), "MSE_low": round(regional_de["MSE_low"], 4),
        "R2_high": round(regional_de["R2_high"], 4), "MSE_high": round(regional_de["MSE_high"], 4),
        "Latency_ms": round(latency_ms, 3), "Train_Time_s": round(train_time, 2),
    }
    all_results.append(result)
    pred_dict["M6_DeepEnsemble"] = y_pred_de
    de.save()
    print(f"  R2={result['R2']}, MSE={result['MSE']}")
    if result["R2"] > best_model_info["R2"]:
        best_model_info = {"R2": result["R2"], "model": de, "name": "M6_DeepEnsemble", "file": "deep_ensemble/"}

    # ============ Uncertainty analysis ============
    print("\n" + "=" * 60)
    print("  Uncertainty Analysis (DeepEnsemble)")
    print("=" * 60)
    mu_test, sigma_test = de.predict_with_uncertainty(X_test_mm)
    mu_test = mu_test * 100.0
    sigma_test = sigma_test * 100.0
    print(f"  Mean sigma (test): {sigma_test.mean():.4f}")
    print(f"  Max sigma  (test): {sigma_test.max():.4f}")
    mu_train, sigma_train = de.predict_with_uncertainty(X_train_mm[:5000])
    mu_train = mu_train * 100.0
    sigma_train = sigma_train * 100.0
    print(f"  Mean sigma (train, 5k): {sigma_train.mean():.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sigma_test, bins=50, alpha=0.6, label="Test", color="coral")
    ax.hist(sigma_train, bins=50, alpha=0.6, label="Train (5k)", color="steelblue")
    ax.set_xlabel("Prediction Std ($\\sigma$)")
    ax.set_ylabel("Frequency")
    ax.set_title("DeepEnsemble: Prediction Uncertainty Distribution")
    ax.legend()
    path = os.path.join(MODELS_FIGURES_DIR, "uncertainty_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Uncertainty plot saved: {path}")

    # ============ Compare ============
    print("\n" + "=" * 60)
    print("  Model Comparison Summary")
    print("=" * 60)
    df = compare_models(all_results, save_csv=True, plot_bar=True)

    plot_predicted_vs_actual(y_test, pred_dict, MODELS_FIGURES_DIR)
    plot_predicted_vs_actual_stratified(y_test, pred_dict, MODELS_FIGURES_DIR)

    # Training curves
    trainers_dict = {"M4_MLP": [m4]}
    trainers_dict["M6_DeepEnsemble"] = de.trainers
    plot_training_curves(trainers_dict, MODELS_FIGURES_DIR)

    # ============ Training time comparison ============
    plot_training_time_comparison(df, MODELS_FIGURES_DIR)

    # ============ Best model ============
    print("\n" + "=" * 60)
    print(f"  BEST MODEL: {best_model_info['name']} (R2={best_model_info['R2']:.4f})")
    print(f"  File: {best_model_info['file']}")
    print("=" * 60)

    # ============ Physics consistency ============
    print("\n" + "=" * 60)
    print("  Physics Consistency: Response Surfaces")
    print("=" * 60)
    run_response_surfaces(best_model_info, data)

    return df, best_model_info


def run_response_surfaces(best_info, data):
    scaler_mm, scaler_ss = load_scalers()
    model = best_info["model"]
    model_name = best_info["name"]

    is_nn = model_name in ("M4_MLP", "M5_TabNet", "M6_DeepEnsemble")
    model_scaler = scaler_mm if is_nn else scaler_ss

    feat_mins = scaler_mm.data_min_
    feat_maxs = scaler_mm.data_max_

    train_orig = scaler_mm.inverse_transform(data["X_train_mm"])

    variable_pairs = [
        ("speed_kmh", "wing_angle_deg"),
    ]
    n_grid = 80

    for v1, v2 in variable_pairs:
        i1 = FEATURE_COLS.index(v1)
        i2 = FEATURE_COLS.index(v2)
        i_drs = FEATURE_COLS.index("drs_active")

        # Pre-compute KNN lookup for downforce/drag given (speed, wing, drs)
        K = 10
        ref_xy = train_orig[:, [i1, i2, i_drs]]
        ref_rest = np.delete(train_orig, [i1, i2, i_drs], axis=1)

        g1 = np.linspace(feat_mins[i1], feat_maxs[i1], n_grid)
        g2 = np.linspace(feat_mins[i2], feat_maxs[i2], n_grid)
        A, B = np.meshgrid(g1, g2, indexing="ij")

        for drs in [0, 1]:
            feats_orig = np.zeros((n_grid * n_grid, 5), dtype=np.float32)
            feats_orig[:, i1] = A.ravel()
            feats_orig[:, i2] = B.ravel()
            feats_orig[:, i_drs] = drs

            # For each grid point, find K nearest neighbours in (v1, v2, drs) space
            test_xy = feats_orig[:, [i1, i2, i_drs]]
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=K)
            nn.fit(ref_xy)
            _, indices = nn.kneighbors(test_xy)

            # Use median of K neighbours for the other features
            other_cols = [j for j in range(5) if j not in (i1, i2, i_drs)]
            for idx, j in enumerate(other_cols):
                neighbour_vals = ref_rest[:, idx][indices]
                feats_orig[:, j] = np.median(neighbour_vals, axis=1)

            feats_scaled = model_scaler.transform(feats_orig)
            pred = model.predict(feats_scaled)
            if is_nn:
                pred = pred * 100.0
            Z = pred.reshape(n_grid, n_grid).T

            fig, ax = plt.subplots(figsize=(8, 6))
            c = ax.contourf(g1, g2, Z, levels=30, cmap="RdYlGn")
            plt.colorbar(c, ax=ax, label="Predicted stability_index")
            ax.set_xlabel(v1)
            ax.set_ylabel(v2)
            ax.set_title(f"Response Surface: {v1} × {v2} (DRS={drs}) — {model_name}")
            path = os.path.join(MODELS_FIGURES_DIR, f"response_surface_{v1}_{v2}_DRS{drs}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Response surface saved: {path}")


def plot_training_time_comparison(df, output_dir=None):
    if output_dir is None:
        output_dir = MODELS_FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    df_sorted = df.sort_values("Train_Time_s", ascending=True)
    models = df_sorted["Model"].tolist()
    times = df_sorted["Train_Time_s"].tolist()

    colors = []
    for t in times:
        if t < 1:
            colors.append("#2ca02c")
        elif t < 60:
            colors.append("#ff7f0e")
        else:
            colors.append("#d62728")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars1 = ax1.barh(models, times, color=colors, edgecolor="white")
    ax1.set_xlabel("Training Time (seconds)")
    ax1.set_title("Training Time Comparison")
    for bar, t in zip(bars1, times):
        label = f"{t:.1f}s" if t >= 1 else f"{t:.2f}s"
        ax1.text(bar.get_width() * 0.02, bar.get_y() + bar.get_height() / 2,
                 label, va="center", fontsize=9, color="white", fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    bars2 = ax2.barh(models, times, color=colors, edgecolor="white")
    ax2.set_xscale("log")
    ax2.set_xlabel("Training Time (s, log scale)")
    ax2.set_title("Training Time Comparison (Log Scale)")
    for bar, t in zip(bars2, times):
        label = f"{t:.1f}s" if t >= 1 else f"{t:.2f}s"
        ax2.text(bar.get_width() * 0.60, bar.get_y() + bar.get_height() / 2,
                 label, va="center", fontsize=9, color="white", fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", label="< 1s (ideal)"),
        Patch(facecolor="#ff7f0e", label="1-60s (moderate)"),
        Patch(facecolor="#d62728", label="> 60s (slow)"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.subplots_adjust(left=0.22, right=0.96, wspace=0.35)
    path = os.path.join(output_dir, "training_time_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Training time comparison saved: {path}")


def main():
    print("=" * 60)
    print("  T03: Surrogate Model Comparison")
    print("=" * 60)

    data = load_data()
    print(f"\n  Data loaded: train={data['X_train_mm'].shape[0]}, "
          f"valid={data['X_valid_mm'].shape[0]}, test={data['X_test_mm'].shape[0]}")
    print(f"  Sample weight range: [{data['sample_weight'].min():.4f}, {data['sample_weight'].max():.4f}]")

    df, best = run_all(data)

    print("\n" + "=" * 60)
    print("  T03 Complete!")
    print("=" * 60)

    return df, best


if __name__ == "__main__":
    main()
