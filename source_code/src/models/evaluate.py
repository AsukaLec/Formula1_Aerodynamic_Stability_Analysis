import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.metrics import compute_metrics, compute_regional_metrics, measure_inference_latency
from src.utils.config import MODELS_COMPARISON_CSV, MODELS_FIGURES_DIR, INFERENCE_WARMUP, INFERENCE_REPEATS, FEATURE_COLS


def _make_dir(path):
    os.makedirs(os.path.dirname(path) if "." in os.path.basename(path) else path, exist_ok=True)


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name=None,
                   y_scaler=None, sample_weight=None):
    if model_name is None:
        model_name = getattr(model, "name", "Unknown")

    t0 = time.time()
    if hasattr(model, "train"):
        train_kwargs = {"X": X_train, "y": y_train}
        if sample_weight is not None:
            train_kwargs["sample_weight"] = sample_weight
        model.train(**train_kwargs)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)

    if y_scaler is not None:
        y_pred = y_scaler(y_pred)

    metrics = compute_metrics(y_test, y_pred)
    regional = compute_regional_metrics(y_test, y_pred, threshold=95.0)

    X_sample = X_test[:1] if X_test.ndim == 2 else X_test[:1].reshape(1, -1)
    try:
        latency = measure_inference_latency(model, X_sample, INFERENCE_WARMUP, INFERENCE_REPEATS)
        latency_ms = latency["latency_ms"]
    except Exception:
        latency_ms = float("nan")

    result = {
        "Model": model_name,
        "R2": round(metrics["R2"], 4),
        "MSE": round(metrics["MSE"], 4),
        "MAE": round(metrics["MAE"], 4),
        "RMSE": round(metrics["RMSE"], 4),
        "R2_low": round(regional["R2_low"], 4),
        "MSE_low": round(regional["MSE_low"], 4),
        "R2_high": round(regional["R2_high"], 4),
        "MSE_high": round(regional["MSE_high"], 4),
        "Latency_ms": round(latency_ms, 3),
        "Train_Time_s": round(train_time, 2),
    }
    return result, y_pred


def compare_models(results, save_csv=True, plot_bar=True):
    df = pd.DataFrame(results)
    if save_csv:
        _make_dir(MODELS_COMPARISON_CSV)
        df.to_csv(MODELS_COMPARISON_CSV, index=False)
        print(f"\n  Model comparison saved: {MODELS_COMPARISON_CSV}")

    if plot_bar:
        plot_model_comparison(df)

    print("\n" + df.to_string(index=False))
    return df


def plot_model_comparison(df):
    _make_dir(MODELS_FIGURES_DIR)
    import numpy as np

    panels = [
        (["R2", "R2_low"], "R2 Summary", False, "model_comparison_r2.png"),
        (["MSE", "MAE", "RMSE"], "Error Metrics (log scale)", True, "model_comparison_error.png"),
        (["Latency_ms"], "Inference Latency (ms, log scale)", True, "model_comparison_latency.png"),
    ]
    eps = 1e-4

    for cols, title, use_log, filename in panels:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(df))
        width = 0.8 / len(cols)

        for j, col in enumerate(cols):
            values = df[col].values.astype(float)
            ax.bar(x + j * width, values, width, label=col, zorder=2)

        if use_log:
            ax.set_yscale("log")
            vals = df[cols].values.flatten()
            vals = vals[np.isfinite(vals) & (vals > 0)]
            lo = max(vals.min() * 0.5, eps) if len(vals) > 0 else eps
            hi = vals.max() * 2 if len(vals) > 0 else 1
            ax.set_ylim(lo, hi)

        for container in ax.containers:
            for bar_patch in container:
                val = bar_patch.get_height()
                if not np.isfinite(val) or val <= 0:
                    continue
                label = f"{val:.3g}" if use_log else (f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}")
                y_pos = val * 1.25 if use_log else val + 0.02 * ax.get_ylim()[1]
                ax.text(bar_patch.get_x() + bar_patch.get_width() / 2, y_pos,
                        label, ha="center", va="bottom", fontsize=8)

        ax.set_title(title, fontsize=13)
        ax.set_xticks(x + width * (len(cols) - 1) / 2)
        ax.set_xticklabels(df["Model"], rotation=30, ha="right")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, zorder=0)

        fig.tight_layout()
        path = os.path.join(MODELS_FIGURES_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Comparison chart saved: {path}")


def plot_predicted_vs_actual(y_test, y_pred_dict, output_dir=None):
    if output_dir is None:
        output_dir = MODELS_FIGURES_DIR
    _make_dir(output_dir)

    n = len(y_pred_dict)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)

    for idx, (name, y_pred) in enumerate(y_pred_dict.items()):
        ax = axes[idx // cols][idx % cols]
        ax.scatter(y_test, y_pred, s=1, alpha=0.3, color="steelblue")
        lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Actual stability_index")
        ax.set_ylabel("Predicted stability_index")
        ax.set_title(name)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    for idx in range(n, rows * cols):
        ax = axes[idx // cols][idx % cols]
        ax.axis("off")

    fig.tight_layout()
    path = os.path.join(output_dir, "predicted_vs_actual.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Predicted vs actual chart saved: {path}")


def plot_predicted_vs_actual_stratified(
    y_test, y_pred_dict, output_dir=None,
    bins=None, labels=None, n_per_bin=500,
):
    from src.utils.config import STABILITY_BINS, STABILITY_LABELS

    if bins is None:
        bins = STABILITY_BINS
    if labels is None:
        labels = STABILITY_LABELS
    if output_dir is None:
        output_dir = MODELS_FIGURES_DIR
    _make_dir(output_dir)

    y_test = np.asarray(y_test)
    bin_idx = np.digitize(y_test, bins[1:-1])

    bin_colors = ["#d62728", "#ff7f0e", "#1f77b4", "#7f7f7f"]

    n_models = len(y_pred_dict)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5 * rows), squeeze=False)

    for idx, (name, y_pred) in enumerate(y_pred_dict.items()):
        ax = axes[idx // cols][idx % cols]
        y_pred = np.asarray(y_pred)

        all_indices = []
        for k in range(len(labels)):
            mask = bin_idx == k
            bin_indices = np.where(mask)[0]
            n_take = min(n_per_bin, len(bin_indices))
            if n_take > 0:
                rng = np.random.RandomState(42)
                sampled = rng.choice(bin_indices, size=n_take, replace=False)
                all_indices.append(sampled)
        all_indices = np.concatenate(all_indices)

        for k in range(len(labels)):
            mask_in_sample = bin_idx[all_indices] == k
            idx_k = all_indices[mask_in_sample]
            if len(idx_k) == 0:
                continue
            ax.scatter(y_test[idx_k], y_pred[idx_k], s=8, alpha=0.7,
                       color=bin_colors[k], label=labels[k], edgecolors="none")

        lo, hi = 0, 100
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Actual stability_index")
        ax.set_ylabel("Predicted stability_index")
        ax.set_title(name)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left", markerscale=1.2)

    for idx in range(n_models, rows * cols):
        ax = axes[idx // cols][idx % cols]
        ax.axis("off")

    fig.tight_layout()
    path = os.path.join(output_dir, "predicted_vs_actual_stratified.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Stratified predicted vs actual chart saved: {path}")


def cross_validate(model, X, y, n_folds=5, y_scaler=None, sample_weight=None):
    """K-Fold cross-validation for model evaluation.
    
    Returns dict with mean and std of R2, MSE, MAE across folds.
    """
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    r2_scores, mse_scores, mae_scores = [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        sw = sample_weight[train_idx] if sample_weight is not None else None
        
        if hasattr(model, "train"):
            train_kwargs = {"X": X_tr, "y": y_tr}
            if sw is not None:
                train_kwargs["sample_weight"] = sw
            model.train(**train_kwargs)
        else:
            model.fit(X_tr, y_tr, sample_weight=sw)
        
        y_pred = model.predict(X_val)
        if y_scaler is not None:
            y_pred = y_scaler(y_pred)
        
        from src.utils.metrics import compute_metrics
        m = compute_metrics(y_val, y_pred)
        r2_scores.append(m["R2"])
        mse_scores.append(m["MSE"])
        mae_scores.append(m["MAE"])
    
    return {
        "R2_mean": np.mean(r2_scores),
        "R2_std": np.std(r2_scores),
        "R2_scores": r2_scores,
        "MSE_mean": np.mean(mse_scores),
        "MSE_std": np.std(mse_scores),
        "MAE_mean": np.mean(mae_scores),
        "MAE_std": np.std(mae_scores),
    }


def leave_region_out_validate(model, X, y, n_regions=5, y_scaler=None, sample_weight=None):
    """Leave-region-out cross-validation based on speed x wing grid.
    
    Splits sample space into n_regions x n_regions grid based on first two
    features (speed_kmh, wing_angle_deg). Each region is held out in turn.
    
    Returns dict with mean/std R2 and per-region scores.
    """
    r2_scores, mse_scores = [], []
    region_labels = []
    
    speed = X[:, 0]
    wing = X[:, 1]
    s_bins = np.linspace(speed.min(), speed.max(), n_regions + 1)
    w_bins = np.linspace(wing.min(), wing.max(), n_regions + 1)
    
    s_idx = np.digitize(speed, s_bins[1:-1])
    w_idx = np.digitize(wing, w_bins[1:-1])
    region_id = s_idx * n_regions + w_idx
    
    unique_regions = np.unique(region_id)
    
    for region in unique_regions:
        test_mask = region_id == region
        train_mask = ~test_mask
        
        if test_mask.sum() < 10:
            continue
        
        X_tr, X_val = X[train_mask], X[test_mask]
        y_tr, y_val = y[train_mask], y[test_mask]
        
        sw = sample_weight[train_mask] if sample_weight is not None else None
        
        if hasattr(model, "train"):
            train_kwargs = {"X": X_tr, "y": y_tr}
            if sw is not None:
                train_kwargs["sample_weight"] = sw
            model.train(**train_kwargs)
        else:
            model.fit(X_tr, y_tr, sample_weight=sw)
        
        y_pred = model.predict(X_val)
        if y_scaler is not None:
            y_pred = y_scaler(y_pred)
        
        from src.utils.metrics import compute_metrics
        m = compute_metrics(y_val, y_pred)
        r2_scores.append(m["R2"])
        mse_scores.append(m["MSE"])
        
        s_lo = s_bins[s_idx[test_mask][0]]
        s_hi = s_bins[s_idx[test_mask][0] + 1]
        w_lo = w_bins[w_idx[test_mask][0]]
        w_hi = w_bins[w_idx[test_mask][0] + 1]
        region_labels.append(f"v{s_lo:.0f}-{s_hi:.0f}_a{w_lo:.0f}-{w_hi:.0f}")
    
    return {
        "R2_mean": np.mean(r2_scores),
        "R2_std": np.std(r2_scores),
        "R2_scores": r2_scores,
        "MSE_mean": np.mean(mse_scores),
        "MSE_std": np.std(mse_scores),
        "n_regions": len(r2_scores),
        "region_labels": region_labels,
    }


def plot_training_curves(trainers_dict, output_dir=None):
    if output_dir is None:
        output_dir = MODELS_FIGURES_DIR
    _make_dir(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, trainer_list in trainers_dict.items():
        if not trainer_list:
            continue
        for i, trainer in enumerate(trainer_list):
            label = f"{name}_{i}" if len(trainer_list) > 1 else name
            if trainer._train_losses:
                axes[0].plot(trainer._train_losses, label=label, alpha=0.7)
            if trainer._val_losses:
                axes[1].plot(trainer._val_losses, label=label, alpha=0.7)

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "training_loss_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training loss curves saved: {path}")
