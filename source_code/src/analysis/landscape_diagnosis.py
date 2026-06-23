"""Landscape Diagnosis Module.

Tasks:
  A — Wing Sensitivity Analysis (∂Stability/∂Wing)
  B — Local Gradient Analysis (speed × wing stability surface)
  E — Dataset Structure Diagnosis (correlations, MI, scatter plots)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import FEATURE_COLS, RANDOM_STATE, PROCESSED_DIR, SCENARIOS_FIGURES_DIR

os.makedirs(SCENARIOS_FIGURES_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(SCENARIOS_FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# Task A: Wing Sensitivity Analysis
# ============================================================================


def wing_sensitivity_sweep(model, scaler, model_type="xgb", n_wing=16):
    """Sweep wing_angle at 27 fixed (speed, downforce, drag) triplets.

    Returns
    -------
    results : list of dict
        [{speed, df_val, drag_val, wing, stability, sigma}, ...]
    """
    wing_values = np.linspace(20, 35, n_wing)
    speeds = [200, 280, 345]
    df_vals = [2000, 4000, 6000]
    drag_vals = [100, 250, 400]
    drs = 1

    results = []
    for speed in speeds:
        for dfv in df_vals:
            for drag in drag_vals:
                for wing in wing_values:
                    X = np.array([[speed, wing, drs, dfv, drag]], dtype=np.float32)
                    if scaler is not None:
                        X_s = scaler.transform(X)
                    else:
                        X_s = X
                    pred = model.predict(X_s)
                    if hasattr(pred, "numpy"):
                        pred = pred.numpy()
                    stab = float(np.clip(np.asarray(pred).flat[0], 0, 100))
                    results.append({
                        "speed": speed, "df_val": dfv, "drag_val": drag,
                        "wing": round(wing, 1), "stability": round(stab, 4), "sigma": 0,
                    })

    return results


def plot_wing_sensitivity(results, output_name="diagnosis_wing_sensitivity.png"):
    """Plot wing → stability for all 27 fixed configs, colored by speed tier."""
    df = pd.DataFrame(results)
    speed_colors = {200: "#2ca02c", 280: "#ff7f0e", 345: "#d62728"}
    speed_labels = {200: "v=200 (low)", 280: "v=280 (mid)", 345: "v=345 (high)"}

    fig, ax = plt.subplots(figsize=(11, 6))

    for speed in [200, 280, 345]:
        sub = df[df["speed"] == speed]
        grouped = sub.groupby(["df_val", "drag_val"])
        for (dfv, drag), grp in grouped:
            grp_sorted = grp.sort_values("wing")
            ax.plot(grp_sorted["wing"], grp_sorted["stability"],
                    color=speed_colors[speed], alpha=0.35, linewidth=0.8)

    # Highlight mean per speed
    for speed in [200, 280, 345]:
        sub = df[df["speed"] == speed]
        mean_by_wing = sub.groupby("wing")["stability"].mean()
        ax.plot(mean_by_wing.index, mean_by_wing.values,
                color=speed_colors[speed], linewidth=2.5, label=speed_labels[speed])

    ax.set_xlabel("Wing Angle (°)", fontsize=12)
    ax.set_ylabel("Predicted Stability", fontsize=12)
    ax.set_title("Task A: Wing Sensitivity Analysis\n"
                 "(27 configs: 3 speed × 3 downforce × 3 drag, dashed = per-speed mean)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, output_name)


def compute_sensitivity_stats(results):
    """Return per-speed sensitivity stats: max delta stability across wing range."""
    df = pd.DataFrame(results)
    stats = []
    for speed in [200, 280, 345]:
        sub = df[df["speed"] == speed]
        by_wing = sub.groupby("wing")["stability"].mean()
        delta = float(by_wing.max() - by_wing.min())
        slope = float(np.polyfit(by_wing.index, by_wing.values, 1)[0])
        stats.append({"speed": speed, "delta": delta, "slope": slope,
                      "max_stability": float(by_wing.max()),
                      "min_stability": float(by_wing.min())})
    return stats


# ============================================================================
# Task B: Local Gradient Analysis
# ============================================================================


def local_gradient_heatmap(model, scaler, n_speed=25, n_wing=16):
    """2D grid scan: speed × wing, return stability matrix and gradient magnitude."""
    speed_range = np.linspace(100, 350, n_speed)
    wing_range = np.linspace(20, 35, n_wing)
    drs, downforce, drag = 1, 4000, 200

    Z = np.zeros((n_wing, n_speed))
    for i, wing in enumerate(wing_range):
        for j, speed in enumerate(speed_range):
            X = np.array([[speed, wing, drs, downforce, drag]], dtype=np.float32)
            X_s = scaler.transform(X) if scaler else X
            pred = model.predict(X_s)
            if hasattr(pred, "numpy"):
                pred = pred.numpy()
            Z[i, j] = float(np.clip(np.asarray(pred).flat[0], 0, 100))

    # Wing-direction gradient (row-wise difference)
    grad = np.abs(np.gradient(Z, wing_range, axis=0))

    return Z, grad, speed_range, wing_range


def plot_gradient_analysis(Z, grad, speed_range, wing_range):
    """Plot stability surface + gradient magnitude."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: stability surface
    im1 = ax1.imshow(Z, aspect="auto", origin="lower",
                     extent=[speed_range[0], speed_range[-1], wing_range[0], wing_range[-1]],
                     cmap="YlOrRd")
    ax1.set_xlabel("Speed (km/h)", fontsize=11)
    ax1.set_ylabel("Wing Angle (°)", fontsize=11)
    ax1.set_title("Predicted Stability Surface\n(DRS=1, df=4000N, drag=200N)",
                  fontsize=11, fontweight="bold")
    plt.colorbar(im1, ax=ax1, shrink=0.75).set_label("Stability", fontsize=9)

    # Right: gradient magnitude
    im2 = ax2.imshow(grad, aspect="auto", origin="lower",
                     extent=[speed_range[0], speed_range[-1], wing_range[0], wing_range[-1]],
                     cmap="YlOrBr")
    ax2.set_xlabel("Speed (km/h)", fontsize=11)
    ax2.set_ylabel("Wing Angle (°)", fontsize=11)
    ax2.set_title("|d(Stability)/d(Wing)| Gradient\n(higher = wing has more impact)",
                  fontsize=11, fontweight="bold")
    plt.colorbar(im2, ax=ax2, shrink=0.75).set_label("Gradient Magnitude", fontsize=9)

    fig.suptitle("Task B: Speed × Wing Sensitivity Heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "diagnosis_speed_wing_heatmap.png")

    return float(grad.mean()), float(grad.max())


# ============================================================================
# Task E: Dataset Structure Diagnosis
# ============================================================================


def dataset_structure_diagnosis(train_path=None):
    """Analyse feature inter-relationships and wing's role in the dataset.

    Returns
    -------
    dict with keys: corr_pearson, corr_spearman, mutual_info, partial_corr, wing_speed_interaction
    """
    if train_path is None:
        train_path = os.path.join(PROCESSED_DIR, "train.csv")

    train = pd.read_csv(train_path)
    feat_cols = ["speed_kmh", "wing_angle_deg", "drs_active", "downforce_n", "drag_n"]
    stability = train["stability_index"].values
    X = train[feat_cols].values

    # Correlations
    corr_p = train[feat_cols + ["stability_index"]].corr(method="pearson")
    corr_s = train[feat_cols + ["stability_index"]].corr(method="spearman")

    # Mutual Information
    from sklearn.feature_selection import mutual_info_regression
    mi = mutual_info_regression(X, stability, random_state=RANDOM_STATE)
    mi_dict = {feat_cols[i]: float(mi[i]) for i in range(len(feat_cols))}

    # Partial correlation: wing vs stability, controlling for downforce
    from scipy.stats import pearsonr
    from sklearn.linear_model import LinearRegression
    ctrl = train[["downforce_n"]].values
    lr = LinearRegression()
    wing_resid = train["wing_angle_deg"].values - lr.fit(ctrl, train["wing_angle_deg"].values).predict(ctrl)
    stab_resid = stability - lr.fit(ctrl, stability).predict(ctrl)
    partial_r, partial_p = pearsonr(wing_resid, stab_resid)

    # Wing × Speed interaction on stability
    # Bin speed, then compute wing-stability correlation per bin
    speed_bins = [(100, 180), (180, 260), (260, 350)]
    interaction_stats = []
    for lo, hi in speed_bins:
        mask = (train["speed_kmh"] >= lo) & (train["speed_kmh"] < hi)
        sub = train[mask]
        if len(sub) > 10:
            r, p = pearsonr(sub["wing_angle_deg"], sub["stability_index"])
            interaction_stats.append({
                "speed_bin": f"{lo}-{hi}", "n": len(sub),
                "r_wing_stability": round(r, 4), "p_value": round(p, 6),
            })

    # Scatter plots
    plot_dataset_scatters(train)

    return {
        "corr_pearson": corr_p,
        "corr_spearman": corr_s,
        "mutual_info": mi_dict,
        "partial_corr_wing_stability_controlled_downforce": float(partial_r),
        "wing_speed_interaction": interaction_stats,
    }


def plot_dataset_scatters(train):
    """Generate wing×downforce, wing×drag, wing×stability scatter + MI heatmap."""
    feat_cols = ["speed_kmh", "wing_angle_deg", "drs_active", "downforce_n", "drag_n"]
    stability = train["stability_index"].values
    X = train[feat_cols].values

    # --- Scatter: Wing vs Downforce, colored by stability ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    sub = train.sample(min(5000, len(train)), random_state=RANDOM_STATE)

    sc = axes[0].scatter(sub["wing_angle_deg"], sub["downforce_n"],
                         c=sub["stability_index"], cmap="YlOrRd", s=3, alpha=0.4, edgecolors="none")
    axes[0].set_xlabel("Wing Angle (°)"); axes[0].set_ylabel("Downforce (N)")
    axes[0].set_title("Wing vs Downforce\n(colored by stability)", fontweight="bold")
    plt.colorbar(sc, ax=axes[0], shrink=0.7).set_label("Stability", fontsize=8)

    sc2 = axes[1].scatter(sub["wing_angle_deg"], sub["drag_n"],
                          c=sub["stability_index"], cmap="YlOrRd", s=3, alpha=0.4, edgecolors="none")
    axes[1].set_xlabel("Wing Angle (°)"); axes[1].set_ylabel("Drag (N)")
    axes[1].set_title("Wing vs Drag\n(colored by stability)", fontweight="bold")
    plt.colorbar(sc2, ax=axes[1], shrink=0.7).set_label("Stability", fontsize=8)

    sc3 = axes[2].scatter(sub["wing_angle_deg"], sub["stability_index"],
                          c=sub["speed_kmh"], cmap="viridis", s=3, alpha=0.4, edgecolors="none")
    axes[2].set_xlabel("Wing Angle (°)"); axes[2].set_ylabel("Stability Index")
    axes[2].set_title("Wing vs Stability\n(colored by speed)", fontweight="bold")
    plt.colorbar(sc3, ax=axes[2], shrink=0.7).set_label("Speed (km/h)", fontsize=8)

    fig.suptitle("Task E: Dataset Structure Diagnosis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "diagnosis_dataset_scatters.png")

    # --- MI Heatmap ---
    from sklearn.feature_selection import mutual_info_regression
    all_feat = feat_cols + ["stability_index"]
    mi_matrix = np.zeros((len(all_feat), len(all_feat)))
    for i, f1 in enumerate(all_feat):
        for j, f2 in enumerate(all_feat):
            if i == j:
                mi_matrix[i, j] = 0
            else:
                mi_matrix[i, j] = mutual_info_regression(
                    train[[f1]].values, train[f2].values, random_state=RANDOM_STATE
                )[0]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mi_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(all_feat))); ax.set_xticklabels(all_feat, rotation=45, ha="right")
    ax.set_yticks(range(len(all_feat))); ax.set_yticklabels(all_feat)
    ax.set_title("Mutual Information Matrix", fontsize=12, fontweight="bold")
    for i in range(len(all_feat)):
        for j in range(len(all_feat)):
            ax.text(j, i, f"{mi_matrix[i,j]:.3f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8).set_label("Mutual Information", fontsize=9)
    fig.tight_layout()
    _save(fig, "diagnosis_mi_heatmap.png")


# ============================================================================
# Final Figure: Speed × Wing Contour + PSO Top-5% Overlay
# ============================================================================


def plot_speed_wing_contour_with_pso(model, scaler, pso_top5_positions=None):
    """Generate the conclusive contour plot.

    Background: speed×wing stability contour map (full landscape)
    Overlay: PSO top-5% solution positions (where PSO actually went)

    This is the single most informative figure — it simultaneously shows:
      1. The model DOES perceive wing (contours descend in bottom-right)
      2. PSO correctly avoids low-stability regions
      3. In the high-stability basin (top-left), wing has no effect
    """
    n_speed = 50
    n_wing = 30
    speed_range = np.linspace(100, 350, n_speed)
    wing_range = np.linspace(20, 35, n_wing)
    drs, downforce, drag = 1, 4000, 200

    Z = np.zeros((n_wing, n_speed))
    for i, wing in enumerate(wing_range):
        for j, speed in enumerate(speed_range):
            X = np.array([[speed, wing, drs, downforce, drag]], dtype=np.float32)
            X_s = scaler.transform(X) if scaler else X
            pred = model.predict(X_s)
            if hasattr(pred, "numpy"):
                pred = pred.numpy()
            Z[i, j] = float(np.clip(np.asarray(pred).flat[0], 0, 100))

    fig, ax = plt.subplots(figsize=(12, 7))

    # Contour fill
    S, W = np.meshgrid(speed_range, wing_range)
    levels = np.linspace(50, 95, 19)
    cf = ax.contourf(S, W, Z, levels=levels, cmap="YlOrRd", alpha=0.85)
    cs = ax.contour(S, W, Z, levels=levels, colors="gray", linewidths=0.4, alpha=0.5)

    # Label key contours
    label_levels = [55, 65, 75, 85, 95]
    ax.clabel(cs, levels=label_levels, inline=True, fontsize=8, fmt="%.0f")

    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Predicted Stability (0-100)", fontsize=10)

    # Overlay PSO top-5% positions
    if pso_top5_positions is not None and len(pso_top5_positions) > 0:
        pos = np.asarray(pso_top5_positions)
        # Clip to plot range
        speed_clip = np.clip(pos[:, 0], speed_range[0], speed_range[-1])
        wing_clip = np.clip(pos[:, 1], wing_range[0], wing_range[-1])
        ax.scatter(speed_clip, wing_clip, s=12, c="darkred", marker="o",
                   edgecolors="white", linewidth=0.3, alpha=0.6, zorder=5,
                   label=f"PSO Top-5% solutions (n={len(pos)})")

        # Draw convex hull or outline around PSO cluster
        from scipy.spatial import ConvexHull
        try:
            hull_points = np.column_stack([speed_clip, wing_clip])
            hull = ConvexHull(hull_points)
            hull_xy = hull_points[hull.vertices]
            ax.fill(hull_xy[:, 0], hull_xy[:, 1], alpha=0.12, color="red", zorder=4)
            ax.plot(np.append(hull_xy[:, 0], hull_xy[0, 0]),
                    np.append(hull_xy[:, 1], hull_xy[0, 1]),
                    "--", color="darkred", linewidth=1.2, alpha=0.7, zorder=4)
        except Exception:
            pass

    # Annotations
    ax.set_xlabel("Speed (km/h)", fontsize=12)
    ax.set_ylabel("Wing Angle (°)", fontsize=12)
    ax.set_title("Speed × Wing Stability Landscape\n"
                 "Background: full stability contour  |  "
                 "Red dots: where PSO actually searched (Top-5%)",
                 fontsize=13, fontweight="bold")

    # Key annotation: PSO basin
    ax.annotate(
        "High-stability basin\n(PSO converges here)\nwing has ~0 effect",
        xy=(150, 22), fontsize=9, color="darkgreen",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5),
    )
    ax.annotate(
        "Low-stability region\n(high speed + high wing)\nmodel DOES perceive wing here",
        xy=(320, 33), fontsize=9, color="darkred",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5),
    )

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.15)

    fig.tight_layout()
    _save(fig, "diagnosis_speed_wing_pso_overlay.png")

    return Z, speed_range, wing_range
