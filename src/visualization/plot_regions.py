"""Visualization for High-Performance Region Discovery.

Generates diagnostic and presentation-quality plots for region analysis results.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.utils.config import HPR_REGIONS_FIGURES_DIR, FEATURE_COLS

os.makedirs(HPR_REGIONS_FIGURES_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(HPR_REGIONS_FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_candidate_distribution(fitnesses_all, gbest):
    """Histogram of all candidate fitnesses with percentile thresholds marked."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(fitnesses_all, bins=80, color="steelblue", alpha=0.7, edgecolor="white")
    for p, color in [(0.01, "red"), (0.05, "orange"), (0.10, "green")]:
        thresh = np.percentile(fitnesses_all, 100 * (1 - p))
        ax.axvline(thresh, color=color, linestyle="--", linewidth=1.5,
                   label=f"Top {int(p*100)}%: >= {thresh:.2f}")
    ax.axvline(gbest, color="black", linestyle="-", linewidth=1.5, label=f"gbest: {gbest:.2f}")
    ax.set_xlabel("Fitness (stability_index)")
    ax.set_ylabel("Count")
    ax.set_title("Candidate Solution Fitness Distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "candidate_fitness_distribution.png")


def plot_region_parameter_ranges(regions, feature_names=None):
    """Horizontal bar chart showing per-parameter P5-P95 ranges for each region."""
    if feature_names is None:
        feature_names = FEATURE_COLS

    fig, axes = plt.subplots(1, len(regions), figsize=(5 * len(regions), 6),
                             squeeze=False)
    for ax_idx, (label, region) in enumerate(regions.items()):
        ax = axes[0][ax_idx]
        stats = region["stats"]
        params = stats["parameter"].tolist()
        means = stats["mean"].values
        p5 = stats["p5"].values
        p95 = stats["p95"].values
        errors_low = means - p5
        errors_high = p95 - means

        y = range(len(params))
        ax.barh(y, means, color="steelblue", alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(params)
        ax.set_xlabel("Value")
        ax.set_title(f"{label} (n={region['positions'].shape[0]})")
        ax.grid(alpha=0.3, axis="x")

    fig.suptitle("High-Performance Parameter Ranges (mean with P5-P95 spread)", fontsize=13)
    fig.tight_layout()
    _save(fig, "region_parameter_ranges.png")


def plot_region_scatter_matrix(positions_top5, fitnesses_top5, feature_names=None):
    """Scatter matrix of Top 5% positions coloured by fitness."""
    if feature_names is None:
        feature_names = FEATURE_COLS

    n_feat = positions_top5.shape[1]
    labels = feature_names[:n_feat]
    n = len(labels)

    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    if n == 1:
        axes = np.array([[axes]])

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            if i == j:
                ax.hist(positions_top5[:, i], bins=30, color="steelblue", alpha=0.7)
                ax.set_xlabel(labels[i])
            elif i < j:
                ax.axis("off")
            else:
                sc = ax.scatter(positions_top5[:, j], positions_top5[:, i],
                                c=fitnesses_top5, s=2, alpha=0.5, cmap="RdYlGn")
                if i == n - 1:
                    ax.set_xlabel(labels[j])
                if j == 0:
                    ax.set_ylabel(labels[i])

    fig.suptitle(f"Top 5% Parameter Scatter Matrix (n={positions_top5.shape[0]})", fontsize=12)
    fig.tight_layout()
    _save(fig, "region_scatter_matrix.png")


def plot_cluster_visualization(positions, labels, centers_df, feature_names=None,
                               n_top_features=3):
    """2D projection of clusters using the most important 2 features."""
    if feature_names is None:
        feature_names = FEATURE_COLS

    unique = sorted(set(labels))
    n_clusters = len([u for u in unique if u >= 0])
    if n_clusters < 2:
        return

    # Use the 2 features with largest range across cluster centroids
    centroids = np.array([centers_df[centers_df["cluster"] == u]["centroid"].iloc[0]
                          for u in unique if u >= 0])
    if len(centroids) == 0:
        return
    ranges = centroids.max(axis=0) - centroids.min(axis=0)
    top_indices = np.argsort(ranges)[-2:]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for lbl in unique:
        mask = labels == lbl
        if lbl == -1:
            ax.scatter(positions[mask, top_indices[0]], positions[mask, top_indices[1]],
                       s=5, alpha=0.3, color="gray", label="Noise")
        else:
            ax.scatter(positions[mask, top_indices[0]], positions[mask, top_indices[1]],
                       s=15, alpha=0.6, color=colors[lbl % 10], label=f"Cluster {lbl}")

    # Mark cluster centres
    for lbl in unique:
        if lbl >= 0:
            cnt = centers_df[centers_df["cluster"] == lbl]
            cpos = cnt["centroid"].iloc[0]
            ax.scatter(cpos[top_indices[0]], cpos[top_indices[1]], s=200,
                       marker="*", color=colors[lbl % 10], edgecolors="black", linewidths=0.8)

    ax.set_xlabel(feature_names[top_indices[0]])
    ax.set_ylabel(feature_names[top_indices[1]])
    ax.set_title(f"Solution Clusters (n={len(positions)}, {n_clusters} clusters)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    _save(fig, "cluster_projection.png")


def plot_region_report(discovery_result, feature_names=None):
    """Generate all region discovery plots from a RegionDiscovery.run_from_trials() result."""
    if feature_names is None:
        feature_names = FEATURE_COLS

    fitnesses_all = discovery_result["fitnesses_all"]
    gbest = discovery_result["gbest"]
    regions = discovery_result["regions"]

    print("\n  --- Region Discovery Plots ---")

    # 1. Fitness distribution
    plot_candidate_distribution(fitnesses_all, gbest)

    # 2. Parameter range bars
    if len(regions) >= 2:
        plot_region_parameter_ranges(regions, feature_names)

    # 3. Top 5% scatter matrix
    top5 = regions.get("top_5pct")
    if top5 is not None and len(top5["positions"]) >= 50:
        plot_region_scatter_matrix(top5["positions"], top5["fitnesses"], feature_names)

    # 4. Cluster projection
    labels = discovery_result.get("cluster_labels")
    centers_df = discovery_result.get("cluster_centers")
    if labels is not None and len(labels) > 0:
        top5 = regions.get("top_5pct")
        if top5 is not None:
            plot_cluster_visualization(top5["positions"], labels, centers_df, feature_names)
