"""High-Performance Region Discovery.

Post-processes PSO candidate solutions to discover:
  - High-performance parameter regions (Top 1%, 5%, 10%)
  - Multi-modal solution clusters (K-Means, DBSCAN)
  - Engineering interpretable parameter ranges

Design philosophy: NOT a PSO algorithm improvement, but "Optimization Result
Analysis / Design Space Exploration" — maps where in parameter space good
solutions live, not just what the single best one is.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

from src.utils.config import (
    FEATURE_COLS, HPR_ABSOLUTE_THRESHOLD, HPR_PERCENTILES,
    HPR_DBSCAN_EPS, HPR_DBSCAN_MIN_SAMPLES,
)


def collect_from_result(result):
    """Flatten PSO candidate list-of-batches into (N, dim) positions and (N,) fitnesses.

    Parameters
    ----------
    result : dict
        PSO result dict with "candidates" key: list of (pos_array, fit_array) tuples.

    Returns
    -------
    positions : np.ndarray shape (N, dim)
    fitnesses : np.ndarray shape (N,)
    """
    batches = result.get("candidates", [])
    if not batches:
        raise ValueError("result has no 'candidates' key. Run PSO with collect_candidates=True.")
    pos_list = [b[0] for b in batches]
    fit_list = [b[1] for b in batches]
    return np.vstack(pos_list), np.concatenate(fit_list)


def collect_from_trials(results):
    """Merge candidates from multiple independent PSO trials.

    Parameters
    ----------
    results : list of dict
        List of PSO result dicts, each containing "candidates".

    Returns
    -------
    positions : np.ndarray
    fitnesses : np.ndarray
    """
    all_pos, all_fit = [], []
    for r in results:
        if "candidates" not in r:
            continue
        pos, fit = collect_from_result(r)
        all_pos.append(pos)
        all_fit.append(fit)
    return np.vstack(all_pos), np.concatenate(all_fit)


def filter_by_percentile(positions, fitnesses, top_p=0.05):
    """Return positions and fitnesses in the top-P% by fitness."""
    threshold = np.percentile(fitnesses, 100 * (1 - top_p))
    mask = fitnesses >= threshold
    return positions[mask], fitnesses[mask], threshold


def filter_by_absolute(positions, fitnesses, threshold=HPR_ABSOLUTE_THRESHOLD):
    """Return positions where fitness >= absolute threshold."""
    mask = fitnesses >= threshold
    return positions[mask], fitnesses[mask], threshold


def filter_by_fraction_of_gbest(positions, fitnesses, gbest, fraction=0.99):
    """Return positions where fitness >= fraction * gbest_fitness."""
    threshold = fraction * gbest
    mask = fitnesses >= threshold
    return positions[mask], fitnesses[mask], threshold


def compute_region_stats(positions, feature_names=None):
    """Compute per-parameter statistics for a set of positions.

    Returns DataFrame with columns: parameter, mean, std, min, max, p5, p50, p95, range
    """
    if feature_names is None:
        feature_names = FEATURE_COLS
    pos = np.asarray(positions)
    rows = []
    for d in range(pos.shape[1]):
        col = pos[:, d]
        rows.append({
            "parameter": feature_names[d],
            "mean": np.mean(col),
            "std": np.std(col),
            "min": np.min(col),
            "max": np.max(col),
            "p5": np.percentile(col, 5),
            "p50": np.percentile(col, 50),
            "p95": np.percentile(col, 95),
        })
    df = pd.DataFrame(rows)
    df["range_p5_p95"] = df["p95"] - df["p5"]
    df["range_description"] = df.apply(
        lambda r: f"[{r['p5']:.1f}, {r['p95']:.1f}]", axis=1
    )
    return df


def cluster_solutions(positions, fitnesses, method="dbscan", n_clusters=None,
                      eps=None, min_samples=HPR_DBSCAN_MIN_SAMPLES,
                      random_state=42):
    """Cluster high-performance positions.

    Parameters
    ----------
    positions : np.ndarray shape (N, dim)
    fitnesses : np.ndarray shape (N,), used for sorting clusters by quality
    method : str
        "kmeans" or "dbscan"
    n_clusters : int or None
        For kmeans. None = auto-detect via silhouette score (2..10).
    eps : float or None
        For DBSCAN. None = auto-detect via k-distance graph (90th percentile).
    min_samples : int
        For DBSCAN.
    random_state : int

    Returns
    -------
    labels : np.ndarray shape (N,) — cluster labels (-1 = noise for DBSCAN)
    centers_df : pd.DataFrame — per-cluster centroid statistics
    """
    scaler = StandardScaler()
    pos_scaled = scaler.fit_transform(positions)

    if method == "kmeans":
        if n_clusters is None:
            scores = []
            for k in range(2, min(11, len(positions) // 5)):
                km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                labels = km.fit_predict(pos_scaled)
                if len(set(labels)) > 1:
                    scores.append((k, silhouette_score(pos_scaled, labels)))
            n_clusters = max(scores, key=lambda x: x[1])[0] if scores else 2

        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = km.fit_predict(pos_scaled)

    elif method == "dbscan":
        if eps is None:
            nn = NearestNeighbors(n_neighbors=min(2 * min_samples, len(positions)))
            nn.fit(pos_scaled)
            dists = np.sort(nn.kneighbors(pos_scaled)[0][:, -1])
            eps = np.percentile(dists, 90)

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(pos_scaled)

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    centers_df = _build_cluster_centers(positions, fitnesses, labels)
    return labels, centers_df


def _build_cluster_centers(positions, fitnesses, labels):
    """Compute per-cluster centroid and statistics."""
    unique_labels = sorted(set(labels))
    rows = []
    for lbl in unique_labels:
        mask = labels == lbl
        cluster_pos = positions[mask]
        cluster_fit = fitnesses[mask]
        centroid = cluster_pos.mean(axis=0)
        rows.append({
            "cluster": lbl,
            "n_samples": mask.sum(),
            "mean_fitness": cluster_fit.mean(),
            "std_fitness": cluster_fit.std(),
            "centroid": centroid,
        })
    df = pd.DataFrame(rows)
    return df.sort_values("mean_fitness", ascending=False)


def describe_clusters(centers_df, feature_names=None):
    """Generate human-readable descriptions of each cluster's parameter profile.

    Returns list of dicts with "cluster", "n_samples", "mean_fitness", "description".
    """
    if feature_names is None:
        feature_names = FEATURE_COLS

    descriptions = []
    for _, row in centers_df.iterrows():
        centroid = row["centroid"]
        desc = {
            "cluster": int(row["cluster"]),
            "n_samples": int(row["n_samples"]),
            "mean_fitness": f"{row['mean_fitness']:.2f}",
        }
        parts = []
        for d, name in enumerate(feature_names):
            parts.append(f"{name}={centroid[d]:.2f}")
        desc["description"] = ", ".join(parts)
        descriptions.append(desc)
    return descriptions


class RegionDiscovery:
    """High-level interface for region discovery from a PSO result or multiple trials."""

    def __init__(self, feature_names=None):
        self.feature_names = feature_names or FEATURE_COLS

    def run_from_trials(self, trial_results, gbest=None):
        """Full pipeline: collect -> filter -> stats -> cluster.

        Parameters
        ----------
        trial_results : list of dict
            List of PSO result dicts with "candidates".
        gbest : float or None
            Overall best fitness across trials. If None, computed from candidates.

        Returns
        -------
        dict with keys:
            positions_all, fitnesses_all, gbest
            regions : dict {percentile_label: {positions, fitnesses, threshold, stats_df}}
            cluster_labels, cluster_centers_df
            descriptions
        """
        positions_all, fitnesses_all = collect_from_trials(trial_results)
        if gbest is None:
            gbest = fitnesses_all.max()

        regions = {}
        for p in HPR_PERCENTILES:
            label = f"top_{int(p*100):d}pct"
            pos, fit, thresh = filter_by_percentile(positions_all, fitnesses_all, top_p=p)
            stats = compute_region_stats(pos, self.feature_names)
            regions[label] = {
                "positions": pos, "fitnesses": fit,
                "threshold": thresh, "stats": stats,
            }
            if len(pos) < 5:
                # Not enough samples for deeper analysis; skip
                continue

        # Cluster the Top 5% region
        top5 = regions.get("top_5pct")
        if top5 is not None and len(top5["positions"]) >= 10:
            labels, centers_df = cluster_solutions(
                top5["positions"], top5["fitnesses"],
                method="dbscan", min_samples=HPR_DBSCAN_MIN_SAMPLES,
            )
            descriptions = describe_clusters(centers_df, self.feature_names)
        else:
            labels = np.array([])
            centers_df = pd.DataFrame()
            descriptions = []

        return {
            "positions_all": positions_all,
            "fitnesses_all": fitnesses_all,
            "gbest": gbest,
            "regions": regions,
            "cluster_labels": labels,
            "cluster_centers": centers_df,
            "cluster_descriptions": descriptions,
        }
