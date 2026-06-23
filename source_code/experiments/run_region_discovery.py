#!/usr/bin/env python
"""
High-Performance Region Discovery Experiment
=============================================
Runs N independent PSO trials, collects all candidate solutions,
discovers high-performance regions and solution clusters.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import (
    RANDOM_STATE, FEATURE_COLS, DEEP_ENSEMBLE_M,
    HPR_N_TRIALS, HPR_PERCENTILES, HPR_ABSOLUTE_THRESHOLD,
    MLP_HIDDEN_UNITS, NN_LR, NN_WEIGHT_DECAY,
    NN_BATCH_SIZE, NN_MAX_EPOCHS, NN_EARLY_STOP_PATIENCE,
)
from src.optimization.fitness import load_xgb_fitness, load_ensemble_fitness
from src.optimization.pso_adaptive import PSOAdaptive
from src.optimization.pso_risk_sensitive import DEFAULT_BOUNDS, DISCRETE_INDICES
from src.analysis.region_discovery import (
    collect_from_trials, filter_by_percentile, filter_by_absolute,
    compute_region_stats, cluster_solutions, describe_clusters,
    RegionDiscovery,
)
from src.visualization.plot_regions import plot_region_report


def run_trials(n_trials=HPR_N_TRIALS, max_iter=60, seed=RANDOM_STATE,
               model="xgb", verbose=False):
    """Run multiple independent PSO trials and return all results."""
    if model == "xgb":
        fitness = load_xgb_fitness()
    else:
        fitness = load_ensemble_fitness(lambda_risk=1.0)

    results = []
    for t in range(n_trials):
        trial_seed = seed + t * 100
        pso = PSOAdaptive(
            n_particles=50, bounds=DEFAULT_BOUNDS,
            discrete_indices=DISCRETE_INDICES,
            w_start=0.9, w_end=0.4, alpha=1.0,
            max_iter=max_iter, seed=trial_seed, maximise=True,
        )
        result = pso.optimize(fitness, verbose=verbose, collect_candidates=True)
        results.append(result)

    return results


def print_cluster_table(descriptions):
    """Pretty-print cluster descriptions."""
    print()
    print(f"  {'Cluster':>8s}  {'Samples':>8s}  {'Fitness':>8s}  {'Description'}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*60}")
    for d in descriptions:
        print(f"  {d['cluster']:>8}  {d['n_samples']:>8d}  {d['mean_fitness']:>8s}  {d['description']}")


def print_region_stats(stats, label):
    """Pretty-print region parameter statistics."""
    print(f"\n  --- {label} Parameter Ranges ---")
    print(f"  {'Parameter':>18s}  {'Mean':>8s}  {'Std':>8s}  {'P5':>8s}  {'P95':>8s}  {'Range'}")
    print(f"  {'-'*18}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for _, row in stats.iterrows():
        print(f"  {row['parameter']:>18s}  {row['mean']:>8.2f}  {row['std']:>8.2f}  "
              f"{row['p5']:>8.2f}  {row['p95']:>8.2f}  {row['range_p5_p95']:>8.2f}")


def main():
    print("=" * 60)
    print("  High-Performance Region Discovery")
    print("=" * 60)

    # ---- PART 1: Run trials ----
    print(f"\n  Running {HPR_N_TRIALS} independent PSO trials (XGBoost)...")
    t0 = time.time()
    trial_results = run_trials(n_trials=HPR_N_TRIALS, max_iter=60, model="xgb", verbose=False)
    t_elapsed = time.time() - t0
    print(f"  Completed in {t_elapsed:.1f}s ({t_elapsed/HPR_N_TRIALS:.2f}s/trial)")

    # Collect all candidates
    positions_all, fitnesses_all = collect_from_trials(trial_results)
    gbest = fitnesses_all.max()
    print(f"  Total candidates: {len(positions_all):,d}")
    print(f"  Overall gbest fitness: {gbest:.4f}")

    # ---- PART 2: Percentile regions ----
    print(f"\n  --- Top-P% Regions ---")
    for p in HPR_PERCENTILES:
        label = f"Top {int(p*100)}%"
        pos, fit, thresh = filter_by_percentile(positions_all, fitnesses_all, top_p=p)
        stats = compute_region_stats(pos)
        print(f"\n  {label} (threshold >= {thresh:.2f}, n={pos.shape[0]})")
        print_region_stats(stats, label)

    # ---- PART 3: Absolute threshold region ----
    print(f"\n  --- Absolute Threshold (>= {HPR_ABSOLUTE_THRESHOLD}) ---")
    pos_abs, fit_abs, thresh_abs = filter_by_absolute(positions_all, fitnesses_all, HPR_ABSOLUTE_THRESHOLD)
    print(f"  n={len(pos_abs):,d} solutions with fitness >= {HPR_ABSOLUTE_THRESHOLD}")

    # ---- PART 4: Clustering ----
    print(f"\n  --- Solution Clustering (Top 5%) ---")
    pos_top5, fit_top5, _ = filter_by_percentile(positions_all, fitnesses_all, top_p=0.05)
    if len(pos_top5) >= 10:
        labels, centers_df = cluster_solutions(pos_top5, fit_top5, method="dbscan")
        n_clusters = len(set(l for l in labels if l >= 0))
        print(f"  Found {n_clusters} clusters ({len(labels)} samples, "
              f"{sum(labels == -1)} noise)")
        if not centers_df.empty:
            descriptions = describe_clusters(centers_df)
            print_cluster_table(descriptions)
    else:
        print("  Not enough samples for clustering")

    # ---- PART 5: Full pipeline via RegionDiscovery ----
    print(f"\n  --- Full Pipeline (RegionDiscovery) ---")
    rd = RegionDiscovery()
    discovery = rd.run_from_trials(trial_results, gbest=gbest)

    # ---- PART 6: Plots ----
    plot_region_report(discovery)

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"  Region Discovery Complete")
    print(f"{'=' * 60}")
    print(f"  Searched {len(positions_all):,d} candidate solutions across {HPR_N_TRIALS} trials")
    print(f"  Overall gbest: {gbest:.2f}")
    for p in HPR_PERCENTILES:
        label = f"top_{int(p*100):d}pct"
        reg = discovery["regions"].get(label)
        if reg:
            print(f"  {label:>10s}: threshold={reg['threshold']:.2f}, n={reg['positions'].shape[0]:,d}")

    if not discovery.get("cluster_centers", pd.DataFrame()).empty:
        n_clus = len(set(l for l in discovery["cluster_labels"] if l >= 0))
        print(f"  Solution clusters found: {n_clus}")

    return discovery


if __name__ == "__main__":
    import pandas as pd
    discovery = main()
