#!/usr/bin/env python
"""
T11: Computational Intelligence Algorithm Comparison
=====================================================
Compares PSO, GA, DE, SA on the F1 aerodynamic optimisation problem
using a common XGBoost-based fitness function.

Output:
  figures/16_ci_comparison.png  — convergence curves + boxplot + timing
  reports/ci_comparison.csv     — summary stats table
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.optimization.pso_base import PSOBase
from src.optimization.pso_adaptive import PSOAdaptive
from src.optimization.ga import GA
from src.optimization.de import DE
from src.optimization.sa import SA
from src.optimization.fitness import load_xgb_fitness
from src.utils.config import (
    FIGURES_DIR, REPORTS_DIR,
    PSO_PARAM_BOUNDS, PSO_DISCRETE_INDICES,
    RANDOM_STATE,
)

FEATURE_COLS = ["speed_kmh", "wing_angle_deg", "drs_active", "downforce_n", "drag_n"]
BOUNDS = np.array(PSO_PARAM_BOUNDS, dtype=np.float64)
DISCRETE = list(PSO_DISCRETE_INDICES)

N_TRIALS = 20
N_DIM = len(BOUNDS)


def run_trial(algo_cls, algo_name, algo_kwargs, fitness_fn, trial_id, max_iter=100):
    """Run a single trial and return results dict."""
    seed = RANDOM_STATE + trial_id * 100
    t0 = time.time()

    if algo_name == "PSO":
        pso = algo_cls(
            n_particles=50, bounds=BOUNDS, discrete_indices=DISCRETE,
            max_iter=max_iter, early_stop_iters=12, tol=1e-6,
            maximise=True, seed=seed, **algo_kwargs,
        )
        result = pso.optimize(fitness_fn, verbose=False)
        elapsed = time.time() - t0
        best_pos = result["gbest_pos"]
        best_fitness = result["gbest_fitness"]
        n_iters = result["n_iter"]
        history_fitness = result["history"]["gbest_fitness"]

    elif algo_name == "GA":
        ga = algo_cls(
            pop_size=50, bounds=BOUNDS, discrete_indices=DISCRETE,
            max_iter=max_iter, early_stop_iters=12, tol=1e-6,
            maximise=True, seed=seed, **algo_kwargs,
        )
        ga.set_fitness(fitness_fn)
        best_pos, best_fitness, history, n_iters = ga.optimize(verbose=False)
        elapsed = time.time() - t0
        history_fitness = history["best_fitness"]

    elif algo_name == "DE":
        de = algo_cls(
            pop_size=50, bounds=BOUNDS, discrete_indices=DISCRETE,
            max_iter=max_iter, early_stop_iters=12, tol=1e-6,
            maximise=True, seed=seed, **algo_kwargs,
        )
        de.set_fitness(fitness_fn)
        best_pos, best_fitness, history, n_iters = de.optimize(verbose=False)
        elapsed = time.time() - t0
        history_fitness = history["best_fitness"]

    elif algo_name == "SA":
        sa = algo_cls(
            bounds=BOUNDS, discrete_indices=DISCRETE,
            max_iter=max_iter * 20, early_stop_iters=12 * 20,
            tol=1e-6, maximise=True, seed=seed, **algo_kwargs,
        )
        sa.set_fitness(fitness_fn)
        best_pos, best_fitness, history, n_iters = sa.optimize(verbose=False)
        elapsed = time.time() - t0
        history_fitness = history["best_fitness"]

    else:
        raise ValueError(f"Unknown algo: {algo_name}")

    return {
        "best_pos": best_pos,
        "best_fitness": float(best_fitness),
        "n_iters": n_iters,
        "elapsed_ms": elapsed * 1000,
        "history": history_fitness,
    }


def compute_stats(trials):
    fitness = [t["best_fitness"] for t in trials]
    iters = [t["n_iters"] for t in trials]
    times = [t["elapsed_ms"] for t in trials]
    return {
        "fitness_mean": np.mean(fitness),
        "fitness_std": np.std(fitness),
        "fitness_max": np.max(fitness),
        "iters_mean": np.mean(iters),
        "iters_std": np.std(iters),
        "time_ms_mean": np.mean(times),
        "time_ms_std": np.std(times),
        "convergence_rate": np.mean([1.0 if t["n_iters"] < 100 else 0.0 for t in trials]),
    }


def plot_comparison(all_trials, stats, out_path):
    """Plot convergence curves + boxplot in a 2x2 panel."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (1,1) Convergence curves
    ax = axes[0, 0]
    colors = {"PSO": "#2B5CD9", "GA": "#E8642D", "DE": "#2CA02C", "SA": "#9467BD"}
    algo_order = ["PSO", "GA", "DE", "SA"]
    for algo in algo_order:
        hist = all_trials[algo]
        max_len = max(len(h["history"]) for h in hist)
        matrix = np.full((len(hist), max_len), np.nan)
        for i, h in enumerate(hist):
            matrix[i, :len(h["history"])] = h["history"]
        mean_curve = np.nanmean(matrix, axis=0)
        std_curve = np.nanstd(matrix, axis=0)
        x = np.arange(max_len)
        ax.plot(x, mean_curve, color=colors[algo], linewidth=2, label=algo)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                         color=colors[algo], alpha=0.12)
    ax.set_xlabel("Iteration / Step")
    ax.set_ylabel("Best Fitness")
    ax.set_title("Convergence Curves (20 trials, mean +/- 1 std)")
    ax.legend()
    ax.grid(alpha=0.3)

    # (1,2) Fitness boxplot
    ax = axes[0, 1]
    fitness_data = [np.array([t["best_fitness"] for t in all_trials[algo]]) for algo in algo_order]
    bp = ax.boxplot(fitness_data, tick_labels=algo_order, patch_artist=True)
    for patch, algo in zip(bp["boxes"], algo_order):
        patch.set_facecolor(colors[algo])
        patch.set_alpha(0.6)
    ax.set_ylabel("Best Fitness")
    ax.set_title("Fitness Distribution (20 trials)")
    ax.grid(alpha=0.3, axis="y")

    # (2,1) Iterations to converge
    ax = axes[1, 0]
    x_pos = np.arange(len(algo_order))
    iters_mean = [stats[algo]["iters_mean"] for algo in algo_order]
    iters_std = [stats[algo]["iters_std"] for algo in algo_order]
    bars = ax.bar(x_pos, iters_mean, yerr=iters_std, capsize=5,
                  color=[colors[a] for a in algo_order], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algo_order)
    ax.set_ylabel("Iterations")
    ax.set_title("Convergence Iterations (mean +/- std)")
    ax.grid(alpha=0.3, axis="y")

    # (2,2) Time per run
    ax = axes[1, 1]
    times_mean = [stats[algo]["time_ms_mean"] for algo in algo_order]
    times_std = [stats[algo]["time_ms_std"] for algo in algo_order]
    bars = ax.bar(x_pos, times_mean, yerr=times_std, capsize=5,
                  color=[colors[a] for a in algo_order], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algo_order)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Execution Time (mean +/- std)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Computational Intelligence Algorithm Comparison\n"
                 "F1 Aerodynamic Stability Optimisation (XGBoost surrogate)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


def main():
    print("=" * 60)
    print("  T11: CI Algorithm Comparison (PSO vs GA vs DE vs SA)")
    print("=" * 60)

    print("\n  Loading XGBoost fitness function...")
    fitness_fn = load_xgb_fitness()
    print("    XGBoost loaded (StandardScaler)")

    # Algorithm configurations
    algos = {
        "PSO": {
            "cls": PSOAdaptive,
            "kwargs": {"w_start": 0.9, "w_end": 0.4, "alpha": 1.0},
        },
        "GA": {
            "cls": GA,
            "kwargs": {"p_crossover": 0.8, "p_mutation": 0.1, "tournament_k": 3, "elitism": 2},
        },
        "DE": {
            "cls": DE,
            "kwargs": {"F": 0.8, "CR": 0.9},
        },
        "SA": {
            "cls": SA,
            "kwargs": {"T_start": 100.0, "T_end": 0.01, "cooling_rate": 0.95, "steps_per_temp": 20},
        },
    }

    all_trials = {}
    stats = {}

    print(f"\n  Running {N_TRIALS} trials per algorithm...")

    for algo_name, cfg in algos.items():
        print(f"\n  --- {algo_name} ---")
        trials = []
        for trial_id in range(N_TRIALS):
            t = run_trial(cfg["cls"], algo_name, cfg["kwargs"], fitness_fn, trial_id)
            trials.append(t)
            if (trial_id + 1) % 5 == 0:
                print(f"    Trial {trial_id + 1}/{N_TRIALS}  "
                      f"fitness={t['best_fitness']:.4f}  iters={t['n_iters']}")

        all_trials[algo_name] = trials
        st = compute_stats(trials)
        stats[algo_name] = st
        print(f"    Results: fitness={st['fitness_mean']:.4f} +/- {st['fitness_std']:.4f}, "
              f"iters={st['iters_mean']:.1f}, time={st['time_ms_mean']:.1f}ms, "
              f"conv={st['convergence_rate']:.0%}")

    # Plot
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, "16_ci_comparison.png")
    plot_comparison(all_trials, stats, fig_path)

    # Save CSV
    os.makedirs(REPORTS_DIR, exist_ok=True)
    csv_path = os.path.join(REPORTS_DIR, "ci_comparison.csv")
    rows = []
    for algo_name in ["PSO", "GA", "DE", "SA"]:
        s = stats[algo_name]
        rows.append({
            "Algorithm": algo_name,
            "Fitness_Mean": s["fitness_mean"],
            "Fitness_Std": s["fitness_std"],
            "Fitness_Max": s["fitness_max"],
            "Iters_Mean": s["iters_mean"],
            "Iters_Std": s["iters_std"],
            "Time_ms_Mean": s["time_ms_mean"],
            "Convergence_Rate": s["convergence_rate"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # Summary table
    print("\n" + "=" * 60)
    print("  CI Algorithm Comparison Summary")
    print("=" * 60)
    print(f"\n{'Algorithm':>12s}  {'Fitness':>10s}  {'Iters':>8s}  {'Time_ms':>8s}  {'Conv':>6s}")
    print("-" * 50)
    for algo_name in ["PSO", "GA", "DE", "SA"]:
        s = stats[algo_name]
        print(f"{algo_name:>12s}  {s['fitness_mean']:>8.4f} +/- {s['fitness_std']:.4f}"
              f"  {s['iters_mean']:>6.1f}  {s['time_ms_mean']:>6.1f}  "
              f"  {s['convergence_rate']:>5.0%}")

    # PSO vs best competitor
    best_fitness = max(stats[a]["fitness_mean"] for a in stats)
    best_algo = [a for a in stats if stats[a]["fitness_mean"] == best_fitness][0]
    best_iters = min(stats[a]["iters_mean"] for a in stats)
    fastest_algo = [a for a in stats if stats[a]["iters_mean"] == best_iters][0]
    best_time = min(stats[a]["time_ms_mean"] for a in stats)
    fastest_time = [a for a in stats if stats[a]["time_ms_mean"] == best_time][0]

    print(f"\n  Best fitness:     {best_algo} ({best_fitness:.4f})")
    print(f"  Fastest converge: {fastest_algo} ({best_iters:.1f} iters)")
    print(f"  Lowest latency:   {fastest_time} ({best_time:.1f} ms)")
    print(f"\n  CSV saved: {csv_path}")
    print(f"  Figure saved: {fig_path}")
    print("\n  T11 Complete!")


if __name__ == "__main__":
    main()
