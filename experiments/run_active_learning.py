#!/usr/bin/env python
"""
T05: Active Learning Iterative Refinement
===========================================
Shallow (5.1): PSO -> uncertainty sampling -> NN -> retrain -> compare metrics
Medium (5.2): Multi-round PSO + fine-tuning loop

Usage:
  /mnt/e/python313/python.exe experiments/run_active_learning.py          # shallow only
  /mnt/e/python313/python.exe experiments/run_active_learning.py --medium # shallow + medium
"""
import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.optimization.active_learning import (
    ActiveLearner, UncertaintySampler, generate_report,
)
from src.utils.config import (
    AL_FIGURES_DIR, PROCESSED_DIR, DEEP_ENSEMBLE_DIR,
    AL_TOP_K, AL_NN_NEIGHBORS, RANDOM_STATE,
    PSO_PARAM_BOUNDS, PSO_DISCRETE_INDICES,
    PSO_N_PARTICLES, PSO_W_START, PSO_W_END, PSO_MAX_ITER,
)

FEATURE_COLS = ["speed_kmh", "wing_angle_deg", "drs_active", "downforce_n", "drag_n"]


def run_shallow_experiment(verbose=True):
    print("=" * 65)
    print("  T05.1 Shallow: One-Round Post-hoc Refinement")
    print("=" * 65)

    learner = ActiveLearner()
    t0 = time.time()
    result = learner.run_shallow(top_k=AL_TOP_K, n_neighbors=AL_NN_NEIGHBORS,
                                  verbose=verbose)
    elapsed = time.time() - t0

    print(f"\n[Shallow] Done in {elapsed:.1f}s")
    print(f"  R2: {result['baseline_metrics']['R2']:.4f} -> {result['refined_metrics']['R2']:.4f} "
          f"(d{result['delta_r2']:+.4f})")
    print(f"  MSE: {result['baseline_metrics']['MSE']:.4f} -> {result['refined_metrics']['MSE']:.4f} "
          f"(d{result['delta_mse']:+.4f})")

    return result


def run_medium_experiment(verbose=True):
    print("\n" + "=" * 65)
    print("  T05.2 Medium: Multi-Round Interleaved Refinement")
    print("=" * 65)

    learner = ActiveLearner()
    t0 = time.time()
    result = learner.run_medium(
        n_rounds=3, pso_interval=10, top_k=10,
        n_neighbors=AL_NN_NEIGHBORS, verbose=verbose,
    )
    elapsed = time.time() - t0

    print(f"\n[Medium] Done in {elapsed:.1f}s")
    metrics_list = result["round_metrics"]
    if len(metrics_list) >= 2:
        first = metrics_list[0]
        last = metrics_list[-1]
        delta_r2 = last["R2"] - first["R2"]
        print(f"  R2: {first['R2']:.4f} -> {last['R2']:.4f} (d{delta_r2:+.4f})")

    return result


def plot_medium_convergence(medium_result):
    """Plot R2 progression across rounds."""
    os.makedirs(AL_FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(AL_FIGURES_DIR, "active_learning_convergence.png")

    if medium_result is None:
        return None

    metrics_list = medium_result["round_metrics"]
    rounds = [m["round"] for m in metrics_list]
    r2_values = [m["R2"] for m in metrics_list]
    mse_values = [m["MSE"] for m in metrics_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(rounds, r2_values, "o-", color="steelblue", linewidth=2, markersize=8)
    ax1.set_xlabel("Active Learning Round")
    ax1.set_ylabel("R2 (Test Set)")
    ax1.set_title("Model R2 vs Active Learning Rounds")
    ax1.grid(alpha=0.3)

    ax2.plot(rounds, mse_values, "s-", color="darkorange", linewidth=2, markersize=8)
    ax2.set_xlabel("Active Learning Round")
    ax2.set_ylabel("MSE (Test Set)")
    ax2.set_title("Model MSE vs Active Learning Rounds")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Convergence plot saved: {fig_path}")
    return fig_path


def plot_shallow_uncertainty_distribution(shallow_result):
    """Histogram of sigma values among collected candidates."""
    if shallow_result is None:
        return None

    os.makedirs(AL_FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(AL_FIGURES_DIR, "uncertainty_distribution.png")

    sigma = shallow_result["top_sigma_sigma"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(sigma, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(np.median(sigma), color="red", linestyle="--", label=f"Median sigma={np.median(sigma):.2f}")
    ax.set_xlabel("Uncertainty sigma(x)")
    ax.set_ylabel("Count")
    ax.set_title(f"Top-{len(sigma)} High-Uncertainty Candidate Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Uncertainty distribution plot saved: {fig_path}")
    return fig_path


def compare_pso_convergence(shallow_result, verbose=True):
    """Run full PSO with baseline vs refined model and compare gbest convergence."""
    print("\n  --- PSO Convergence Comparison (Baseline vs Refined) ---")

    from src.optimization.pso_adaptive import PSOAdaptive
    from src.optimization.fitness import FitnessRiskSensitive, ModelWrapper, load_ensemble_fitness
    import pickle

    learner = ActiveLearner()
    learner._ensure_data_loaded()

    bounds = np.array(PSO_PARAM_BOUNDS, dtype=np.float64)

    # --- Baseline PSO ---
    if verbose:
        print("  Running PSO with baseline model...")
    fitness_baseline = load_ensemble_fitness(lambda_risk=0.0)
    pso_baseline = PSOAdaptive(
        n_particles=PSO_N_PARTICLES, bounds=bounds,
        discrete_indices=list(PSO_DISCRETE_INDICES),
        w_start=PSO_W_START, w_end=PSO_W_END,
        max_iter=60, seed=RANDOM_STATE, maximise=True,
    )
    result_baseline = pso_baseline.optimize(fitness_baseline, verbose=True)

    # --- Refined PSO ---
    if verbose:
        print("\n  Running PSO with refined model...")
    from src.models.deep_ensemble import DeepEnsemble
    from src.utils.config import MLP_HIDDEN_UNITS, DEEP_ENSEMBLE_M, NN_LR, NN_WEIGHT_DECAY, NN_BATCH_SIZE, NN_MAX_EPOCHS, NN_EARLY_STOP_PATIENCE

    refined_ensemble = DeepEnsemble.load(
        input_dim=5, hidden_units=MLP_HIDDEN_UNITS, M=DEEP_ENSEMBLE_M,
        lr=NN_LR, weight_decay=NN_WEIGHT_DECAY,
        batch_size=NN_BATCH_SIZE, max_epochs=NN_MAX_EPOCHS,
        patience=NN_EARLY_STOP_PATIENCE,
        load_dir=shallow_result["model_dir"],
    )
    refined_wrapper = ModelWrapper(refined_ensemble, learner._scaler,
                                    model_type="ensemble", y_transform="x100")
    fitness_refined = FitnessRiskSensitive(refined_wrapper, lambda_risk=0.0)
    pso_refined = PSOAdaptive(
        n_particles=PSO_N_PARTICLES, bounds=bounds,
        discrete_indices=list(PSO_DISCRETE_INDICES),
        w_start=PSO_W_START, w_end=PSO_W_END,
        max_iter=60, seed=RANDOM_STATE, maximise=True,
    )
    result_refined = pso_refined.optimize(fitness_refined, verbose=True)

    # --- Plot comparison ---
    os.makedirs(AL_FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(AL_FIGURES_DIR, "pso_convergence_comparison.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result_baseline["history"]["iteration"],
            result_baseline["history"]["gbest_fitness"],
            label="Baseline Model", linewidth=2, color="steelblue")
    ax.plot(result_refined["history"]["iteration"],
            result_refined["history"]["gbest_fitness"],
            label="Refined Model (AL)", linewidth=2, color="darkorange")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gbest Fitness")
    ax.set_title("PSO Convergence: Baseline vs Active-Learning Refined Model")
    ax.legend()
    ax.grid(alpha=0.3)

    # Annotate final values
    b_last = result_baseline["history"]["gbest_fitness"][-1]
    r_last = result_refined["history"]["gbest_fitness"][-1]
    ax.axhline(y=b_last, linestyle=":", color="steelblue", alpha=0.5)
    ax.axhline(y=r_last, linestyle=":", color="darkorange", alpha=0.5)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PSO convergence comparison saved: {fig_path}")

    if verbose:
        print(f"  Baseline PSO: final gbest={b_last:.4f}, iters={result_baseline['n_iter']}")
        print(f"  Refined PSO:  final gbest={r_last:.4f}, iters={result_refined['n_iter']}")

    return {
        "baseline_gbest": b_last,
        "refined_gbest": r_last,
        "baseline_iters": result_baseline["n_iter"],
        "refined_iters": result_refined["n_iter"],
        "baseline_history": result_baseline["history"],
        "refined_history": result_refined["history"],
    }


def main():
    parser = argparse.ArgumentParser(description="T05: Active Learning")
    parser.add_argument("--medium", action="store_true", help="Run medium (multi-round) experiment")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    args = parser.parse_args()

    shallow_result = None
    medium_result = None

    # --- 5.1 Shallow ---
    shallow_result = run_shallow_experiment(verbose=args.verbose)

    # --- 5.2 Medium ---
    if args.medium:
        medium_result = run_medium_experiment(verbose=args.verbose)

    # --- Plots ---
    print("\n" + "=" * 65)
    print("  Generating Figures")
    print("=" * 65)

    plot_shallow_uncertainty_distribution(shallow_result)

    if medium_result:
        plot_medium_convergence(medium_result)

    # --- PSO Convergence Comparison ---
    pso_conv_result = compare_pso_convergence(shallow_result, verbose=args.verbose)

    # --- Report ---
    print("\n" + "=" * 65)
    print("  Generating Report")
    print("=" * 65)

    report_path = generate_report(shallow_result, medium_result, pso_conv_result)
    print(f"  Report saved: {report_path}")

    # --- Final Summary ---
    print("\n" + "=" * 65)
    print("  T05 Active Learning Complete")
    print("=" * 65)

    if shallow_result:
        delta = shallow_result["delta_r2"]
        status = "improved" if delta > 0 else "no change" if delta == 0 else "degraded"
        print(f"  [5.1 Shallow] R2 {status}: {shallow_result['baseline_metrics']['R2']:.4f} -> "
              f"{shallow_result['refined_metrics']['R2']:.4f} (d{delta:+.4f})")

    if pso_conv_result:
        print(f"  [PSO Conv.] Baseline gbest={pso_conv_result['baseline_gbest']:.4f} ({pso_conv_result['baseline_iters']} iters), "
              f"Refined gbest={pso_conv_result['refined_gbest']:.4f} ({pso_conv_result['refined_iters']} iters)")

    if medium_result:
        m = medium_result["round_metrics"]
        if len(m) >= 2:
            delta = m[-1]["R2"] - m[0]["R2"]
            status = "improved" if delta > 0 else "no change" if delta == 0 else "degraded"
            print(f"  [5.2 Medium] R2 {status}: {m[0]['R2']:.4f} -> {m[-1]['R2']:.4f} (d{delta:+.4f})")


if __name__ == "__main__":
    main()
