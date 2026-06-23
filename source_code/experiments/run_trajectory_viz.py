#!/usr/bin/env python
"""
PCA Trajectory Visualization for PSO Particle Migration.
=========================================================
Runs 1 PSO trial per scenario (S1 Monza, S3 Balanced) with
collect_candidates=True, then projects all particle positions
to 2D via PCA and draws migration trajectories.

Key visuals:
  - hexbin background: fitness landscape in PCA space
  - faded dots: all particles at all iterations
  - top-5 particle trajectories: light-blue to dark-blue lines
  - gbest trajectory: red line with interval markers
  - gold star at final optimum
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import (
    RANDOM_STATE, SCENARIO_N_PARTICLES, SCENARIO_MAX_ITER,
    PSO_W_START, PSO_W_END, PSO_W_ALPHA,
    PSO_C1, PSO_C2, PSO_TOL, PSO_EARLY_STOP_ITERS,
    SCENARIO_LAMBDA_RISK,
)
from src.scenarios.scenario_def import SCENARIO_S1_MONZA, SCENARIO_S3_BALANCED
from src.scenarios.scenario_runner import _make_scenario_fitness
from src.optimization.pso_adaptive import PSOAdaptive
from src.optimization.pso_risk_sensitive import DISCRETE_INDICES
from src.analysis.explainability import collect_trajectory_data, pca_reduce
from src.visualization.plot_scenarios import plot_trajectory


def run_single_pso_candidates(scenario, seed=RANDOM_STATE):
    """Run 1 PSO trial with collect_candidates=True."""
    fitness = _make_scenario_fitness(scenario, use_risk=True, lambda_risk=SCENARIO_LAMBDA_RISK)

    pso = PSOAdaptive(
        n_particles=SCENARIO_N_PARTICLES,
        bounds=scenario.bounds,
        discrete_indices=scenario.discrete_indices,
        w_start=PSO_W_START,
        w_end=PSO_W_END,
        alpha=PSO_W_ALPHA,
        c1=PSO_C1,
        c2=PSO_C2,
        max_iter=SCENARIO_MAX_ITER,
        early_stop_iters=PSO_EARLY_STOP_ITERS,
        tol=PSO_TOL,
        seed=seed,
        maximise=True,
    )
    result = pso.optimize(fitness, verbose=True, collect_candidates=True)
    return result


def main():
    print("=" * 60)
    print("  T06-Ext: PSO Particle Migration Trajectory (PCA)")
    print("=" * 60)

    scenarios = [
        (SCENARIO_S1_MONZA, 4200),
        (SCENARIO_S3_BALANCED, 4300),
    ]

    # ---- Phase 1: Run PSO with collect_candidates ----
    print("\n  Phase 1: Running PSO with candidate collection...")
    results = {}
    traj_data = {}

    for scenario, seed in scenarios:
        print(f"\n  --- {scenario.name} ---")
        result = run_single_pso_candidates(scenario, seed=seed)
        results[scenario.code] = result
        td = collect_trajectory_data(result)
        traj_data[scenario.code] = td

    # ---- Phase 2: Joint PCA fit ----
    print("\n  Phase 2: Fitting joint PCA on all particle positions...")
    all_positions = []
    for code in traj_data:
        positions_list, _, _, _ = traj_data[code]
        all_positions.append(np.vstack(positions_list))
    all_positions = np.vstack(all_positions)
    print(f"  Total PCA samples: {all_positions.shape[0]:,d} rows × {all_positions.shape[1]} dims")

    pca_model, scaler, explained_var = pca_reduce(all_positions)
    print(f"  PC1: {explained_var[0]*100:.1f}%  |  PC2: {explained_var[1]*100:.1f}%  "
          f"|  Total: {explained_var.sum()*100:.1f}%")
    pc1_loadings = np.abs(pca_model.components_[0])
    pc2_loadings = np.abs(pca_model.components_[1])
    from src.utils.config import FEATURE_COLS
    print(f"  PC1 dominant: {FEATURE_COLS[np.argmax(pc1_loadings)]} ({max(pc1_loadings):.3f})")
    print(f"  PC2 dominant: {FEATURE_COLS[np.argmax(pc2_loadings)]} ({max(pc2_loadings):.3f})")

    # ---- Phase 3: Plot trajectories ----
    print("\n  Phase 3: Generating trajectory plots...")
    for scenario, _ in scenarios:
        code = scenario.code
        td = traj_data[code]
        plot_trajectory(td, scenario.name, pca_model, scaler, explained_var)

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"  Trajectory Visualization Complete")
    print(f"{'=' * 60}")
    for scenario, _ in scenarios:
        code = scenario.code
        td = traj_data[code]
        _, _, gbest_fits, _ = td
        print(f"  {code}: {len(td[0])} iters, "
              f"gbest start={gbest_fits[0]:.2f}, final={gbest_fits[-1]:.4f}")
    print(f"\n  Figures saved: figures/scenarios/trajectory_*.png")


if __name__ == "__main__":
    main()
