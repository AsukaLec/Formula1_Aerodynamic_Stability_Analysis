#!/usr/bin/env python
"""
T07: Porpoising Risk Heatmap Experiment
========================================
Computes local gradient norms of the DeepEnsemble surrogate model in
(speed_kmh, wing_angle_deg) space as a proxy for porpoising risk.
Generates risk heatmaps overlaid with T06 scenario optimal solutions.
"""
import os
import sys
import json
import time
import pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import (
    PROCESSED_DIR, MODELS_OUTPUT_DIR, DEEP_ENSEMBLE_DIR,
    RANDOM_STATE, FEATURE_COLS,
    MLP_HIDDEN_UNITS, DEEP_ENSEMBLE_M,
    NN_LR, NN_WEIGHT_DECAY, NN_BATCH_SIZE, NN_MAX_EPOCHS, NN_EARLY_STOP_PATIENCE,
    SCENARIOS_DIR, FIGURES_DIR,
)
from src.models.deep_ensemble import DeepEnsemble
from src.visualization.plot_porpoising import (
    compute_porpoising_risk, plot_risk_heatmaps,
    compute_xgboost_risk, PORPOISING_FIGURES_DIR,
)


def load_ensemble_refined():
    """Load the refined DeepEnsemble model from deep_ensemble_refined/."""
    load_dir = os.path.join(MODELS_OUTPUT_DIR, "deep_ensemble_refined")
    print(f"  Loading refined DeepEnsemble from: {load_dir}")
    ensemble = DeepEnsemble.load(
        input_dim=5, hidden_units=MLP_HIDDEN_UNITS, M=DEEP_ENSEMBLE_M,
        lr=NN_LR, weight_decay=NN_WEIGHT_DECAY,
        batch_size=NN_BATCH_SIZE, max_epochs=NN_MAX_EPOCHS,
        patience=NN_EARLY_STOP_PATIENCE,
        load_dir=load_dir,
    )
    ensemble.eval()
    print(f"  Loaded {ensemble.M} ensemble members on {ensemble.trainers[0].device}")
    return ensemble


def compute_stability_surface(ensemble, scaler_mm, X_train_orig, n_grid=80, knn_k=10):
    """Compute predicted stability_index surface for DRS=0 and DRS=1.
    Grid spans PSO_PARAM_BOUNDS to cover full search space."""
    from sklearn.neighbors import NearestNeighbors
    from src.utils.config import PSO_PARAM_BOUNDS

    i_speed = FEATURE_COLS.index("speed_kmh")
    i_wing = FEATURE_COLS.index("wing_angle_deg")
    i_drs = FEATURE_COLS.index("drs_active")

    grid_speed = np.linspace(PSO_PARAM_BOUNDS[i_speed][0], PSO_PARAM_BOUNDS[i_speed][1], n_grid)
    grid_wing = np.linspace(PSO_PARAM_BOUNDS[i_wing][0], PSO_PARAM_BOUNDS[i_wing][1], n_grid)
    A_speed, A_wing = np.meshgrid(grid_speed, grid_wing, indexing="ij")

    ref_xy = X_train_orig[:, [i_speed, i_wing, i_drs]]
    other_indices = [j for j in range(5) if j not in (i_speed, i_wing, i_drs)]
    ref_rest = X_train_orig[:, other_indices]

    nn = NearestNeighbors(n_neighbors=knn_k)
    nn.fit(ref_xy)

    surfaces = {}
    for drs in [0, 1]:
        feats_orig = np.zeros((n_grid * n_grid, 5), dtype=np.float32)
        feats_orig[:, i_speed] = A_speed.ravel()
        feats_orig[:, i_wing] = A_wing.ravel()
        feats_orig[:, i_drs] = drs

        test_xy = feats_orig[:, [i_speed, i_wing, i_drs]]
        _, indices = nn.kneighbors(test_xy)

        for idx, j in enumerate(other_indices):
            neighbour_vals = ref_rest[:, idx][indices]
            feats_orig[:, j] = np.median(neighbour_vals, axis=1)

        feats_scaled = scaler_mm.transform(feats_orig)
        pred = ensemble.predict(feats_scaled) * 100.0
        surfaces[f"surface_drs{drs}"] = pred.reshape(n_grid, n_grid)

    return surfaces


def load_scenario_best_solutions():
    """Load best_solution.json from each scenario directory."""
    solutions = []
    scenario_names = {
        "S1_monza": "S1 Monza",
        "S2_monaco": "S2 Monaco",
        "S3_balanced": "S3 Balanced",
        "S4_wet": "S4 Wet",
    }
    for key, name in scenario_names.items():
        path = os.path.join(SCENARIOS_DIR, key, "best_solution.json")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(path, "r") as f:
            data = json.load(f)
        sol = {
            "scenario": key,
            "scenario_name": name,
            "speed_kmh": data["gbest_pos_dict"]["speed_kmh"],
            "wing_angle_deg": data["gbest_pos_dict"]["wing_angle_deg"],
            "drs_active": int(data["gbest_pos_dict"]["drs_active"]),
            "gbest_fitness": data["gbest_fitness"],
        }
        solutions.append(sol)
        print(f"  Loaded {key}: v={sol['speed_kmh']:.1f} km/h, "
              f"α={sol['wing_angle_deg']:.1f}°, DRS={sol['drs_active']}, "
              f"fitness={sol['gbest_fitness']:.2f}")
    return solutions


def print_analysis(risk_data, best_solutions, train_speed_bounds, train_wing_bounds):
    """Print quantitative analysis of risk vs optimal solutions."""
    risk_drs0 = risk_data["risk_drs0"]
    risk_drs1 = risk_data["risk_drs1"]
    grid_speed = risk_data["grid_speed"]
    grid_wing = risk_data["grid_wing"]

    print()
    print("=" * 60)
    print("  Risk Analysis: Scenario Optimal Solutions")
    print("=" * 60)
    print(f"  Grid span:  speed [{grid_speed[0]:.0f}, {grid_speed[-1]:.0f}] km/h")
    print(f"              wing  [{grid_wing[0]:.0f}, {grid_wing[-1]:.0f}] deg")
    print(f"  Training data span: speed [{train_speed_bounds[0]:.0f}, {train_speed_bounds[1]:.0f}], "
          f"wing [{train_wing_bounds[0]:.0f}, {train_wing_bounds[1]:.0f}]")
    print(f"  Risk range (DRS=0): {risk_drs0.min():.4f} – {risk_drs0.max():.4f}")
    print(f"  Risk range (DRS=1): {risk_drs1.min():.4f} – {risk_drs1.max():.4f}")
    print(f"  Risk median (DRS=0): {np.median(risk_drs0):.4f}")
    print(f"  Risk median (DRS=1): {np.median(risk_drs1):.4f}")

    speed_min_t, speed_max_t = train_speed_bounds
    wing_min_t, wing_max_t = train_wing_bounds

    for sol in best_solutions:
        drs = sol["drs_active"]
        risk_grid = risk_drs0 if drs == 0 else risk_drs1

        si = np.abs(grid_speed - sol["speed_kmh"]).argmin()
        wi = np.abs(grid_wing - sol["wing_angle_deg"]).argmin()
        risk_at_opt = risk_grid[si, wi]

        risk_flat = risk_grid.ravel()
        pct = (risk_flat < risk_at_opt).mean() * 100

        label = "LOW" if risk_at_opt < np.median(risk_grid) else "HIGH"

        # OOD check
        ood_speed = sol["speed_kmh"] < speed_min_t or sol["speed_kmh"] > speed_max_t
        ood_wing = sol["wing_angle_deg"] < wing_min_t or sol["wing_angle_deg"] > wing_max_t
        ood_flags = []
        if ood_speed:
            ood_flags.append("OOD-speed")
        if ood_wing:
            ood_flags.append("OOD-wing")
        ood_tag = f" [{' + '.join(ood_flags)}]" if ood_flags else ""

        print(f"\n  [{sol['scenario_name']}] DRS={drs}{ood_tag}")
        print(f"    Position:  v={sol['speed_kmh']:.1f} km/h, α={sol['wing_angle_deg']:.1f}°")
        print(f"    Risk at optimum: {risk_at_opt:.4f} ({label} risk, {pct:.1f}th percentile)")
        print(f"    Fitness: {sol['gbest_fitness']:.2f}")

    # Physical intuition check
    print()
    print("=" * 60)
    print("  Physical Intuition Verification")
    print("=" * 60)

    # High-speed + small wing angle region (danger zone)
    speed_high_mask = grid_speed[:, None] > 250
    wing_low_mask = grid_wing[None, :] < 10
    danger_drs0 = risk_drs0[speed_high_mask & wing_low_mask]
    danger_drs1 = risk_drs1[speed_high_mask & wing_low_mask]
    print(f"  High-speed (>250 km/h) + Small wing (<10°) — DRS=0: "
          f"mean_risk={danger_drs0.mean():.4f}")
    print(f"  High-speed (>250 km/h) + Small wing (<10°) — DRS=1: "
          f"mean_risk={danger_drs1.mean():.4f}")

    # Low-speed + large wing angle region (safe zone)
    speed_low_mask = grid_speed[:, None] < 150
    wing_high_mask = grid_wing[None, :] > 25
    safe_drs0 = risk_drs0[speed_low_mask & wing_high_mask]
    safe_drs1 = risk_drs1[speed_low_mask & wing_high_mask]
    print(f"  Low-speed (<150 km/h) + Large wing (>25°)  — DRS=0: "
          f"mean_risk={safe_drs0.mean():.4f}")
    print(f"  Low-speed (<150 km/h) + Large wing (>25°)  — DRS=1: "
          f"mean_risk={safe_drs1.mean():.4f}")

    # DRS open vs closed comparison
    drs_diff = risk_drs1 - risk_drs0
    print(f"\n  DRS ON - OFF risk difference: range [{drs_diff.min():.4f}, {drs_diff.max():.4f}]")
    print(f"  Mean risk increase from DRS ON: {drs_diff.mean():.4f}")

    ratio = risk_drs1.mean() / risk_drs0.mean() if risk_drs0.mean() > 0 else float("inf")
    print(f"  DRS ON/OFF risk ratio: {ratio:.2f}")

    # Overall assessment
    print()
    if danger_drs0.mean() > safe_drs0.mean():
        print("  [PASS] High-speed + small wing shows higher risk than low-speed + large wing.")
        print("  This aligns with F1 aerodynamic intuition.")
    else:
        print("  [WARNING] High-speed region does NOT show elevated risk.")
        print("  The surrogate model may not fully capture porpoising dynamics.")

    if drs_diff.mean() > 0:
        print("  [PASS] DRS ON increases risk on average — consistent with reduced rear grip.")
    else:
        print("  [NOTE] DRS ON does not increase risk overall in this model.")


def main():
    print("=" * 60)
    print("  T07: Porpoising Risk Heatmap Experiment")
    print("=" * 60)

    os.makedirs(PORPOISING_FIGURES_DIR, exist_ok=True)

    # ============ 1. Load Model ============
    print("\n[1/5] Loading DeepEnsemble refined model...")
    t0 = time.time()
    ensemble = load_ensemble_refined()
    print(f"  Done ({time.time() - t0:.1f}s)")

    # ============ 2. Load Data & Scaler ============
    print("\n[2/5] Loading data and scalers...")
    scaler_path = os.path.join(PROCESSED_DIR, "scaler_mm.pkl")
    with open(scaler_path, "rb") as f:
        scaler_mm = pickle.load(f)

    X_train_mm = np.load(os.path.join(PROCESSED_DIR, "X_train_mm.npy")).astype(np.float32)
    X_train_orig = scaler_mm.inverse_transform(X_train_mm)
    print(f"  Train data: {X_train_orig.shape}")
    print(f"  Scaler ranges: speed=[{scaler_mm.data_min_[0]:.0f}, {scaler_mm.data_max_[0]:.0f}]")
    print(f"                 wing=[{scaler_mm.data_min_[1]:.0f}, {scaler_mm.data_max_[1]:.0f}]")

    # ============ 3. Load Scenario Solutions ============
    print("\n[3/5] Loading scenario best solutions...")
    best_solutions = load_scenario_best_solutions()

    # ============ 4. Compute Risk Grids ============
    print("\n[4/5] Computing porpoising risk grids (autograd)...")
    print("  This computes gradient norm for 80x80 grid × 2 DRS states...")
    t0 = time.time()
    risk_data = compute_porpoising_risk(
        ensemble, scaler_mm, X_train_orig,
        n_grid=80, knn_k=10,
    )
    print(f"  Done ({time.time() - t0:.1f}s)")
    print(f"  DRS=0 risk: mean={risk_data['risk_drs0'].mean():.4f}, max={risk_data['risk_drs0'].max():.4f}")
    print(f"  DRS=1 risk: mean={risk_data['risk_drs1'].mean():.4f}, max={risk_data['risk_drs1'].max():.4f}")

    # ============ 4.5 Compute Stability Surface (for overlay context) ============
    print("\n[4.5] Computing predicted stability surfaces...")
    t0 = time.time()
    stability_surface = compute_stability_surface(
        ensemble, scaler_mm, X_train_orig,
        n_grid=80, knn_k=10,
    )
    print(f"  Done ({time.time() - t0:.1f}s)")

    # ============ 5. Analysis & Plotting ============
    print("\n[5/5] Generating heatmaps and analysis...")
    plot_risk_heatmaps(risk_data, best_solutions, stability_surface, PORPOISING_FIGURES_DIR)
    train_speed_bounds = risk_data.get(
        "train_speed_bounds",
        (scaler_mm.data_min_[0], scaler_mm.data_max_[0]))
    train_wing_bounds = risk_data.get(
        "train_wing_bounds",
        (scaler_mm.data_min_[1], scaler_mm.data_max_[1]))
    print_analysis(risk_data, best_solutions, train_speed_bounds, train_wing_bounds)

    print()
    print("=" * 60)
    print("  T07 Complete")
    print(f"  Output: {PORPOISING_FIGURES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
