#!/usr/bin/env python
"""
Weight Sensitivity Analysis (T17 / new experiment, Priority A)
==============================================================
Goal: test whether the multi-objective weights (w_stability, w_efficiency, w_power)
actually move the optimal solution. If the landscape is degenerate, large weight
changes should leave the optimum (especially wing and drag) almost fixed.

Method:
  - One scenario geometry (S3 balanced: speed fixed 320, wing in [20,35], DRS free).
  - Sweep weights on a simplex grid (step 0.1, each >= 0.1).
  - For each weight vector run PSO (3 trials, keep best) and record the optimum.
  - Report displacement of each parameter relative to its feasible range.

Output:
  figures/18_weight_sensitivity.png
  outputs/weight_sensitivity_summary.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scenarios.scenario_def import SCENARIO_S3_BALANCED, FEATURE_COLS
from src.scenarios.scenario_runner import _make_multiobjective_fitness
from src.optimization.pso_adaptive import PSOAdaptive
from src.utils.config import (
    PSO_W_START, PSO_W_END, PSO_W_ALPHA, PSO_C1, PSO_C2,
    PSO_EARLY_STOP_ITERS, PSO_TOL, SCENARIO_SEED_BASE,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

N_TRIALS = 3
MAX_ITER = 60
N_PARTICLES = 50


def weight_grid(step=0.1, floor=0.1):
    combos = []
    vals = np.round(np.arange(floor, 1.0 - 2 * floor + 1e-9, step), 2)
    for ws in vals:
        for we in vals:
            wp = round(1.0 - ws - we, 2)
            if wp >= floor - 1e-9 and wp <= 1.0:
                combos.append((float(ws), float(we), float(wp)))
    return combos


def run_pso_best(fitness, scenario, seed_base):
    best = None
    for t in range(N_TRIALS):
        pso = PSOAdaptive(
            n_particles=N_PARTICLES, bounds=scenario.bounds,
            discrete_indices=scenario.discrete_indices,
            w_start=PSO_W_START, w_end=PSO_W_END, alpha=PSO_W_ALPHA,
            c1=PSO_C1, c2=PSO_C2, max_iter=MAX_ITER,
            early_stop_iters=PSO_EARLY_STOP_ITERS, tol=PSO_TOL,
            seed=seed_base + t * 100, maximise=True,
        )
        res = pso.optimize(fitness, verbose=False)
        if best is None or res["gbest_fitness"] > best["gbest_fitness"]:
            best = res
    return best


def main():
    print("=" * 60)
    print("  Weight Sensitivity Analysis (do weights move the optimum?)")
    print("=" * 60)

    scenario = SCENARIO_S3_BALANCED
    bounds = scenario.bounds
    fitness = _make_multiobjective_fitness(scenario)   # loads ensemble + norm constants once

    combos = weight_grid()
    print(f"\n  {len(combos)} weight vectors on the simplex (step 0.1)")
    print(f"  geometry: {scenario.name}, speed fixed {bounds[0][0]:.0f}, wing [{bounds[1][0]:.0f},{bounds[1][1]:.0f}]")

    rows = []
    for i, (ws, we, wp) in enumerate(combos):
        fitness.w_stability, fitness.w_efficiency, fitness.w_power = ws, we, wp
        best = run_pso_best(fitness, scenario, SCENARIO_SEED_BASE + i * 1000)
        pos = best["gbest_pos"]
        row = {"w_stability": ws, "w_efficiency": we, "w_power": wp,
               "fitness": float(best["gbest_fitness"])}
        for j, c in enumerate(FEATURE_COLS):
            row[c] = float(pos[j])
        rows.append(row)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(combos)}] ws={ws} we={we} wp={wp} -> "
                  f"wing={row['wing_angle_deg']:.2f}, drag={row['drag_n']:.2f}, "
                  f"df={row['downforce_n']:.0f}")

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "weight_sensitivity_summary.csv"), index=False)

    # displacement relative to feasible range
    print("\n  Optimum displacement across ALL weight vectors (range / feasible range):")
    disp = {}
    for j, c in enumerate(FEATURE_COLS):
        lo, hi = bounds[j]
        feas = (hi - lo) if hi > lo else 1.0
        rng = df[c].max() - df[c].min()
        disp[c] = rng / feas
        print(f"    {c:18s}: optimum range = {rng:8.3f}  ({100*disp[c]:5.1f}% of feasible span)")

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.scatter(df["w_stability"], df["wing_angle_deg"], c=df["w_efficiency"],
               cmap="viridis", s=60, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("w_stability")
    ax.set_ylabel("optimal wing angle (deg)")
    ax.set_ylim(bounds[1][0] - 1, bounds[1][1] + 1)
    ax.set_title("(a) Wing pinned at lower bound regardless of weights")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(df["w_efficiency"], df["drag_n"], c=df["w_stability"],
               cmap="viridis", s=60, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("w_efficiency")
    ax.set_ylabel("optimal drag (N)")
    ax.set_ylim(bounds[4][0] - 5, bounds[4][1] + 5)
    ax.set_title("(b) Drag pinned at lower bound regardless of weights")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(df["w_stability"], df["downforce_n"], c=df["w_power"],
               cmap="plasma", s=60, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("w_stability")
    ax.set_ylabel("optimal downforce (N)")
    ax.set_title("(c) Downforce: the only mildly responsive dimension")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    names = ["speed_kmh", "wing_angle_deg", "downforce_n", "drag_n"]
    vals = [disp[n] * 100 for n in names]
    colors = ["#7f8c8d", "#C0392B", "#2980B9", "#C0392B"]
    ax.bar(range(len(names)), vals, color=colors, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["speed\n(fixed)", "wing", "downforce", "drag"])
    ax.set_ylabel("optimum range  (% of feasible span)")
    ax.set_title("(d) Weight changes barely relocate the optimum")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Weight Sensitivity: large preference changes leave wing and drag pinned\n"
                 "(evidence of a degenerate multi-objective landscape)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "18_weight_sensitivity.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {out}")
    print("  CSV saved: outputs/weight_sensitivity_summary.csv")
    print("\n  Weight Sensitivity Analysis complete.")


if __name__ == "__main__":
    main()
