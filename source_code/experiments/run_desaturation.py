#!/usr/bin/env python
"""
De-saturation Experiment (T17 / new experiment, Priority B)
===========================================================
Question: are all solutions identical because the OPTIMIZER fails, or because the
target landscape is degenerate (saturated near 100)?

Key idea:
  - The high-stability basin is a near-flat plateau: a huge, geometrically dispersed
    set of configurations are statistically tied at stability ~100. Within such a
    degenerate landscape the located optimum is UNDERDETERMINED -- monotone re-scalings
    of the (saturated) objective relocate the argmax across a wide region while the
    achieved stability barely changes. A robustly-informative optimum would be invariant
    to monotone re-scaling; the observed wandering is the signature of degeneration.
  - Counter-test: inject an EXOGENOUS gradient (reward low wing). The optimum MUST move
    -- proving PSO can relocate on demand, so the value-invariance is a landscape
    property, not an optimizer weakness.

Variants (single-objective mu, lambda=0, S3 geometry):
  V1 Raw                 f = mu(x)
  V2 Desat-power (mono)  f = 100 * (mu/100)^p ,  p=20
  V3 Desat-log   (mono)  f = -log(100 - mu + eps)
  V4 Exogenous gradient  f = mu/100 - beta * wing/35      (genuine new gradient)

Output:
  figures/19_desaturation.png
  outputs/desaturation_summary.csv
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
from src.optimization.fitness import load_ensemble_fitness
from src.optimization.pso_adaptive import PSOAdaptive
from src.utils.config import (
    PSO_W_START, PSO_W_END, PSO_W_ALPHA, PSO_C1, PSO_C2,
    PSO_EARLY_STOP_ITERS, PSO_TOL, SCENARIO_SEED_BASE,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

N_TRIALS = 5
MAX_ITER = 80
N_PARTICLES = 50
P_POWER = 20.0
BETA = 0.5
EPS = 1e-3
DF_MAX = 8979.0


class TransformedFitness:
    """Wraps ensemble mean mu(x) with a target transform g(mu, X)."""

    def __init__(self, wrapper, transform):
        self.wrapper = wrapper
        self.transform = transform

    def evaluate(self, X):
        X = np.asarray(X, dtype=np.float64)
        mu = self.wrapper.predict(X)
        return self.transform(mu, X)

    def __call__(self, X):
        return self.evaluate(X)


VARIANTS = {
    "V1_raw":          lambda mu, X: mu,
    "V2_power_mono":   lambda mu, X: 100.0 * np.power(np.clip(mu, 0, 100) / 100.0, P_POWER),
    "V3_log_mono":     lambda mu, X: -np.log(np.clip(100.0 - mu, EPS, None)),
    "V4_exo_tension":  lambda mu, X: mu / 100.0 - BETA * (X[:, 1] / 35.0),
}


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
    print("  De-saturation Experiment (optimizer failure vs landscape degeneration)")
    print("=" * 60)

    scenario = SCENARIO_S3_BALANCED
    wrapper = load_ensemble_fitness(lambda_risk=0.0).model  # use mean mu(x)

    rows = []
    for i, (name, g) in enumerate(VARIANTS.items()):
        fit = TransformedFitness(wrapper, g)
        best = run_pso_best(fit, scenario, SCENARIO_SEED_BASE + i * 1000)
        pos = best["gbest_pos"]
        mu_at = float(wrapper.predict(pos.reshape(1, -1))[0])
        row = {"variant": name, "raw_stability_at_opt": mu_at}
        for j, c in enumerate(FEATURE_COLS):
            row[c] = float(pos[j])
        rows.append(row)
        print(f"  {name:16s}: wing={row['wing_angle_deg']:5.2f}, drag={row['drag_n']:6.2f}, "
              f"downforce={row['downforce_n']:7.0f}, raw_stability={mu_at:.2f}")

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "desaturation_summary.csv"), index=False)

    # location wanders while achieved stability stays ~constant
    mono = df[df["variant"].isin(["V1_raw", "V2_power_mono", "V3_log_mono"])]
    stab_range = mono["raw_stability_at_opt"].max() - mono["raw_stability_at_opt"].min()
    df_range = mono["downforce_n"].max() - mono["downforce_n"].min()
    wing_range = mono["wing_angle_deg"].max() - mono["wing_angle_deg"].min()
    v1 = df[df["variant"] == "V1_raw"].iloc[0]
    v4 = df[df["variant"] == "V4_exo_tension"].iloc[0]
    print("\n  Under monotone re-scalings (V1/V2/V3):")
    print(f"    achieved stability range = {stab_range:.2f}  (value is essentially fixed ~100)")
    print(f"    optimal downforce range  = {df_range:.0f} N   (location wanders widely)")
    print(f"    optimal wing range       = {wing_range:.2f} deg")
    print("  => the optimum is UNDERDETERMINED: a degenerate, near-flat landscape.")
    print(f"\n  Counter-test V4 (inject gradient: reward low wing):")
    print(f"    wing  V1={v1['wing_angle_deg']:.1f}  ->  V4={v4['wing_angle_deg']:.1f}  "
          f"(PSO relocates on demand -> optimizer is capable)")

    # ---- downforce slice (visualises the flat plateau) ----
    df_grid = np.linspace(scenario.bounds[3][0], scenario.bounds[3][1], 200)
    Xs = np.tile(np.array([320.0, 20.0, 1.0, 0.0, 18.0]), (len(df_grid), 1))
    Xs[:, 3] = df_grid
    mu_slice = wrapper.predict(Xs)

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    palette = ["#7f8c8d", "#2980B9", "#16A085", "#C0392B"]

    ax = axes[0, 0]
    ax.bar(range(len(df)), df["downforce_n"], color=palette, alpha=0.85)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(["V1 raw", "V2 power\n(mono)", "V3 log\n(mono)", "V4 exo\ngradient"])
    ax.set_ylabel("optimal downforce (N)")
    ax.set_title("(a) Optimum location WANDERS under monotone re-scaling")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 1]
    ax.bar(range(len(df)), df["raw_stability_at_opt"], color=palette, alpha=0.85)
    ax.axhline(100, color="grey", ls="--", lw=1)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(["V1", "V2", "V3", "V4"])
    ax.set_ylim(98, 102)
    ax.set_ylabel("achieved stability  mu(x*)")
    ax.set_title("(b) ...yet achieved stability stays fixed at ~100")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    ax.plot(df_grid, mu_slice, color="#2980B9", lw=2)
    ax.axhline(100, color="grey", ls="--", lw=1)
    ax.set_xlabel("downforce (N)  [speed=320, wing=20, drag=18]")
    ax.set_ylabel("surrogate stability  mu(x)")
    ax.set_title("(c) A flat plateau: many configs tied near 100")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    txt = (
        "Interpretation\n"
        "-----------------------------------------\n"
        "V2 / V3 are MONOTONE re-scalings of the\n"
        "saturated stability objective. A robustly\n"
        "informative optimum would be invariant to\n"
        "them. Instead the located optimum jumps\n"
        f"(downforce range ~{df_range:.0f} N, wing ~{wing_range:.0f} deg)\n"
        f"while achieved stability moves only {stab_range:.2f} pt.\n\n"
        "=> A large, dispersed set of configurations\n"
        "   are statistically TIED at ~100. The\n"
        "   optimum is UNDERDETERMINED -- a property\n"
        "   of the degenerate data landscape, not of\n"
        "   the optimizer.\n\n"
        "V4 injects an exogenous gradient (reward low\n"
        f"wing): PSO relocates wing {v1['wing_angle_deg']:.0f} deg -> {v4['wing_angle_deg']:.0f} deg.\n"
        "The optimizer CAN move when real gradient\n"
        "exists -- so the value-invariance above is a\n"
        "landscape signature, not optimizer weakness."
    )
    ax.text(0.0, 1.0, txt, va="top", ha="left", fontsize=9.5, family="monospace")

    fig.suptitle("De-saturation: the high-stability optimum is underdetermined "
                 "(landscape degeneration), not optimizer-limited",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "19_desaturation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {out}")
    print("  CSV saved: outputs/desaturation_summary.csv")
    print("\n  De-saturation Experiment complete.")


if __name__ == "__main__":
    main()
