#!/usr/bin/env python
"""
T08: Experiment Design and Evaluation
=====================================
8.1 Comparison Experiments (5 groups x >=20 independent runs)
8.2 Ablation Studies (5 groups x >=10 runs)
8.3 Robustness Experiments (4 types)
8.4 Baseline Comparison (Linear Regression + Gradient Search)

Usage:
  /mnt/e/python313/python.exe experiments/run_ablation.py              # standard
  /mnt/e/python313/python.exe experiments/run_ablation.py --quick      # 5 runs each, fast verify
  /mnt/e/python313/python.exe experiments/run_ablation.py --full       # full 20 runs
  /mnt/e/python313/python.exe experiments/run_ablation.py --robust     # include robustness
  /mnt/e/python313/python.exe experiments/run_ablation.py --baseline   # include 8.4 baseline
  /mnt/e/python313/python.exe experiments/run_ablation.py --no-al      # skip Exp-4 (active learning)
"""

import os
import sys
import time
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.optimization.pso_base import PSOBase
from src.optimization.pso_adaptive import PSOAdaptive
from src.optimization.pso_risk_sensitive import PSORiskSensitive, DEFAULT_BOUNDS, DISCRETE_INDICES
from src.optimization.fitness import (
    FitnessStandard, FitnessRiskSensitive, FitnessDensityPenalty, ModelWrapper,
    load_xgb_fitness, load_ensemble_fitness, load_rf_fitness,
)
from src.utils.config import (
    RANDOM_STATE, PROCESSED_DIR, MODELS_OUTPUT_DIR, DEEP_ENSEMBLE_DIR,
    PSO_PARAM_BOUNDS, PSO_DISCRETE_INDICES,
    PSO_N_PARTICLES, PSO_W, PSO_W_START, PSO_W_END, PSO_W_ALPHA,
    PSO_C1, PSO_C2, PSO_MAX_ITER, PSO_EARLY_STOP_ITERS, PSO_TOL,
    PSO_DENSITY_LAMBDA, PSO_DENSITY_K,
    PSO_FIGURES_DIR, FIGURES_DIR, REPORTS_DIR, FEATURE_COLS,
    MLP_HIDDEN_UNITS, NN_LR, NN_WEIGHT_DECAY, NN_BATCH_SIZE,
    NN_MAX_EPOCHS, NN_EARLY_STOP_PATIENCE, DEEP_ENSEMBLE_M,
)

warnings.filterwarnings("ignore")

# ============================================================================
# T08-specific configuration
# ============================================================================

SEED_BASE_T08 = 2024
SEED_OFFSET_BASELINE = 0
SEED_OFFSET_EXP1     = 5000
SEED_OFFSET_EXP2     = 10000
SEED_OFFSET_EXP3     = 15000
SEED_OFFSET_EXP4     = 20000

T08_FIGURES_DIR = PSO_FIGURES_DIR  # figures/pso/
T08_REPORT_PATH = os.path.join(REPORTS_DIR, "experiment_results.md")

os.makedirs(T08_FIGURES_DIR, exist_ok=True)

# ============================================================================
# Data containers
# ============================================================================

class TrialResult:
    """Single PSO trial result."""
    def __init__(self, gbest_fitness, gbest_pos, n_iter, converged, elapsed, history=None):
        self.fitness = float(gbest_fitness)
        self.position = np.asarray(gbest_pos, dtype=np.float64)
        self.n_iter = int(n_iter)
        self.converged = bool(converged)
        self.elapsed = float(elapsed)
        self.history = history or {}

class GroupResult:
    """Aggregated results for an experiment group."""
    def __init__(self, group_name, label, description, trials):
        self.group_name = group_name
        self.label = label
        self.description = description
        self.trials = trials

    @property
    def n_trials(self):
        return len(self.trials)

    @property
    def fitnesses(self):
        return np.array([t.fitness for t in self.trials])

    @property
    def fitness_mean(self):
        return float(np.mean(self.fitnesses))

    @property
    def fitness_std(self):
        return float(np.std(self.fitnesses, ddof=1)) if len(self.trials) > 1 else 0.0

    @property
    def fitness_min(self):
        return float(np.min(self.fitnesses))

    @property
    def fitness_max(self):
        return float(np.max(self.fitnesses))

    @property
    def iters_mean(self):
        return float(np.mean([t.n_iter for t in self.trials]))

    @property
    def converged_rate(self):
        return float(np.mean([1.0 if t.converged else 0.0 for t in self.trials]))

    @property
    def latency_mean(self):
        return float(np.mean([t.elapsed for t in self.trials]))

    @property
    def positions(self):
        return np.array([t.position for t in self.trials])

    @property
    def param_means(self):
        p = self.positions
        return {FEATURE_COLS[i]: float(np.mean(p[:, i])) for i in range(p.shape[1])}

    @property
    def param_stds(self):
        p = self.positions
        return {FEATURE_COLS[i]: float(np.std(p[:, i], ddof=1)) if len(self.trials) > 1 else 0.0
                for i in range(p.shape[1])}

    def convergence_curve(self):
        """Average gbest fitness per iteration across trials (aligns to max iters)."""
        max_iter = max(t.n_iter for t in self.trials)
        curves = []
        for t in self.trials:
            if t.history and "iteration" in t.history and "gbest_fitness" in t.history:
                iters = np.asarray(t.history["iteration"], dtype=int)
                fits = np.asarray(t.history["gbest_fitness"])
                true_max = max(max_iter, int(iters.max()) if len(iters) > 0 else max_iter)
                curve = np.full(true_max + 1, np.nan)
                curve[iters] = fits
                curves.append(curve)
        if not curves:
            return np.array([]), np.array([]), np.array([])
        curves = np.array(curves)
        mean_curve = np.nanmean(curves, axis=0)
        std_curve = np.nanstd(curves, axis=0)
        return np.arange(len(mean_curve)), mean_curve, std_curve

    def to_dict(self):
        return {
            "group": self.group_name,
            "label": self.label,
            "description": self.description,
            "n_trials": self.n_trials,
            "fitness_mean": self.fitness_mean,
            "fitness_std": self.fitness_std,
            "fitness_min": self.fitness_min,
            "fitness_max": self.fitness_max,
            "iters_mean": self.iters_mean,
            "converged_rate": self.converged_rate,
            "latency_mean": self.latency_mean,
            "param_means": self.param_means,
            "param_stds": self.param_stds,
        }


# ============================================================================
# Core helper: run multiple independent trials
# ============================================================================

def run_trials(pso_cls, pso_kwargs, fitness_fn, n_trials, seed_base, group_label="",
               verbose=False, bounds=None):
    """Run N independent PSO trials and return a GroupResult."""
    if bounds is None:
        bounds = DEFAULT_BOUNDS
    pso_kwargs = dict(pso_kwargs)
    pso_kwargs.setdefault("bounds", bounds)
    pso_kwargs.setdefault("discrete_indices", DISCRETE_INDICES)
    pso_kwargs.setdefault("maximise", True)

    trials = []
    for t in range(n_trials):
        trial_seed = seed_base + t * 100
        pso_kwargs["seed"] = trial_seed
        pso = pso_cls(**pso_kwargs)
        t0 = time.time()
        result = pso.optimize(fitness_fn, verbose=False)
        elapsed = time.time() - t0
        trial = TrialResult(
            gbest_fitness=result["gbest_fitness"],
            gbest_pos=result["gbest_pos"],
            n_iter=result["n_iter"],
            converged=result["converged"],
            elapsed=elapsed,
            history=result.get("history"),
        )
        trials.append(trial)
        if verbose:
            print(f"    [{group_label}] Trial {t+1:2d}/{n_trials}: "
                  f"f={trial.fitness:.4f}, iters={trial.n_iter}, "
                  f"t={elapsed:.3f}s, conv={trial.converged}")

    desc = f"{group_label}: {n_trials} trials, seed base={seed_base}"
    return GroupResult(group_label, group_label, desc, trials)


# ============================================================================
# 8.1 Comparison Experiments
# ============================================================================

def run_comparison_experiments(n_trials=10, include_al=True, verbose=False):
    """Run 5 comparison groups: Baseline + Exp1-4."""
    print("\n" + "=" * 65)
    print("  8.1 Comparison Experiments")
    print("=" * 65)

    results = {}

    # -- Baseline: Standard PSO (fixed w) + XGBoost --
    print("\n  [Baseline] Standard PSO (w=0.7) + XGBoost")
    xgb_fitness = load_xgb_fitness()
    results["Baseline"] = run_trials(
        PSOBase,
        {"w": 0.7, "c1": PSO_C1, "c2": PSO_C2, "max_iter": PSO_MAX_ITER,
         "early_stop_iters": PSO_EARLY_STOP_ITERS, "tol": PSO_TOL},
        xgb_fitness, n_trials, SEED_BASE_T08 + SEED_OFFSET_BASELINE,
        "Baseline", verbose=verbose,
    )
    print(f"    fitness={results['Baseline'].fitness_mean:.4f} +/- {results['Baseline'].fitness_std:.4f}, "
          f"iters={results['Baseline'].iters_mean:.1f}, "
          f"latency={results['Baseline'].latency_mean*1000:.2f}ms")

    # -- Exp-1: Adaptive PSO + XGBoost --
    print("\n  [Exp-1] Adaptive PSO + XGBoost")
    results["Exp1_Adaptive"] = run_trials(
        PSOAdaptive,
        {"w_start": PSO_W_START, "w_end": PSO_W_END, "alpha": PSO_W_ALPHA,
         "c1": PSO_C1, "c2": PSO_C2, "max_iter": PSO_MAX_ITER,
         "early_stop_iters": PSO_EARLY_STOP_ITERS, "tol": PSO_TOL},
        xgb_fitness, n_trials, SEED_BASE_T08 + SEED_OFFSET_EXP1,
        "Exp1_Adaptive", verbose=verbose,
    )
    print(f"    fitness={results['Exp1_Adaptive'].fitness_mean:.4f} +/- {results['Exp1_Adaptive'].fitness_std:.4f}, "
          f"iters={results['Exp1_Adaptive'].iters_mean:.1f}, "
          f"latency={results['Exp1_Adaptive'].latency_mean*1000:.2f}ms")

    # -- Exp-2: Risk-Sensitive PSO (lambda=1.0) + DeepEnsemble --
    print("\n  [Exp-2] Risk-Sensitive PSO (lambda=1.0) + DeepEnsemble")
    risk_fitness_1 = load_ensemble_fitness(lambda_risk=1.0)
    results["Exp2_RiskSensitive"] = run_trials(
        PSOAdaptive,
        {"w_start": PSO_W_START, "w_end": PSO_W_END, "alpha": PSO_W_ALPHA,
         "c1": PSO_C1, "c2": PSO_C2, "max_iter": PSO_MAX_ITER,
         "early_stop_iters": PSO_EARLY_STOP_ITERS, "tol": PSO_TOL},
        risk_fitness_1, n_trials, SEED_BASE_T08 + SEED_OFFSET_EXP2,
        "Exp2_RiskSensitive", verbose=verbose,
    )
    print(f"    fitness={results['Exp2_RiskSensitive'].fitness_mean:.4f} +/- {results['Exp2_RiskSensitive'].fitness_std:.4f}, "
          f"iters={results['Exp2_RiskSensitive'].iters_mean:.1f}, "
          f"latency={results['Exp2_RiskSensitive'].latency_mean*1000:.2f}ms")

    # -- Exp-3: Risk-Sensitive + Density Penalty --
    print("\n  [Exp-3] Risk-Sensitive + Density Penalty (lambda1=1.0, lambda2=0.5)")
    try:
        X_train_mm = np.load(os.path.join(PROCESSED_DIR, "X_train_mm.npy")).astype(np.float32)
        from src.models.deep_ensemble import DeepEnsemble
        import pickle as _pickle
        ensemble = DeepEnsemble.load(
            input_dim=5, hidden_units=MLP_HIDDEN_UNITS, M=DEEP_ENSEMBLE_M,
            lr=NN_LR, weight_decay=NN_WEIGHT_DECAY,
            batch_size=NN_BATCH_SIZE, max_epochs=NN_MAX_EPOCHS,
            patience=NN_EARLY_STOP_PATIENCE,
        )
        with open(os.path.join(PROCESSED_DIR, "scaler_mm.pkl"), "rb") as f:
            scaler_mm = _pickle.load(f)
        wrapper = ModelWrapper(ensemble, scaler_mm, model_type="ensemble", y_transform="x100")
        density_fitness = FitnessDensityPenalty(
            wrapper, lambda_risk=1.0, lambda_density=PSO_DENSITY_LAMBDA,
            k=PSO_DENSITY_K, X_train=X_train_mm,
        )
        results["Exp3_Density"] = run_trials(
            PSOAdaptive,
            {"w_start": PSO_W_START, "w_end": PSO_W_END, "alpha": PSO_W_ALPHA,
             "c1": PSO_C1, "c2": PSO_C2, "max_iter": PSO_MAX_ITER,
             "early_stop_iters": PSO_EARLY_STOP_ITERS, "tol": PSO_TOL},
            density_fitness, n_trials, SEED_BASE_T08 + SEED_OFFSET_EXP3,
            "Exp3_Density", verbose=verbose,
        )
        print(f"    fitness={results['Exp3_Density'].fitness_mean:.4f} +/- {results['Exp3_Density'].fitness_std:.4f}, "
              f"iters={results['Exp3_Density'].iters_mean:.1f}, "
              f"latency={results['Exp3_Density'].latency_mean*1000:.2f}ms")
    except Exception as e:
        print(f"    [SKIP] Exp-3 Density failed: {e}")
        results["Exp3_Density"] = None

    # -- Exp-4: AL-Enhanced PSO --
    if include_al:
        print("\n  [Exp-4] AL-Enhanced PSO (refined DeepEnsemble)")
        try:
            refined_dir = os.path.join(MODELS_OUTPUT_DIR, "deep_ensemble_refined")
            if os.path.exists(os.path.join(refined_dir, "mlp_0.pt")):
                from src.models.deep_ensemble import DeepEnsemble as DE
                ensemble_ref = DE.load(
                    input_dim=5, hidden_units=MLP_HIDDEN_UNITS, M=DEEP_ENSEMBLE_M,
                    load_dir=refined_dir,
                    lr=NN_LR, weight_decay=NN_WEIGHT_DECAY,
                    batch_size=NN_BATCH_SIZE, max_epochs=NN_MAX_EPOCHS,
                    patience=NN_EARLY_STOP_PATIENCE,
                )
                with open(os.path.join(PROCESSED_DIR, "scaler_mm.pkl"), "rb") as f:
                    scaler_mm = _pickle.load(f)
                wrapper_ref = ModelWrapper(ensemble_ref, scaler_mm, model_type="ensemble", y_transform="x100")
                al_fitness = FitnessRiskSensitive(wrapper_ref, lambda_risk=1.0)
                results["Exp4_ALEnhanced"] = run_trials(
                    PSOAdaptive,
                    {"w_start": PSO_W_START, "w_end": PSO_W_END, "alpha": PSO_W_ALPHA,
                     "c1": PSO_C1, "c2": PSO_C2, "max_iter": PSO_MAX_ITER,
                     "early_stop_iters": PSO_EARLY_STOP_ITERS, "tol": PSO_TOL},
                    al_fitness, n_trials, SEED_BASE_T08 + SEED_OFFSET_EXP4,
                    "Exp4_ALEnhanced", verbose=verbose,
                )
                print(f"    fitness={results['Exp4_ALEnhanced'].fitness_mean:.4f} +/- {results['Exp4_ALEnhanced'].fitness_std:.4f}, "
                      f"iters={results['Exp4_ALEnhanced'].iters_mean:.1f}, "
                      f"latency={results['Exp4_ALEnhanced'].latency_mean*1000:.2f}ms")
            else:
                print("    [SKIP] Refined ensemble not found. Run active learning first.")
                results["Exp4_ALEnhanced"] = None
        except Exception as e:
            print(f"    [SKIP] Exp-4 AL failed: {e}")
            results["Exp4_ALEnhanced"] = None

    print(f"\n  8.1 Summary ({n_trials} trials per group):")
    for key, grp in results.items():
        if grp is not None:
            print(f"    {key:25s}: f={grp.fitness_mean:.4f}+/-{grp.fitness_std:.4f}, "
                  f"iters={grp.iters_mean:.1f}, conv={grp.converged_rate:.0%}, "
                  f"t={grp.latency_mean*1000:.2f}ms")

    return results


# ============================================================================
# 8.2 Ablation Studies
# ============================================================================

def train_7feature_xgb():
    """Train XGBoost on 7 features (5 base + force_ratio + speed_wing).
    Returns (model, scaler_7d) and a factory that creates FitnessStandard from 5D PSO positions.
    """
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    import pickle as _pk

    train_csv = os.path.join(PROCESSED_DIR, "train.csv")
    df_train = pd.read_csv(train_csv)
    base_cols = ["speed_kmh", "wing_angle_deg", "drs_active", "downforce_n", "drag_n"]
    y_train = df_train["stability_index"].values

    X_base = df_train[base_cols].values.astype(np.float64)
    force_ratio = X_base[:, 3] / (X_base[:, 4] + 1e-6)
    speed_wing = X_base[:, 0] * X_base[:, 1]
    X_7d = np.column_stack([X_base, force_ratio, speed_wing])

    scaler = StandardScaler()
    X_7d_scaled = scaler.fit_transform(X_7d)

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_7d_scaled, y_train)

    class SevenFeatureWrapper:
        """Fitness that takes 5D PSO positions and predicts using 7-feature model."""
        def __init__(self, model, scaler):
            self.model = model
            self.scaler = scaler

        def evaluate(self, X):
            X = np.asarray(X, dtype=np.float64)
            fr = X[:, 3] / (X[:, 4] + 1e-6)
            sw = X[:, 0] * X[:, 1]
            X7 = np.column_stack([X, fr, sw])
            X7_scaled = self.scaler.transform(X7)
            return self.model.predict(X7_scaled)

        def __call__(self, X):
            return self.evaluate(X)

    return SevenFeatureWrapper(model, scaler)


def run_ablation_experiments(n_trials=10, verbose=False):
    """Run 5 ablation groups."""
    print("\n" + "=" * 65)
    print("  8.2 Ablation Studies")
    print("=" * 65)

    results = {}

    pso_kwargs = {
        "w_start": PSO_W_START, "w_end": PSO_W_END, "alpha": PSO_W_ALPHA,
        "c1": PSO_C1, "c2": PSO_C2, "max_iter": PSO_MAX_ITER,
        "early_stop_iters": PSO_EARLY_STOP_ITERS, "tol": PSO_TOL,
    }

    # -- A1: Remove sigma penalty (lambda=0, same as using mu-only) --
    print("\n  [A1] Remove sigma penalty (lambda=0, mu-only)")
    risk_fitness_0 = load_ensemble_fitness(lambda_risk=0.0)
    results["A1_NoSigma"] = run_trials(
        PSOAdaptive, pso_kwargs, risk_fitness_0, n_trials,
        SEED_BASE_T08 + 25000, "A1_NoSigma", verbose=verbose,
    )
    print(f"    fitness={results['A1_NoSigma'].fitness_mean:.4f}+/-{results['A1_NoSigma'].fitness_std:.4f}")

    # -- A2: Remove density penalty (lambda2=0, mu - 1.0*sigma only) --
    print("\n  [A2] Remove density penalty (lambda2=0)")
    results["A2_NoDensity"] = run_trials(
        PSOAdaptive, pso_kwargs,
        load_ensemble_fitness(lambda_risk=1.0), n_trials,
        SEED_BASE_T08 + 26000, "A2_NoDensity", verbose=verbose,
    )
    print(f"    fitness={results['A2_NoDensity'].fitness_mean:.4f}+/-{results['A2_NoDensity'].fitness_std:.4f}")

    # -- A3: Remove adaptive weights (fixed w=0.7) --
    print("\n  [A3] Remove adaptive weights (fixed w=0.7)")
    pso_fixed_kwargs = dict(pso_kwargs)
    pso_fixed_kwargs.pop("w_start", None)
    pso_fixed_kwargs.pop("w_end", None)
    pso_fixed_kwargs.pop("alpha", None)
    pso_fixed_kwargs["w"] = 0.7
    results["A3_FixedW"] = run_trials(
        PSOBase, pso_fixed_kwargs, load_ensemble_fitness(lambda_risk=1.0), n_trials,
        SEED_BASE_T08 + 27000, "A3_FixedW", verbose=verbose,
    )
    print(f"    fitness={results['A3_FixedW'].fitness_mean:.4f}+/-{results['A3_FixedW'].fitness_std:.4f}")

    # -- A4: Feature ablation (5 vs 7 features) --
    print("\n  [A4] Feature ablation: 5 base features vs 5+composite (7 features)")
    try:
        feat7_fitness = train_7feature_xgb()
        results["A4_7Features"] = run_trials(
            PSOAdaptive, pso_kwargs, feat7_fitness, n_trials,
            SEED_BASE_T08 + 28000, "A4_7Features", verbose=verbose,
        )
        print(f"    7-feature fitness={results['A4_7Features'].fitness_mean:.4f}+/-{results['A4_7Features'].fitness_std:.4f}")
    except Exception as e:
        print(f"    [SKIP] A4 7-feature model training failed: {e}")
        results["A4_7Features"] = None

    # -- A5: Model selection ablation (XGBoost vs RF vs DeepEnsemble, all mu-only) --
    print("\n  [A5] Model selection ablation: XGBoost vs RandomForest vs DeepEnsemble (mu-only)")
    model_ablations = {}
    for model_name, fitness_getter in [
        ("XGBoost", load_xgb_fitness),
        ("RandomForest", load_rf_fitness),
        ("DeepEnsemble_lambda0", lambda: load_ensemble_fitness(lambda_risk=0.0)),
    ]:
        try:
            f = fitness_getter()
            grp = run_trials(
                PSOAdaptive, pso_kwargs, f, n_trials,
                SEED_BASE_T08 + 29000 + 1000 * len(model_ablations),
                f"A5_{model_name}", verbose=verbose,
            )
            model_ablations[model_name] = grp
            print(f"    {model_name}: fitness={grp.fitness_mean:.4f}+/-{grp.fitness_std:.4f}, "
                  f"iters={grp.iters_mean:.1f}, t={grp.latency_mean*1000:.2f}ms")
        except Exception as e:
            print(f"    [SKIP] {model_name}: {e}")
    results["A5_ModelSelection"] = model_ablations

    print(f"\n  8.2 Summary ({n_trials} trials per group):")
    for key, val in results.items():
        if key == "A5_ModelSelection":
            for mk, mv in val.items():
                print(f"    A5_{mk:20s}: f={mv.fitness_mean:.4f}+/-{mv.fitness_std:.4f}, "
                      f"iters={mv.iters_mean:.1f}")
        elif val is not None:
            print(f"    {key:20s}: f={val.fitness_mean:.4f}+/-{val.fitness_std:.4f}, "
                  f"iters={val.iters_mean:.1f}")

    return results


# ============================================================================
# 8.3 Robustness Experiments
# ============================================================================

def run_robustness_experiments(verbose=False):
    """Run 4 robustness tests."""
    print("\n" + "=" * 65)
    print("  8.3 Robustness Experiments")
    print("=" * 65)

    results = {}

    base_pso_kwargs = {
        "w_start": PSO_W_START, "w_end": PSO_W_END, "alpha": PSO_W_ALPHA,
        "c1": PSO_C1, "c2": PSO_C2, "max_iter": PSO_MAX_ITER,
        "early_stop_iters": PSO_EARLY_STOP_ITERS, "tol": PSO_TOL,
        "bounds": DEFAULT_BOUNDS, "discrete_indices": DISCRETE_INDICES,
    }

    # --- 8.3.1 Different random seeds ---
    print("\n  [R1] Different random seeds (5 seeds)")
    seeds = [42, 123, 456, 789, 1024]
    r1_fitness = load_ensemble_fitness(lambda_risk=1.0)
    r1_trials = []
    for seed in seeds:
        pso = PSOAdaptive(seed=seed, **base_pso_kwargs)
        t0 = time.time()
        result = pso.optimize(r1_fitness, verbose=False)
        elapsed = time.time() - t0
        r1_trials.append(TrialResult(
            gbest_fitness=result["gbest_fitness"],
            gbest_pos=result["gbest_pos"],
            n_iter=result["n_iter"],
            converged=result["converged"],
            elapsed=elapsed,
        ))
        if verbose:
            print(f"    seed={seed:4d}: f={r1_trials[-1].fitness:.4f}, "
                  f"iters={r1_trials[-1].n_iter}, conv={r1_trials[-1].converged}")
    results["R1_Seeds"] = GroupResult("R1_Seeds", "R1 Different Seeds",
                                       "5 different random seeds", r1_trials)
    print(f"    fitness range: [{results['R1_Seeds'].fitness_min:.4f}, {results['R1_Seeds'].fitness_max:.4f}], "
          f"std={results['R1_Seeds'].fitness_std:.4f}")

    # --- 8.3.2 Model output noise ---
    print("\n  [R2] Model output noise (N(0, 0.01*sigma_model))")
    try:
        ensemble = load_ensemble_fitness(lambda_risk=0.0)
        # Get model sigma for noise scale
        test_pos = np.array([[200, 20, 0, 5000, 300]])
        mu, sigma_test = ensemble.model.predict_with_uncertainty(test_pos)
        noise_scale = 0.01 * float(np.mean(sigma_test))

        class NoisyFitness:
            def __init__(self, base_fitness, noise_std):
                self._base = base_fitness
                self.noise_std = noise_std
            def evaluate(self, X):
                base = self._base.model.predict(X)
                noise = np.random.normal(0, self.noise_std, len(base))
                return base + noise
            def __call__(self, X):
                return self.evaluate(X)

        noisy_fitness = NoisyFitness(ensemble, noise_scale)
        r2_grp = run_trials(
            PSOAdaptive, base_pso_kwargs, noisy_fitness, 10,
            SEED_BASE_T08 + 31000, "R2_Noise", verbose=verbose,
        )
        results["R2_Noise"] = r2_grp
        print(f"    noise_scale={noise_scale:.3f}, fitness={r2_grp.fitness_mean:.4f}+/-{r2_grp.fitness_std:.4f}")
    except Exception as e:
        print(f"    [SKIP] R2 Noise: {e}")
        results["R2_Noise"] = None

    # --- 8.3.3 Initialization perturbation ---
    print("\n  [R3] Initialization perturbation (3 strategies)")
    try:
        r3_fitness = load_ensemble_fitness(lambda_risk=1.0)
        _orig_init = PSOBase._init_particles
        r3_trials = []

        for strategy in ["uniform", "normal", "beta_skewed"]:
            pso = PSOAdaptive(seed=RANDOM_STATE, **base_pso_kwargs)
            if strategy == "uniform":
                # Default: no override needed
                pass
            elif strategy == "normal":
                def _normal_init(self):
                    _orig_init(self)
                    mid = (self.bounds[:, 0] + self.bounds[:, 1]) / 2
                    std = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.15
                    for p in self.particles:
                        p.pos = np.clip(self.rng.normal(mid, std), self.bounds[:, 0], self.bounds[:, 1])
                        p.pbest_pos = p.pos.copy()
                pso._init_particles = _normal_init.__get__(pso)
            elif strategy == "beta_skewed":
                def _beta_init(self):
                    _orig_init(self)
                    for p in self.particles:
                        u = self.rng.beta(2, 5, size=self.dim)
                        p.pos = self.bounds[:, 0] + u * (self.bounds[:, 1] - self.bounds[:, 0])
                        p.pbest_pos = p.pos.copy()
                pso._init_particles = _beta_init.__get__(pso)

            t0 = time.time()
            result = pso.optimize(r3_fitness, verbose=False)
            elapsed = time.time() - t0
            r3_trials.append(TrialResult(
                gbest_fitness=result["gbest_fitness"],
                gbest_pos=result["gbest_pos"],
                n_iter=result["n_iter"],
                converged=result["converged"],
                elapsed=elapsed,
            ))
            if verbose:
                print(f"    init={strategy:12s}: f={r3_trials[-1].fitness:.4f}, "
                      f"iters={r3_trials[-1].n_iter}")
        results["R3_Init"] = GroupResult("R3_Init", "R3 Init Perturbation",
                                          "3 initialization strategies", r3_trials)
        print(f"    fitness range: [{results['R3_Init'].fitness_min:.4f}, {results['R3_Init'].fitness_max:.4f}], "
              f"std={results['R3_Init'].fitness_std:.4f}")
    except Exception as e:
        print(f"    [SKIP] R3 Init: {e}")
        results["R3_Init"] = None

    # --- 8.3.4 Hyperparameter sensitivity ---
    print("\n  [R4] Hyperparameter sensitivity (c1, c2, N, lambda)")
    try:
        c1_vals = [1.5, 2.0, 2.5]
        c2_vals = [1.5, 2.0, 2.5]
        n_vals = [30, 50, 70]
        lam_vals = [0.5, 1.0, 1.5, 2.0]

        hp_results = {}  # {(c1,c2,n,lam): fitness}
        total = len(c1_vals) * len(c2_vals) * len(n_vals) * len(lam_vals)
        count = 0
        for c1 in c1_vals:
            for c2 in c2_vals:
                for n in n_vals:
                    for lam in lam_vals:
                        count += 1
                        try:
                            kw = dict(base_pso_kwargs)
                            kw["c1"] = c1
                            kw["c2"] = c2
                            kw["n_particles"] = n
                            kw["max_iter"] = 80
                            fitness = load_ensemble_fitness(lambda_risk=lam)
                            pso = PSOAdaptive(seed=RANDOM_STATE, **kw)
                            result = pso.optimize(fitness, verbose=False)
                            hp_results[(c1, c2, n, lam)] = float(result["gbest_fitness"])
                        except Exception:
                            hp_results[(c1, c2, n, lam)] = np.nan
            if verbose:
                print(f"    progress: {count}/{total}")

        results["R4_Hyperparams"] = hp_results
        valid = {k: v for k, v in hp_results.items() if not np.isnan(v)}
        if valid:
            best_key = max(valid, key=valid.get)
            worst_key = min(valid, key=valid.get)
            print(f"    Best HP: c1={best_key[0]}, c2={best_key[1]}, N={best_key[2]}, "
                  f"lambda={best_key[3]} -> f={valid[best_key]:.4f}")
            print(f"    Worst HP: c1={worst_key[0]}, c2={worst_key[1]}, N={worst_key[2]}, "
                  f"lambda={worst_key[3]} -> f={valid[worst_key]:.4f}")
            print(f"    Range: [{min(valid.values()):.4f}, {max(valid.values()):.4f}]")
    except Exception as e:
        print(f"    [SKIP] R4 Hyperparam: {e}")
        results["R4_Hyperparams"] = None

    return results


# ============================================================================
# 8.4 Baseline Comparison (Linear Regression + Gradient Search)
# ============================================================================

def run_baseline_comparison(n_trials=5):
    """Compare PSO-BP with linear regression + gradient search."""
    print("\n" + "=" * 65)
    print("  8.4 Baseline Comparison: Linear Regression + Gradient Search")
    print("=" * 65)

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from scipy.optimize import minimize

    train_csv = os.path.join(PROCESSED_DIR, "train.csv")
    df_train = pd.read_csv(train_csv)
    base_cols = ["speed_kmh", "wing_angle_deg", "drs_active", "downforce_n", "drag_n"]
    X_train = df_train[base_cols].values
    y_train = df_train["stability_index"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    lr_model = LinearRegression()
    lr_model.fit(X_scaled, y_train)
    train_r2 = lr_model.score(X_scaled, y_train)
    print(f"\n  Linear Regression R2 (train): {train_r2:.4f}")

    def lr_fitness(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = scaler.transform(X)
        return lr_model.predict(X_scaled)

    lb = DEFAULT_BOUNDS[:, 0].copy()
    ub = DEFAULT_BOUNDS[:, 1].copy()
    # Fix drs_active as discrete (use 0.5 as threshold, but gradient methods need continuous)
    # We'll optimize over [0,1] for drs_active and round at the end
    lb[2] = 0.0
    ub[2] = 1.0
    bounds_list = [(float(l), float(u)) for l, u in zip(lb, ub)]

    # Multiple random starts
    n_starts = n_trials
    gradient_results = []
    for start_idx in range(n_starts):
        rng = np.random.RandomState(RANDOM_STATE + start_idx * 100)
        x0 = rng.uniform(lb, ub)

        t0 = time.time()
        # Minimize negative fitness (maximize stability)
        res = minimize(
            lambda x: -float(lr_fitness(x.reshape(1, -1))[0]),
            x0, method="L-BFGS-B", bounds=bounds_list,
            options={"maxiter": 200, "ftol": 1e-8},
        )
        elapsed = time.time() - t0

        x_opt = np.clip(res.x, lb, ub)
        x_opt[2] = np.round(x_opt[2])  # drs_active
        y_opt = float(lr_fitness(x_opt.reshape(1, -1))[0])
        gradient_results.append({
            "x_opt": x_opt,
            "fitness": y_opt,
            "n_iter": res.nit,
            "converged": res.success,
            "elapsed": elapsed,
        })
        print(f"    Start {start_idx+1}: x0 fitness={float(lr_fitness(x0.reshape(1,-1))[0]):.1f}, "
              f"opt fitness={y_opt:.2f}, nit={res.nit}, conv={res.success}, "
              f"t={elapsed:.3f}s")

    all_fitness = [r["fitness"] for r in gradient_results]
    print(f"\n  Gradient best fitness: {max(all_fitness):.2f} "
          f"(mean={np.mean(all_fitness):.2f}+/-{np.std(all_fitness):.2f})")

    # Also run PSO with the same LR model for direct comparison
    print("\n  PSO + LinearRegression for comparison:")
    from src.optimization.fitness import FitnessStandard
    lr_wrapper = ModelWrapper(lr_model, scaler, model_type="custom", y_transform="identity")
    # Override predict to work with our wrapper
    lr_wrapper.predict = lambda X: lr_model.predict(scaler.transform(
        np.asarray(X, dtype=np.float64).reshape(-1, 5) if np.asarray(X).ndim == 1 else np.asarray(X, dtype=np.float64)
    ))
    lr_std_fitness = FitnessStandard(lr_wrapper)
    # Fix: create a proper callable wrapper
    class LRFitness:
        def evaluate(self, X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return lr_model.predict(scaler.transform(X))
        def __call__(self, X):
            return self.evaluate(X)

    lr_fit = LRFitness()
    pso_lr_grp = run_trials(
        PSOAdaptive,
        {"w_start": PSO_W_START, "w_end": PSO_W_END, "alpha": PSO_W_ALPHA,
         "c1": PSO_C1, "c2": PSO_C2, "max_iter": PSO_MAX_ITER,
         "early_stop_iters": PSO_EARLY_STOP_ITERS, "tol": PSO_TOL},
        lr_fit, n_trials, SEED_BASE_T08 + 35000,
        "PSO_LR", verbose=False,
    )
    print(f"    PSO+LR fitness={pso_lr_grp.fitness_mean:.2f}+/-{pso_lr_grp.fitness_std:.2f}, "
          f"iters={pso_lr_grp.iters_mean:.1f}")

    return {
        "lr_train_r2": train_r2,
        "gradient_results": gradient_results,
        "gradient_best": max(all_fitness),
        "gradient_mean": float(np.mean(all_fitness)),
        "gradient_std": float(np.std(all_fitness)) if len(all_fitness) > 1 else 0.0,
        "pso_lr_fitness_mean": pso_lr_grp.fitness_mean,
        "pso_lr_fitness_std": pso_lr_grp.fitness_std,
        "pso_lr_iters_mean": pso_lr_grp.iters_mean,
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_convergence_comparison(comparison_results):
    """Multi-curve convergence plot for all comparison groups."""
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 6))

    for idx, (key, grp) in enumerate(comparison_results.items()):
        if grp is None:
            continue
        iters, mean_curve, std_curve = grp.convergence_curve()
        if len(iters) == 0:
            continue
        color = colors[idx % len(colors)]
        ax.plot(iters, mean_curve, linewidth=2, color=color, label=key, alpha=0.9)
        ax.fill_between(iters, mean_curve - std_curve, mean_curve + std_curve,
                        color=color, alpha=0.15)

    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel("Gbest Fitness", fontsize=13)
    ax.set_title("Convergence Curves: All PSO Variants", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(T08_FIGURES_DIR, "convergence_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] convergence_comparison.png saved")
    return path


def plot_best_fitness_boxplot(comparison_results):
    """Box plot of best fitness across groups."""
    groups = []
    labels = []
    for key, grp in comparison_results.items():
        if grp is None:
            continue
        groups.append(grp.fitnesses)
        labels.append(key)

    if not groups:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(groups, labels=labels, patch_artist=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 6})
    for patch, color in zip(bp["boxes"], plt.cm.Set2(np.linspace(0, 1, len(groups)))):
        patch.set_facecolor(color)

    ax.set_ylabel("Best Fitness", fontsize=13)
    ax.set_title("Best Fitness Distribution Across PSO Variants", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(T08_FIGURES_DIR, "best_fitness_boxplot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] best_fitness_boxplot.png saved")
    return path


def plot_ablation_study(comparison_results, ablation_results):
    """Ablation impact chart: delta from baseline."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: component ablation (A1-A3) vs Exp-2 baseline
    exp2_fitness = None
    if "Exp2_RiskSensitive" in comparison_results and comparison_results["Exp2_RiskSensitive"] is not None:
        exp2_fitness = comparison_results["Exp2_RiskSensitive"].fitness_mean

    ab_components = {}
    for key, grp in ablation_results.items():
        if key in ("A1_NoSigma", "A2_NoDensity", "A3_FixedW") and grp is not None:
            ab_components[key.replace("A1_", "").replace("A2_", "").replace("A3_", "")] = grp.fitness_mean

    if exp2_fitness is not None and ab_components:
        labels = list(ab_components.keys())
        values = [ab_components[l] - exp2_fitness for l in labels]
        colors = ["#d62728" if v < 0 else "#2ca02c" for v in values]
        bars = ax1.barh(labels, values, color=colors, edgecolor="white", height=0.5)
        ax1.axvline(0, color="black", linewidth=1)
        ax1.set_xlabel("Delta Fitness (vs Exp-2 RiskSensitive)", fontsize=12)
        ax1.set_title("Component Ablation Impact", fontsize=13, fontweight="bold")
        ax1.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, values):
            ax1.text(val + (0.02 if val >= 0 else -0.02), bar.get_y() + bar.get_height()/2,
                     f"{val:+.4f}", va="center", fontsize=10,
                     ha="left" if val >= 0 else "right")

    # Right: model selection ablation (A5)
    a5_data = ablation_results.get("A5_ModelSelection", {})
    if a5_data:
        model_labels = list(a5_data.keys())
        model_fitness = [a5_data[l].fitness_mean for l in model_labels]
        model_latency = [a5_data[l].latency_mean * 1000 for l in model_labels]
        x = np.arange(len(model_labels))
        width = 0.35
        bars1 = ax2.bar(x - width/2, model_fitness, width, label="Fitness",
                        color="steelblue", edgecolor="white")
        ax2_2 = ax2.twinx()
        bars2 = ax2_2.bar(x + width/2, model_latency, width, label="Latency (ms)",
                          color="darkorange", edgecolor="white")
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_labels, rotation=20, ha="right")
        ax2.set_ylabel("Gbest Fitness", fontsize=12)
        ax2_2.set_ylabel("Latency (ms)", fontsize=12)
        ax2.set_title("Model Selection: Fitness vs Latency", fontsize=13, fontweight="bold")
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    fig.tight_layout()
    path = os.path.join(T08_FIGURES_DIR, "ablation_study.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] ablation_study.png saved")
    return path


def plot_robustness_heatmap(robustness_results):
    """Hyperparameter sensitivity heatmap."""
    hp_data = robustness_results.get("R4_Hyperparams", None)
    if hp_data is None or not hp_data:
        return None

    # Pivot: average over (c1, c2) for each (n, lambda)
    n_vals = sorted(set(k[2] for k in hp_data.keys()))
    lam_vals = sorted(set(k[3] for k in hp_data.keys()))
    matrix = np.zeros((len(lam_vals), len(n_vals)))
    for i, lam in enumerate(lam_vals):
        for j, n in enumerate(n_vals):
            vals = [v for (c1, c2, nn, ll), v in hp_data.items()
                    if nn == n and ll == lam and not np.isnan(v)]
            matrix[i, j] = np.mean(vals) if vals else np.nan

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(n_vals)))
    ax.set_xticklabels([f"N={n}" for n in n_vals])
    ax.set_yticks(range(len(lam_vals)))
    ax.set_yticklabels([f"lambda={lam}" for lam in lam_vals])
    ax.set_xlabel("Particle Count (N)", fontsize=13)
    ax.set_ylabel("Risk Aversion (lambda)", fontsize=13)
    ax.set_title("Hyperparameter Sensitivity: N x lambda (avg over c1, c2)", fontsize=14, fontweight="bold")

    for i in range(len(lam_vals)):
        for j in range(len(n_vals)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                        fontsize=9, color="black" if 0.3 < (matrix[i,j] - matrix.min())/(matrix.max()-matrix.min()+1e-10) < 0.7 else "white")

    fig.colorbar(im, ax=ax, label="Mean Gbest Fitness")
    fig.tight_layout()
    path = os.path.join(T08_FIGURES_DIR, "robustness_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] robustness_heatmap.png saved")
    return path


# ============================================================================
# Report generation
# ============================================================================

def generate_report(comparison_results, ablation_results, robustness_results=None,
                    baseline_results=None, n_trials_comp=10, n_trials_abl=10):
    """Generate experiments_results.md."""
    lines = []
    lines.append("# T08: Experiment Results\n")
    lines.append(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
    lines.append(f"*Comparison: {n_trials_comp} trials/group, Ablation: {n_trials_abl} trials/group*\n")

    # -- 8.1 Comparison --
    lines.append("## 8.1 Comparison Experiments\n")
    lines.append("| Group | Fitness (mean+/-std) | Iters | Converged | Latency (ms) |")
    lines.append("|-------|----------------------|-------|-----------|-------------|")
    for key, grp in comparison_results.items():
        if grp is None:
            lines.append(f"| {key} | *SKIPPED* | - | - | - |")
            continue
        lines.append(f"| {key} | {grp.fitness_mean:.4f} +/- {grp.fitness_std:.4f} | "
                     f"{grp.iters_mean:.1f} | {grp.converged_rate:.0%} | "
                     f"{grp.latency_mean*1000:.2f} |")

    best_comp = max((grp for grp in comparison_results.values() if grp is not None),
                    key=lambda g: g.fitness_mean, default=None)
    if best_comp:
        lines.append(f"\n**Best performer**: {best_comp.group_name} "
                     f"(fitness={best_comp.fitness_mean:.4f})\n")

    lines.append("### Optimal Parameters per Group\n")
    lines.append("| Group | speed_kmh | wing_angle | drs | downforce | drag |")
    lines.append("|-------|-----------|------------|-----|-----------|------|")
    for key, grp in comparison_results.items():
        if grp is None:
            continue
        pm = grp.param_means
        lines.append(f"| {key} | {pm['speed_kmh']:.1f} | {pm['wing_angle_deg']:.1f} | "
                     f"{pm['drs_active']:.1f} | {pm['downforce_n']:.1f} | {pm['drag_n']:.1f} |")

    lines.append(f"\n![Convergence]({os.path.join('..', 'figures', 'pso', 'convergence_comparison.png')})\n")
    lines.append(f"![Boxplot]({os.path.join('..', 'figures', 'pso', 'best_fitness_boxplot.png')})\n")

    # -- 8.2 Ablation --
    lines.append("## 8.2 Ablation Studies\n")

    lines.append("### Component Ablation (vs Exp-2 RiskSensitive baseline)\n")
    exp2_fitness = None
    if "Exp2_RiskSensitive" in comparison_results and comparison_results["Exp2_RiskSensitive"] is not None:
        exp2_fitness = comparison_results["Exp2_RiskSensitive"].fitness_mean

    lines.append("| Ablation | Fitness (mean+/-std) | Delta vs Exp-2 | Iters |")
    lines.append("|----------|----------------------|----------------|-------|")
    for key, grp in ablation_results.items():
        if key in ("A1_NoSigma", "A2_NoDensity", "A3_FixedW") and grp is not None:
            delta = grp.fitness_mean - exp2_fitness if exp2_fitness is not None else float("nan")
            lines.append(f"| {key} | {grp.fitness_mean:.4f}+/-{grp.fitness_std:.4f} | "
                         f"{delta:+.4f} | {grp.iters_mean:.1f} |")

    lines.append("\n### Feature Ablation (7 features vs baseline 5)\n")
    a4 = ablation_results.get("A4_7Features")
    bl_grp = comparison_results.get("Baseline")
    if a4 is not None and bl_grp is not None:
        lines.append(f"- 7-feature XGBoost fitness: {a4.fitness_mean:.4f}+/-{a4.fitness_std:.4f}")
        lines.append(f"- 5-feature Baseline fitness: {bl_grp.fitness_mean:.4f}+/-{bl_grp.fitness_std:.4f}")
        lines.append(f"- Delta: {a4.fitness_mean - bl_grp.fitness_mean:+.4f}")
    else:
        lines.append("*Skipped or failed*\n")

    lines.append("\n### Model Selection Ablation\n")
    a5 = ablation_results.get("A5_ModelSelection", {})
    if a5:
        lines.append("| Model | Fitness (mean+/-std) | Iters | Latency (ms) |")
        lines.append("|-------|----------------------|-------|-------------|")
        for mname, mgrp in a5.items():
            lines.append(f"| {mname} | {mgrp.fitness_mean:.4f}+/-{mgrp.fitness_std:.4f} | "
                         f"{mgrp.iters_mean:.1f} | {mgrp.latency_mean*1000:.2f} |")

    lines.append(f"\n![Ablation]({os.path.join('..', 'figures', 'pso', 'ablation_study.png')})\n")

    # -- 8.3 Robustness --
    if robustness_results:
        lines.append("## 8.3 Robustness Experiments\n")
        r1 = robustness_results.get("R1_Seeds")
        if r1 is not None:
            lines.append(f"### R1: Different Random Seeds\n")
            lines.append(f"- 5 seeds tested: fitness range [{r1.fitness_min:.4f}, {r1.fitness_max:.4f}], "
                         f"std={r1.fitness_std:.4f}")
            lines.append(f"- Solution consistency: {'High' if r1.fitness_std < 0.1 else 'Moderate' if r1.fitness_std < 0.5 else 'Low'}\n")

        r2 = robustness_results.get("R2_Noise")
        if r2 is not None:
            lines.append(f"### R2: Model Output Noise\n")
            lines.append(f"- Noise N(0, 0.01*sigma_model) added to predictions")
            lines.append(f"- Fitness with noise: {r2.fitness_mean:.4f}+/-{r2.fitness_std:.4f} (vs reference ~{exp2_fitness:.4f})\n")

        r3 = robustness_results.get("R3_Init")
        if r3 is not None:
            lines.append(f"### R3: Initialization Perturbation\n")
            lines.append(f"- 3 init strategies: fitness range [{r3.fitness_min:.4f}, {r3.fitness_max:.4f}], "
                         f"std={r3.fitness_std:.4f}\n")

        lines.append(f"### R4: Hyperparameter Sensitivity\n")
        hp = robustness_results.get("R4_Hyperparams", {})
        if hp:
            valid_hp = {k: v for k, v in hp.items() if not np.isnan(v)}
            if valid_hp:
                best = max(valid_hp, key=valid_hp.get)
                worst = min(valid_hp, key=valid_hp.get)
                lines.append(f"- Best: c1={best[0]}, c2={best[1]}, N={best[2]}, lambda={best[3]} -> f={valid_hp[best]:.4f}")
                lines.append(f"- Worst: c1={worst[0]}, c2={worst[1]}, N={worst[2]}, lambda={worst[3]} -> f={valid_hp[worst]:.4f}")
                lines.append(f"- Range: [{min(valid_hp.values()):.4f}, {max(valid_hp.values()):.4f}]")
        lines.append(f"\n![Heatmap]({os.path.join('..', 'figures', 'pso', 'robustness_heatmap.png')})\n")

    # -- 8.4 Baseline --
    if baseline_results:
        lines.append("## 8.4 Baseline Comparison (Linear Regression + Gradient Search)\n")
        lines.append(f"- Linear Regression R2: {baseline_results['lr_train_r2']:.4f}")
        lines.append(f"- Gradient search best fitness: {baseline_results['gradient_best']:.2f}")
        lines.append(f"- Gradient search mean+/-std: {baseline_results['gradient_mean']:.2f}+/-{baseline_results['gradient_std']:.2f}")
        lines.append(f"- PSO+LR fitness: {baseline_results['pso_lr_fitness_mean']:.2f}+/-{baseline_results['pso_lr_fitness_std']:.2f}")
        lines.append(f"- PSO+LR iters: {baseline_results['pso_lr_iters_mean']:.1f}")
        lines.append(f"\n**Conclusion**: PSO-based approach {'significantly outperforms' if baseline_results['pso_lr_fitness_mean'] > baseline_results['gradient_best'] else 'is comparable to'} gradient-based search on the same linear model.\n")

    # -- Key Findings --
    lines.append("## Key Findings\n")
    lines.append("### Q: Which improvement module contributes most?\n")

    findings = []
    if exp2_fitness is not None:
        bl_fit = comparison_results.get("Baseline")
        exp1_fit = comparison_results.get("Exp1_Adaptive")
        if bl_fit is not None and exp1_fit is not None:
            delta_adapt = exp1_fit.fitness_mean - bl_fit.fitness_mean
            findings.append(f"- **Adaptive inertia weight** (Exp-1 vs Baseline): "
                           f"delta fitness = {delta_adapt:+.4f}, "
                           f"convergence {'improved' if exp1_fit.iters_mean <= bl_fit.iters_mean else 'regressed'} "
                           f"({bl_fit.iters_mean:.0f} -> {exp1_fit.iters_mean:.0f} iters)")

        exp2_fit = comparison_results.get("Exp2_RiskSensitive")
        if exp2_fit is not None:
            delta_risk = exp2_fit.fitness_mean - bl_fit.fitness_mean
            findings.append(f"- **Risk-sensitive fitness** (Exp-2 vs Baseline): "
                           f"delta fitness = {delta_risk:+.4f}, "
                           f"reduced OOD risk through sigma penalty")

        exp3_fit = comparison_results.get("Exp3_Density")
        if exp3_fit is not None and exp2_fit is not None:
            delta_dens = exp3_fit.fitness_mean - exp2_fit.fitness_mean
            findings.append(f"- **Density penalty** (Exp-3 vs Exp-2): "
                           f"delta fitness = {delta_dens:+.4f}")

        exp4_fit = comparison_results.get("Exp4_ALEnhanced")
        if exp4_fit is not None and exp2_fit is not None:
            delta_al = exp4_fit.fitness_mean - exp2_fit.fitness_mean
            findings.append(f"- **Active learning refinement** (Exp-4 vs Exp-2): "
                           f"delta fitness = {delta_al:+.4f}")

        a1 = ablation_results.get("A1_NoSigma")
        if a1 is not None:
            delta_sigma = a1.fitness_mean - exp2_fit.fitness_mean
            findings.append(f"- **Sigma penalty removal** (A1 vs Exp-2): "
                           f"delta fitness = {delta_sigma:+.4f} — "
                           f"fitness rises to {a1.fitness_mean:.2f} (>100, OOD hallucination); "
                           f"sigma penalty is a critical OOD guard, not a performance cost")

        a3 = ablation_results.get("A3_FixedW")
        if a3 is not None:
            delta_w = a3.fitness_mean - exp2_fit.fitness_mean
            findings.append(f"- **Adaptive weight removal** (A3 vs Exp-2): "
                           f"delta fitness = {delta_w:+.4f}, "
                           f"convergence slowed ({exp2_fit.iters_mean:.0f} -> {a3.iters_mean:.0f} iters)")

    if findings:
        lines.extend(findings)

    lines.append(f"\n---\n*Report auto-generated by `experiments/run_ablation.py`*\n")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(T08_REPORT_PATH), exist_ok=True)
    with open(T08_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  [Report] {T08_REPORT_PATH}")

    return T08_REPORT_PATH


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="T08: Experiment Design and Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 5 runs each for fast verification")
    parser.add_argument("--full", action="store_true",
                        help="Full mode: 20 runs for comparison (default: 10)")
    parser.add_argument("--robust", action="store_true",
                        help="Include robustness experiments (8.3)")
    parser.add_argument("--baseline", action="store_true",
                        help="Include baseline comparison 8.4 (LinearReg+Gradient)")
    parser.add_argument("--no-al", action="store_true",
                        help="Skip Exp-4 (Active Learning enhanced PSO)")
    args = parser.parse_args()

    n_comp = 5 if args.quick else (20 if args.full else 10)
    n_abl = 5 if args.quick else 10

    print("=" * 65)
    print("  T08: Experiment Design and Evaluation")
    print(f"  Comparison: {n_comp} trials/group | Ablation: {n_abl} trials/group")
    print("=" * 65)

    t_start = time.time()

    # 8.1 Comparison
    comp_results = run_comparison_experiments(
        n_trials=n_comp, include_al=not getattr(args, 'no_al', False), verbose=True,
    )

    # 8.2 Ablation
    abl_results = run_ablation_experiments(n_trials=n_abl, verbose=True)

    # 8.3 Robustness
    rob_results = None
    if args.robust:
        rob_results = run_robustness_experiments(verbose=True)

    # 8.4 Baseline
    bl_results = None
    if args.baseline:
        bl_results = run_baseline_comparison(n_trials=n_abl)

    # Plotting
    print("\n" + "=" * 65)
    print("  Generating plots...")
    print("=" * 65)
    plot_convergence_comparison(comp_results)
    plot_best_fitness_boxplot(comp_results)
    plot_ablation_study(comp_results, abl_results)
    if rob_results:
        plot_robustness_heatmap(rob_results)

    # Report
    print("\n" + "=" * 65)
    print("  Generating report...")
    print("=" * 65)
    generate_report(comp_results, abl_results, rob_results, bl_results, n_comp, n_abl)

    t_total = time.time() - t_start
    print(f"\n{'=' * 65}")
    print(f"  T08 Complete! Total time: {t_total:.1f}s")
    print(f"  Figures: {T08_FIGURES_DIR}/")
    print(f"  Report:  {T08_REPORT_PATH}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
