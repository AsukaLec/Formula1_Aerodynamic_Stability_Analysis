import os, json, time
import numpy as np
import pandas as pd

from src.utils.config import (
    RANDOM_STATE, SCENARIOS_DIR, SCENARIO_N_TRIALS,
    SCENARIO_N_PARTICLES, SCENARIO_MAX_ITER, SCENARIO_SEED_BASE,
    SCENARIO_LAMBDA_RISK, FEATURE_COLS,
    PSO_W_START, PSO_W_END, PSO_W_ALPHA,
    PSO_C1, PSO_C2, PSO_TOL, PSO_EARLY_STOP_ITERS,
)
from src.optimization.fitness import (
    FitnessRiskSensitive, FitnessStandard, FitnessMultiObjective, ModelWrapper,
    load_xgb_fitness, load_ensemble_fitness,
)
from src.optimization.pso_risk_sensitive import DEFAULT_BOUNDS, DISCRETE_INDICES
from src.optimization.pso_adaptive import PSOAdaptive


class ScenarioFitness:
    """Composite fitness for a specific scenario: base model fitness minus scenario penalties."""

    def __init__(self, base_fitness, scenario):
        """
        Parameters
        ----------
        base_fitness : FitnessStandard or FitnessRiskSensitive
            Model-based fitness object with .evaluate(X) method.
        scenario : ScenarioDefinition
            Scenario with .penalize(X) method.
        """
        self._base = base_fitness
        self._scenario = scenario

    def evaluate(self, X):
        base = self._base.evaluate(X)
        penalty = self._scenario.penalize(X)
        return base - penalty

    def __call__(self, X):
        return self.evaluate(X)


def _make_scenario_fitness(scenario, use_risk=True, lambda_risk=SCENARIO_LAMBDA_RISK):
    """Build composite fitness for a scenario."""
    if use_risk:
        base = load_ensemble_fitness(lambda_risk=lambda_risk)
    else:
        base = load_xgb_fitness()
    return ScenarioFitness(base, scenario)


def _make_multiobjective_fitness(scenario, lambda_risk=SCENARIO_LAMBDA_RISK):
    """Build multi-objective fitness with scenario-specific weights.

    Parameters
    ----------
    scenario : ScenarioDefinition — must have a .weights dict.
    lambda_risk : float

    Returns
    -------
    FitnessMultiObjective
    """
    from src.models.deep_ensemble import DeepEnsemble
    from src.utils.config import (
        MLP_HIDDEN_UNITS, NN_LR, NN_WEIGHT_DECAY, NN_BATCH_SIZE,
        NN_MAX_EPOCHS, NN_EARLY_STOP_PATIENCE, DEEP_ENSEMBLE_M, PROCESSED_DIR,
    )
    import numpy as np

    ensemble = DeepEnsemble.load(
        input_dim=5, hidden_units=MLP_HIDDEN_UNITS, M=DEEP_ENSEMBLE_M,
        lr=NN_LR, weight_decay=NN_WEIGHT_DECAY,
        batch_size=NN_BATCH_SIZE, max_epochs=NN_MAX_EPOCHS,
        patience=NN_EARLY_STOP_PATIENCE,
    )
    with open(os.path.join(PROCESSED_DIR, "scaler_mm.pkl"), "rb") as f:
        import pickle; scaler = pickle.load(f)
    wrapper = ModelWrapper(ensemble, scaler, model_type="ensemble", y_transform="x100")

    w = scenario.weights
    fitness = FitnessMultiObjective(
        wrapper,
        w_stability=w.get("w_stability", 0.5),
        w_efficiency=w.get("w_efficiency", 0.3),
        w_power=w.get("w_power", 0.2),
        lambda_risk=lambda_risk,
    )

    # Compute normalisation constants from training data
    import pandas as pd
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    max_eff = (train["downforce_n"] / (train["drag_n"] + 1e-6)).quantile(0.99)
    max_pow = (train["drag_n"] * train["speed_kmh"]).quantile(0.99)
    fitness._set_norm_constants(max_eff, max_pow)

    return fitness


def run_scenario_trials_multiobj(
    scenario,
    n_trials=SCENARIO_N_TRIALS,
    n_particles=SCENARIO_N_PARTICLES,
    max_iter=SCENARIO_MAX_ITER,
    seed_base=SCENARIO_SEED_BASE,
    verbose=True,
):
    """Run N PSO trials for a scenario using multi-objective fitness.

    Returns
    -------
    results, stats (same format as run_scenario_trials, with extra 'components' key)
    """
    fitness = _make_multiobjective_fitness(scenario)

    results = []
    best_positions = []
    best_fitnesses = []
    best_components = []

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  {scenario.name}  [MULTI-OBJECTIVE]")
        w = scenario.weights
        print(f"  weights: ws={w['w_stability']}, we={w['w_efficiency']}, wp={w['w_power']}")
        print(f"  speed fixed: {scenario.bounds[0][0]:.0f} km/h")
        print(f"  wing range: [{scenario.bounds[1][0]:.0f}, {scenario.bounds[1][1]:.0f}]°")
        print(f"{'=' * 60}")

    for t in range(n_trials):
        trial_seed = seed_base + t * 100
        pso = PSOAdaptive(
            n_particles=n_particles,
            bounds=scenario.bounds,
            discrete_indices=scenario.discrete_indices,
            w_start=PSO_W_START, w_end=PSO_W_END, alpha=PSO_W_ALPHA,
            c1=PSO_C1, c2=PSO_C2,
            max_iter=max_iter,
            early_stop_iters=PSO_EARLY_STOP_ITERS,
            tol=PSO_TOL,
            seed=trial_seed,
            maximise=True,
        )
        result = pso.optimize(fitness, verbose=False, collect_candidates=True)
        result["trial_seed"] = trial_seed
        result["scenario_name"] = scenario.name

        # Compute objective components at the gbest position
        stab, eff, power, sigma, _ = fitness.evaluate_components(
            result["gbest_pos"].reshape(1, -1)
        )
        result["components"] = {
            "stability": float(stab[0]), "efficiency": float(eff[0]),
            "power": float(power[0]), "sigma": float(sigma[0]),
        }

        results.append(result)
        best_positions.append(result["gbest_pos"])
        best_fitnesses.append(result["gbest_fitness"])
        best_components.append(result["components"])

        if verbose:
            comp = result["components"]
            print(f"  Trial {t+1:2d}/{n_trials}: f={result['gbest_fitness']:.4f}, "
                  f"stab={comp['stability']:.1f}, eff={comp['efficiency']:.1f}, "
                  f"pow={comp['power']:.3f}, iters={result['n_iter']}")

    best_arr = np.array(best_positions)
    fit_arr = np.array(best_fitnesses)
    comp_keys = ["stability", "efficiency", "power", "sigma"]
    comp_means = {k: float(np.mean([c[k] for c in best_components])) for k in comp_keys}
    comp_stds = {k: float(np.std([c[k] for c in best_components], ddof=1)) for k in comp_keys}

    stats = {
        "scenario": scenario.code,
        "n_trials": n_trials,
        "model_type": "multi_objective_ensemble",
        "lambda_risk": SCENARIO_LAMBDA_RISK,
        "weights": dict(scenario.weights),
        "fitness_mean": float(np.mean(fit_arr)),
        "fitness_std": float(np.std(fit_arr, ddof=1)),
        "param_means": {FEATURE_COLS[i]: float(np.mean(best_arr[:, i])) for i in range(len(FEATURE_COLS))},
        "param_stds":  {FEATURE_COLS[i]: float(np.std(best_arr[:, i], ddof=1)) for i in range(len(FEATURE_COLS))},
        "iters_mean": float(np.mean([r["n_iter"] for r in results])),
        "converged_rate": float(np.mean([1.0 if r["converged"] else 0.0 for r in results])),
        "comp_means": comp_means,
        "comp_stds": comp_stds,
    }

    return results, stats


def run_scenario_trials(
    scenario,
    n_trials=SCENARIO_N_TRIALS,
    n_particles=SCENARIO_N_PARTICLES,
    max_iter=SCENARIO_MAX_ITER,
    seed_base=SCENARIO_SEED_BASE,
    use_risk=True,
    verbose=False,
):
    """Run N independent PSO trials for a single scenario.

    Returns
    -------
    results : list of dict
        One per trial, each with keys: gbest_pos, gbest_fitness, history, n_iter, converged, trial_seed.
    stats : dict
        Aggregated statistics across trials.
    """
    fitness = _make_scenario_fitness(scenario, use_risk=use_risk)

    results = []
    best_positions = []
    best_fitnesses = []

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  {scenario.name}")
        print(f"  {scenario.description}")
        print(f"  model={'RiskSensitive' if use_risk else 'XGBoost'}, trials={n_trials}, "
              f"particles={n_particles}, max_iter={max_iter}")
        print(f"{'=' * 60}")

    for t in range(n_trials):
        trial_seed = seed_base + t * 100
        pso = PSOAdaptive(
            n_particles=n_particles,
            bounds=scenario.bounds,
            discrete_indices=scenario.discrete_indices,
            w_start=PSO_W_START,
            w_end=PSO_W_END,
            alpha=PSO_W_ALPHA,
            c1=PSO_C1,
            c2=PSO_C2,
            max_iter=max_iter,
            early_stop_iters=PSO_EARLY_STOP_ITERS,
            tol=PSO_TOL,
            seed=trial_seed,
            maximise=True,
        )
        result = pso.optimize(fitness, verbose=False)

        result["trial_seed"] = trial_seed
        result["scenario_name"] = scenario.name
        results.append(result)
        best_positions.append(result["gbest_pos"])
        best_fitnesses.append(result["gbest_fitness"])

        if verbose:
            print(f"  Trial {t+1:2d}/{n_trials}: seed={trial_seed}, "
                  f"fitness={result['gbest_fitness']:.4f}, iters={result['n_iter']}, "
                  f"conv={result['converged']}")

    best_arr = np.array(best_positions)
    fit_arr = np.array(best_fitnesses)

    stats = {
        "scenario": scenario.code,
        "n_trials": n_trials,
        "model_type": "risk_sensitive_ensemble" if use_risk else "xgb_baseline",
        "lambda_risk": SCENARIO_LAMBDA_RISK if use_risk else None,
        "fitness_mean": float(np.mean(fit_arr)),
        "fitness_std": float(np.std(fit_arr, ddof=1)),
        "fitness_min": float(np.min(fit_arr)),
        "fitness_max": float(np.max(fit_arr)),
        "param_means": {FEATURE_COLS[i]: float(np.mean(best_arr[:, i])) for i in range(len(FEATURE_COLS))},
        "param_stds":  {FEATURE_COLS[i]: float(np.std(best_arr[:, i], ddof=1)) for i in range(len(FEATURE_COLS))},
        "param_mins":  {FEATURE_COLS[i]: float(np.min(best_arr[:, i])) for i in range(len(FEATURE_COLS))},
        "param_maxs":  {FEATURE_COLS[i]: float(np.max(best_arr[:, i])) for i in range(len(FEATURE_COLS))},
        "iters_mean": float(np.mean([r["n_iter"] for r in results])),
        "converged_rate": float(np.mean([1.0 if r["converged"] else 0.0 for r in results])),
    }

    return results, stats


def save_scenario_results(scenario, results, stats):
    """Save scenario optimization results to disk."""
    out_dir = os.path.join(SCENARIOS_DIR, scenario.code)
    os.makedirs(out_dir, exist_ok=True)

    best_idx = np.argmax([r["gbest_fitness"] for r in results])
    best = results[best_idx]

    best_solution = {
        "scenario": scenario.code,
        "scenario_name": scenario.name,
        "gbest_pos": best["gbest_pos"].tolist(),
        "gbest_pos_dict": {FEATURE_COLS[i]: best["gbest_pos"][i] for i in range(len(FEATURE_COLS))},
        "gbest_fitness": float(best["gbest_fitness"]),
        "n_iter": best["n_iter"],
        "converged": best["converged"],
        "trial_seed": best["trial_seed"],
    }
    with open(os.path.join(out_dir, "best_solution.json"), "w") as f:
        json.dump(best_solution, f, indent=2)

    stats_path = os.path.join(out_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    conv_data = {"trial": [], "iteration": [], "gbest_fitness": []}
    for t, r in enumerate(results):
        for step, fit in zip(r["history"]["iteration"], r["history"]["gbest_fitness"]):
            conv_data["trial"].append(t)
            conv_data["iteration"].append(step)
            conv_data["gbest_fitness"].append(fit)
    pd.DataFrame(conv_data).to_csv(os.path.join(out_dir, "convergence.csv"), index=False)

    trials_data = []
    for r in results:
        row = {"trial_seed": r["trial_seed"], "gbest_fitness": r["gbest_fitness"],
               "n_iter": r["n_iter"], "converged": r["converged"]}
        for i, col in enumerate(FEATURE_COLS):
            row[col] = r["gbest_pos"][i]
        trials_data.append(row)
    pd.DataFrame(trials_data).to_csv(os.path.join(out_dir, "trials.csv"), index=False)

    print(f"  Results saved: {out_dir}/")
    return out_dir


def run_all_scenarios(scenarios, n_trials=SCENARIO_N_TRIALS, use_risk=True, verbose=True):
    """Run multi-trial PSO optimization for all given scenarios.

    Returns
    -------
    all_results : dict
        {scenario.code: (results_list, stats_dict)}
    """
    all_results = {}

    for scenario in scenarios:
        results, stats = run_scenario_trials(
            scenario, n_trials=n_trials, use_risk=use_risk, verbose=verbose,
        )
        all_results[scenario.code] = (results, stats)
        save_scenario_results(scenario, results, stats)

    return all_results


def build_comparison_csv(all_results):
    """Build a scenario comparison CSV from aggregated stats."""
    rows = []
    for code, (_, stats) in all_results.items():
        row = {
            "scenario": code,
            "scenario_name": stats.get("scenario", code),
            "fitness_mean": stats["fitness_mean"],
            "fitness_std": stats["fitness_std"],
            "iters_mean": stats["iters_mean"],
            "converged_rate": stats["converged_rate"],
        }
        for col in FEATURE_COLS:
            row[f"{col}_mean"] = stats["param_means"].get(col)
            row[f"{col}_std"] = stats["param_stds"].get(col)
        # Multi-objective components
        if "comp_means" in stats:
            for k, v in stats["comp_means"].items():
                row[f"comp_{k}_mean"] = v
            for k, v in stats.get("comp_stds", {}).items():
                row[f"comp_{k}_std"] = v
            row["w_stability"] = stats.get("weights", {}).get("w_stability", "")
            row["w_efficiency"] = stats.get("weights", {}).get("w_efficiency", "")
            row["w_power"] = stats.get("weights", {}).get("w_power", "")
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(SCENARIOS_DIR, "scenario_comparison.csv")
    os.makedirs(SCENARIOS_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n  Comparison saved: {path}")
    return df


def run_all_scenarios_multiobj(scenarios, n_trials=SCENARIO_N_TRIALS, verbose=True):
    """Run multi-objective PSO for all given scenarios.

    Returns
    -------
    all_results : dict {scenario.code: (results_list, stats_dict)}
    """
    all_results = {}
    for scenario in scenarios:
        results, stats = run_scenario_trials_multiobj(
            scenario, n_trials=n_trials, verbose=verbose,
        )
        all_results[scenario.code] = (results, stats)
        save_scenario_results(scenario, results, stats)
    return all_results
