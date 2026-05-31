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
    FitnessRiskSensitive, FitnessStandard, ModelWrapper,
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
            "fitness_min": stats["fitness_min"],
            "fitness_max": stats["fitness_max"],
            "iters_mean": stats["iters_mean"],
            "converged_rate": stats["converged_rate"],
        }
        for col in FEATURE_COLS:
            row[f"{col}_mean"] = stats["param_means"].get(col)
            row[f"{col}_std"] = stats["param_stds"].get(col)
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(SCENARIOS_DIR, "scenario_comparison.csv")
    os.makedirs(SCENARIOS_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n  Comparison saved: {path}")
    return df
