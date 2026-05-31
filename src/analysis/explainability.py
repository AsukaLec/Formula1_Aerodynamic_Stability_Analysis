import os, pickle, time
import numpy as np
import pandas as pd

from src.utils.config import (
    PROCESSED_DIR, MODELS_OUTPUT_DIR, FEATURE_COLS, RANDOM_STATE,
    SCENARIO_SENSITIVITY_DELTA, SCENARIO_LAMBDA_RISK,
)

# --- SHAP Analysis ---------------------------------------------------------


def load_xgb_model():
    """Load XGBoost model and scaler for SHAP analysis."""
    import xgboost as xgb
    with open(os.path.join(MODELS_OUTPUT_DIR, "M3_XGBoost.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "scaler_ss.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def compute_shap_values(model=None, scaler=None, X_sample=None, n_sample=2000):
    """Compute SHAP values using TreeExplainer on XGBoost.

    Parameters
    ----------
    model : XGBoost model, optional
    scaler : StandardScaler, optional
    X_sample : np.ndarray (n, 5), optional. Raw (unscaled) background data.
    n_sample : int. Number of background samples if X_sample not provided.

    Returns
    -------
    shap_values : np.ndarray (n, 5)
    X_background : np.ndarray
    explainer : shap.TreeExplainer
    """
    import shap
    if model is None or scaler is None:
        model, scaler = load_xgb_model()

    if X_sample is None:
        X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
        n = min(n_sample, len(X_train))
        X_sample = X_train[FEATURE_COLS].sample(n, random_state=RANDOM_STATE).values

    X_scaled = scaler.transform(X_sample)
    explainer = shap.TreeExplainer(model, X_scaled, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_scaled)

    return shap_values, X_sample, explainer


def compute_shap_local(model, scaler, positions, position_labels=None, n_background=2000):
    """Compute SHAP values for specific positions (e.g. optimal solutions).

    Parameters
    ----------
    positions : np.ndarray (n, 5) — raw (unscaled) position vectors.
    position_labels : list of str or None
    n_background : int. Number of training samples to use as SHAP background.

    Returns
    -------
    shap_values : np.ndarray (n, 5)
    expected_value : float
    """
    import shap
    import pandas as pd
    if position_labels is None:
        position_labels = [f"Sample_{i+1}" for i in range(len(positions))]

    # Use training data as background (NOT the query points themselves)
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    n = min(n_background, len(X_train))
    X_bg = X_train[FEATURE_COLS].sample(n, random_state=RANDOM_STATE).values
    X_bg_scaled = scaler.transform(X_bg)

    X_scaled = scaler.transform(positions)
    explainer = shap.TreeExplainer(model, X_bg_scaled, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_scaled)

    return shap_values, explainer.expected_value


# --- Permutation Importance -------------------------------------------------


def compute_permutation_importance(model, scaler, X_test, y_test, n_repeats=10):
    """Compute permutation importance for a model + scaler pipeline.

    Parameters
    ----------
    model : XGBoost or other model with .predict(X) -> np.ndarray
    scaler : StandardScaler or other with .transform(X)
    X_test : np.ndarray (n, 5) — raw (unscaled) values
    y_test : np.ndarray (n,)
    n_repeats : int

    Returns
    -------
    result : dict with keys: importances_mean, importances_std, feature_names
    """
    import xgboost as xgb
    from sklearn.base import BaseEstimator, RegressorMixin, clone
    from sklearn.inspection import permutation_importance

    class _WrappedModel(BaseEstimator, RegressorMixin):
        """Minimal sklearn-compatible wrapper for permutation_importance."""
        def __init__(self, model=None, scaler=None):
            self.model = model
            self.scaler = scaler

        def fit(self, X, y):
            return self

        def predict(self, X):
            X_s = X if self.scaler is None else self.scaler.transform(X)
            preds = self.model.predict(X_s)
            if hasattr(preds, "numpy"):
                return preds.numpy()
            return np.asarray(preds, dtype=np.float64)

        def get_params(self, deep=True):
            return {"model": self.model, "scaler": self.scaler}

    wrapped = _WrappedModel(model=model, scaler=scaler)
    result = permutation_importance(
        wrapped, X_test, y_test, n_repeats=n_repeats,
        random_state=RANDOM_STATE, scoring="r2",
    )
    return {
        "importances_mean": result.importances_mean,
        "importances_std": result.importances_std,
        "feature_names": FEATURE_COLS,
    }


# --- Constraint Sensitivity (Tornado) ----------------------------------------
# Perturb each constraint bound ±delta, run PSO once, observe ∆fitness


def _run_single_pso_with_bounds(bounds, fitness_fn, seed=RANDOM_STATE):
    """Run one PSO trial with given bounds, return gbest_fitness."""
    from src.optimization.pso_adaptive import PSOAdaptive
    from src.optimization.pso_risk_sensitive import DISCRETE_INDICES
    from src.utils.config import (
        SCENARIO_N_PARTICLES, PSO_W_START, PSO_W_END, PSO_W_ALPHA,
        PSO_C1, PSO_C2, SCENARIO_MAX_ITER as MAX_ITER,
        PSO_TOL, PSO_EARLY_STOP_ITERS,
    )

    pso = PSOAdaptive(
        n_particles=30,  # fewer particles for speed
        bounds=bounds,
        discrete_indices=DISCRETE_INDICES,
        w_start=PSO_W_START,
        w_end=PSO_W_END,
        alpha=PSO_W_ALPHA,
        c1=PSO_C1,
        c2=PSO_C2,
        max_iter=MAX_ITER // 2,
        early_stop_iters=PSO_EARLY_STOP_ITERS,
        tol=PSO_TOL,
        seed=seed,
        maximise=True,
    )
    result = pso.optimize(fitness_fn, verbose=False)
    return result["gbest_fitness"]


def compute_sensitivity(scenario, n_constraints=None, delta=SCENARIO_SENSITIVITY_DELTA):
    """Perturb each constraint of a scenario ±delta ratio and measure fitness change.

    For each adjustable constraint (speed/wing bounds, penalty weights), increase
    and decrease the bound by delta*, re-run PSO, and compute ∆fitness from baseline.

    Parameters
    ----------
    scenario : ScenarioDefinition
    n_constraints : int or None. Number of top constraints to analyse.
    delta : float. Fractional perturbation (e.g. 0.10 = ±10%).

    Returns
    -------
    tornado_data : list of dict
        [{parameter, direction, original_value, perturbed_value, fitness, delta_fitness, delta_pct}, ...]
    """
    import copy
    from src.scenarios.scenario_runner import _make_scenario_fitness

    fitness_base = _make_scenario_fitness(scenario, use_risk=True)
    baseline_fitness = _run_single_pso_with_bounds(scenario.bounds, fitness_base)

    # Build list of constraints to perturb
    perturbations = []
    bounds = scenario.bounds.copy()

    # Perturb speed constraint
    if bounds[0][0] > 80:  # speed lower bound
        perturbations.append(("speed_lb", 0, 0, bounds[0][0], delta))
    if bounds[0][1] < 360:  # speed upper bound
        perturbations.append(("speed_ub", 0, 1, bounds[0][1], delta))

    # Perturb wing_angle constraint
    if bounds[1][0] > 0:
        perturbations.append(("wing_lb", 1, 0, bounds[1][0], delta))
    if bounds[1][1] < 45:
        perturbations.append(("wing_ub", 1, 1, bounds[1][1], delta))

    # Perturb downforce constraint (for S2)
    if bounds[3][0] > 0:
        perturbations.append(("downforce_lb", 3, 0, bounds[3][0], delta))
    if bounds[3][1] < 10000:
        perturbations.append(("downforce_ub", 3, 1, bounds[3][1], delta))

    # Perturb drag constraint
    if bounds[4][0] > 0:
        perturbations.append(("drag_lb", 4, 0, bounds[4][0], delta))
    if bounds[4][1] < 600:
        perturbations.append(("drag_ub", 4, 1, bounds[4][1], delta))

    if n_constraints:
        perturbations = perturbations[:n_constraints]

    tornado_data = []
    for name, dim, bound_idx, orig_val, frac in perturbations:
        for direction, sign in [("decrease", -1), ("increase", +1)]:
            new_val = orig_val * (1 + sign * frac)
            # Clamp to global bounds
            from src.utils.config import PSO_PARAM_BOUNDS
            global_lb, global_ub = PSO_PARAM_BOUNDS[dim]
            new_val = np.clip(new_val, global_lb, global_ub)

            perturbed_bounds = bounds.copy()
            if bound_idx == 0:
                perturbed_bounds[dim][0] = new_val
            else:
                perturbed_bounds[dim][1] = new_val

            # Rebuild scenario with perturbed bounds
            from src.scenarios.scenario_def import ScenarioDefinition
            bounds_overrides = {}
            for d in range(len(perturbed_bounds)):
                if not np.array_equal(perturbed_bounds[d], np.array(PSO_PARAM_BOUNDS[d])):
                    bounds_overrides[d] = perturbed_bounds[d].tolist()
            pert_scenario = ScenarioDefinition(
                name=f"{scenario.name}_pert",
                code=f"{scenario.code}_pert",
                description="Perturbed for sensitivity",
                bounds_overrides=bounds_overrides,
                fixed_params=scenario.fixed_params,
                penalty_terms=scenario._penalty_terms,
            )
            fitness_pert = _make_scenario_fitness(pert_scenario, use_risk=True)
            pert_fitness = _run_single_pso_with_bounds(pert_scenario.bounds, fitness_pert)

            tornado_data.append({
                "parameter": name,
                "direction": direction,
                "original_value": round(orig_val, 2),
                "perturbed_value": round(new_val, 2),
                "fitness": round(pert_fitness, 4),
                "delta_fitness": round(pert_fitness - baseline_fitness, 4),
                "delta_pct": round(100 * (pert_fitness - baseline_fitness) / abs(baseline_fitness), 2) if baseline_fitness != 0 else 0,
            })

    return tornado_data, baseline_fitness


# --- Counterfactual Analysis (Optional) --------------------------------------


def counterfactual_search(model_wrapper, optimal_solution, target_threshold=95.0,
                          max_steps=500, step_size=0.02, n_directions=50):
    """Search for the minimum parameter perturbation that drops stability below threshold.

    Uses random search in parameter space to find the closest point to the optimal
    solution where predicted stability falls below target_threshold.

    Parameters
    ----------
    model_wrapper : ModelWrapper
    optimal_solution : np.ndarray (5,)
    target_threshold : float. Stability index threshold.
    max_steps : int. Maximum random perturbation steps.
    step_size : float. Initial perturbation magnitude per step.
    n_directions : int. Directions to explore per step.

    Returns
    -------
    dict with: found, distance, perturbed_point, perturbed_stability
    """
    rng = np.random.RandomState(RANDOM_STATE)
    opt_fitness = model_wrapper.predict(optimal_solution.reshape(1, -1))[0]

    if opt_fitness < target_threshold:
        return {"found": True, "distance": 0.0, "perturbed_point": optimal_solution.tolist(),
                "perturbed_stability": opt_fitness}

    best_dist = np.inf
    best_point = None
    best_stability = None

    for step in range(max_steps):
        scale = step_size * (1 + step * 0.05)
        unit_directions = rng.randn(n_directions, len(optimal_solution))
        unit_directions /= np.linalg.norm(unit_directions, axis=1, keepdims=True) + 1e-10
        candidates = optimal_solution + scale * unit_directions

        preds = model_wrapper.predict(candidates)
        for i, pred in enumerate(preds):
            if pred < target_threshold:
                dist = np.linalg.norm(candidates[i] - optimal_solution)
                if dist < best_dist:
                    best_dist = dist
                    best_point = candidates[i].copy()
                    best_stability = pred

        if best_point is not None and step > 10:
            break

    if best_point is not None:
        return {
            "found": True,
            "distance": float(best_dist),
            "perturbed_point": best_point.tolist(),
            "perturbed_stability": float(best_stability),
            "original_stability": float(opt_fitness),
        }
    else:
        return {
            "found": False,
            "distance": None,
            "perturbed_point": None,
            "perturbed_stability": None,
            "original_stability": float(opt_fitness),
        }


# --- Decision Rule Extraction (Optional) -------------------------------------


def extract_decision_rules(model_wrapper=None, n_samples=5000, max_depth=3):
    """Train a shallow decision tree on high-stability regions and extract rules.

    Parameters
    ----------
    model_wrapper : ModelWrapper, optional
    n_samples : int. Number of samples to generate for rule extraction.
    max_depth : int. Tree depth for readability.

    Returns
    -------
    tree : DecisionTreeRegressor
    rules_text : str. Human-readable rules.
    """
    from sklearn.tree import DecisionTreeRegressor, export_text

    # Generate samples within reasonable bounds
    rng = np.random.RandomState(RANDOM_STATE)
    X_synthetic = np.column_stack([
        rng.uniform(100, 365, n_samples),
        rng.uniform(5, 35, n_samples),
        rng.choice([0, 1], n_samples),
        rng.uniform(105, 8979, n_samples),
        rng.uniform(18, 516, n_samples),
    ])

    if model_wrapper is not None:
        y_synthetic = model_wrapper.predict(X_synthetic)
    else:
        # Use XGBoost as default
        model, scaler = load_xgb_model()
        X_scaled = scaler.transform(X_synthetic)
        y_synthetic = model.predict(X_scaled)

    # Only keep high-stability samples to learn rules for success
    high_mask = y_synthetic >= 95.0
    X_high = X_synthetic[high_mask]
    y_high = y_synthetic[high_mask]

    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=RANDOM_STATE)
    tree.fit(X_high, y_high)

    rules_text = export_text(tree, feature_names=FEATURE_COLS, decimals=2)

    return tree, rules_text
