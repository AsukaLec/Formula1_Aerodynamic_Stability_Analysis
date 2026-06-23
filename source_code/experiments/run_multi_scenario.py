#!/usr/bin/env python
"""
T06: Multi-Scenario Optimization & Explainability Analysis
===========================================================
1. Define F1 track scenarios (S1 Monza high-speed, S2 Monaco high-downforce, + optional)
2. Run risk-sensitive PSO with scenario constraints (>= 10 trials each)
3. SHAP analysis on XGBoost (global + local waterfall)
4. Permutation importance
5. Constraint sensitivity (tornado chart)
6. Scenario comparison (radar chart, convergence curves)
7. Optional: counterfactual analysis, decision rule extraction
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import (
    SCENARIOS_DIR, SCENARIOS_FIGURES_DIR, FEATURE_COLS, RANDOM_STATE,
    SCENARIO_N_TRIALS, SCENARIO_LAMBDA_RISK,
    PROCESSED_DIR, PSO_PARAM_BOUNDS,
)
from src.scenarios.scenario_def import (
    SCENARIO_S1_MONZA, SCENARIO_S2_MONACO,
    SCENARIO_S3_BALANCED, SCENARIO_S4_WET,
    ALL_SCENARIOS, OPTIONAL_SCENARIOS,
)
from src.scenarios.scenario_runner import (
    run_scenario_trials, save_scenario_results, build_comparison_csv,
    run_all_scenarios, run_all_scenarios_multiobj, _make_scenario_fitness,
)
from src.analysis.explainability import (
    load_xgb_model,
    compute_shap_values, compute_shap_local,
    compute_permutation_importance,
    compute_sensitivity,
    counterfactual_search,
    extract_decision_rules,
)
from src.visualization.plot_scenarios import (
    plot_shap_waterfall, plot_scenario_report,
    plot_pareto_frontier, plot_parallel_coordinates,
    plot_multiobj_convergence, plot_landscape_multimodality,
)
from src.optimization.fitness import load_xgb_fitness


def run_sensitivity_analysis(scenarios, all_results):
    """Run constraint sensitivity for each scenario and return tornado data."""
    print("\n" + "=" * 60)
    print("  Phase 3: Constraint Sensitivity Analysis")
    print("=" * 60)

    tornado_list = []
    for scenario in scenarios:
        code = scenario.code
        print(f"\n  --- {scenario.name} ---")
        t0 = time.time()
        td, bl = compute_sensitivity(scenario)
        t_elapsed = time.time() - t0
        print(f"    Baseline fitness: {bl:.4f}")
        print(f"    Sensitivity computed in {t_elapsed:.1f}s ({len(td)} perturbations)")
        if td:
            # Print sorted by impact magnitude
            impacts = {}
            for entry in td:
                key = entry["parameter"]
                impacts[key] = max(impacts.get(key, 0), abs(entry["delta_fitness"]))
            sorted_params = sorted(impacts.keys(), key=lambda k: impacts[k], reverse=True)
            for sp in sorted_params:
                low = next(e["delta_fitness"] for e in td if e["parameter"] == sp and e["direction"] == "decrease")
                high = next(e["delta_fitness"] for e in td if e["parameter"] == sp and e["direction"] == "increase")
                print(f"      {sp:>16s}: low={low:+.4f}, high={high:+.4f}")
            tornado_list.append((td, bl, scenario.name))
    return tornado_list


def run_shap_analysis(all_results):
    """Global SHAP analysis + local waterfall for each scenario's best solution."""
    print("\n" + "=" * 60)
    print("  Phase 4: SHAP Explainability Analysis")
    print("=" * 60)

    model, scaler = load_xgb_model()
    print("  XGBoost model loaded for SHAP analysis")

    print("  Computing global SHAP values...")
    shap_vals, X_bg, explainer = compute_shap_values(
        model=model, scaler=scaler, n_sample=2000,
    )
    print(f"  Global SHAP: shape={shap_vals.shape}, "
          f"top features: {', '.join(FEATURE_COLS[i] for i in np.argsort(np.abs(shap_vals).mean(0))[::-1])}")

    # Local SHAP for each scenario's single best solution
    print("\n  Computing local SHAP waterfall for each scenario...")
    shap_waterfalls = []
    for code in sorted(all_results.keys()):
        results, stats = all_results[code]
        best_idx = np.argmax([r["gbest_fitness"] for r in results])
        best_pos = results[best_idx]["gbest_pos"].reshape(1, -1)
        sv_local, expected_val = compute_shap_local(model, scaler, best_pos, position_labels=[code])
        shap_waterfalls.append((sv_local, expected_val, best_pos, [code]))

    return shap_vals, X_bg, explainer, shap_waterfalls


def run_permutation_importance():
    """Compute permutation importance for XGBoost on test set."""
    print("\n  Computing permutation importance...")
    import pandas as pd
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))[FEATURE_COLS].values
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    model, scaler = load_xgb_model()

    pi_result = compute_permutation_importance(model, scaler, X_test, y_test)
    print(f"  Top feature by permutation importance: {FEATURE_COLS[np.argmax(pi_result['importances_mean'])]}")
    return pi_result


def run_counterfactual_analysis(all_results):
    """Optional: find min perturbation to drop below stability threshold for each scenario."""
    print("\n" + "=" * 60)
    print("  Phase 5 (Optional): Counterfactual Analysis")
    print("=" * 60)

    xgb_fitness = load_xgb_fitness()
    cf_results = {}
    for code in sorted(all_results.keys()):
        results, stats = all_results[code]
        best_idx = np.argmax([r["gbest_fitness"] for r in results])
        best_pos = results[best_idx]["gbest_pos"]
        cf = counterfactual_search(xgb_fitness.model, best_pos, target_threshold=95.0)
        cf_results[code] = cf
        if cf["found"]:
            print(f"  {code}: min perturbation dist = {cf['distance']:.3f} drops stability "
                  f"from {cf['original_stability']:.1f} to {cf['perturbed_stability']:.1f}")
        else:
            print(f"  {code}: no counterfactual found within search budget (stability={cf['original_stability']:.1f})")
    return cf_results


def run_decision_rules():
    """Optional: extract interpretable rules from a shallow tree."""
    print("\n  Extracting decision rules (shallow tree)...")
    xgb_fitness = load_xgb_fitness()
    tree, rules = extract_decision_rules(xgb_fitness.model, n_samples=5000, max_depth=3)
    print("  Decision rules for high-stability region (stability >= 95):")
    print("-" * 50)
    for line in rules.strip().split("\n"):
        print(f"  {line}")
    return tree, rules


def print_summary(all_results, shap_waterfalls, pi_result, tornado_data_list,
                  cf_results=None, include_optional=False):
    """Print comprehensive T06 summary."""
    print("\n" + "=" * 60)
    print("  T06 Multi-Scenario Optimization -- Summary")
    print("=" * 60)

    for code in sorted(all_results.keys()):
        _, stats = all_results[code]
        print(f"\n  --- {code} ---")
        print(f"    Fitness:  {stats['fitness_mean']:.4f} +/- {stats['fitness_std']:.4f}")
        print(f"    Iters:    {stats['iters_mean']:.1f} (converged {stats['converged_rate']*100:.0f}%)")
        print(f"    Optimal params:")
        for col in FEATURE_COLS:
            mean = stats["param_means"][col]
            std = stats["param_stds"][col]
            print(f"      {col:>18s}: {mean:8.2f} +/- {std:6.2f}")

    print(f"\n  --- SHAP Top Features (global) ---")
    if pi_result:
        order = np.argsort(pi_result["importances_mean"])[::-1]
        for i in order:
            print(f"    {FEATURE_COLS[i]:>18s}: {pi_result['importances_mean'][i]:.4f}")

    print(f"\n  --- Constraint Sensitivity Top Impacts ---")
    if tornado_data_list:
        for td, bl, sn in tornado_data_list:
            if td:
                impacts = {}
                for entry in td:
                    key = entry["parameter"]
                    impacts[key] = max(impacts.get(key, 0), abs(entry["delta_fitness"]))
                top = sorted(impacts.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"    {sn}: {', '.join(f'{k}={v:.3f}' for k, v in top)}")

    print(f"\n  Outputs saved to:")
    print(f"    {os.path.join(os.path.dirname(__file__), '..', 'outputs', 'scenarios')}/")
    print(f"    {SCENARIOS_FIGURES_DIR}/")
    print(f"\n  T06 Complete!")


def main(include_optional=False, run_counterfactual=True, run_rules=True):
    print("=" * 60)
    print("  T06: Multi-Scenario Optimization & Explainability")
    print("=" * 60)

    scenarios = ALL_SCENARIOS.copy()
    if include_optional:
        scenarios += OPTIONAL_SCENARIOS
        print(f"  Running {len(scenarios)} scenarios (including optional)")
    else:
        print(f"  Running {len(scenarios)} core scenarios")

    # ---- Phase 0: Unconstrained Baseline ----
    print("\n  Phase 0: Computing unconstrained baseline (full bounds, no penalties)")
    from src.scenarios.scenario_def import ScenarioDefinition
    unconstrained_scenario = ScenarioDefinition(
        name="Baseline Unconstrained",
        code="baseline",
        description="Full parameter space, no scenario constraints",
    )
    bl_results, bl_stats = run_scenario_trials(
        unconstrained_scenario, n_trials=SCENARIO_N_TRIALS,
        use_risk=True, verbose=False,
    )
    show_baseline = {
        "label": "Unconstrained Optimum",
        "params": bl_stats["param_means"],
    }
    print(f"  Baseline fitness: {bl_stats['fitness_mean']:.4f}")

    # ---- Phase 1: Scenario Optimization ----
    print(f"\n  Phase 1: Running PSO trials ({SCENARIO_N_TRIALS} per scenario, "
          f"lambda={SCENARIO_LAMBDA_RISK})")
    t0 = time.time()
    all_results = run_all_scenarios(scenarios, n_trials=SCENARIO_N_TRIALS, use_risk=True, verbose=True)
    t_phase1 = time.time() - t0
    print(f"\n  Phase 1 completed in {t_phase1:.1f}s")

    # ---- Phase 2: Comparison CSV ----
    print("\n  Phase 2: Building comparison CSV")
    comp_df = build_comparison_csv(all_results)

    # ---- Phase 3: Constraint Sensitivity ----
    tornado_data_list = run_sensitivity_analysis(scenarios, all_results)

    # ---- Phase 4: SHAP Analysis ----
    shap_vals, X_bg, explainer, shap_waterfalls = run_shap_analysis(all_results)

    # ---- Phase 4.5: Permutation Importance ----
    pi_result = run_permutation_importance()

    # ---- Phase 5: Counterfactual (Optional) ----
    cf_results = None
    if run_counterfactual:
        cf_results = run_counterfactual_analysis(all_results)

    # ---- Phase 6: Decision Rules (Optional) ----
    if run_rules:
        run_decision_rules()

    # ---- Phase 7: Plotting ----
    print("\n" + "=" * 60)
    print("  Phase 7: Generating plots...")
    print("=" * 60)

    # Waterfall plots
    for sv_local, expected_val, best_pos, labels in shap_waterfalls:
        plot_shap_waterfall(sv_local, expected_val, best_pos, labels,
                            scenario_name=labels[0])

    # Full report with global bounds and baseline
    global_bounds = np.array(PSO_PARAM_BOUNDS, dtype=np.float64)
    plot_scenario_report(
        all_results=all_results,
        shap_values=shap_vals,
        X_background=X_bg,
        tornado_data_list=tornado_data_list,
        pi_result=pi_result,
        global_bounds=global_bounds,
        show_baseline=show_baseline,
    )

    # ---- Summary ----
    print_summary(all_results, shap_waterfalls, pi_result, tornado_data_list, cf_results)

    return all_results


def main_multiobj(include_optional=True, run_landscape=True):
    """Run multi-objective PSO with scenario-specific weights."""
    print("=" * 60)
    print("  T06 v2: Multi-Objective Scenario Optimization")
    print("=" * 60)

    scenarios = ALL_SCENARIOS.copy()
    if include_optional:
        scenarios += OPTIONAL_SCENARIOS
    print(f"  Running {len(scenarios)} scenarios (multi-objective weights)")

    # ---- Phase 1: Multi-objective PSO ----
    print(f"\n  Phase 1: Running multi-objective PSO ({SCENARIO_N_TRIALS} per scenario, lambda={SCENARIO_LAMBDA_RISK})")
    t0 = time.time()
    all_results = run_all_scenarios_multiobj(scenarios, n_trials=SCENARIO_N_TRIALS, verbose=True)
    t_phase1 = time.time() - t0
    print(f"\n  Phase 1 completed in {t_phase1:.1f}s")

    # ---- Phase 2: Comparison CSV ----
    print("\n  Phase 2: Building comparison CSV")
    build_comparison_csv(all_results)

    # ---- Phase 3: Landscape Analysis ----
    if run_landscape:
        print("\n  Phase 3: Landscape Analysis (random sampling + PCA/t-SNE)...")
        t0 = time.time()
        plot_landscape_multimodality(all_results_multiobj=all_results, n_samples=30000)
        print(f"  Landscape completed in {time.time()-t0:.1f}s")

    # ---- Phase 4: Multi-objective Plots ----
    print("\n  Phase 4: Generating multi-objective plots...")

    print("  [Plot] Pareto frontiers...")
    plot_pareto_frontier(all_results)

    print("  [Plot] Parallel coordinates...")
    plot_parallel_coordinates(all_results)

    print("  [Plot] Multi-objective convergence...")
    plot_multiobj_convergence(all_results)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  T06 v2 Multi-Objective Summary")
    print("=" * 60)
    for code in sorted(all_results.keys()):
        _, stats = all_results[code]
        print(f"  {code}: fitness={stats['fitness_mean']:.4f}+/-{stats['fitness_std']:.4f}, "
              f"stab={stats['comp_means']['stability']:.1f}, "
              f"eff={stats['comp_means']['efficiency']:.1f}, "
              f"pow={stats['comp_means']['power']:.3f}")
        w = stats.get("weights", {})
        print(f"         weights: ws={w.get('w_stability','')} we={w.get('w_efficiency','')} wp={w.get('w_power','')}")
        for col in FEATURE_COLS:
            print(f"         {col:>18s}: {stats['param_means'][col]:.1f}")

    print(f"\n  Outputs: {SCENARIOS_DIR}/")
    print(f"  Figures:  {SCENARIOS_FIGURES_DIR}/")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="T06: Multi-Scenario Optimization")
    parser.add_argument("--core", action="store_true", help="Core only: S1+S2")
    parser.add_argument("--no-cf", action="store_true", help="Skip counterfactual analysis (v1 only)")
    parser.add_argument("--no-rules", action="store_true", help="Skip decision rule extraction (v1 only)")
    parser.add_argument("--multiobj", action="store_true", help="Run multi-objective mode (v2)")
    parser.add_argument("--both", action="store_true", help="Run both v1 single-obj and v2 multi-obj")
    args = parser.parse_args()

    if args.multiobj:
        main_multiobj(include_optional=not args.core)
    elif args.both:
        print("\n===== PHASE A: Single-Objective (v1) =====")
        main(include_optional=not args.core, run_counterfactual=not args.no_cf, run_rules=not args.no_rules)
        print("\n\n===== PHASE B: Multi-Objective (v2) =====")
        main_multiobj(include_optional=not args.core)
    else:
        main(
            include_optional=not args.core,
            run_counterfactual=not args.no_cf,
            run_rules=not args.no_rules,
        )
