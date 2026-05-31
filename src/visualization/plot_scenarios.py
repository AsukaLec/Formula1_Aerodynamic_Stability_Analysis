import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from src.utils.config import (
    SCENARIOS_FIGURES_DIR, FEATURE_COLS, PSO_PARAM_BOUNDS,
    SCENARIO_SENSITIVITY_DELTA,
)

os.makedirs(SCENARIOS_FIGURES_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(SCENARIOS_FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# Radar Chart
# ============================================================================


def plot_scenario_radar(all_results, global_bounds=None, show_baseline=None):
    """Radar chart comparing optimal parameters across scenarios.

    Normalises each parameter to global PSO bounds (NOT per-column min/max)
    so different numbers of scenarios produce consistent, interpretable shapes.

    Parameters
    ----------
    all_results : dict {code: (results_list, stats_dict)}
    global_bounds : np.ndarray (5, 2) or None. If None, uses PSO_PARAM_BOUNDS.
    show_baseline : dict or None. Optional "Unconstrained" reference with keys
                    'code' and 'params' (mean parameter vector).
    """
    if global_bounds is None:
        global_bounds = np.array(PSO_PARAM_BOUNDS, dtype=np.float64)
    else:
        global_bounds = np.asarray(global_bounds, dtype=np.float64)

    scenarios = []
    param_vals_list = []

    for code in sorted(all_results.keys()):
        _, stats = all_results[code]
        scenarios.append(code)
        vals = [stats["param_means"][col] for col in FEATURE_COLS]
        param_vals_list.append(vals)

    param_vals = np.array(param_vals_list)

    # Normalise to global bounds
    lower = global_bounds[:, 0]
    upper = global_bounds[:, 1]
    param_norm = (param_vals - lower) / (upper - lower + 1e-10)
    param_norm = np.clip(param_norm, 0, 1)

    angles = np.linspace(0, 2 * np.pi, len(FEATURE_COLS), endpoint=False).tolist()
    angles += angles[:1]

    n_scenarios = len(scenarios) + (1 if show_baseline else 0)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(n_scenarios, 2)))
    if n_scenarios <= 3:
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_scenarios, 2)))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, (sc_name, vals_norm) in enumerate(zip(scenarios, param_norm)):
        vals_plot = vals_norm.tolist() + [vals_norm[0]]
        ax.fill(angles, vals_plot, alpha=0.06, color=colors[i])
        ax.plot(angles, vals_plot, "o-", linewidth=2.2, color=colors[i],
                label=sc_name, markersize=6, markerfacecolor="white",
                markeredgewidth=1.5, markeredgecolor=colors[i])

    if show_baseline is not None:
        b_params = np.array([show_baseline["params"][col] for col in FEATURE_COLS])
        b_norm = np.clip((b_params - lower) / (upper - lower + 1e-10), 0, 1)
        b_plot = b_norm.tolist() + [b_norm[0]]
        ax.plot(angles, b_plot, "--", linewidth=1.5, color="gray",
                label=show_baseline.get("label", "Unconstrained"), alpha=0.7)
        ax.fill(angles, b_plot, alpha=0.03, color="gray")

    # Annotate the bound values on each axis
    axis_labels = []
    for i, col in enumerate(FEATURE_COLS):
        lo, hi = lower[i], upper[i]
        if col == "drs_active":
            axis_labels.append(f"{col}\n[{int(lo)},{int(hi)}]")
        elif col in ("speed_kmh", "downforce_n", "drag_n"):
            axis_labels.append(f"{col}\n[{lo:.0f},{hi:.0f}]")
        else:
            axis_labels.append(f"{col}\n[{lo:.1f},{hi:.1f}]")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title("Multi-Scenario Optimal Parameter Comparison\n(Normalized to Global Bounds)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.08), fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    _save(fig, "scenario_radar.png")


# ============================================================================
# Convergence Comparison
# ============================================================================


def plot_convergence_comparison(all_results):
    """Overlay gbest convergence curves (mean +/- std) for all scenarios."""
    fig, ax = plt.subplots(figsize=(10, 6))
    codes = sorted(all_results.keys())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(codes)))

    for idx, code in enumerate(codes):
        results, _ = all_results[code]
        max_len = max(len(r["history"]["gbest_fitness"]) for r in results)
        curves = []
        for r in results:
            fit = r["history"]["gbest_fitness"]
            curves.append(np.resize(fit, max_len))
        curves = np.array(curves)
        mean_fit = curves.mean(axis=0)
        std_fit = curves.std(axis=0)
        iters = np.arange(len(mean_fit))

        ax.plot(iters, mean_fit, color=colors[idx], linewidth=2, label=code)
        ax.fill_between(iters, mean_fit - std_fit, mean_fit + std_fit,
                        color=colors[idx], alpha=0.15)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Gbest Fitness (stability index)", fontsize=12)
    ax.set_title("PSO Convergence: Multi-Scenario Comparison\n(mean +/- 1 std across trials)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "convergence_comparison.png")


# ============================================================================
# Constraint Sensitivity (Tornado)
# ============================================================================


def plot_sensitivity_tornado(tornado_data, baseline_fitness, scenario_name):
    """Tornado chart: left = decrease constraint, right = increase constraint.

    Each parameter gets TWO horizontal bars:
      - LEFT  (red)   : impact when constraint is DECREASED  (tightened)
      - RIGHT (green) : impact when constraint is INCREASED  (loosened)

    Labels are placed at the bar ends for readability.
    """
    if not tornado_data:
        print("  [WARN] No tornado data to plot.")
        return

    import pandas as pd
    df = pd.DataFrame(tornado_data)

    # Aggregate: for each parameter, get "decrease" and "increase" delta values
    param_data = {}
    for _, row in df.iterrows():
        p = row["parameter"]
        if p not in param_data:
            param_data[p] = {"decrease": 0.0, "increase": 0.0}
        param_data[p][row["direction"]] = row["delta_fitness"]

    # Sort by total impact range
    items = sorted(param_data.items(), key=lambda kv: abs(kv[1]["increase"] - kv[1]["decrease"]))

    param_names = [k for k, _ in items]
    dec_values = [v["decrease"] for _, v in items]
    inc_values = [v["increase"] for _, v in items]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(items)) + 1))
    y_pos = np.arange(len(items))

    bar_height = 0.35

    # Left bars: decrease (negative impact typically on left)
    ax.barh(y_pos + bar_height / 2, dec_values, height=bar_height,
            color="tomato", edgecolor="white", alpha=0.85, label="Decrease constraint")
    # Right bars: increase
    ax.barh(y_pos - bar_height / 2, inc_values, height=bar_height,
            color="mediumseagreen", edgecolor="white", alpha=0.85, label="Increase constraint")

    # Value labels
    for i, (dec, inc) in enumerate(zip(dec_values, inc_values)):
        fmt = "+.4f" if max(abs(dec), abs(inc)) < 0.01 else "+.3f"
        if dec < 0:
            ax.text(dec - 0.003, i + bar_height / 2, f"{dec:{fmt}}",
                    ha="right", va="center", fontsize=8.5, color="darkred")
        else:
            ax.text(dec + 0.003, i + bar_height / 2, f"{dec:{fmt}}",
                    ha="left", va="center", fontsize=8.5, color="darkred")
        if inc < 0:
            ax.text(inc - 0.003, i - bar_height / 2, f"{inc:{fmt}}",
                    ha="right", va="center", fontsize=8.5, color="darkgreen")
        else:
            ax.text(inc + 0.003, i - bar_height / 2, f"{inc:{fmt}}",
                    ha="left", va="center", fontsize=8.5, color="darkgreen")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names, fontsize=10)
    ax.axvline(x=0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel(f"Change in Gbest Fitness (baseline = {baseline_fitness:.4f})", fontsize=11)
    ax.set_title(f"Constraint Sensitivity: {scenario_name}\n"
                 f"(Perturb each bound by +/-{SCENARIO_SENSITIVITY_DELTA*100:.0f}%)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    _save(fig, f"constraint_sensitivity_{scenario_name.split()[0].lower()}.png")


# ============================================================================
# SHAP Summary Plot
# ============================================================================


def plot_shap_summary(shap_values, X_background, feature_names=None):
    """SHAP summary bar plot with logarithmic x-axis."""
    if feature_names is None:
        feature_names = FEATURE_COLS

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs_shap)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(feature_names)))

    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, mean_abs_shap[order], color=colors, edgecolor="white", alpha=0.88)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=11)
    ax.set_xlabel("Mean |SHAP Value|  (log scale)", fontsize=11)
    ax.set_title("SHAP Feature Importance (XGBoost TreeExplainer)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.grid(axis="x", alpha=0.25)

    # Value labels with decimal places scaling
    for i, (pos, val) in enumerate(zip(y_pos, mean_abs_shap[order])):
        if val >= 10:
            label = f"{val:.1f}"
        elif val >= 1:
            label = f"{val:.2f}"
        else:
            label = f"{val:.4f}"
        ax.text(val * 1.08, pos, label, va="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    _save(fig, "shap_summary.png")


# ============================================================================
# SHAP Waterfall
# ============================================================================


def plot_shap_waterfall(shap_values, expected_value, positions, labels=None,
                        scenario_name="", feature_names=None):
    """Diverging bar chart of SHAP contributions centered at x=0.

    Uses symlog x-axis (linthresh=0.15) so that both large features
    (downforce ~5.7) and small features (drag ~0.1, wing ~0.07)
    are visible simultaneously.
    """
    if feature_names is None:
        feature_names = FEATURE_COLS
    if labels is None:
        labels = [scenario_name]

    for idx, label in enumerate(labels):
        sv = shap_values[idx]
        pos = positions[idx]
        base = expected_value
        final_pred = base + sv.sum()

        # Sort by absolute SHAP descending (top = most influential)
        order = np.argsort(np.abs(sv))[::-1]
        sorted_sv = sv[order]
        sorted_names = [feature_names[i] for i in order]
        sorted_pos = pos[order]

        fig, ax = plt.subplots(figsize=(10, 5.5))
        bar_height = 0.55
        y_pos = np.arange(len(sorted_sv))

        # Symlog: linear within threshold, log beyond
        linthresh = max(abs(sv).min() * 2, 0.05) if abs(sv).min() > 1e-10 else 0.05

        # Draw bar from 0 to shap_val for each feature
        for i, (shap_val, name) in enumerate(zip(sorted_sv, sorted_names)):
            color = "mediumseagreen" if shap_val >= 0 else "tomato"
            ax.barh(i, shap_val, height=bar_height, color=color,
                    edgecolor="white", alpha=0.88)

        ax.axvline(x=0, color="black", linewidth=1.2, alpha=0.5)

        # Use symlog so small SHAP values are not invisible
        abs_max = abs(sv).max()
        ax.set_xscale("symlog", linthresh=linthresh, linscale=0.5)
        ax.set_xlim(-abs_max * 1.35, abs_max * 1.35)

        # Value annotations on each bar
        for i, (shap_val, feat_name, feat_val) in enumerate(
                zip(sorted_sv, sorted_names, sorted_pos)):
            # Format feature value
            if abs(feat_val) < 1 and feat_val != 0:
                fv = f"{feat_val:.3f}"
            elif feat_val == int(feat_val):
                fv = f"{int(feat_val)}"
            else:
                fv = f"{feat_val:.1f}"

            # Format SHAP value
            if abs(shap_val) >= 1:
                shap_str = f"{shap_val:+.2f}"
            elif abs(shap_val) >= 0.01:
                shap_str = f"{shap_val:+.4f}"
            else:
                shap_str = f"{shap_val:+.6f}"

            annotation = f"{shap_str}  ({feat_name}={fv})"

            # Place label near bar end — use data-space offset
            offset = abs_max * 0.03
            if shap_val >= 0:
                ax.text(shap_val + offset, i, annotation,
                        va="center", fontsize=8.5, color="#1a1a1a")
            else:
                ax.text(shap_val - offset, i, annotation,
                        va="center", ha="right", fontsize=8.5, color="#1a1a1a")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=10)
        ax.set_xlabel("SHAP Contribution  (symlog scale)", fontsize=11)
        ax.set_title(f"SHAP Waterfall: {label}\n"
                     f"f(x) = {final_pred:.2f}  =  E[f(x)]={base:.2f}  +  sum(SHAP)={sv.sum():+.2f}",
                     fontsize=12, fontweight="bold", loc="center")
        ax.grid(axis="x", alpha=0.22)
        ax.invert_yaxis()

        fig.tight_layout()
        fname = f"shap_waterfall_{label.replace(' ', '_').replace('/', '_')}.png"
        _save(fig, fname)


# ============================================================================
# Permutation Importance
# ============================================================================


def plot_permutation_importance(pi_result):
    """Permutation importance bar chart with logarithmic x-axis."""
    importances = pi_result["importances_mean"]
    stds = pi_result["importances_std"]
    names = pi_result["feature_names"]
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, importances[order], xerr=stds[order],
            color="steelblue", edgecolor="white", alpha=0.88, capsize=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[i] for i in order], fontsize=11)
    ax.set_xlabel("Permutation Importance — R² decrease  (log scale)", fontsize=11)
    ax.set_title("Permutation Feature Importance", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.grid(axis="x", alpha=0.25)

    for i, (pos, val) in enumerate(zip(y_pos, importances[order])):
        if val >= 1:
            label = f"{val:.2f}"
        elif val >= 0.01:
            label = f"{val:.4f}"
        else:
            label = f"{val:.6f}"
        ax.text(val * 1.08, pos, label, va="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    _save(fig, "permutation_importance.png")


# ============================================================================
# Scenario Fitness Comparison
# ============================================================================


def plot_scenario_fitness_comparison(all_results):
    """Bar chart comparing mean fitness (+/-std) across scenarios."""
    codes = sorted(all_results.keys())
    means = [all_results[c][1]["fitness_mean"] for c in codes]
    stds = [all_results[c][1]["fitness_std"] for c in codes]

    fig, ax = plt.subplots(figsize=(9, 5))
    n = len(codes)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(n, 2)))

    bars = ax.bar(range(n), means, yerr=stds, color=colors,
                  edgecolor="white", capsize=6, alpha=0.88)
    ax.set_xticks(range(n))
    ax.set_xticklabels(codes, fontsize=11)
    ax.set_ylabel("Gbest Fitness (stability index)", fontsize=11)
    ax.set_title("Scenario Fitness Comparison\n(mean +/- std across trials)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.15,
                f"{mean:.3f}+/-{std:.4f}", ha="center", fontsize=8.5, fontweight="bold")

    fig.tight_layout()
    _save(fig, "scenario_fitness_comparison.png")


# ============================================================================
# Main plotting entry point
# ============================================================================


def plot_scenario_report(all_results, shap_values=None, X_background=None,
                         tornado_data_list=None, pi_result=None,
                         global_bounds=None, show_baseline=None):
    """Generate all scenario plots."""
    print("\n  [Plot] Scenario radar...")
    plot_scenario_radar(all_results, global_bounds=global_bounds,
                        show_baseline=show_baseline)

    print("  [Plot] Convergence comparison...")
    plot_convergence_comparison(all_results)

    print("  [Plot] Scenario fitness comparison...")
    plot_scenario_fitness_comparison(all_results)

    if shap_values is not None:
        print("  [Plot] SHAP summary...")
        plot_shap_summary(shap_values, X_background)

    if pi_result is not None:
        print("  [Plot] Permutation importance...")
        plot_permutation_importance(pi_result)

    if tornado_data_list:
        for td, bl, sn in tornado_data_list:
            if td:
                print(f"  [Plot] Sensitivity tornado ({sn})...")
                plot_sensitivity_tornado(td, bl, sn)
