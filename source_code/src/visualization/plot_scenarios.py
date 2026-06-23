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


# ============================================================================
# PSO Particle Trajectory (PCA 2D)
# ============================================================================


def plot_trajectory(traj_data, scenario_name, pca_model, scaler, explained_var):
    """Plot PSO particle migration trajectories in PCA 2D space.

    Parameters
    ----------
    traj_data : tuple
        (positions_list, gbest_positions, gbest_fitnesses, fitnesses_list)
        as returned by collect_trajectory_data().
    scenario_name : str
    pca_model : sklearn PCA
    scaler : sklearn StandardScaler
    explained_var : np.ndarray (2,) — PC1, PC2 explained variance ratios.
    """
    positions_list, gbest_positions, gbest_fitnesses, fitnesses_list = traj_data
    n_iters = len(positions_list)
    n_particles = positions_list[0].shape[0]

    # Project gbest trajectory
    gbest_scaled = scaler.transform(gbest_positions)
    gbest_2d = pca_model.transform(gbest_scaled)

    # Project all particles per iteration
    all_scaled = scaler.transform(np.vstack(positions_list))
    all_2d = pca_model.transform(all_scaled)

    # Split back per iteration
    iter_offsets = [0]
    for pos in positions_list:
        iter_offsets.append(iter_offsets[-1] + len(pos))
    per_iter_2d = [all_2d[iter_offsets[i]:iter_offsets[i+1]] for i in range(n_iters)]

    # Build hexbin background colored by mean fitness
    per_iter_fit = fitnesses_list
    all_fitnesses = np.concatenate(per_iter_fit)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Hexbin background
    hb = ax.hexbin(all_2d[:, 0], all_2d[:, 1], C=all_fitnesses,
                   gridsize=40, cmap="YlOrRd", alpha=0.55, reduce_C_function=np.mean,
                   linewidths=0.1, edgecolors="face")
    cbar = plt.colorbar(hb, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Mean Fitness (stability proxy)", fontsize=9)

    # Particle dots: all particles, all iterations, faded
    colors_iter = plt.cm.viridis(np.linspace(0.1, 0.9, n_iters))
    for i in range(n_iters):
        ax.scatter(per_iter_2d[i][:, 0], per_iter_2d[i][:, 1],
                   s=3, alpha=0.08, color=colors_iter[i], edgecolors="none")

    # Top-5 particle trajectories (by final fitness)
    final_fitness = per_iter_fit[-1]
    top_indices = np.argsort(final_fitness)[::-1][:5]

    for pid in top_indices:
        traj = np.array([per_iter_2d[i][pid] for i in range(n_iters)])
        # Color line: light blue to dark blue
        segments = np.linspace(0, 1, n_iters)
        for s in range(n_iters - 1):
            t = segments[s]
            color = (0.45 * (1 - t), 0.6 * (1 - t), 0.85 * (0.4 + 0.6 * t))
            ax.plot(traj[s:s+2, 0], traj[s:s+2, 1], color=color,
                    linewidth=1.2, alpha=0.6)
        # Arrow at final segment
        if n_iters >= 2:
            dx = traj[-1, 0] - traj[-2, 0]
            dy = traj[-1, 1] - traj[-2, 1]
            ax.arrow(traj[-2, 0], traj[-2, 1], dx * 0.9, dy * 0.9,
                     head_width=0.15, head_length=0.2, fc="darkblue",
                     ec="darkblue", alpha=0.7, linewidth=0.5)

    # Gbest trajectory (red)
    ax.plot(gbest_2d[:, 0], gbest_2d[:, 1], "-", color="#e34a33",
            linewidth=2.5, alpha=0.85, label="gbest trajectory", zorder=5)

    # Mark gbest at intervals
    mark_iters = list(range(0, n_iters, max(1, n_iters // 8)))
    for mi in mark_iters:
        ax.plot(gbest_2d[mi, 0], gbest_2d[mi, 1], "o",
                color="#e34a33", markersize=6, alpha=0.7, zorder=6)

    # Final gbest: gold star
    ax.plot(gbest_2d[-1, 0], gbest_2d[-1, 1], "*",
            color="gold", markersize=18, markeredgecolor="darkorange",
            markeredgewidth=1.2, zorder=7)

    # Start point marker
    ax.plot(gbest_2d[0, 0], gbest_2d[0, 1], "s",
            color="white", markersize=7, markeredgecolor="dimgray",
            markeredgewidth=1.2, zorder=5)

    # Annotations
    ax.annotate("Start", (gbest_2d[0, 0], gbest_2d[0, 1]),
                textcoords="offset points", xytext=(8, -8),
                fontsize=8, color="dimgray", fontweight="bold")
    ax.annotate(f"End (f={gbest_fitnesses[-1]:.2f})",
                (gbest_2d[-1, 0], gbest_2d[-1, 1]),
                textcoords="offset points", xytext=(8, 8),
                fontsize=8, color="darkred", fontweight="bold")

    pc1_pct = explained_var[0] * 100
    pc2_pct = explained_var[1] * 100
    ax.set_xlabel(f"PC1 ({pc1_pct:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pc2_pct:.1f}%)", fontsize=11)
    ax.set_title(f"PSO Particle Migration Trajectory: {scenario_name}\n"
                 f"({n_particles} particles × {n_iters} iters, "
                 f"PCA 2D: {pc1_pct+pc2_pct:.1f}% variance retained)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.2)

    # Tight layout with colorbar
    fig.tight_layout()
    _save(fig, f"trajectory_{scenario_name.split()[0].lower()}.png")


# ============================================================================
# Multi-Objective Visualizations (T06 v2)
# ============================================================================


def plot_pareto_frontier(all_results_multiobj):
    """Pareto frontier scatter for each scenario: efficiency vs stability + power vs stability.

    Parameters
    ----------
    all_results_multiobj : dict {code: (results_list, stats_dict)}
        Results from run_scenario_trials_multiobj. Each result MUST have
        collect_candidates=True and result['components'].
    """
    import matplotlib.patches as mpatches

    for code in sorted(all_results_multiobj.keys()):
        results, stats = all_results_multiobj[code]
        scenario_code = code

        # Collect all candidate data: (stability, efficiency_raw, power_raw, fitness)
        all_stab = []; all_eff = []; all_pow = []; all_fit = []
        for r in results:
            batches = r.get("candidates", [])
            for pos_batch, _ in batches:
                # Quick component calculation (simplified for speed)
                df_vals = pos_batch[:, 3]; drag_vals = pos_batch[:, 4]
                speed = pos_batch[0, 0]  # all same for fixed-speed scenario
                stab_batch = r.get("components", {}).get("stability", 90.0)
                eff_batch = df_vals / (drag_vals + 1e-6)
                pow_batch = (drag_vals * speed) / 150000.0
                # Simple linear approximation of fitness
                w = stats.get("weights", {"w_stability": 0.4, "w_efficiency": 0.3, "w_power": 0.3})
                max_eff = max(eff_batch) if len(eff_batch) > 0 else 1
                fit_batch = (
                    w["w_stability"] * 0.95
                    + w["w_efficiency"] * np.clip(eff_batch / (max_eff + 1e-6), 0, 1)
                    - w["w_power"] * np.clip(pow_batch, 0, 1)
                )
                all_stab.extend([stab_batch] * len(pos_batch))
                all_eff.extend(eff_batch.tolist())
                all_pow.extend(pow_batch.tolist())
                all_fit.extend(fit_batch.tolist())

        all_stab = np.array(all_stab); all_eff = np.array(all_eff)
        all_pow = np.array(all_pow); all_fit = np.array(all_fit)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # Left: efficiency vs stability
        sc1 = ax1.scatter(all_eff, all_stab, c=all_fit, cmap="YlOrRd",
                          s=4, alpha=0.25, edgecolors="none")
        # Best point
        best_stab = stats["comp_means"]["stability"]
        best_eff = stats["comp_means"]["efficiency"]
        ax1.plot(best_eff, best_stab, "*", color="darkred",
                 markersize=16, markeredgecolor="white", markeredgewidth=1)
        w = stats.get("weights", {})
        ax1.set_xlabel("Efficiency = downforce / drag (N/N)", fontsize=10)
        ax1.set_ylabel("Stability Index (0-100)", fontsize=10)
        ax1.set_title(f"{code}: Efficiency vs Stability\n"
                      f"weights: ws={w.get('w_stability',''):.1f} "
                      f"we={w.get('w_efficiency',''):.1f} wp={w.get('w_power',''):.1f}",
                      fontsize=10, fontweight="bold")
        ax1.grid(alpha=0.25)
        plt.colorbar(sc1, ax=ax1, shrink=0.7).set_label("Fitness", fontsize=8)

        # Right: power vs stability
        sc2 = ax2.scatter(all_pow, all_stab, c=all_fit, cmap="YlOrRd",
                          s=4, alpha=0.25, edgecolors="none")
        best_pow = stats["comp_means"]["power"]
        ax2.plot(best_pow, best_stab, "*", color="darkred",
                 markersize=16, markeredgecolor="white", markeredgewidth=1)
        ax2.set_xlabel("Power proxy = drag * speed / max_power", fontsize=10)
        ax2.set_ylabel("Stability Index (0-100)", fontsize=10)
        ax2.set_title(f"{code}: Power vs Stability", fontsize=10, fontweight="bold")
        ax2.grid(alpha=0.25)
        plt.colorbar(sc2, ax=ax2, shrink=0.7).set_label("Fitness", fontsize=8)

        fig.tight_layout()
        _save(fig, f"pareto_frontier_{scenario_code}.png")


def plot_parallel_coordinates(all_results_multiobj):
    """Parallel coordinates: 4 scenarios across 9 dimensions.

    Axes: speed, wing, drs, downforce, drag, stability, efficiency, power, fitness_multi
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    dim_names = ["speed", "wing", "drs", "df", "drag", "stab", "eff", "pwr", "fit"]
    colors = ["#e34a33", "#1f78b4", "#2ca02c", "#8c564b"]  # S1-S4

    # Extract per-scenario mean param + component vector
    lines_data = []
    for i, code in enumerate(sorted(all_results_multiobj.keys())):
        _, stats = all_results_multiobj[code]
        row = [
            stats["param_means"]["speed_kmh"],
            stats["param_means"]["wing_angle_deg"],
            stats["param_means"]["drs_active"],
            stats["param_means"]["downforce_n"],
            stats["param_means"]["drag_n"],
            stats["comp_means"]["stability"],
            stats["comp_means"]["efficiency"],
            stats["comp_means"]["power"],
            stats["fitness_mean"],
        ]
        lines_data.append((code, row))

    # Collect all values for min-max normalization per axis
    all_cols = np.array([d[1] for d in lines_data])
    mins = all_cols.min(axis=0)
    maxs = all_cols.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1

    for i, (code, row) in enumerate(lines_data):
        normed = (np.array(row) - mins) / ranges
        ax.plot(range(len(dim_names)), normed, "o-", linewidth=2.2,
                color=colors[i], label=code, markersize=6)

    ax.set_xticks(range(len(dim_names)))
    ax.set_xticklabels(dim_names, fontsize=10)
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("Parallel Coordinates: Multi-Objective Optimal Solutions\n"
                 "(all axes min-max normalized)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "parallel_coordinates.png")


def plot_multiobj_convergence(all_results_multiobj):
    """Multi-objective fitness convergence curves (like the single-obj version)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    codes = sorted(all_results_multiobj.keys())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(codes)))

    for idx, code in enumerate(codes):
        results, _ = all_results_multiobj[code]
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
    ax.set_ylabel("Composite Fitness F(x)", fontsize=12)
    ax.set_title("PSO Convergence: Multi-Objective Scenario Comparison\n"
                 "(mean +/- 1 std across trials)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "convergence_multiobj.png")


def plot_landscape_multimodality(all_results_multiobj=None, n_samples=50000):
    """Landscape analysis via random sampling + PCA/t-SNE.
    Generates a 2x2 subplot proving multi-modality under multi-objective fitness.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Sample from global bounds
    from src.utils.config import PSO_PARAM_BOUNDS
    bounds = np.array(PSO_PARAM_BOUNDS)

    rng = np.random.RandomState(42)
    X_sampled = np.column_stack([
        rng.uniform(b[0], b[1], n_samples) for b in bounds
    ])
    # Clip wing to [20, 35]
    X_sampled[:, 1] = np.clip(X_sampled[:, 1], 20, 35)

    # Compute single-obj (XGBoost) and multi-obj fitness
    from src.optimization.fitness import load_xgb_fitness
    xgb_fitness = load_xgb_fitness()
    stab = xgb_fitness.evaluate(X_sampled)
    stab = np.clip(stab, 0, 100)

    # Multi-objective: use weights from S1 as representative
    downforce = X_sampled[:, 3]; drag = X_sampled[:, 4]; speed = X_sampled[:, 0]
    eff = downforce / (drag + 1e-6)
    pow_norm = (drag * speed) / 150000.0
    eff_norm = np.clip(eff / np.percentile(eff, 99), 0, 1)
    pow_c = np.clip(pow_norm, 0, 1)
    mo_fitness = 0.4 * stab / 100 + 0.3 * eff_norm - 0.3 * pow_c

    # PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sampled)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (1,1): PCA colored by stability
    ax = axes[0, 0]
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=stab, cmap="YlOrRd",
               s=1, alpha=0.3, edgecolors="none")
    ax.set_title(f"PCA: single-obj (stability only)\n"
                 f"PC1={pca.explained_variance_ratio_[0]*100:.1f}% "
                 f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    # (1,2): PCA colored by multi-obj fitness
    ax = axes[0, 1]
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=mo_fitness, cmap="YlOrRd",
               s=1, alpha=0.3, edgecolors="none")
    ax.set_title("PCA: multi-obj fitness (stability + eff - power)",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    # (2,1): t-SNE (subsample for speed)
    from sklearn.manifold import TSNE
    n_tsne = min(5000, n_samples)
    idx_tsne = rng.choice(n_samples, n_tsne, replace=False)
    X_tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_jobs=1).fit_transform(
        X_scaled[idx_tsne])
    ax = axes[1, 0]
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=mo_fitness[idx_tsne], cmap="YlOrRd",
               s=3, alpha=0.4, edgecolors="none")
    ax.set_title(f"t-SNE: multi-obj fitness (n={n_tsne})", fontsize=10, fontweight="bold")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

    # (2,2): Objective space KDE + single vs multi best markers
    ax = axes[1, 1]
    idx_sub = rng.choice(n_samples, min(10000, n_samples), replace=False)
    hb = ax.hexbin(eff[idx_sub], stab[idx_sub], gridsize=40,
                   cmap="YlOrRd", alpha=0.7, mincnt=1)
    # Mark single-obj best (max stability point)
    best_single_idx = np.argmax(stab[idx_sub])
    ax.plot(eff[idx_sub][best_single_idx], stab[idx_sub][best_single_idx],
            "s", color="blue", markersize=10, label="Best stability-only")
    # Mark multi-obj best (max mo_fitness point)
    best_multi_idx = np.argmax(mo_fitness[idx_sub])
    ax.plot(eff[idx_sub][best_multi_idx], stab[idx_sub][best_multi_idx],
            "*", color="darkred", markersize=14, label="Best multi-obj")
    ax.set_xlabel("Efficiency (downforce/drag)", fontsize=10)
    ax.set_ylabel("Stability Index", fontsize=10)
    ax.set_title("Objective Space: stability vs efficiency\n(KDE + best points)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    plt.colorbar(hb, ax=ax, shrink=0.7).set_label("Density", fontsize=8)

    fig.suptitle("Landscape Analysis: Single-Objective vs Multi-Objective",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "landscape_multimodality.png")
