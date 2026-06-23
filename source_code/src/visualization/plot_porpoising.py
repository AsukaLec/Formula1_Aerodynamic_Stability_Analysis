"""Porpoising Risk Heatmap Generation.

Computes local gradient norm of the surrogate model in (speed_kmh, wing_angle_deg)
space as a proxy for porpoising risk, and generates risk heatmaps overlaid with
scenario optimal solutions.

Risk(x) = ||∇f(speed, wing_angle)|| = sqrt((∂f/∂v)² + (∂f/∂α)²)
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from sklearn.neighbors import NearestNeighbors

from src.utils.config import FEATURE_COLS, FIGURES_DIR, PSO_PARAM_BOUNDS

PORPOISING_FIGURES_DIR = os.path.join(FIGURES_DIR, "porpoising_risk")
os.makedirs(PORPOISING_FIGURES_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(PORPOISING_FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _add_ood_overlay(ax, grid_speed, grid_wing, train_speed_bounds, train_wing_bounds):
    """Add hatched overlay indicating OOD regions beyond training data coverage."""
    speed_min, speed_max = train_speed_bounds
    wing_min, wing_max = train_wing_bounds

    ood_kwargs = dict(facecolor="gray", alpha=0.18, hatch="////", zorder=2, linewidth=0)

    gs_min, gs_max = grid_speed[0], grid_speed[-1]
    gw_min, gw_max = grid_wing[0], grid_wing[-1]

    if gs_min < speed_min:
        rect = matplotlib.patches.Rectangle(
            (gs_min, gw_min), speed_min - gs_min, gw_max - gw_min, **ood_kwargs)
        ax.add_patch(rect)
    if gs_max > speed_max:
        rect = matplotlib.patches.Rectangle(
            (speed_max, gw_min), gs_max - speed_max, gw_max - gw_min, **ood_kwargs)
        ax.add_patch(rect)
    if gw_min < wing_min:
        rect = matplotlib.patches.Rectangle(
            (gs_min, gw_min), gs_max - gs_min, wing_min - gw_min, **ood_kwargs)
        ax.add_patch(rect)
    if gw_max > wing_max:
        rect = matplotlib.patches.Rectangle(
            (gs_min, wing_max), gs_max - gs_min, gw_max - wing_max, **ood_kwargs)
        ax.add_patch(rect)

    # OOD label
    ax.text(0.98, 0.02, "hatch = OOD extrapolation",
            transform=ax.transAxes, fontsize=6.5, color="gray",
            ha="right", va="bottom", style="italic")


def compute_porpoising_risk(ensemble, scaler_mm, X_train_orig, n_grid=80, knn_k=10):
    """Compute risk grid for DRS=0 and DRS=1 using autograd on DeepEnsemble.

    Grid bounds use PSO_PARAM_BOUNDS to cover the full search space.
    Training data range is used for gradient chain-rule scaling only.

    Parameters
    ----------
    ensemble : DeepEnsemble (in eval mode)
    scaler_mm : fitted MinMaxScaler
    X_train_orig : np.ndarray of shape (N, 5) in original feature scale
    n_grid : int, grid resolution
    knn_k : int, K neighbours for downforce/drag estimation

    Returns
    -------
    dict with keys:
        "grid_speed": 1D array of speed_kmh values
        "grid_wing": 1D array of wing_angle_deg values
        "risk_drs0": (n_grid, n_grid) risk values for DRS=0
        "risk_drs1": (n_grid, n_grid) risk values for DRS=1
        "train_speed_bounds": (min, max) training data range for OOD detection
        "train_wing_bounds": (min, max) training data range for OOD detection
    """
    ensemble.eval()
    device = ensemble.trainers[0].device

    i_speed = FEATURE_COLS.index("speed_kmh")
    i_wing = FEATURE_COLS.index("wing_angle_deg")
    i_drs = FEATURE_COLS.index("drs_active")

    # PSO parameter bounds for full search space
    grid_speed = np.linspace(PSO_PARAM_BOUNDS[i_speed][0], PSO_PARAM_BOUNDS[i_speed][1], n_grid)
    grid_wing = np.linspace(PSO_PARAM_BOUNDS[i_wing][0], PSO_PARAM_BOUNDS[i_wing][1], n_grid)
    A_speed, A_wing = np.meshgrid(grid_speed, grid_wing, indexing="ij")

    # Training data range for gradient chain-rule scaling
    train_mins = scaler_mm.data_min_
    train_maxs = scaler_mm.data_max_
    speed_range = train_maxs[i_speed] - train_mins[i_speed]
    wing_range = train_maxs[i_wing] - train_mins[i_wing]
    train_speed_bounds = (train_mins[i_speed], train_maxs[i_speed])
    train_wing_bounds = (train_mins[i_wing], train_maxs[i_wing])

    # KNN for downforce/drag estimation
    ref_xy = X_train_orig[:, [i_speed, i_wing, i_drs]]
    other_indices = [j for j in range(5) if j not in (i_speed, i_wing, i_drs)]
    ref_rest = X_train_orig[:, other_indices]

    nn = NearestNeighbors(n_neighbors=knn_k)
    nn.fit(ref_xy)

    risk_results = {}

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

        grad_accum = None
        for trainer in ensemble.trainers:
            trainer.model.eval()
            trainer.model.zero_grad()

            X_t = torch.tensor(feats_scaled, dtype=torch.float32, device=device, requires_grad=True)
            out = trainer.model(X_t)
            out.sum().backward()

            g = X_t.grad.detach().cpu().numpy()

            if grad_accum is None:
                grad_accum = g
            else:
                grad_accum += g

        avg_grad_norm = grad_accum / ensemble.M

        grad_speed_real = 100.0 * avg_grad_norm[:, i_speed] / speed_range
        grad_wing_real = 100.0 * avg_grad_norm[:, i_wing] / wing_range

        risk = np.sqrt(grad_speed_real ** 2 + grad_wing_real ** 2)
        risk = risk.reshape(n_grid, n_grid)
        gs = grad_speed_real.reshape(n_grid, n_grid)
        gw = grad_wing_real.reshape(n_grid, n_grid)

        risk_results[f"risk_drs{drs}"] = risk
        risk_results[f"grad_speed_drs{drs}"] = gs
        risk_results[f"grad_wing_drs{drs}"] = gw

    risk_results["grid_speed"] = grid_speed
    risk_results["grid_wing"] = grid_wing
    risk_results["train_speed_bounds"] = train_speed_bounds
    risk_results["train_wing_bounds"] = train_wing_bounds

    return risk_results


def plot_risk_heatmaps(risk_data, best_solutions, stability_surface=None, output_dir=None):
    """Generate all porpoising risk figures.

    Creates:
      - risk_heatmap_drs0.png       (log-scaled, OOD-annotated)
      - risk_heatmap_drs1.png       (log-scaled, OOD-annotated)
      - risk_vs_optimal_overlay.png  (side-by-side, GridSpec layout)
      - stability_surface_drs0.png  (predicted stability_index, DRS=0)
      - stability_surface_drs1.png  (predicted stability_index, DRS=1)
    """
    output_dir = output_dir or PORPOISING_FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    grid_speed = risk_data["grid_speed"]
    grid_wing = risk_data["grid_wing"]
    risk_drs0 = risk_data["risk_drs0"]
    risk_drs1 = risk_data["risk_drs1"]
    train_speed_bounds = risk_data.get("train_speed_bounds", (grid_speed[0], grid_speed[-1]))
    train_wing_bounds = risk_data.get("train_wing_bounds", (grid_wing[0], grid_wing[-1]))

    all_risk = np.concatenate([risk_drs0.ravel(), risk_drs1.ravel()])
    vmin = np.percentile(all_risk, 10)
    vmax_99 = np.percentile(all_risk, 99)

    log_vmin = max(vmin, max(all_risk[all_risk > 0].min() if (all_risk > 0).any() else 1e-6, 1e-6) / 2)
    log_vmax = vmax_99
    levels = np.logspace(np.log10(log_vmin), np.log10(log_vmax), 40)

    scenario_colors = {"S1_monza": "#e74c3c", "S2_monaco": "#3498db",
                       "S3_balanced": "#2ecc71", "S4_wet": "#9b59b6"}
    scenario_markers = {"S1_monza": "s", "S2_monaco": "o",
                        "S3_balanced": "^", "S4_wet": "D"}
    scenario_edgecolors = {"S1_monza": "darkred", "S2_monaco": "darkblue",
                           "S3_balanced": "darkgreen", "S4_wet": "darkviolet"}

    # ========== Individual heatmaps with physics annotations ==========
    for drs_val, risk_grid, drs_label, filename in [
        (0, risk_drs0, "DRS OFF", "risk_heatmap_drs0.png"),
        (1, risk_drs1, "DRS ON", "risk_heatmap_drs1.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 7))

        c = ax.contourf(grid_speed, grid_wing, risk_grid.T, levels=levels,
                        cmap="YlOrRd", extend="max",
                        norm=matplotlib.colors.LogNorm(vmin=log_vmin, vmax=log_vmax))
        cb = plt.colorbar(c, ax=ax, shrink=0.82)
        cb.set_label("Porpoising Risk = ||grad||  (log scale)", fontsize=10)

        _add_ood_overlay(ax, grid_speed, grid_wing, train_speed_bounds, train_wing_bounds)

        p90 = np.percentile(risk_grid, 90)
        ax.contour(grid_speed, grid_wing, risk_grid.T, levels=[p90],
                   colors="darkred", linewidths=1.5, linestyles="--", alpha=0.8)
        ax.text(grid_speed[-1] - 5, grid_wing[-1] - 1,
                f"p90: {p90:.3f}", fontsize=7, color="darkred",
                ha="right", va="top", alpha=0.8)

        # Zone A: High-speed + small wing
        za_x = grid_speed[int(len(grid_speed) * 0.60)]
        za_w = grid_speed[-1] - za_x
        za_y = grid_wing[0]
        za_h = grid_wing[int(len(grid_wing) * 0.33)] - za_y
        zone_a_rect = matplotlib.patches.Rectangle(
            (za_x, za_y), za_w, za_h,
            linewidth=1.5, edgecolor="#e74c3c", facecolor="none",
            linestyle="-", alpha=0.7)
        ax.add_patch(zone_a_rect)
        ax.annotate("Zone A: High-speed\n    + Small wing\n    (danger zone)",
                    xy=(za_x + za_w * 0.45, za_y + za_h * 0.35),
                    fontsize=7.5, color="#e74c3c", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

        # Zone B: Low-speed + large wing
        zb_x = grid_speed[0]
        zb_w = grid_speed[int(len(grid_speed) * 0.40)] - zb_x
        zb_y = grid_wing[int(len(grid_wing) * 0.67)]
        zb_h = grid_wing[-1] - zb_y
        zone_b_rect = matplotlib.patches.Rectangle(
            (zb_x, zb_y), zb_w, zb_h,
            linewidth=1.5, edgecolor="#3498db", facecolor="none",
            linestyle="-", alpha=0.7)
        ax.add_patch(zone_b_rect)
        ax.annotate("Zone B: Low-speed\n    + Large wing\n    (safe zone)",
                    xy=(zb_x + zb_w * 0.45, zb_y + zb_h * 0.65),
                    fontsize=7.5, color="#3498db", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

        for sol in best_solutions:
            if sol["drs_active"] == drs_val:
                color = scenario_colors.get(sol["scenario"], "black")
                marker = scenario_markers.get(sol["scenario"], "*")
                ec = scenario_edgecolors.get(sol["scenario"], "black")
                ax.scatter(sol["speed_kmh"], sol["wing_angle_deg"],
                           c=color, marker=marker, s=160, edgecolors=ec,
                           linewidths=1.2, zorder=5,
                           label=f"{sol['scenario_name']} ({sol['gbest_fitness']:.1f})")

        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9, ncol=1)
        ax.set_xlabel("Speed (km/h)", fontsize=11)
        ax.set_ylabel("Wing Angle (deg)", fontsize=11)
        ax.set_title(f"Porpoising Risk Heatmap — {drs_label}", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.25)

        fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {os.path.join(output_dir, filename)}")

    # ========== Combined overlay (GridSpec layout, no squeeze) ==========
    fig = plt.figure(figsize=(18, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.04])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    for ax, risk_grid, drs_val, drs_label in [
        (ax0, risk_drs0, 0, "DRS OFF"),
        (ax1, risk_drs1, 1, "DRS ON"),
    ]:
        c = ax.contourf(grid_speed, grid_wing, risk_grid.T, levels=levels,
                        cmap="YlOrRd", extend="max",
                        norm=matplotlib.colors.LogNorm(vmin=log_vmin, vmax=log_vmax))

        _add_ood_overlay(ax, grid_speed, grid_wing, train_speed_bounds, train_wing_bounds)

        p90 = np.percentile(risk_grid, 90)
        ax.contour(grid_speed, grid_wing, risk_grid.T, levels=[p90],
                   colors="darkred", linewidths=1.2, linestyles="--", alpha=0.7)

        za_x = grid_speed[int(len(grid_speed) * 0.60)]
        za_w = grid_speed[-1] - za_x
        za_y = grid_wing[0]
        za_h = grid_wing[int(len(grid_wing) * 0.33)] - za_y
        zone_a = matplotlib.patches.Rectangle(
            (za_x, za_y), za_w, za_h,
            linewidth=1.2, edgecolor="#e74c3c", facecolor="none", linestyle="-", alpha=0.5)
        ax.add_patch(zone_a)
        zb_x = grid_speed[0]
        zb_w = grid_speed[int(len(grid_speed) * 0.40)] - zb_x
        zb_y = grid_wing[int(len(grid_wing) * 0.67)]
        zb_h = grid_wing[-1] - zb_y
        zone_b = matplotlib.patches.Rectangle(
            (zb_x, zb_y), zb_w, zb_h,
            linewidth=1.2, edgecolor="#3498db", facecolor="none", linestyle="-", alpha=0.5)
        ax.add_patch(zone_b)

        for sol in best_solutions:
            if sol["drs_active"] == drs_val:
                color = scenario_colors.get(sol["scenario"], "black")
                marker = scenario_markers.get(sol["scenario"], "*")
                ec = scenario_edgecolors.get(sol["scenario"], "black")
                ax.scatter(sol["speed_kmh"], sol["wing_angle_deg"],
                           c=color, marker=marker, s=130, edgecolors=ec,
                           linewidths=1.0, zorder=5,
                           label=f"{sol['scenario_name']} ({sol['gbest_fitness']:.1f})")

        ax.set_xlabel("Speed (km/h)", fontsize=11)
        ax.set_ylabel("Wing Angle (deg)", fontsize=11)
        ax.set_title(f"Porpoising Risk — {drs_label}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
        ax.grid(alpha=0.25)

    cb = fig.colorbar(c, cax=cax)
    cb.set_label("Risk = ||grad|| (log)", fontsize=10)

    fig.suptitle("Porpoising Risk Heatmaps with Scenario Optimal Solutions  (hatch = OOD extrapolation)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    overlay_path = os.path.join(output_dir, "risk_vs_optimal_overlay.png")
    fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {overlay_path}")

    # ========== Stability surface plots (predicted stability_index) ==========
    if stability_surface is not None:
        for drs_val, surface_grid, drs_label, filename in [
            (0, stability_surface["surface_drs0"], "DRS OFF", "stability_surface_drs0.png"),
            (1, stability_surface["surface_drs1"], "DRS ON", "stability_surface_drs1.png"),
        ]:
            fig, ax = plt.subplots(figsize=(9, 6.5))
            c = ax.contourf(grid_speed, grid_wing, surface_grid.T, levels=30,
                            cmap="RdYlGn")
            cb = plt.colorbar(c, ax=ax, shrink=0.82)
            cb.set_label("Predicted stability_index", fontsize=10)

            _add_ood_overlay(ax, grid_speed, grid_wing, train_speed_bounds, train_wing_bounds)

            for sol in best_solutions:
                if sol["drs_active"] == drs_val:
                    color = scenario_colors.get(sol["scenario"], "black")
                    marker = scenario_markers.get(sol["scenario"], "*")
                    ec = scenario_edgecolors.get(sol["scenario"], "black")
                    ax.scatter(sol["speed_kmh"], sol["wing_angle_deg"],
                               c=color, marker=marker, s=140, edgecolors=ec,
                               linewidths=1.0, zorder=5,
                               label=f"{sol['scenario_name']} ({sol['gbest_fitness']:.1f})")

            ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
            ax.set_xlabel("Speed (km/h)", fontsize=11)
            ax.set_ylabel("Wing Angle (deg)", fontsize=11)
            ax.set_title(f"Predicted Stability Surface — {drs_label}", fontsize=12, fontweight="bold")
            ax.grid(alpha=0.25)

            fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {os.path.join(output_dir, filename)}")


def compute_hessian_analysis(risk_data):
    """Compute Hessian curvature and concavity from gradient fields.

    Uses central finite differences on the gradient grids to estimate
    second-order partial derivatives at interior points.

    Returns dict with:
      - "curvature_drs{0,1}": (n_grid-2, n_grid-2) max eigenvalue magnitude
      - "concavity_drs{0,1}": (n_grid-2, n_grid-2) binary mask (1=concave)
      - "anisotropy_drs{0,1}": (n_grid-2, n_grid-2) eigenvalue ratio
      - "grid_speed_inner", "grid_wing_inner": 1D arrays for interior points
    """
    grid_speed = risk_data["grid_speed"]
    grid_wing = risk_data["grid_wing"]
    dv = grid_speed[1] - grid_speed[0]
    da = grid_wing[1] - grid_wing[0]

    hessian_data = {
        "grid_speed_inner": grid_speed[1:-1],
        "grid_wing_inner": grid_wing[1:-1],
    }

    for drs in [0, 1]:
        gs = risk_data[f"grad_speed_drs{drs}"]  # (n, n)
        gw = risk_data[f"grad_wing_drs{drs}"]

        # Second derivatives via central differences on gradient fields
        # interior: i in [1, n-2], j in [1, n-2]
        n = gs.shape[0]
        d2f_dv2 = (gs[2:, 1:-1] - gs[:-2, 1:-1]) / (2 * dv)       # ∂²f/∂v²
        d2f_da2 = (gw[1:-1, 2:] - gw[1:-1, :-2]) / (2 * da)       # ∂²f/∂α²
        cross_v = (gs[1:-1, 2:] - gs[1:-1, :-2]) / (2 * da)       # ∂(∂f/∂v)/∂α
        cross_a = (gw[2:, 1:-1] - gw[:-2, 1:-1]) / (2 * dv)       # ∂(∂f/∂α)/∂v
        d2f_dvda = (cross_v + cross_a) / 2.0                       # symmetrised

        # Eigenvalues of [[a, b], [b, c]] with a=d2f_dv2, b=d2f_dvda, c=d2f_da2
        trace = d2f_dv2 + d2f_da2
        det = d2f_dv2 * d2f_da2 - d2f_dvda ** 2
        discriminant = np.maximum((trace / 2.0) ** 2 - det, 0)
        sqrt_disc = np.sqrt(discriminant)
        lam1 = trace / 2.0 + sqrt_disc
        lam2 = trace / 2.0 - sqrt_disc

        curvature = np.maximum(np.abs(lam1), np.abs(lam2))          # curvature strength
        concavity = (np.minimum(lam1, lam2) < 0).astype(np.float64) # 1 if concave
        denom = np.maximum(np.abs(lam2), 1e-12)
        anisotropy = np.maximum(np.abs(lam1) / denom, np.abs(lam2) / np.maximum(np.abs(lam1), 1e-12))

        hessian_data[f"curvature_drs{drs}"] = curvature
        hessian_data[f"concavity_drs{drs}"] = concavity
        hessian_data[f"anisotropy_drs{drs}"] = anisotropy

    return hessian_data


def plot_hessian_analysis(hessian_data, best_solutions, output_dir=None):
    """Generate Hessian-based porpoising diagnostic figures.

    Creates:
      - hessian_curvature_drs0.png / drs1.png   — curvature strength
      - hessian_concavity_drs0.png / drs1.png   — concave zones (red=risky)
      - hessian_summary.png                      — 2x3 overview panel
    """
    output_dir = output_dir or PORPOISING_FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    gs = hessian_data["grid_speed_inner"]
    gw = hessian_data["grid_wing_inner"]

    scenario_colors = {"S1_monza": "#e74c3c", "S2_monaco": "#3498db",
                       "S3_balanced": "#2ecc71", "S4_wet": "#9b59b6"}
    scenario_markers = {"S1_monza": "s", "S2_monaco": "o",
                        "S3_balanced": "^", "S4_wet": "D"}
    scenario_edgecolors = {"S1_monza": "darkred", "S2_monaco": "darkblue",
                           "S3_balanced": "darkgreen", "S4_wet": "darkviolet"}

    # Shared limits
    all_curv = np.concatenate([hessian_data["curvature_drs0"].ravel(),
                               hessian_data["curvature_drs1"].ravel()])
    curv_vmax = np.percentile(all_curv, 98)

    for drs_val, curv_grid, conc_grid, drs_label, curv_fn, conc_fn in [
        (0, hessian_data["curvature_drs0"], hessian_data["concavity_drs0"],
         "DRS OFF", "hessian_curvature_drs0.png", "hessian_concavity_drs0.png"),
        (1, hessian_data["curvature_drs1"], hessian_data["concavity_drs1"],
         "DRS ON", "hessian_curvature_drs1.png", "hessian_concavity_drs1.png"),
    ]:
        # --- Curvature ---
        fig, ax = plt.subplots(figsize=(9, 6.5))
        c = ax.contourf(gs, gw, curv_grid.T, levels=30, cmap="YlOrRd",
                        vmin=0, vmax=curv_vmax)
        cb = plt.colorbar(c, ax=ax, shrink=0.82)
        cb.set_label("Hessian Curvature  max(|λ₁|, |λ₂|)", fontsize=10)
        for sol in best_solutions:
            if sol["drs_active"] == drs_val:
                color = scenario_colors.get(sol["scenario"], "black")
                marker = scenario_markers.get(sol["scenario"], "*")
                ec = scenario_edgecolors.get(sol["scenario"], "black")
                ax.scatter(sol["speed_kmh"], sol["wing_angle_deg"],
                           c=color, marker=marker, s=160, edgecolors=ec,
                           linewidths=1.2, zorder=5,
                           label=f"{sol['scenario_name']} ({sol['gbest_fitness']:.2f})")
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
        ax.set_xlabel("Speed (km/h)", fontsize=11)
        ax.set_ylabel("Wing Angle (deg)", fontsize=11)
        ax.set_title(f"Hessian Curvature — {drs_label}", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.25)
        fig.savefig(os.path.join(output_dir, curv_fn), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {os.path.join(output_dir, curv_fn)}")

        # --- Concavity ---
        fig, ax = plt.subplots(figsize=(9, 6.5))
        cmap_conc = matplotlib.colors.ListedColormap(["#2ecc71", "#e74c3c"])
        ax.pcolormesh(gs, gw, conc_grid.T, cmap=cmap_conc, vmin=0, vmax=1, alpha=0.7)
        # Outline concave clusters
        from scipy import ndimage
        if conc_grid.any():
            labeled, n_feat = ndimage.label(conc_grid)
            for i in range(1, n_feat + 1):
                mask = labeled == i
                if mask.sum() > 3:
                    yy, xx = np.where(mask)
                    r_min, r_max = yy.min(), yy.max()
                    c_min, c_max = xx.min(), xx.max()
                    rect = matplotlib.patches.Rectangle(
                        (gs[c_min], gw[r_min]),
                        gs[c_max] - gs[c_min], gw[r_max] - gw[r_min],
                        linewidth=1.2, edgecolor="#c0392b", facecolor="none",
                        linestyle="--", alpha=0.6)
                    ax.add_patch(rect)
        cbar = plt.colorbar(plt.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(0, 1), cmap=cmap_conc),
            ax=ax, shrink=0.82, ticks=[0.25, 0.75])
        cbar.ax.set_yticklabels(["Convex (safe)", "Concave (risky)"])
        for sol in best_solutions:
            if sol["drs_active"] == drs_val:
                color = scenario_colors.get(sol["scenario"], "black")
                marker = scenario_markers.get(sol["scenario"], "*")
                ec = scenario_edgecolors.get(sol["scenario"], "black")
                ax.scatter(sol["speed_kmh"], sol["wing_angle_deg"],
                           c="white", marker=marker, s=160, edgecolors=ec,
                           linewidths=1.2, zorder=5,
                           label=f"{sol['scenario_name']} ({sol['gbest_fitness']:.2f})")
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
        ax.set_xlabel("Speed (km/h)", fontsize=11)
        ax.set_ylabel("Wing Angle (deg)", fontsize=11)
        ax.set_title(f"Concave Regions (Hessian λ < 0) — {drs_label}",
                     fontsize=12, fontweight="bold")
        ax.grid(alpha=0.25)
        fig.savefig(os.path.join(output_dir, conc_fn), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {os.path.join(output_dir, conc_fn)}")

    # --- Summary panel (2x3) ---
    fig = plt.figure(figsize=(20, 12))
    gs_grid = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

    panels = [
        (0, 0, hessian_data["curvature_drs0"], "YlOrRd",
         "Curvature DRS=0", curv_vmax, False),
        (0, 1, hessian_data["curvature_drs1"], "YlOrRd",
         "Curvature DRS=1", curv_vmax, False),
        (0, 2, hessian_data["concavity_drs0"], cmap_conc,
         "Concavity DRS=0", 1, True),
        (1, 0, hessian_data["concavity_drs1"], cmap_conc,
         "Concavity DRS=1", 1, True),
        (1, 1, hessian_data["anisotropy_drs0"], "plasma",
         "Anisotropy DRS=0", np.percentile(hessian_data["anisotropy_drs0"], 98), False),
        (1, 2, hessian_data["anisotropy_drs1"], "plasma",
         "Anisotropy DRS=1", np.percentile(hessian_data["anisotropy_drs1"], 98), False),
    ]

    for row, col, data, cmap, title, vmax, is_categorical in panels:
        ax = fig.add_subplot(gs_grid[row, col])
        if is_categorical:
            ax.pcolormesh(gs, gw, data.T, cmap=cmap, vmin=0, vmax=1, alpha=0.7)
        else:
            ax.contourf(gs, gw, data.T, levels=30, cmap=cmap, vmin=0, vmax=vmax)
        for sol in best_solutions:
            col_sol = scenario_colors.get(sol["scenario"], "black")
            marker = scenario_markers.get(sol["scenario"], "*")
            ec = scenario_edgecolors.get(sol["scenario"], "black")
            ax.scatter(sol["speed_kmh"], sol["wing_angle_deg"],
                       c=col_sol if not is_categorical else "white",
                       marker=marker, s=80, edgecolors=ec, linewidths=0.8, zorder=5)
        ax.set_xlabel("Speed (km/h)", fontsize=9)
        ax.set_ylabel("Wing Angle (deg)", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.2)

    fig.suptitle("Hessian-Based Porpoising Risk Diagnostics  (concave zone = potential instability)",
                 fontsize=14, fontweight="bold")
    summary_path = os.path.join(output_dir, "hessian_summary.png")
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {summary_path}")

    return hessian_data


def compute_xgboost_risk(xgb_model, scaler_ss, X_train_orig, n_grid=80, knn_k=10, eps=1.0):
    """Fallback: compute risk using numerical finite differences on XGBoost model.

    Parameters
    ----------
    xgb_model : XGBoostModel instance
    scaler_ss : fitted StandardScaler
    X_train_orig : np.ndarray of shape (N, 5) in original feature scale
    n_grid : int
    knn_k : int
    eps : float, perturbation step in original units

    Returns
    -------
    Same dict format as compute_porpoising_risk
    """
    i_speed = FEATURE_COLS.index("speed_kmh")
    i_wing = FEATURE_COLS.index("wing_angle_deg")
    i_drs = FEATURE_COLS.index("drs_active")

    feat_mins = scaler_ss.mean_  # not exactly mins, but we use the scaler's data range
    feat_maxs = scaler_ss.mean_
    # For StandardScaler, get ranges from training data
    train_min = X_train_orig.min(axis=0)
    train_max = X_train_orig.max(axis=0)
    speed_min, speed_max = train_min[i_speed], train_max[i_speed]
    wing_min, wing_max = train_min[i_wing], train_max[i_wing]

    grid_speed = np.linspace(speed_min, speed_max, n_grid)
    grid_wing = np.linspace(wing_min, wing_max, n_grid)
    A_speed, A_wing = np.meshgrid(grid_speed, grid_wing, indexing="ij")

    ref_xy = X_train_orig[:, [i_speed, i_wing, i_drs]]
    other_indices = [j for j in range(5) if j not in (i_speed, i_wing, i_drs)]
    ref_rest = X_train_orig[:, other_indices]

    nn = NearestNeighbors(n_neighbors=knn_k)
    nn.fit(ref_xy)

    speed_range = speed_max - speed_min
    wing_range = wing_max - wing_min

    results = {}

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

        # Numerical gradient: f(x+h) - f(x-h) / (2h) in original space
        feats_plus_speed = feats_orig.copy()
        feats_minus_speed = feats_orig.copy()
        feats_plus_speed[:, i_speed] += eps
        feats_minus_speed[:, i_speed] -= eps

        feats_plus_wing = feats_orig.copy()
        feats_minus_wing = feats_orig.copy()
        feats_plus_wing[:, i_wing] += eps
        feats_minus_wing[:, i_wing] -= eps

        def _predict(feats):
            scaled = scaler_ss.transform(feats)
            return xgb_model.predict(scaled)

        f_plus_s = _predict(feats_plus_speed)
        f_minus_s = _predict(feats_minus_speed)
        f_plus_w = _predict(feats_plus_wing)
        f_minus_w = _predict(feats_minus_wing)

        grad_speed = (f_plus_s - f_minus_s) / (2 * eps)
        grad_wing = (f_plus_w - f_minus_w) / (2 * eps)

        risk = np.sqrt(grad_speed ** 2 + grad_wing ** 2)
        risk = risk.reshape(n_grid, n_grid)

        results[f"risk_drs{drs}"] = risk

    results["grid_speed"] = grid_speed
    results["grid_wing"] = grid_wing
    results["speed_range"] = speed_range
    results["wing_range"] = wing_range

    return results
