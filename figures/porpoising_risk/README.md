# Porpoising Risk Analysis — Figure Descriptions

> Experiment: `experiments/run_porpoising_risk.py`
> Generated: 2026-05-31
> Model: DeepEnsemble refined (3 × MLP), PyTorch autograd for gradient

---

## 1. Overview

T07 computes local gradient norms of the surrogate model in `(speed_kmh, wing_angle_deg)` space as a proxy for porpoising risk, then extends to second-order Hessian curvature analysis. 10 figures total.

### 1.1 Data Source

- Model: `outputs/models/deep_ensemble_refined/mlp_{0,1,2}.pt`
- Scaler: `data/processed/scaler_mm.pkl`
- Scenario overlays: `outputs/scenarios/*/best_solution.json` (T06 v2 multi-objective)
- Grid: 80×80 spanning `PSO_PARAM_BOUNDS`: speed [80, 360] km/h, wing [0, 45] deg

### 1.2 Key Design Decisions

- **Gradient via autograd**: Averaged over 3 ensemble members for stability
- **KNN(K=10)** estimates downforce_n/drag_n for each grid point from training data
- **Log-scale colour mapping** (LogNorm) for visibility in right-skewed risk distribution
- **Hatchet OOD overlay**: Grey hatched stripes mark regions beyond training data coverage `[100,365]×[5,35]`
- **GridSpec layout**: Dedicated colourbar axis prevents right-subplot squeeze
- **Hessian extension**: Central finite differences on gradient fields → 2×2 Hessian eigenvalues per grid point

---

## 2. Figure Inventory

### 2.1 First-Order Gradient Risk (5 figures)

| Figure | Description |
|--------|-------------|
| `risk_heatmap_drs0.png` | Log-scale risk with Zone A (high-speed+small wing) and Zone B (low-speed+large wing) annotations, OOD hatched overlay, p90 contour, scenario optimal solutions |
| `risk_heatmap_drs1.png` | Same, DRS=1 |
| `risk_vs_optimal_overlay.png` | Side-by-side DRS=0 / DRS=1 comparison with shared colour scale |
| `stability_surface_drs0.png` | Predicted stability_index surface (`RdYlGn` colourmap) — shows model near-saturation |
| `stability_surface_drs1.png` | Same, DRS=1 |

**Risk metric**: `Risk(x) = ||∇f|| = sqrt((∂f/∂v)² + (∂f/∂α)²)` in real physical units (stability_index per km/h and per degree).

### 2.2 Second-Order Hessian Curvature (5 figures)

| Figure | Description |
|--------|-------------|
| `hessian_curvature_drs0.png` | Curvature strength `max(|λ₁|, |λ₂|)` — warm colours = high curvature |
| `hessian_curvature_drs1.png` | Same, DRS=1 |
| `hessian_concavity_drs0.png` | Binary classification: **red** = concave (λ_min < 0, potentially unstable), **green** = convex (safe). Concave clusters outlined with dashed boxes |
| `hessian_concavity_drs1.png` | Same, DRS=1 |
| `hessian_summary.png` | 2×3 overview panel: Curvature DRS=0/1, Concavity DRS=0/1, Anisotropy DRS=0/1 |

**Key finding**: ~70% of grid points show concave behaviour (λ_min < 0) but curvature magnitude is very low (mean ~0.03), suggesting the MLP prediction surface is so flat that eigenvalue sign is dominated by numerical noise. High anisotropy (386–6324 for scenario solutions) indicates single-direction curvature dominance.

---

## 3. Interpretation Guide

| Symbol | Meaning |
|--------|---------|
| **Zone A** (red rectangle) | High-speed + Small wing — traditional porpoising danger zone |
| **Zone B** (blue rectangle) | Low-speed + Large wing — traditionally stable zone |
| **Gray hatching** | OOD extrapolation beyond training data `[100,365]×[5,35]` |
| **Dashed contour** | p90 risk boundary |
| **Scenario markers** | S1=square, S2=circle, S3=triangle, S4=diamond, with dark edges |

### Physical Intuition Status
- High-speed+small-wing zone shows **higher** mean risk (0.0013) than low-speed+large-wing zone (0.0002) — **passes** F1 aerodynamic intuition
- DRS ON increases risk slightly (ratio 1.01) — consistent with reduced rear grip
