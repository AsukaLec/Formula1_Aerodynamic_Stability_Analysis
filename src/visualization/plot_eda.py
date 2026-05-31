"""T02: Exploratory Data Analysis - Visualization Module

Generates statistical reports and publication-quality figures for the
F1 aerodynamic stability dataset.  Covers histograms, correlation
matrices, scatter plots, joint distributions, CDFs, and imbalance
quantification.
"""
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import os
from scipy.stats import skew, kurtosis

from src.utils.config import (
    PROCESSED_DIR, REPORTS_DIR, EDA_FIGURES_DIR,
    STABILITY_BINS, STABILITY_LABELS, RANDOM_STATE, DTYPE,
)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.15)

PALETTE_DRS = {0: "#2c7bb6", 1: "#d7191c"}
PALETTE_DRS_STR = {"0": "#2c7bb6", "1": "#d7191c"}
PALETTE_BINS = dict(zip(STABILITY_LABELS, ["#ca0020", "#f4a582", "#92c5de", "#0571b0"]))
DRS_LABELS = {0: "DRS OFF", 1: "DRS ON"}
FEATURE_COLS = ["speed_kmh", "wing_angle_deg", "drs_active", "downforce_n", "drag_n"]
TARGET_COL = "stability_index"

SPEED_BINS = [0, 180, 280, 400]
SPEED_LABELS = ["Low [80-180]", "Mid [180-280]", "High [280-360]"]


# ===================================================================
# Helpers
# ===================================================================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_data(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(PROCESSED_DIR, "train.csv")
    df = pd.read_csv(path)
    if "stability_label" in df.columns:
        df["stability_label"] = df["stability_label"].astype("category")
    return df


def _assign_speed_group(df: pd.DataFrame) -> pd.Series:
    return pd.cut(df["speed_kmh"], bins=SPEED_BINS, labels=SPEED_LABELS)


def _format_pct(x, pos=None):
    """Formatter that works without LaTeX."""
    return f"{x:.0f}%"


# ===================================================================
# 2.1  Statistics
# ===================================================================

def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of descriptive statistics for all numeric columns."""
    cols = [c for c in FEATURE_COLS + [TARGET_COL] if c in df.columns]
    desc = df[cols].describe(percentiles=[0.25, 0.50, 0.75]).T
    desc = desc.rename(columns={
        "25%": "Q1", "50%": "median", "75%": "Q3"
    })

    skew_vals = {}
    kurt_vals = {}
    for c in cols:
        arr = df[c].dropna()
        skew_vals[c] = skew(arr)
        kurt_vals[c] = kurtosis(arr, fisher=True)  # excess kurtosis

    desc["skewness"] = pd.Series(skew_vals)
    desc["kurtosis"] = pd.Series(kurt_vals)

    desc = desc.round(4)
    col_order = ["count", "mean", "std", "min", "Q1", "median", "Q3", "max",
                 "skewness", "kurtosis"]
    desc = desc[[c for c in col_order if c in desc.columns]]
    return desc


def generate_statistics_report(
    df: pd.DataFrame | None = None,
    output_path: str | None = None,
) -> str:
    """Write a markdown statistics report and return the path."""
    if df is None:
        df = _load_data()
    if output_path is None:
        output_path = os.path.join(REPORTS_DIR, "statistics_report.md")

    stats_df = compute_statistics(df)
    n_samples = len(df)
    n_features = len(FEATURE_COLS)

    bins_series = pd.cut(
        df[TARGET_COL], bins=STABILITY_BINS, labels=STABILITY_LABELS,
        include_lowest=True, right=False,
    )
    bin_counts = bins_series.value_counts().sort_index()
    bin_pct = (bin_counts / n_samples * 100).round(2)

    lines = [
        "# Statistics Report",
        "",
        f"**Samples**: {n_samples:,}  |  **Features**: {n_features} (+ 1 target)",
        "",
        "## 1. Descriptive Statistics",
        "",
        stats_df.to_markdown(floatfmt=".4f"),
        "",
        "## 2. Stability Index Distribution by Bin",
        "",
        "| Bin     | Count   | Ratio   |",
        "|---------|---------|---------|",
    ]
    for lbl in STABILITY_LABELS:
        cnt = bin_counts.get(lbl, 0)
        pct = bin_pct.get(lbl, 0)
        lines.append(f"| {lbl} | {cnt:,} | {pct:.2f}% |")

    unstable_pct = 100 - bin_pct.get("stable", 0)
    lines += [
        "",
        f"- Unstable (stability < 95): **{unstable_pct:.2f}%**",
        f"- Stable   (stability >= 95): **{bin_pct.get('stable', 0):.2f}%**",
        "",
        "## 3. Key Observations",
        "",
    ]

    # auto-generate observations
    obs = []
    stab_skew = stats_df.loc[TARGET_COL, "skewness"]
    if abs(stab_skew) > 1:
        direction = "left" if stab_skew < 0 else "right"
        obs.append(
            f"- **Severe {direction}-skew** on `stability_index` "
            f"(skewness = {stab_skew:.2f}): the distribution is heavily "
            f"concentrated at the upper bound (100)."
        )
    for col in ["speed_kmh", "wing_angle_deg", "downforce_n", "drag_n"]:
        val = stats_df.loc[col, "skewness"]
        if abs(val) < 0.3:
            obs.append(
                f"- `{col}` is approximately symmetric (skewness = {val:.2f})."
            )
    obs.append(
        f"- `drs_active` is binary; the mean ({stats_df.loc['drs_active', 'mean']:.2f}) "
        f"reflects the proportion of DRS=1 samples."
    )
    for o in obs:
        lines.append(o)

    lines.append("")
    _ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Statistics report saved: {output_path}")
    return output_path


# ===================================================================
# 2.2  Distribution visualizations
# ===================================================================

def plot_distribution_stability(
    df: pd.DataFrame | None = None,
    save_path: str | None = None,
) -> str:
    """Histogram + KDE of stability_index with bin-boundary markers."""
    if df is None:
        df = _load_data()
    if save_path is None:
        save_path = os.path.join(EDA_FIGURES_DIR, "distribution_stability.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    arr = df[TARGET_COL].dropna()
    n = len(arr)
    sk = skew(arr)

    sns.histplot(arr, bins=80, stat="density", kde=True,
                 color=PALETTE_BINS["stable"], edgecolor="white",
                 linewidth=0.3, alpha=0.85, ax=ax)
    ax.lines[0].set_color("#053061")  # KDE line

    # bin boundaries
    for bound in STABILITY_BINS[1:-1]:  # skip 0, skip 101
        ax.axvline(bound, color="#ca0020", linestyle="--", linewidth=1.0, alpha=0.7)

    # annotation box
    textstr = (
        f"N = {n:,}\n"
        f"Skew = {sk:.2f}\n"
        f"Median = {arr.median():.1f}\n"
        f"Mean = {arr.mean():.1f}"
    )
    ax.text(
        0.02, 0.95, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    ax.set_xlabel("Stability Index")
    ax.set_ylabel("Density")
    ax.set_title("Stability Index Distribution (Histogram + KDE)")

    _ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Figure saved: {save_path}")
    return save_path


def plot_distribution_features(
    df: pd.DataFrame | None = None,
    save_path: str | None = None,
) -> str:
    """2x2 grid of histograms/KDEs for the four continuous input features."""
    if df is None:
        df = _load_data()
    if save_path is None:
        save_path = os.path.join(EDA_FIGURES_DIR, "distribution_features.png")

    cont_features = ["speed_kmh", "wing_angle_deg", "downforce_n", "drag_n"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for ax, col in zip(axes, cont_features):
        arr = df[col].dropna()
        sns.histplot(arr, bins=50, stat="density", kde=True,
                     color="#2c7bb6", edgecolor="white",
                     linewidth=0.3, alpha=0.8, ax=ax)
        ax.lines[0].set_color("#053061")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        sk = skew(arr)
        ax.set_title(f"{col}  (skew = {sk:.2f})", fontsize=10)

    fig.suptitle("Continuous Feature Distributions", fontsize=13, y=1.01)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Figure saved: {save_path}")
    return save_path


def plot_stability_by_drs(
    df: pd.DataFrame | None = None,
    save_path: str | None = None,
) -> str:
    """Boxplot + violin plot: stability_index grouped by DRS state."""
    if df is None:
        df = _load_data()
    if save_path is None:
        save_path = os.path.join(EDA_FIGURES_DIR, "stability_by_drs_boxplot.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    df_plot = df.copy()
    df_plot["drs_active"] = df_plot["drs_active"].astype(str)

    for idx, (kind, ax) in enumerate(zip(["box", "violin"], axes)):
        if kind == "box":
            sns.boxplot(x="drs_active", y=TARGET_COL, data=df_plot,
                        hue="drs_active", palette=PALETTE_DRS_STR,
                        legend=False, ax=ax, linewidth=1.0)
        else:
            sns.violinplot(x="drs_active", y=TARGET_COL, data=df_plot,
                           hue="drs_active", palette=PALETTE_DRS_STR,
                           legend=False, ax=ax, inner="quartile",
                           cut=0)
        drs_vals = sorted(df["drs_active"].unique())
        ax.set_xlabel("DRS State")
        ax.set_ylabel("Stability Index")
        ax.set_title(f"{kind.title()}plot of Stability by DRS State")
        ax.set_xticks(range(len(drs_vals)))
        ax.set_xticklabels([DRS_LABELS[int(v)] for v in drs_vals])

    fig.suptitle("Stability Index vs DRS State", fontsize=13, y=1.01)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Figure saved: {save_path}")
    return save_path


def plot_stability_by_speed_group(
    df: pd.DataFrame | None = None,
    save_path: str | None = None,
) -> str:
    """Stability distribution split by three speed groups."""
    if df is None:
        df = _load_data()
    if save_path is None:
        save_path = os.path.join(EDA_FIGURES_DIR, "stability_by_speed_group.png")

    df = df.copy()
    df["speed_group"] = _assign_speed_group(df)
    speed_palette = {"Low [80-180]": "#1b9e77", "Mid [180-280]": "#7570b3",
                     "High [280-360]": "#d95f02"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    sns.boxplot(x="speed_group", y=TARGET_COL, data=df,
                hue="speed_group", palette=speed_palette,
                legend=False, ax=axes[0], order=SPEED_LABELS)
    axes[0].set_xlabel("Speed Group")
    axes[0].set_ylabel("Stability Index")
    axes[0].set_title("Boxplot by Speed Group")
    axes[0].tick_params(axis="x", rotation=20)

    for grp, color in speed_palette.items():
        subset = df[df["speed_group"] == grp][TARGET_COL]
        if len(subset) == 0:
            continue
        sns.kdeplot(subset, label=grp, color=color, linewidth=1.8,
                    ax=axes[1], warn_singular=False)
    axes[1].set_xlabel("Stability Index")
    axes[1].set_ylabel("Density")
    axes[1].set_title("KDE by Speed Group")
    axes[1].legend(fontsize=8, loc="upper left")

    fig.suptitle("Stability Index across Speed Groups", fontsize=13, y=1.01)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Figure saved: {save_path}")
    return save_path


# ===================================================================
# 2.3  Correlation analysis
# ===================================================================

def plot_correlation_heatmap(
    df: pd.DataFrame | None = None,
    save_path: str | None = None,
    method: str = "pearson",
) -> str:
    """Correlation matrix heatmap (Pearson by default)."""
    if df is None:
        df = _load_data()
    if save_path is None:
        suffix = "spearman" if method == "spearman" else "pearson"
        save_path = os.path.join(EDA_FIGURES_DIR, f"correlation_{suffix}_heatmap.png")

    cols = [c for c in FEATURE_COLS + [TARGET_COL] if c in df.columns]
    corr = df[cols].corr(method=method)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # hide upper triangle

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".3f", cmap="RdBu_r",
                vmin=-1, vmax=1, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=ax)
    ax.set_title(f"{method.title()} Correlation Matrix", fontsize=12)

    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Figure saved: {save_path}")
    return save_path


# ===================================================================
# 2.4  Variable relationships
# ===================================================================

def plot_pairplot(
    df: pd.DataFrame | None = None,
    save_path: str | None = None,
    sample_n: int = 8000,
) -> str:
    """Pairplot overview (sampled for performance)."""
    if df is None:
        df = _load_data()
    if save_path is None:
        save_path = os.path.join(EDA_FIGURES_DIR, "pairplot.png")

    cols = [c for c in FEATURE_COLS + [TARGET_COL] if c in df.columns and c != "drs_active"]
    n_available = min(sample_n, len(df))
    sampled = df[cols].sample(n=n_available, random_state=RANDOM_STATE).copy()

    g = sns.pairplot(
        sampled, diag_kind="kde",
        plot_kws={"alpha": 0.25, "s": 6, "edgecolor": "none"},
        diag_kws={"fill": True, "color": "#2c7bb6"},
    )
    g.fig.suptitle("Pairplot Overview (sampled 8k points)", fontsize=14, y=1.01)

    _ensure_dir(os.path.dirname(save_path))
    g.savefig(save_path)
    plt.close(g.fig)
    print(f"  Figure saved: {save_path}")
    return save_path


def plot_speed_vs_stability(
    df: pd.DataFrame | None = None,
    save_path: str | None = None,
) -> str:
    """Scatter: speed_kmh vs stability_index, colored by DRS state."""
    if df is None:
        df = _load_data()
    if save_path is None:
        save_path = os.path.join(EDA_FIGURES_DIR, "speed_vs_stability_scatter.png")

    fig, ax = plt.subplots(figsize=(8, 5))

    for drs_val, color in PALETTE_DRS.items():
        subset = df[df["drs_active"] == drs_val]
        if len(subset) > 5000:
            subset = subset.sample(5000, random_state=RANDOM_STATE)
        ax.scatter(subset["speed_kmh"], subset[TARGET_COL],
                   c=color, label=DRS_LABELS[drs_val],
                   alpha=0.35, s=8, edgecolor="none", rasterized=True)

    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Stability Index")
    ax.set_title("Speed vs Stability Index (by DRS state)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Figure saved: {save_path}")
    return save_path


def plot_wing_vs_stability(
    df: pd.DataFrame | None = None,
    save_path: str | None = None,
) -> str:
    """Scatter: wing_angle_deg vs stability_index."""
    if df is None:
        df = _load_data()
    if save_path is None:
        save_path = os.path.join(EDA_FIGURES_DIR, "wing_vs_stability_scatter.png")

    fig, ax = plt.subplots(figsize=(8, 5))

    for drs_val, color in PALETTE_DRS.items():
        subset = df[df["drs_active"] == drs_val]
        if len(subset) > 5000:
            subset = subset.sample(5000, random_state=RANDOM_STATE)
        ax.scatter(subset["wing_angle_deg"], subset[TARGET_COL],
                   c=color, label=DRS_LABELS[drs_val],
                   alpha=0.35, s=8, edgecolor="none", rasterized=True)

    ax.set_xlabel("Wing Angle (deg)")
    ax.set_ylabel("Stability Index")
    ax.set_title("Wing Angle vs Stability Index (by DRS state)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Figure saved: {save_path}")
    return save_path


def plot_speed_wing_stability(
    df: pd.DataFrame | None = None,
    save_path: str | None = None,
    bin_type: str = "hexbin",
) -> str:
    """Heatmap of mean stability_index over speed × wing_angle space.

    Unlike a density (sample-count) heatmap — which is featureless here
    because ``speed_kmh`` and ``wing_angle_deg`` are independent and
    quasi-uniform — this plot colours each bin by the *average*
    ``stability_index`` of the samples that fall into it.

    Parameters
    ----------
    bin_type : str
        ``"hexbin"`` (hexagonal bins) or ``"hist2d"`` (rectangular grid).
    """
    if df is None:
        df = _load_data()
    if save_path is None:
        save_path = os.path.join(EDA_FIGURES_DIR, f"speed_wing_stability_{bin_type}.png")

    fig, ax = plt.subplots(figsize=(8, 5.5))

    title_kind = "hexbin" if bin_type == "hexbin" else "2D histogram"

    if bin_type == "hist2d":
        counts, x_edges, y_edges = np.histogram2d(
            df["speed_kmh"].values, df["wing_angle_deg"].values, bins=120,
        )
        sums, _, _ = np.histogram2d(
            df["speed_kmh"].values, df["wing_angle_deg"].values, bins=120,
            weights=df[TARGET_COL].values,
        )
        with np.errstate(invalid="ignore"):
            avg = sums / counts  # shape (120, 120)
        avg[counts == 0] = np.nan

        pc = ax.pcolormesh(x_edges, y_edges, avg.T, cmap="RdYlGn",
                           shading="flat", vmin=0, vmax=100)
        cb = fig.colorbar(pc, ax=ax, label="Mean Stability Index")
    else:
        hb = ax.hexbin(
            df["speed_kmh"], df["wing_angle_deg"],
            C=df[TARGET_COL], reduce_C_function=np.mean,
            gridsize=120, cmap="RdYlGn", vmin=0, vmax=100,
            mincnt=1, linewidths=0.3,
        )
        cb = fig.colorbar(hb, ax=ax, label="Mean Stability Index")
    cb.formatter = mticker.FuncFormatter(lambda x, p: f"{x:.1f}")

    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Wing Angle (deg)")
    ax.set_title(f"Mean Stability Index: Speed vs Wing Angle ({title_kind})")

    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Figure saved: {save_path}")
    return save_path


# ===================================================================
# 2.5  Imbalance quantification
# ===================================================================

def plot_imbalance_cdf(
    df: pd.DataFrame | None = None,
    save_path: str | None = None,
) -> str:
    """CDF of stability_index with 4-bin boundary markers."""
    if df is None:
        df = _load_data()
    if save_path is None:
        save_path = os.path.join(EDA_FIGURES_DIR, "imbalance_cdf.png")

    arr = np.sort(df[TARGET_COL].dropna().values)
    n = len(arr)
    cdf_y = np.arange(1, n + 1) / n * 100

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(arr, cdf_y, color="#2c7bb6", linewidth=1.8, label="CDF")

    for bound in STABILITY_BINS[1:-1]:
        ax.axvline(bound, color="#ca0020", linestyle="--", linewidth=1.0, alpha=0.7)
        pctl = (arr < bound).sum() / n * 100
        ax.axhline(pctl, color="#ca0020", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.annotate(
            f"{bound}: {pctl:.1f}%",
            xy=(bound, pctl), xytext=(bound + 2, pctl + 5),
            fontsize=7, color="#ca0020",
            arrowprops=dict(arrowstyle="->", color="#ca0020", lw=0.8),
        )

    # label severe bin zone
    ax.axvspan(0, STABILITY_BINS[1], alpha=0.07, color="#ca0020", label="severe")
    ax.axvspan(STABILITY_BINS[1], STABILITY_BINS[2], alpha=0.07, color="#f4a582",
               label="moderate")
    ax.axvspan(STABILITY_BINS[2], STABILITY_BINS[3], alpha=0.07, color="#92c5de",
               label="mild")
    ax.axvspan(STABILITY_BINS[3], 101, alpha=0.07, color="#0571b0", label="stable")

    ax.set_xlabel("Stability Index")
    ax.set_ylabel("Cumulative %")
    ax.set_title("Cumulative Distribution Function (CDF) of Stability Index")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.0f}%"))
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0, 101)

    fig.tight_layout()
    _ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Figure saved: {save_path}")
    return save_path


# ===================================================================
# Figure Guide (auto-generated README)
# ===================================================================

def generate_figure_guide(output_path: str | None = None) -> str:
    """Write ``figures/eda/README.md`` documenting every EDA figure."""
    if output_path is None:
        output_path = os.path.join(EDA_FIGURES_DIR, "README.md")

    lines = [
        "# EDA Figure Guide",
        "",
        "> Auto-generated by `src/visualization/plot_eda.py::generate_figure_guide()`.",
        "> Run `run_eda()` to regenerate after adding or modifying figures.",
        "",
        "Each section below corresponds to one PNG file in this directory.",
        "",
        "---",
        "",
        "## 1. `distribution_stability.png`",
        "",
        "- **What**: Histogram + KDE of `stability_index`.",
        "- **Axes**: X = stability_index (0-100), Y = density.",
        "- **Dashed lines**: 4-bin boundaries at 30 (severe), 60 (moderate), 95 (mild→stable).",
        "- **Text box**: N (sample count), skewness, median, mean.",
        "- **Interpretation**: The massive spike at 100 reveals a *ceiling effect* — 86% of",
        "  samples sit at the maximum. The long tail toward 0 shows a rare but",
        "  severe instability regime. A negative skew value confirms the left-leaning tail.",
        "",
        "---",
        "",
        "## 2. `distribution_features.png`",
        "",
        "- **What**: 2×2 grid of histograms + KDE for the 4 continuous input features.",
        "- **Axes**: X = feature value, Y = density. Each subplot title shows feature name and skewness.",
        "- **Interpretation**: `speed_kmh` and `wing_angle_deg` are approximately uniform,",
        "  reflecting the dataset design. `downforce_n` and `drag_n` are right-skewed",
        "  (long tail toward high values), consistent with aerodynamic physics.",
        "",
        "---",
        "",
        "## 3. `stability_by_drs_boxplot.png`",
        "",
        "- **What**: Side-by-side boxplot and violin plot of `stability_index` grouped by DRS state.",
        "- **Left panel (boxplot)**: Box = IQR (Q1-Q3), line = median, whiskers = 1.5×IQR.",
        "- **Right panel (violin)**: Width = density at that stability value, white dot = median.",
        "- **Colors**: Blue = DRS OFF (0), Red = DRS ON (1).",
        "- **Interpretation**: Both groups have medians at 100. DRS ON shows a slightly",
        "  wider lower tail, hinting that DRS activation may correlate with unstable conditions.",
        "",
        "---",
        "",
        "## 4. `stability_by_speed_group.png`",
        "",
        "- **What**: Boxplot + KDE of `stability_index` across 3 speed ranges.",
        "- **Groups**: Low [80-180] (green), Mid [180-280] (purple), High [280-360] (orange).",
        "- **Interpretation**: Low-speed group shows zero variance (all samples at 100).",
        "  Mid and High groups have visible lower tails, indicating instability is",
        "  concentrated in higher speed regimes — consistent with aerodynamic porpoising physics.",
        "",
        "---",
        "",
        "## 5. `correlation_pearson_heatmap.png`",
        "",
        "- **What**: Lower-triangle heatmap of Pearson (linear) correlation coefficients.",
        "- **Scale**: -1 (perfect negative) ↔ 0 (none) ↔ +1 (perfect positive).",
        "- **Key values**: `downforce_n` ~ `drag_n` = 0.94 (highly collinear),",
        "  `speed_kmh` ~ `downforce_n` = 0.82, `speed_kmh` ~ `drag_n` = 0.78.",
        "- **Implication**: Strong collinearity among aerodynamic forces may require",
        "  regularization or dimensionality reduction during modeling.",
        "",
        "---",
        "",
        "## 6. `correlation_spearman_heatmap.png`",
        "",
        "- **What**: Same layout as #5 but using Spearman rank correlation.",
        "- **Why**: Sensitive to monotonic (not necessarily linear) relationships.",
        "- **Key difference**: `downforce_n` ~ `drag_n` = 0.98 vs Pearson 0.94,",
        "  confirming near-perfect monotonic coupling between the two forces.",
        "",
        "---",
        "",
        "## 7. `pairplot.png`",
        "",
        "- **What**: Scatter matrix of 5 continuous variables (8k random samples).",
        "- **Diagonal**: KDE of each variable.",
        "- **Off-diagonal**: Scatter plots between variable pairs.",
        "- **Purpose**: Quick overview of pairwise relationships — look for",
        "  non-linear patterns, clusters, or outliers across the feature space.",
        "",
        "---",
        "",
        "## 8. `speed_vs_stability_scatter.png`",
        "",
        "- **What**: Scatter plot of `speed_kmh` (X) vs `stability_index` (Y),",
        "  colored by DRS state.",
        "- **Blue**: DRS OFF, **Red**: DRS ON. Sample limited to 5k per group for clarity.",
        "- **Interpretation**: High-speed region (300-365 km/h) shows a wider spread",
        "  of stability values including low-stability (unstable) points.",
        "",
        "---",
        "",
        "## 9. `wing_vs_stability_scatter.png`",
        "",
        "- **What**: Scatter plot of `wing_angle_deg` (X) vs `stability_index` (Y),",
        "  colored by DRS state.",
        "- **Interpretation**: Wing angle alone shows no strong deterministic",
        "  relationship with stability; the scatter is near-uniform across angles.",
        "",
        "---",
        "",
        "## 10A. `speed_wing_stability_hexbin.png`",
        "",
        "- **What**: Hexbin heatmap of **mean stability_index** across the",
        "  `speed_kmh` × `wing_angle_deg` parameter space.",
        "- **Color**: `RdYlGn` — red = low stability (unstable), green = high stability.",
        "- **Why not sample-count density**: `speed_kmh` and `wing_angle_deg` are",
        "  independent and quasi-uniform, so a density (count-based) heatmap is",
        "  featurelessly flat.  This plot shows the *outcome* (stability) instead.",
        "- **Interpretation**: The green-to-red gradient reveals which speed/angle",
        "  combinations correlate with aerodynamic stability.  Look for red patches",
        "  in high-speed, low-angle regions as possible porpoising risk zones.",
        "",
        "---",
        "",
        "## 10B. `speed_wing_stability_hist2d.png`",
        "",
        "- **What**: Rectangular-grid (2D histogram) version of the same analysis.",
        "- **Color**: Same `RdYlGn` scale, identical meaning to 10A.",
        "- **Comparison with hexbin**: The rectangular grid aligns cleanly with",
        "  the original Cartesian axes, making it easier to read out exact",
        "  (speed, angle) bin coordinates.",
        "",
        "---",
        "",
        "## 11. `imbalance_cdf.png`",
        "",
        "- **What**: Cumulative Distribution Function (CDF) of `stability_index`.",
        "- **X**: stability_index, **Y**: cumulative percentage.",
        "- **Dashed vertical lines**: Bin boundaries at 30, 60, 95.",
        "- **Shaded zones**: 4 stability bins (severe/moderate/mild/stable) mapped",
        "  to red → orange → light blue → dark blue.",
        "- **Annotations**: For each boundary, the cumulative % of samples below that",
        "  threshold is labeled.",
        "- **Key insight**: The sharp jump at 95+ shows ~86% of samples are stable.",
        "  The CDF rises gradually below 95, indicating the unstable regime spans",
        "  a continuous range of severity.",
        "",
        "---",
        "",
        "## Summary — Imbalance at a Glance",
        "",
        "| Bin      | Range   | % of Data |",
        "|----------|---------|-----------|",
        "| severe   | [0, 30) | ~6.4%    |",
        "| moderate | [30, 60)| ~3.0%    |",
        "| mild     | [60, 95)| ~4.6%    |",
        "| stable   | [95,101)| ~86.0%   |",
        "",
        "**Unstable total** (stability < 95): ~14.0%",
        "",
        "**Stable : Unstable ratio**: ~6.2 : 1",
    ]

    _ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Figure guide saved: {output_path}")
    return output_path


# ===================================================================
# Orchestrator
# ===================================================================

def run_eda(data_path: str | None = None):
    """Run the full EDA pipeline: statistics + all figures.

    Parameters
    ----------
    data_path : str or None
        Path to a CSV file.  If None, ``data/processed/train.csv`` is used.
    """
    print("=" * 60)
    print("  T02: Exploratory Data Analysis")
    print("=" * 60)

    df = _load_data(data_path)
    print(f"\n  Data loaded: {len(df):,} samples, {len(df.columns)} columns")

    # --- 2.1 Statistics ---
    print("\n--- 2.1   Descriptive Statistics ---")
    stats = compute_statistics(df)
    print(stats.to_string(float_format=lambda x: f"{x:.4f}"))
    generate_statistics_report(df)
    print()

    # --- 2.2 Distributions ---
    print("--- 2.2   Distribution Visualizations ---")
    plot_distribution_stability(df)
    plot_distribution_features(df)
    plot_stability_by_drs(df)
    plot_stability_by_speed_group(df)
    print()

    # --- 2.3 Correlations ---
    print("--- 2.3   Correlation Analysis ---")
    plot_correlation_heatmap(df, method="pearson")

    # Spearman (optional but valuable for non-linear relationships)
    spearman_path = os.path.join(EDA_FIGURES_DIR, "correlation_spearman_heatmap.png")
    plot_correlation_heatmap(df, save_path=spearman_path, method="spearman")
    print()

    # --- 2.4 Relationships ---
    print("--- 2.4   Variable Relationships ---")
    plot_pairplot(df)
    plot_speed_vs_stability(df)
    plot_wing_vs_stability(df)
    plot_speed_wing_stability(df)
    plot_speed_wing_stability(df, bin_type="hist2d")
    print()

    # --- 2.5 Imbalance ---
    print("--- 2.5   Imbalance Quantification ---")
    plot_imbalance_cdf(df)
    print()

    # --- Figure Guide ---
    print("--- Figure Guide ---")
    generate_figure_guide()
    print()

    # --- Summary ---
    n_stable = (df[TARGET_COL] >= 95).sum()
    n_unstable = len(df) - n_stable
    ratio = n_stable / n_unstable if n_unstable > 0 else float("inf")
    print("=" * 60)
    print("  T02 Complete — Summary")
    print(f"  Stable / Unstable ratio: {ratio:.1f}:1")
    print(f"  Unstable proportion: {n_unstable / len(df) * 100:.2f}%")
    print(f"  Figures saved to: {EDA_FIGURES_DIR}")
    print(f"  Report saved to: {os.path.join(REPORTS_DIR, 'statistics_report.md')}")
    print("=" * 60)


if __name__ == "__main__":
    run_eda()
