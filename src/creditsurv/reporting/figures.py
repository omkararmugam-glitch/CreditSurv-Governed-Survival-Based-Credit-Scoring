"""Figures. Matplotlib only, written to disk; nothing is displayed.

Every figure uses a colourblind-safe palette, direct labelling in place of
legends where it fits, and works in print. Each function returns the path it
wrote so callers can record it in an audit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "plot_survival_curves",
    "plot_km_by_group",
    "plot_time_dependent_auc",
    "plot_survshap_curves",
    "plot_importance_comparison",
    "plot_segment_heatmap",
    "plot_propensity_overlap",
    "plot_calibration",
    "plot_missing_share",
    "plot_numeric_distributions",
    "plot_correlation_heatmap",
    "plot_default_rate_bars",
]

# Okabe-Ito: distinguishable under the common forms of colour vision deficiency.
PALETTE = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
)


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )
    return plt


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def plot_survival_curves(
    survival: np.ndarray,
    times: np.ndarray,
    path: str | Path,
    *,
    labels: list[str] | None = None,
    title: str = "Predicted survival curves",
    max_curves: int = 8,
) -> Path:
    """Per-borrower survival curves -- the output a binary model cannot produce."""
    plt = _setup()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n = min(len(survival), max_curves)
    for i in range(n):
        ax.plot(
            times, survival[i], color=PALETTE[i % len(PALETTE)], linewidth=1.8,
            label=labels[i] if labels else f"borrower {i + 1}",
        )
    ax.set_xlabel("Months since origination")
    ax.set_ylabel("P(no default yet)")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return _save(fig, path)


def plot_km_by_group(
    duration: np.ndarray,
    event: np.ndarray,
    group: pd.Series,
    path: str | Path,
    *,
    title: str = "Kaplan-Meier by group",
    max_groups: int = 8,
) -> Path:
    """Observed KM curves by segment, with at-risk counts in the labels."""
    from lifelines import KaplanMeierFitter

    plt = _setup()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    levels = list(pd.Series(group).value_counts().head(max_groups).index)
    for i, level in enumerate(sorted(map(str, levels))):
        m = (pd.Series(group).astype(str) == level).to_numpy()
        if m.sum() < 5:
            continue
        km = KaplanMeierFitter().fit(np.asarray(duration)[m], np.asarray(event)[m])
        km.plot_survival_function(
            ax=ax, ci_show=False, color=PALETTE[i % len(PALETTE)],
            label=f"{level} (n={m.sum():,})", linewidth=1.8,
        )
    ax.set_xlabel("Months since origination")
    ax.set_ylabel("P(no default yet)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def plot_time_dependent_auc(
    tables: dict[str, pd.DataFrame], path: str | Path, *, title: str = "Time-dependent AUC"
) -> Path:
    """AUC against horizon for several models. Unreliable points are hollow."""
    plt = _setup()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (name, tbl) in enumerate(tables.items()):
        colour = PALETTE[i % len(PALETTE)]
        ax.plot(tbl["time"], tbl["auc"], "-", color=colour, linewidth=1.8, label=name)
        rel = tbl["reliable"].to_numpy(dtype=bool) if "reliable" in tbl else np.ones(len(tbl), bool)
        ax.plot(tbl["time"][rel], tbl["auc"][rel], "o", color=colour, markersize=5)
        if (~rel).any():
            ax.plot(
                tbl["time"][~rel], tbl["auc"][~rel], "o", markerfacecolor="white",
                markeredgecolor=colour, markersize=5,
            )
    ax.axhline(0.5, color="#888888", linestyle=":", linewidth=1)
    ax.set_xlabel("Horizon (months)")
    ax.set_ylabel("Cumulative/dynamic AUC")
    ax.set_title(f"{title}\n(hollow markers: censoring weights unreliable)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, path)


def plot_survshap_curves(
    expl, path: str | Path, *, obs: int = 0, top_k: int = 6,
    title: str | None = None,
) -> Path:
    """The signature SurvSHAP(t) plot: attribution as a function of time.

    A flat line means the feature's effect does not depend on loan age, so a
    scalar SHAP value loses nothing. A line that crosses zero means the feature
    protects early and harms later (or the reverse) -- which is exactly what a
    single number cannot express.
    """
    plt = _setup()
    curve = expl.curve(obs)
    top = expl.importance(obs=obs).head(top_k).index

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, feature in enumerate(top):
        ax.plot(
            curve.index, curve[feature], color=PALETTE[i % len(PALETTE)],
            linewidth=1.9, label=feature,
        )
    ax.axhline(0.0, color="#444444", linewidth=1)
    ax.set_xlabel("Months since origination")
    ax.set_ylabel("Attribution to S(t | x)")
    ax.set_title(title or f"SurvSHAP(t) attributions, borrower {obs}")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return _save(fig, path)


def plot_importance_comparison(
    survshap: pd.Series, naive: pd.Series, path: str | Path, *, top_k: int = 12
) -> Path:
    """Side-by-side importance for Stage 3(a)."""
    plt = _setup()
    features = list(survshap.head(top_k).index)
    a = survshap.reindex(features).to_numpy()
    b = naive.reindex(features).to_numpy()

    y = np.arange(len(features))
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(features) + 1.6))
    ax.barh(y - 0.2, a, height=0.38, color=PALETTE[0], label="SurvSHAP(t)")
    ax.barh(y + 0.2, b, height=0.38, color=PALETTE[1], label="naive SHAP")
    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |attribution|")
    ax.set_title("Feature importance: time-dependent vs time-agnostic")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, path)


def plot_segment_heatmap(
    importance: pd.DataFrame, path: str | Path, *, top_k: int = 12,
    title: str = "Importance by segment",
) -> Path:
    """Feature x segment importance heatmap, row-normalised.

    Row normalisation is the point: it shows whether a feature's *relative*
    standing changes between segments, which is the drift question.
    """
    plt = _setup()
    top = importance.mean(axis=1).sort_values(ascending=False).head(top_k).index
    mat = importance.loc[top]
    row_max = mat.max(axis=1).replace(0.0, np.nan)
    norm = mat.div(row_max, axis=0)

    fig, ax = plt.subplots(figsize=(1.1 * len(mat.columns) + 3.4, 0.4 * len(top) + 1.8))
    im = ax.imshow(norm.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([str(c) for c in mat.columns], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(list(top), fontsize=8)
    ax.set_title(f"{title}\n(each row scaled to its own maximum)", fontsize=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="relative importance", fraction=0.03)
    return _save(fig, path)


def plot_propensity_overlap(
    accepted_propensity: np.ndarray,
    rejected_propensity: np.ndarray,
    path: str | Path,
    *,
    support_range: tuple[float, float] | None = None,
) -> Path:
    """Overlap of accept-propensity distributions -- the Stage 4 gate, visualised.

    If the two distributions barely intersect, reject inference is extrapolation,
    and this figure is the clearest way to see that.
    """
    plt = _setup()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, 1, 60)
    ax.hist(accepted_propensity, bins=bins, density=True, alpha=0.6,
            color=PALETTE[0], label=f"accepted (n={len(accepted_propensity):,})")
    ax.hist(rejected_propensity, bins=bins, density=True, alpha=0.6,
            color=PALETTE[1], label=f"rejected (n={len(rejected_propensity):,})")
    if support_range:
        for x in support_range:
            ax.axvline(x, color="#444444", linestyle="--", linewidth=1.1)
        ax.text(
            float(np.mean(support_range)), ax.get_ylim()[1] * 0.92,
            "common support", ha="center", fontsize=8, color="#444444",
        )
    ax.set_xlabel("P(accepted | common features)")
    ax.set_ylabel("Density")
    ax.set_title("Accept-propensity overlap")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, path)


def plot_calibration(
    calibration: pd.DataFrame, path: str | Path, *, t: float = 24.0
) -> Path:
    """Predicted vs Kaplan-Meier-observed survival by predicted decile."""
    plt = _setup()
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], ":", color="#888888", linewidth=1.2, label="perfect")
    ax.plot(
        calibration["predicted_survival"], calibration["observed_survival"],
        "o-", color=PALETTE[0], linewidth=1.8, markersize=6, label="model",
    )
    ax.set_xlabel(f"Predicted S({t:.0f} months)")
    ax.set_ylabel(f"Observed S({t:.0f} months), Kaplan-Meier")
    ax.set_title("Calibration by predicted decile")
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# EDA charts (scripts/01b_eda.py and the dashboard's Data Profile tab). These
# describe data; none of them is produced from a model.
# ---------------------------------------------------------------------------


def plot_missing_share(missing: pd.DataFrame, path: str | Path, *, top: int = 30) -> Path:
    """Share of rows missing, worst columns first."""
    plt = _setup()
    data = missing[missing["missing_share"] > 0].head(top).iloc[::-1]
    if data.empty:
        fig, ax = plt.subplots(figsize=(6, 1.6))
        ax.text(0.5, 0.5, "No missing values in any column", ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.26 * len(data))))
    ax.barh(data["column"], data["missing_share"] * 100, color=PALETTE[0])
    ax.set_xlabel("% of rows missing")
    ax.set_title(f"Missing values (top {len(data)} of "
                 f"{int((missing['missing_share'] > 0).sum())} affected columns)")
    return _save(fig, path)


def plot_numeric_distributions(df: pd.DataFrame, columns, path: str | Path, *,
                               bins: int = 40) -> Path:
    """Histograms on a shared grid, each clipped at its own 0.5/99.5 percentiles so
    one extreme value cannot flatten the whole distribution."""
    plt = _setup()
    cols = [c for c in columns if c in df.columns][:12]
    if not cols:
        fig, ax = plt.subplots(figsize=(6, 1.6))
        ax.text(0.5, 0.5, "No numeric columns", ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)
    ncol = 3
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.5 * nrow))
    for ax, col in zip(np.atleast_1d(axes).ravel(), cols):
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        if v.empty:
            ax.axis("off")
            continue
        lo, hi = v.quantile(0.005), v.quantile(0.995)
        ax.hist(v.clip(lo, hi), bins=bins, color=PALETTE[0])
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in np.atleast_1d(axes).ravel()[len(cols):]:
        ax.axis("off")
    fig.suptitle("Numeric distributions (clipped at 0.5/99.5 percentiles for display)",
                 fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def plot_correlation_heatmap(matrix: pd.DataFrame, path: str | Path, *,
                             max_features: int = 40) -> Path:
    """Pearson correlations. Features beyond ``max_features`` are dropped from the
    picture (the full matrix is in the CSV) because the labels stop being legible."""
    plt = _setup()
    m = matrix.iloc[:max_features, :max_features]
    fig, ax = plt.subplots(figsize=(min(13, 0.34 * len(m) + 3),
                                    min(12, 0.34 * len(m) + 2.4)))
    im = ax.imshow(m.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(m)), m.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(m)), m.index, fontsize=6)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")
    ax.set_title(f"Correlation of numeric features"
                 + (f" (first {max_features} of {len(matrix)})"
                    if len(matrix) > max_features else ""))
    return _save(fig, path)


def plot_default_rate_bars(table: pd.DataFrame, column: str, path: str | Path, *,
                           top: int = 20) -> Path:
    """Observed default rate by segment, with segment sizes in the labels and small
    segments marked, since a rate over a handful of loans is noise."""
    plt = _setup()
    data = table.head(top).copy()
    if data.empty or "default_rate" not in data:
        return None
    labels = [f"{r[column]} (n={int(r['loans']):,})"
              + ("*" if r.get("small_segment") else "") for _, r in data.iterrows()]
    colours = [PALETTE[1] if r.get("small_segment") else PALETTE[0]
               for _, r in data.iterrows()]
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.32 * len(data))))
    ax.barh(labels[::-1], (data["default_rate"] * 100).to_numpy()[::-1],
            color=colours[::-1])
    ax.set_xlabel("observed default rate (%)")
    ax.set_title(f"Default rate by {column}"
                 + ("   * small segment" if data.get("small_segment", pd.Series()).any()
                    else ""))
    return _save(fig, path)
