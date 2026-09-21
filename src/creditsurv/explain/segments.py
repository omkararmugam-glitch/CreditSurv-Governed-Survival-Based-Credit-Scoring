"""Stage 3(c): is feature importance stable across borrower segments?

Aggregates SurvSHAP(t) attributions within segments -- loan grade, purpose,
income band -- and asks whether the drivers of risk are the same everywhere or
drift.

Why it matters beyond curiosity: if importance is stable, one global explanation
is a fair summary of the model and a single set of adverse-action reason templates
serves everyone. If it drifts, a global explanation is misleading for some
borrowers, and a lender relying on global importance to justify decisions would be
describing a model it does not actually have. Under ECOA that is a fair-lending
exposure, since the reasons given to one group would be systematically less
accurate than those given to another.

Measures
--------
* **Rank correlation between each segment and the global ranking** -- how much
  reordering the segment shows.
* **Top-k membership churn** -- whether the features a borrower would actually be
  told about change.
* **Coefficient of variation of importance across segments**, per feature --
  which specific features are unstable.
* **Stability verdict** against pre-registered thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .survshap import SurvShapExplanation

__all__ = [
    "SegmentStabilityResult",
    "segment_importance",
    "analyse_segment_stability",
    "bootstrap_segment_ranks",
]


def segment_importance(
    expl: SurvShapExplanation, segment: pd.Series, *, min_size: int = 30
) -> pd.DataFrame:
    """Time-integrated importance per feature, per segment level.

    Returns a features x segments frame. Levels with fewer than ``min_size``
    observations are dropped, because a Shapley ranking from a handful of
    borrowers is noise and would dominate any instability measure.
    """
    if len(segment) != expl.n_observations:
        raise ValueError(
            f"segment has {len(segment)} rows but explanation has "
            f"{expl.n_observations} observations"
        )

    span = max(float(expl.times[-1] - expl.times[0]), 1e-12)
    # (obs, feat): area under |phi_j(t)| for each observation
    area = np.trapezoid(np.abs(expl.phi), expl.times, axis=2) / span
    per_obs = pd.DataFrame(area, columns=list(expl.feature_names))
    per_obs["_segment"] = np.asarray(segment)

    counts = per_obs["_segment"].value_counts()
    keep = counts[counts >= min_size].index
    per_obs = per_obs[per_obs["_segment"].isin(keep)]
    if per_obs.empty:
        return pd.DataFrame(index=list(expl.feature_names))

    return per_obs.groupby("_segment", observed=True).mean().T


@dataclass
class SegmentStabilityResult:
    segment_name: str
    importance: pd.DataFrame
    segment_sizes: pd.Series
    rank_correlation: pd.Series
    topk_overlap: pd.Series
    feature_cv: pd.Series
    global_importance: pd.Series
    k: int

    @property
    def min_rank_correlation(self) -> float:
        return float(self.rank_correlation.min()) if len(self.rank_correlation) else np.nan

    @property
    def mean_topk_overlap(self) -> float:
        return float(self.topk_overlap.mean()) if len(self.topk_overlap) else np.nan

    def verdict(
        self, *, rank_floor: float = 0.85, overlap_floor: float = 0.7, cv_ceiling: float = 0.5
    ) -> str:
        """Plain reading against thresholds fixed before seeing results."""
        if len(self.rank_correlation) == 0:
            return "INCONCLUSIVE: no segment met the minimum size requirement."

        unstable = self.feature_cv[self.feature_cv > cv_ceiling]
        worst = self.rank_correlation.idxmin()

        rank_failed = self.min_rank_correlation < rank_floor
        overlap_failed = self.mean_topk_overlap < overlap_floor

        if rank_failed or overlap_failed:
            # Name the criterion that actually triggered. Reporting both as if
            # both had failed misdescribes the result -- rank correlation is
            # often comfortably above its floor while top-k membership churns.
            breaches = []
            if rank_failed:
                breaches.append(
                    f"rank correlation {self.min_rank_correlation:.3f} is below its "
                    f"{rank_floor} floor (worst segment '{worst}')"
                )
            if overlap_failed:
                breaches.append(
                    f"mean top-{self.k} overlap {self.mean_topk_overlap:.3f} is below "
                    f"its {overlap_floor} floor"
                )
            passed = []
            if not rank_failed:
                passed.append(
                    f"rank correlation {self.min_rank_correlation:.3f} passed"
                )
            if not overlap_failed:
                passed.append(
                    f"top-{self.k} overlap {self.mean_topk_overlap:.3f} passed"
                )

            msg = f"DRIFTS: {'; '.join(breaches)}."
            if passed:
                msg += f" ({'; '.join(passed)}.)"
            msg += (
                " Which features a borrower would be told about changes between "
                f"segments, so one global explanation is not a fair summary of "
                f"{self.segment_name}."
            )
            if len(unstable):
                msg += f" Least stable features: {', '.join(unstable.head(3).index)}."
            else:
                msg += (
                    f" No individual feature exceeded CV {cv_ceiling}, so the drift is "
                    "spread across many features rather than driven by a few."
                )
            return msg

        return (
            f"STABLE: minimum rank correlation {self.min_rank_correlation:.3f} "
            f"(floor {rank_floor}), mean top-{self.k} overlap "
            f"{self.mean_topk_overlap:.3f} (floor {overlap_floor}), "
            f"{len(unstable)} feature(s) above CV {cv_ceiling}. Importance is "
            f"consistent across {self.segment_name}, so a global explanation is a "
            f"fair summary."
        )

    def summary(self) -> dict:
        return {
            "segment": self.segment_name,
            "n_levels": int(len(self.rank_correlation)),
            "levels": list(map(str, self.rank_correlation.index)),
            "segment_sizes": {str(k): int(v) for k, v in self.segment_sizes.items()},
            # Per-level values, not just aggregates: a min/mean hides which segment
            # drifted, and the rare high-risk levels are precisely the ones a
            # fair-lending review needs to see individually.
            "rank_correlation_by_level": {
                str(k): round(float(v), 4) for k, v in self.rank_correlation.items()
            },
            "topk_overlap_by_level": {
                str(k): round(float(v), 4) for k, v in self.topk_overlap.items()
            },
            "topk_by_level": {
                str(level): list(
                    self.importance[level].sort_values(ascending=False).head(self.k).index
                )
                for level in self.importance.columns
            },
            "min_rank_correlation": round(self.min_rank_correlation, 4),
            "mean_rank_correlation": round(float(self.rank_correlation.mean()), 4),
            "mean_topk_overlap": round(self.mean_topk_overlap, 4),
            "k": self.k,
            "least_stable_features": list(self.feature_cv.head(5).index),
            "most_stable_features": list(self.feature_cv.tail(5).index),
            "global_top5": list(self.global_importance.head(5).index),
            "verdict": self.verdict(),
        }


def analyse_segment_stability(
    expl: SurvShapExplanation,
    segment: pd.Series,
    *,
    segment_name: str = "segment",
    k: int = 5,
    min_size: int = 30,
) -> SegmentStabilityResult:
    """Compare per-segment importance rankings against the global ranking."""
    from scipy.stats import spearmanr

    imp = segment_importance(expl, segment, min_size=min_size)
    global_imp = expl.importance().sort_values(ascending=False)

    sizes = pd.Series(np.asarray(segment)).value_counts()
    sizes = sizes[sizes.index.isin(imp.columns)]

    features = list(global_imp.index)
    global_vec = global_imp.reindex(features).to_numpy()
    global_top = set(global_imp.head(k).index)

    rank_corr, overlap = {}, {}
    for level in imp.columns:
        vec = imp[level].reindex(features).to_numpy()
        ok = np.isfinite(vec) & np.isfinite(global_vec)
        rank_corr[level] = (
            float(spearmanr(vec[ok], global_vec[ok]).statistic) if ok.sum() > 2 else np.nan
        )
        seg_top = set(imp[level].sort_values(ascending=False).head(k).index)
        union = seg_top | global_top
        overlap[level] = float(len(seg_top & global_top) / len(union)) if union else np.nan

    # Per-feature instability across segments.
    mean_imp = imp.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = (imp.std(axis=1) / mean_imp.replace(0.0, np.nan)).fillna(0.0)

    return SegmentStabilityResult(
        segment_name=segment_name,
        importance=imp,
        segment_sizes=sizes,
        rank_correlation=pd.Series(rank_corr).sort_values(),
        topk_overlap=pd.Series(overlap).sort_values(),
        feature_cv=cv.sort_values(ascending=False),
        global_importance=global_imp,
        k=k,
    )


def bootstrap_segment_ranks(
    per_obs: pd.DataFrame,
    segment: pd.Series,
    *,
    target: str,
    reference_levels: list[str],
    features: list[str],
    n_boot: int = 2000,
    seed: int = 20260921,
    ci: float = 0.95,
) -> pd.DataFrame:
    """Stratified bootstrap of a feature's rank in one segment vs. a reference set.

    ``per_obs`` is borrowers x features of time-integrated |attribution| (what
    :func:`segment_importance` averages). Each replicate resamples borrowers *with
    replacement within each level*, preserving every level's size, recomputes each
    level's mean importance, and ranks all features within that level (1 = most
    important). Resampling within level is what makes this the right bootstrap
    for an equal-allocation design: it asks how much the ranking would move had a
    different 36 borrowers been drawn from the same grade.

    Reported per feature:

    * ``rank_target`` -- the feature's rank within the target level.
    * ``rank_reference`` -- its rank within the pooled reference levels.
    * ``rank_shift`` = target minus reference. The central question is whether its
      interval excludes 0.
    * ``share_ratio`` -- the feature's share of the level's total attribution,
      target over reference. Ranks are discrete and move at near-ties, so this
      continuous companion checks the rank result isn't a tie-break artefact.
      Shares rather than raw magnitudes, because riskier borrowers carry larger
      attributions overall and raw ratios would confound that with reordering.
    * ``p_outside_reference_range`` -- share of replicates in which the target rank
      falls outside the min-max rank across the individual reference levels.
    """
    rng = np.random.default_rng(seed)
    seg = np.asarray(segment).astype(str)
    all_feats = list(per_obs.columns)
    values = per_obs.to_numpy(dtype=float)
    levels = [target, *reference_levels]
    rows_by_level = {lvl: np.flatnonzero(seg == lvl) for lvl in levels}
    missing = [lvl for lvl, r in rows_by_level.items() if len(r) == 0]
    if missing:
        raise ValueError(f"levels with no observations: {missing}")
    fidx = [all_feats.index(f) for f in features]

    def stats(sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        means = {lvl: values[idx].mean(axis=0) for lvl, idx in sample.items()}
        ref_pool = values[np.concatenate([sample[l] for l in reference_levels])].mean(axis=0)

        def ranks(v: np.ndarray) -> np.ndarray:
            # rank 1 = largest; ties share the better rank
            order = (-v).argsort(kind="stable")
            r = np.empty_like(order)
            r[order] = np.arange(1, len(v) + 1)
            return r

        r_t = ranks(means[target])[fidx]
        r_ref = ranks(ref_pool)[fidx]
        per_ref = np.stack([ranks(means[l])[fidx] for l in reference_levels])
        share_t = means[target][fidx] / means[target].sum()
        share_r = ref_pool[fidx] / ref_pool.sum()
        outside = (r_t < per_ref.min(axis=0)) | (r_t > per_ref.max(axis=0))
        return {
            "rank_target": r_t, "rank_reference": r_ref, "rank_shift": r_t - r_ref,
            "share_ratio": share_t / share_r, "outside": outside,
            "ref_min": per_ref.min(axis=0), "ref_max": per_ref.max(axis=0),
        }

    point = stats(rows_by_level)
    draws = {k: [] for k in point}
    for _ in range(n_boot):
        sample = {lvl: rng.choice(idx, size=len(idx), replace=True)
                  for lvl, idx in rows_by_level.items()}
        s = stats(sample)
        for k in draws:
            draws[k].append(s[k])
    draws = {k: np.asarray(v, dtype=float) for k, v in draws.items()}

    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    out = []
    for j, f in enumerate(features):
        row = {"feature": f, "n_boot": n_boot, "ci": ci}
        for k in ("rank_target", "rank_reference", "rank_shift", "share_ratio"):
            row[k] = float(point[k][j])
            row[f"{k}_lo"] = float(np.quantile(draws[k][:, j], lo_q))
            row[f"{k}_hi"] = float(np.quantile(draws[k][:, j], hi_q))
        row["reference_rank_range"] = f"{int(point['ref_min'][j])}-{int(point['ref_max'][j])}"
        row["p_outside_reference_range"] = float(draws["outside"][:, j].mean())
        row["shift_ci_excludes_zero"] = bool(row["rank_shift_lo"] > 0 or row["rank_shift_hi"] < 0)
        row["share_ci_excludes_one"] = bool(row["share_ratio_lo"] > 1 or row["share_ratio_hi"] < 1)
        out.append(row)
    return pd.DataFrame(out)
