"""Stage 3(a): does naive SHAP actually disagree with SurvSHAP(t) *here*?

The literature reports that collapsing a survival prediction to a scalar
misrepresents skewed survival outcomes. This module measures whether that
transfers to this dataset instead of assuming it does. A finding of "they largely
agree on Lending Club" is a legitimate result and is reported as such.

Four measures, because they fail in different ways:

* **Spearman rank correlation** of feature importances. Catches wholesale
  reordering, but is insensitive to swaps among the unimportant tail.
* **Top-k Jaccard overlap.** What a reader of an adverse-action notice actually
  sees is the top few reasons, so disagreement concentrated in the top 5 matters
  far more than disagreement at rank 30.
* **Sign agreement.** Whether the two methods agree on the *direction* of each
  feature's effect. A sign flip in a top reason is the most serious possible
  disagreement -- it would mean telling an applicant the opposite of the truth.
* **Time-varying share.** SurvSHAP(t)'s own diagnostic: how much of a feature's
  attribution varies over time rather than sitting at a constant level. A feature
  with a near-flat curve loses nothing in the scalar collapse; one that crosses
  zero loses everything. This is the mechanism by which disagreement, if present,
  arises -- so it explains the other three numbers rather than just restating
  them.

Sign convention
---------------
SurvSHAP(t) explains ``S(t|x)`` (survival), naive SHAP explains ``1 - S(T|x)``
(default). Their signs are therefore mirrored, and every comparison below negates
the survival-side attributions first. Forgetting this inverts every conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .naive_shap import NaiveShapExplanation
from .survshap import SurvShapExplanation

__all__ = [
    "ComparisonResult",
    "compare_explanations",
    "time_variation_share",
    "top_k_overlap",
]


def top_k_overlap(a: pd.Series, b: pd.Series, k: int) -> float:
    """Jaccard overlap of the top-``k`` features of two importance Series.

    Both inputs are sorted descending here rather than assumed to arrive sorted.
    Relying on ``head(k)`` of an unsorted Series takes the first *k positions*,
    not the *k largest*, which silently reports agreement between rankings that
    are in fact reversed.
    """
    top_a = set(a.sort_values(ascending=False).head(k).index)
    top_b = set(b.sort_values(ascending=False).head(k).index)
    union = top_a | top_b
    return float(len(top_a & top_b) / len(union)) if union else float("nan")


def time_variation_share(expl: SurvShapExplanation) -> pd.Series:
    """Per feature, the share of attribution magnitude that is time-varying.

    Computed as ``mean_t |phi(t) - mean_t phi(t)| / mean_t |phi(t)|``. Zero means
    a perfectly flat curve -- the scalar collapse is lossless for that feature.
    Values near or above 1 mean the curve's shape carries most of the signal, and
    a feature whose curve changes sign will score high.
    """
    phi = expl.phi                             # (obs, feat, time)
    mean_over_time = phi.mean(axis=2, keepdims=True)
    deviation = np.abs(phi - mean_over_time).mean(axis=2)
    magnitude = np.abs(phi).mean(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(magnitude > 1e-12, deviation / magnitude, np.nan)

    # A feature with zero attribution for every observation yields an all-NaN
    # column; averaging that is not an error, it is 0 variation, so it is handled
    # explicitly rather than by letting numpy warn on an empty slice.
    usable = ~np.all(np.isnan(share), axis=0)
    out = np.zeros(share.shape[1], dtype=float)
    if usable.any():
        out[usable] = np.nanmean(share[:, usable], axis=0)
    return pd.Series(out, index=list(expl.feature_names)).sort_values(ascending=False)


@dataclass
class ComparisonResult:
    spearman: float
    pearson: float
    top5_overlap: float
    top10_overlap: float
    sign_agreement: float
    sign_disagreements: tuple[str, ...]
    top5_sign_disagreements: tuple[str, ...]
    survshap_importance: pd.Series
    naive_importance: pd.Series
    time_variation: pd.Series
    per_feature: pd.DataFrame
    n_observations: int
    at_month: float

    def verdict(self, *, spearman_floor: float = 0.8, overlap_floor: float = 0.6) -> str:
        """A plain-language reading against pre-registered thresholds.

        Thresholds are fixed here so the conclusion cannot be reverse-engineered
        from whatever the data happened to produce.
        """
        if self.top5_sign_disagreements:
            return (
                "MATERIAL DISAGREEMENT: the two methods disagree on the direction "
                f"of {len(self.top5_sign_disagreements)} top-5 feature(s) "
                f"({', '.join(self.top5_sign_disagreements)}). Naive SHAP would "
                "state the wrong reason to an applicant."
            )
        if self.spearman < spearman_floor or self.top5_overlap < overlap_floor:
            return (
                f"MEANINGFUL DISAGREEMENT: Spearman {self.spearman:.3f} "
                f"(floor {spearman_floor}), top-5 overlap {self.top5_overlap:.3f} "
                f"(floor {overlap_floor}). Ranking differs enough to change which "
                "reasons get reported."
            )
        return (
            f"BROAD AGREEMENT: Spearman {self.spearman:.3f}, top-5 overlap "
            f"{self.top5_overlap:.3f}, sign agreement "
            f"{self.sign_agreement:.3f}. On this dataset the scalar collapse does "
            "not materially change which features are named, so the literature's "
            "concern does not reproduce here at the level of reported reasons."
        )

    def summary(self) -> dict:
        return {
            "n_observations": self.n_observations,
            "at_month": self.at_month,
            "spearman": round(self.spearman, 4),
            "pearson": round(self.pearson, 4),
            "top5_overlap": round(self.top5_overlap, 4),
            "top10_overlap": round(self.top10_overlap, 4),
            "sign_agreement": round(self.sign_agreement, 4),
            "sign_disagreements": list(self.sign_disagreements),
            "top5_sign_disagreements": list(self.top5_sign_disagreements),
            "mean_time_variation_share": round(float(self.time_variation.mean()), 4),
            "most_time_varying": list(self.time_variation.head(5).index),
            "verdict": self.verdict(),
        }


def compare_explanations(
    surv: SurvShapExplanation,
    naive: NaiveShapExplanation,
    *,
    k_small: int = 5,
    k_large: int = 10,
) -> ComparisonResult:
    """Compare a SurvSHAP(t) explanation against naive SHAP on the same model."""
    from scipy.stats import pearsonr, spearmanr

    if surv.feature_names != naive.feature_names:
        raise ValueError("explanations must share the same feature ordering")

    features = list(surv.feature_names)
    surv_imp = surv.importance()
    naive_imp = naive.importance()

    common = surv_imp.reindex(features).to_numpy()
    other = naive_imp.reindex(features).to_numpy()
    ok = np.isfinite(common) & np.isfinite(other)

    spearman = float(spearmanr(common[ok], other[ok]).statistic) if ok.sum() > 2 else np.nan
    pearson = float(pearsonr(common[ok], other[ok]).statistic) if ok.sum() > 2 else np.nan

    # Align sign conventions: survival-side attributions are negated so that, on
    # both sides, positive means "increases default risk".
    surv_signed = (-surv.signed_importance()).reindex(features)
    naive_signed = naive.signed_importance().reindex(features)

    both_nonzero = (surv_signed.abs() > 1e-9) & (naive_signed.abs() > 1e-9)
    agree = np.sign(surv_signed) == np.sign(naive_signed)
    sign_agreement = (
        float((agree & both_nonzero).sum() / both_nonzero.sum())
        if both_nonzero.any()
        else float("nan")
    )
    disagreements = tuple(
        f for f in features if both_nonzero.get(f, False) and not agree.get(f, True)
    )
    top5 = set(surv_imp.head(k_small).index) | set(naive_imp.head(k_small).index)
    top5_disagreements = tuple(f for f in disagreements if f in top5)

    time_var = time_variation_share(surv)

    per_feature = pd.DataFrame(
        {
            "survshap_importance": surv_imp.reindex(features),
            "naive_importance": naive_imp.reindex(features),
            "survshap_rank": surv_imp.rank(ascending=False).reindex(features),
            "naive_rank": naive_imp.rank(ascending=False).reindex(features),
            "survshap_signed": surv_signed,
            "naive_signed": naive_signed,
            "sign_agrees": agree.reindex(features),
            "time_variation_share": time_var.reindex(features),
        }
    )
    per_feature["rank_shift"] = (
        per_feature["survshap_rank"] - per_feature["naive_rank"]
    ).abs()

    return ComparisonResult(
        spearman=spearman,
        pearson=pearson,
        top5_overlap=top_k_overlap(surv_imp, naive_imp, k_small),
        top10_overlap=top_k_overlap(surv_imp, naive_imp, k_large),
        sign_agreement=sign_agreement,
        sign_disagreements=disagreements,
        top5_sign_disagreements=top5_disagreements,
        survshap_importance=surv_imp,
        naive_importance=naive_imp,
        time_variation=time_var,
        per_feature=per_feature.sort_values("survshap_importance", ascending=False),
        n_observations=surv.n_observations,
        at_month=naive.at_month,
    )
