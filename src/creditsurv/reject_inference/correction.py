"""Reject-inference correction, applied only when the diagnostic gate allows it.

Deliberately simple. The spec asks for reweighting or augmentation and warns
against over-engineering, and there is a substantive reason to keep it simple
here: with only ~6 common features (and the score column two-thirds missing on
the rejected side), an elaborate method would be inventing precision that the
overlapping information cannot support.

The method
----------
**Inverse-propensity reweighting.** Fit ``P(accepted | x)`` on the common
features, then weight each accepted loan by ``1 / P(accepted | x)``. An accepted
loan that looks like a typical *rejected* applicant gets a large weight, because
it stands in for many unobserved rejects. This is Horvitz-Thompson weighting, and
it makes the accepted sample resemble the full applicant population in the
common-feature marginals -- without fabricating a single outcome label.

What is deliberately NOT done
-----------------------------
No outcome is imputed for any rejected applicant. Those applications were never
funded, so no ground truth exists at any time horizon, and assigning them
synthetic default labels would misrepresent the data. Reweighting changes how the
*observed* outcomes are counted; it never manufactures new ones.

Honest limits
------------
Reweighting corrects bias only on the features used to build the weights. Income,
revolving utilisation, home ownership and credit-history depth are absent from the
rejected file entirely, so any selection operating through them is untouched and
*invisible* to this correction. Weight clipping is applied and reported, since
unclipped inverse-propensity weights in a thin-overlap region produce estimates
driven by a handful of observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["RejectInferenceWeights", "fit_reweighting", "effective_sample_size"]


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size, ``(sum w)^2 / sum(w^2)``.

    The honest count of how much information survives weighting. A large drop
    from the nominal n means the estimate rests on relatively few observations.
    """
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w) & (w > 0)]
    if len(w) == 0:
        return 0.0
    return float(w.sum() ** 2 / np.square(w).sum())


@dataclass
class RejectInferenceWeights:
    """Weights for the accepted sample, plus everything needed to judge them."""

    weights: np.ndarray
    propensity: np.ndarray
    features_used: tuple[str, ...]
    clip_quantiles: tuple[float, float]
    clip_bounds: tuple[float, float]
    n_clipped: int
    nominal_n: int
    effective_n: float
    propensity_auc: float
    unavailable_features: tuple[str, ...] = field(default=())

    @property
    def ess_ratio(self) -> float:
        return float(self.effective_n / self.nominal_n) if self.nominal_n else float("nan")

    def summary(self) -> dict:
        w = self.weights
        return {
            "features_used": list(self.features_used),
            "unavailable_features": list(self.unavailable_features),
            "propensity_auc": round(self.propensity_auc, 4),
            "weight_mean": round(float(np.mean(w)), 4),
            "weight_min": round(float(np.min(w)), 4),
            "weight_max": round(float(np.max(w)), 4),
            "weight_p99": round(float(np.quantile(w, 0.99)), 4),
            "clip_quantiles": list(self.clip_quantiles),
            "clip_bounds": [round(b, 4) for b in self.clip_bounds],
            "n_clipped": self.n_clipped,
            "nominal_n": self.nominal_n,
            "effective_n": round(self.effective_n, 1),
            "ess_ratio": round(self.ess_ratio, 4),
            "caveat": (
                "Corrects selection only on the features listed in features_used. "
                "Selection operating through "
                f"{', '.join(self.unavailable_features) or 'unobserved features'} "
                "is not addressed and is invisible to this correction."
            ),
        }


def fit_reweighting(
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    *,
    features: list[str] | None = None,
    clip_quantiles: tuple[float, float] = (0.01, 0.99),
    unavailable_features: tuple[str, ...] = (),
    seed: int = 20260921,
    max_n: int = 200_000,
) -> RejectInferenceWeights:
    """Fit inverse-propensity weights for the accepted sample.

    Parameters
    ----------
    clip_quantiles:
        Weights are clipped to these quantiles of their own distribution. Without
        this, a single accepted loan in a thin-overlap region can dominate the
        entire weighted estimate.
    unavailable_features:
        Features known to drive selection but absent from the rejected file.
        Recorded so the correction's blind spot is stated in its own summary
        rather than only in prose elsewhere.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    cols = features or [c for c in accepted.columns if c in rejected.columns]
    if not cols:
        raise ValueError("no common features available for reweighting")

    rng = np.random.default_rng(seed)
    acc_fit = accepted[cols]
    rej_fit = rejected[cols]
    if len(acc_fit) > max_n:
        acc_fit = acc_fit.iloc[rng.choice(len(acc_fit), max_n, replace=False)]
    if len(rej_fit) > max_n:
        rej_fit = rej_fit.iloc[rng.choice(len(rej_fit), max_n, replace=False)]

    X = pd.concat([acc_fit, rej_fit], ignore_index=True)
    y = np.concatenate([np.ones(len(acc_fit), dtype=int), np.zeros(len(rej_fit), dtype=int)])

    medians = X.median()
    scaler = StandardScaler().fit(X.fillna(medians))
    clf = LogisticRegression(max_iter=2000, random_state=seed).fit(
        scaler.transform(X.fillna(medians)), y
    )
    auc = float(roc_auc_score(y, clf.predict_proba(scaler.transform(X.fillna(medians)))[:, 1]))

    # Score the FULL accepted set, not the fitting subsample.
    p_acc = clf.predict_proba(scaler.transform(accepted[cols].fillna(medians)))[:, 1]
    p_acc = np.clip(p_acc, 1e-6, 1 - 1e-6)

    raw = 1.0 / p_acc
    lo, hi = np.quantile(raw, list(clip_quantiles))
    clipped = np.clip(raw, lo, hi)
    n_clipped = int(((raw < lo) | (raw > hi)).sum())
    # Normalise to mean 1 so weighted counts stay on the original scale.
    weights = clipped / clipped.mean()

    return RejectInferenceWeights(
        weights=weights.astype("float64"),
        propensity=p_acc,
        features_used=tuple(cols),
        clip_quantiles=clip_quantiles,
        clip_bounds=(float(lo), float(hi)),
        n_clipped=n_clipped,
        nominal_n=len(accepted),
        effective_n=effective_sample_size(weights),
        propensity_auc=auc,
        unavailable_features=unavailable_features,
    )
