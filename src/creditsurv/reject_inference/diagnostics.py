"""Stage 4 gate: is there meaningful selection bias to correct?

Reject inference is applied only if this diagnostic says it is warranted. The
thresholds live in :class:`~creditsurv.config.DiagnosticConfig` and were fixed
before any result was seen, so the conclusion cannot be reverse-engineered.

Why effect sizes and not p-values
---------------------------------
There are 2.26M accepted and 27.6M rejected applications. At that size *every*
Kolmogorov-Smirnov test returns p < 1e-300 for any difference whatsoever,
including differences far too small to matter. A p-value-based gate would
therefore always fire and would be pure theatre. p-values are computed and
reported for completeness but never used to decide.

Why rank statistics for two of the features
-------------------------------------------
`fico_range_low` (accepted) and `risk_score` (rejected) are *different
instruments*, and the accepted/rejected DTI fields use different definitions. A
standardised mean difference between them conflates measurement with selection --
a large SMD could mean the populations differ, or merely that the two columns are
not on the same scale. For features graded ``partial`` in
:data:`creditsurv.io.schema.COMMON_FEATURE_MAP`, the verdict therefore uses a
**rank AUC** (Mann-Whitney), which is invariant to any monotone rescaling and so
survives the instrument mismatch. The SMD is still reported, flagged as
indicative only.

Common support
--------------
Even large distributional differences do not license reject inference. What
licenses it is *overlap*: rejected applicants must occupy a region of feature
space where accepted outcomes were actually observed, or any inferred outcome is
extrapolation dressed up as correction. Lending Club used near-deterministic
score and DTI cutoffs, so thin overlap is a live possibility and
:attr:`SelectionDiagnostic.gate_decision` will block correction if it occurs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "FeatureDiagnostic",
    "SelectionDiagnostic",
    "standardized_mean_difference",
    "rank_auc",
    "ks_statistic",
    "separability",
    "common_support",
    "run_selection_diagnostic",
    "parse_rejected_dti",
]


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen-style SMD using the pooled standard deviation.

    Meaningful only when ``a`` and ``b`` measure the same quantity on the same
    scale.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if pooled <= 1e-12:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def rank_auc(a: np.ndarray, b: np.ndarray, *, max_n: int = 200_000, seed: int = 0) -> float:
    """P(random draw from ``a`` exceeds a random draw from ``b``), ties at 0.5.

    The Mann-Whitney statistic normalised to [0, 1]. Scale-free, so it is valid
    across the FICO / Risk_Score instrument mismatch. 0.5 means the two
    distributions are interchangeable by rank; distance from 0.5 is the effect
    size.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")

    rng = np.random.default_rng(seed)
    if len(a) > max_n:
        a = rng.choice(a, max_n, replace=False)
    if len(b) > max_n:
        b = rng.choice(b, max_n, replace=False)

    from scipy.stats import mannwhitneyu

    try:
        u = mannwhitneyu(a, b, alternative="two-sided").statistic
    except ValueError:
        return float("nan")
    return float(u / (len(a) * len(b)))


def ks_statistic(a: np.ndarray, b: np.ndarray, *, max_n: int = 200_000, seed: int = 0):
    """Two-sample KS statistic and p-value. Only the statistic is used to decide."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    if len(a) > max_n:
        a = rng.choice(a, max_n, replace=False)
    if len(b) > max_n:
        b = rng.choice(b, max_n, replace=False)

    from scipy.stats import ks_2samp

    res = ks_2samp(a, b)
    return float(res.statistic), float(res.pvalue)


def parse_rejected_dti(
    values: pd.Series, *, clip_upper: float = 100.0, clip_lower: float = 0.0
) -> pd.Series:
    """Parse the rejected file's DTI text and clip it to a plausible range.

    The raw column runs from -1% to 50,000,031%. Those are self-reported junk, not
    signal. The clip is a parameter so the choice appears in the audit, and the
    share of rows affected is reported by the caller rather than hidden.
    """
    numeric = pd.to_numeric(
        values.astype("string").str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )
    return numeric.clip(lower=clip_lower, upper=clip_upper).astype("float32")


def separability(
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    *,
    seed: int = 20260921,
    max_n: int = 200_000,
) -> tuple[float, np.ndarray, np.ndarray, pd.Series]:
    """Train a classifier to tell accepted from rejected.

    Returns ``(auc, accepted_propensity, rejected_propensity, coefficients)``.

    The AUC is a single summary of how strong selection was: 0.5 means the two
    populations are indistinguishable on the common features (no selection bias
    detectable), while values near 1.0 mean selection was essentially
    deterministic in these features -- which is *bad* news for reject inference,
    because it implies little overlap to borrow information across.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    cols = [c for c in accepted.columns if c in rejected.columns]
    if not cols:
        raise ValueError("no common columns between accepted and rejected")

    acc = accepted[cols]
    rej = rejected[cols]
    if len(acc) > max_n:
        acc = acc.iloc[rng.choice(len(acc), max_n, replace=False)]
    if len(rej) > max_n:
        rej = rej.iloc[rng.choice(len(rej), max_n, replace=False)]

    X = pd.concat([acc, rej], ignore_index=True)
    y = np.concatenate([np.ones(len(acc), dtype=int), np.zeros(len(rej), dtype=int)])

    medians = X.median()
    X = X.fillna(medians)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    clf = LogisticRegression(max_iter=2000, random_state=seed).fit(Xs, y)
    p = clf.predict_proba(Xs)[:, 1]
    auc = float(roc_auc_score(y, p))
    coefs = pd.Series(clf.coef_[0], index=cols).sort_values(key=np.abs, ascending=False)

    # Score the full inputs so common support is computed on everything.
    full_acc = scaler.transform(accepted[cols].fillna(medians))
    full_rej = scaler.transform(rejected[cols].fillna(medians))
    return (
        auc,
        clf.predict_proba(full_acc)[:, 1],
        clf.predict_proba(full_rej)[:, 1],
        coefs,
    )


def common_support(
    accepted_propensity: np.ndarray,
    rejected_propensity: np.ndarray,
    *,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> dict:
    """Share of rejected applicants inside the accepted propensity range.

    The range is trimmed to the 1st-99th percentile of accepted propensities, so a
    handful of outlying accepted loans cannot manufacture apparent overlap.
    """
    acc = np.asarray(accepted_propensity, dtype=float)
    rej = np.asarray(rejected_propensity, dtype=float)
    acc = acc[np.isfinite(acc)]
    rej = rej[np.isfinite(rej)]
    if len(acc) < 2 or len(rej) < 2:
        return {"share_in_support": float("nan")}

    lo, hi = np.quantile(acc, [lower_q, upper_q])
    inside = (rej >= lo) & (rej <= hi)
    return {
        "accepted_propensity_range": (float(lo), float(hi)),
        "accepted_median": float(np.median(acc)),
        "rejected_median": float(np.median(rej)),
        "share_in_support": float(inside.mean()),
        "n_rejected_in_support": int(inside.sum()),
        "n_rejected": int(len(rej)),
    }


@dataclass
class FeatureDiagnostic:
    feature: str
    comparability: str
    n_accepted: int
    n_rejected: int
    accepted_mean: float
    rejected_mean: float
    smd: float
    rank_auc: float
    ks: float
    ks_pvalue: float
    primary_metric: str
    verdict: str

    def as_row(self) -> dict:
        return {
            "feature": self.feature,
            "comparability": self.comparability,
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "accepted_mean": round(self.accepted_mean, 3),
            "rejected_mean": round(self.rejected_mean, 3),
            "smd": round(self.smd, 4),
            "rank_auc": round(self.rank_auc, 4),
            "ks": round(self.ks, 4),
            "ks_pvalue": f"{self.ks_pvalue:.3e}",
            "primary_metric": self.primary_metric,
            "verdict": self.verdict,
        }


@dataclass
class SelectionDiagnostic:
    """The complete Stage 4 gate result."""

    features: list[FeatureDiagnostic]
    separability_auc: float
    separability_coefficients: pd.Series
    support: dict
    thresholds: dict
    n_accepted: int
    n_rejected: int
    notes: list[str] = field(default_factory=list)

    @property
    def substantial_features(self) -> list[str]:
        return [f.feature for f in self.features if f.verdict == "substantial"]

    @property
    def notable_features(self) -> list[str]:
        return [f.feature for f in self.features if f.verdict in {"notable", "substantial"}]

    @property
    def bias_detected(self) -> bool:
        """Whether meaningful distributional distortion is present."""
        strong_auc = self.separability_auc >= self.thresholds["separability_auc_strong"]
        return bool(self.substantial_features or strong_auc)

    @property
    def support_adequate(self) -> bool:
        share = self.support.get("share_in_support", float("nan"))
        return bool(np.isfinite(share) and share >= self.thresholds["min_common_support"])

    @property
    def gate_decision(self) -> str:
        """``apply_correction``, ``no_correction_needed`` or ``correction_unjustified``."""
        if not self.bias_detected:
            return "no_correction_needed"
        if not self.support_adequate:
            return "correction_unjustified"
        return "apply_correction"

    def rationale(self) -> str:
        share = self.support.get("share_in_support", float("nan"))
        decision = self.gate_decision
        if decision == "no_correction_needed":
            return (
                f"No substantial distortion. Separability AUC {self.separability_auc:.3f} "
                f"is below the {self.thresholds['separability_auc_strong']} threshold and "
                f"no like-for-like feature exceeds |SMD| "
                f"{self.thresholds['smd_substantial']}. Reject inference is not "
                f"warranted, so none is applied."
            )
        if decision == "correction_unjustified":
            return (
                f"Distortion IS present (separability AUC {self.separability_auc:.3f}; "
                f"substantial on {self.substantial_features or 'AUC alone'}) but common "
                f"support is only {share:.1%}, below the pre-registered "
                f"{self.thresholds['min_common_support']:.0%} floor. Rejected "
                f"applicants largely occupy regions where no accepted outcome was "
                f"observed, so any inferred outcome would be extrapolation rather "
                f"than correction. NO correction applied -- this is the finding, not "
                f"a failure."
            )
        return (
            f"Distortion present (separability AUC {self.separability_auc:.3f}; "
            f"substantial on {self.substantial_features or 'AUC alone'}) AND common "
            f"support is {share:.1%}, above the {self.thresholds['min_common_support']:.0%} "
            f"floor. A reweighting correction is warranted and will be applied."
        )

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([f.as_row() for f in self.features])

    def summary(self) -> dict:
        return {
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "separability_auc": round(self.separability_auc, 4),
            "top_selection_drivers": list(self.separability_coefficients.head(3).index),
            "common_support_share": (
                None
                if not np.isfinite(self.support.get("share_in_support", np.nan))
                else round(float(self.support["share_in_support"]), 4)
            ),
            "substantial_features": self.substantial_features,
            "notable_features": self.notable_features,
            "bias_detected": self.bias_detected,
            "support_adequate": self.support_adequate,
            "gate_decision": self.gate_decision,
            "rationale": self.rationale(),
            "thresholds": self.thresholds,
            "notes": self.notes,
            "features": [f.as_row() for f in self.features],
        }


def _verdict(
    value: float, *, notable: float, substantial: float, centre: float = 0.0
) -> str:
    if not np.isfinite(value):
        return "not_comparable"
    dev = abs(value - centre)
    if dev >= substantial:
        return "substantial"
    if dev >= notable:
        return "notable"
    return "negligible"


def run_selection_diagnostic(
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    *,
    comparability: dict[str, str] | None = None,
    smd_notable: float = 0.10,
    smd_substantial: float = 0.25,
    ks_substantial: float = 0.20,
    rank_auc_notable: float = 0.55,
    rank_auc_substantial: float = 0.65,
    separability_auc_strong: float = 0.75,
    separability_auc_weak: float = 0.60,
    min_common_support: float = 0.05,
    seed: int = 20260921,
) -> SelectionDiagnostic:
    """Run the full gate on aligned accepted / rejected frames.

    Both frames must already share column names (see
    :data:`creditsurv.io.schema.REJECTED_RENAMES`). ``comparability`` marks which
    features are ``"good"`` (like-for-like, judged on SMD) versus ``"partial"``
    (different instruments, judged on rank AUC).
    """
    comparability = comparability or {}
    cols = [c for c in accepted.columns if c in rejected.columns]
    if not cols:
        raise ValueError("accepted and rejected share no columns")

    diagnostics: list[FeatureDiagnostic] = []
    notes: list[str] = []

    for col in cols:
        a = pd.to_numeric(accepted[col], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(rejected[col], errors="coerce").to_numpy(dtype=float)
        grade = comparability.get(col, "good")

        smd = standardized_mean_difference(a, b)
        rauc = rank_auc(a, b, seed=seed)
        ks, ks_p = ks_statistic(a, b, seed=seed)

        if grade == "partial":
            primary = "rank_auc"
            verdict = _verdict(
                rauc, notable=rank_auc_notable - 0.5,
                substantial=rank_auc_substantial - 0.5, centre=0.5
            )
        else:
            primary = "smd"
            verdict = _verdict(smd, notable=smd_notable, substantial=smd_substantial)

        a_fin, b_fin = a[np.isfinite(a)], b[np.isfinite(b)]
        diagnostics.append(
            FeatureDiagnostic(
                feature=col,
                comparability=grade,
                n_accepted=int(len(a_fin)),
                n_rejected=int(len(b_fin)),
                accepted_mean=float(a_fin.mean()) if len(a_fin) else float("nan"),
                rejected_mean=float(b_fin.mean()) if len(b_fin) else float("nan"),
                smd=smd,
                rank_auc=rauc,
                ks=ks,
                ks_pvalue=ks_p,
                primary_metric=primary,
                verdict=verdict,
            )
        )
        if grade == "partial":
            notes.append(
                f"{col}: graded 'partial' -- judged on rank AUC ({rauc:.3f}), not SMD "
                f"({smd:.3f}), because the accepted and rejected columns are different "
                f"instruments and an SMD between them conflates scale with selection."
            )

    auc, p_acc, p_rej, coefs = separability(accepted[cols], rejected[cols], seed=seed)
    support = common_support(p_acc, p_rej)

    if auc < separability_auc_weak:
        notes.append(
            f"Separability AUC {auc:.3f} is below the weak threshold "
            f"{separability_auc_weak}: on these {len(cols)} common features the two "
            f"populations are barely distinguishable."
        )

    return SelectionDiagnostic(
        features=diagnostics,
        separability_auc=auc,
        separability_coefficients=coefs,
        support=support,
        thresholds={
            "smd_notable": smd_notable,
            "smd_substantial": smd_substantial,
            "ks_substantial": ks_substantial,
            "rank_auc_notable": rank_auc_notable,
            "rank_auc_substantial": rank_auc_substantial,
            "separability_auc_strong": separability_auc_strong,
            "separability_auc_weak": separability_auc_weak,
            "min_common_support": min_common_support,
        },
        n_accepted=len(accepted),
        n_rejected=len(rejected),
        notes=notes,
    )
