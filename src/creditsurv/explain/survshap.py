"""SurvSHAP(t): time-dependent Shapley attributions for survival predictions.

Reimplementation of the method of Krzyzinski et al. (2023), *Knowledge-Based
Systems* 262:110234. The reference package (`survshap` 0.4.2) cannot be installed
on this machine -- see FINDINGS.md section 0 -- so the method is built directly on
`shap.KernelExplainer`.

The idea
--------
Ordinary SHAP explains a scalar. A survival model does not produce a scalar; it
produces a function of time. SurvSHAP(t) therefore computes a Shapley value for
each feature *at each time point*, decomposing

    S(t | x) - E[S(t)] = sum_j phi_j(t)

so an attribution becomes a curve rather than a number. A feature can matter
early and not late, or flip sign as the loan seasons -- structure that collapsing
to one number destroys.

Correctness
-----------
Because this is a reimplementation rather than the reference package, it is
checked against the **efficiency axiom** of Shapley values: at every `t`, the
attributions plus the base value must equal the model's prediction.
:func:`check_efficiency` measures the residual and
``tests/test_survshap.py`` asserts it. That is a sharp, falsifiable property --
an implementation that got the weighting or the masking wrong would fail it.

Cost
----
KernelSHAP is a weighted least-squares problem over coalitions and needs
`nsamples * n_background` model evaluations per explained observation. This is
why explanations are computed on a sample of borrowers rather than the whole
population, and why the background set is summarised by k-means. Expect roughly
a second per borrower at the defaults here.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STABLE_NSAMPLES_FLOOR = 600
"""Empirically-determined setting at which the between-draw Spearman of the full
importance ranking reaches ~0.96 on this 73-feature space.

From the sweep in ``outputs/tables/03_nsamples_study.csv`` (40 borrowers, two
independent coalition draws, background held fixed):

===========  ==================  ====================
``nsamples``  between-draw rho    seconds/borrower
===========  ==================  ====================
146 (= 2p)   0.883               0.85
210          0.883               0.82
350          0.890               1.18
600          0.960               1.85
1200         0.973               4.50
2194         0.969               8.38
===========  ==================  ====================

Top-5 and top-10 membership were identical between draws at *every* setting, so
the features named in an adverse-action notice are stable even at the floor. It is
the tail of the ranking that needs the extra samples.
"""

__all__ = [
    "STABLE_NSAMPLES_FLOOR",
    "SurvShapExplanation",
    "explain_survshap",
    "check_efficiency",
    "summarise_background",
    "aggregate_importance",
]


def summarise_background(
    X: pd.DataFrame,
    n: int = 50,
    seed: int = 20260921,
    max_fit_rows: int = 50_000,
) -> pd.DataFrame:
    """Reduce a background set to ``n`` representative rows.

    KernelSHAP cost is linear in background size, so this is the main lever on
    runtime. Numeric columns are k-means summarised; categorical columns take the
    modal value of each cluster, since a mean is meaningless for them.

    ``max_fit_rows`` caps the rows k-means actually clusters. On the full dataset
    the background would otherwise be 451k rows, and clustering that to find 100
    centroids costs far more than it improves them -- a random subsample locates
    the same centroids to well within the Monte Carlo noise of the explanation
    itself.
    """
    if len(X) <= n:
        return X.copy()

    if len(X) > max_fit_rows:
        X = X.sample(n=max_fit_rows, random_state=seed)

    num_cols = [c for c in X.columns if str(X[c].dtype) != "category"]
    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]

    if not num_cols:
        return X.sample(n=n, random_state=seed)

    from sklearn.cluster import KMeans

    num = X[num_cols].astype("float64")
    filled = num.fillna(num.median())
    sd = filled.std().replace(0.0, 1.0)
    z = (filled - filled.mean()) / sd

    labels = KMeans(n_clusters=n, n_init=4, random_state=seed).fit_predict(z)
    rows = []
    for k in range(n):
        m = labels == k
        if not m.any():
            continue
        # A cluster can be entirely missing for a structurally-missing column
        # (mths_since_last_record is 84% NaN on the real data). NaN is retained
        # there rather than invented, since missingness is signal for this model;
        # taking a median of nothing is handled explicitly to avoid numpy's
        # empty-slice warning firing once per cluster per column.
        row = {}
        for c in num_cols:
            vals = num.loc[m, c]
            row[c] = float(vals.median()) if vals.notna().any() else np.nan
        for c in cat_cols:
            modes = X.loc[m, c].mode()
            row[c] = modes.iloc[0] if len(modes) else X[c].iloc[0]
        rows.append(row)

    out = pd.DataFrame(rows)
    for c in cat_cols:
        out[c] = pd.Categorical(out[c], categories=X[c].cat.categories)
    return out[list(X.columns)].reset_index(drop=True)


@dataclass
class SurvShapExplanation:
    """Time-dependent attributions for one or more observations.

    ``phi`` has shape ``(n_observations, n_features, n_times)``.
    """

    phi: np.ndarray
    times: np.ndarray
    base: np.ndarray
    prediction: np.ndarray
    feature_names: tuple[str, ...]
    feature_values: pd.DataFrame
    nsamples: int = 0
    n_background: int = 0

    def __post_init__(self) -> None:
        self.phi = np.asarray(self.phi, dtype=float)
        if self.phi.ndim != 3:
            raise ValueError(f"phi must be 3-D, got shape {self.phi.shape}")

    @property
    def n_observations(self) -> int:
        return self.phi.shape[0]

    def curve(self, obs: int = 0) -> pd.DataFrame:
        """Attribution curves for one observation: rows = time, cols = feature."""
        return pd.DataFrame(
            self.phi[obs].T, index=self.times, columns=list(self.feature_names)
        ).rename_axis("time")

    def importance(self, obs: int | None = None) -> pd.Series:
        """Time-integrated absolute importance per feature.

        The paper's local variable importance: the area under ``|phi_j(t)|``,
        normalised by the time span so it reads on the same scale as the survival
        probability itself.
        """
        return aggregate_importance(self, obs=obs)

    def signed_importance(self, obs: int | None = None) -> pd.Series:
        """Time-integrated *signed* attribution, to show direction of effect.

        Positive means the feature pushes survival up (lowers risk).
        """
        phi = self.phi if obs is None else self.phi[[obs]]
        span = max(float(self.times[-1] - self.times[0]), 1e-12)
        area = np.trapezoid(phi, self.times, axis=2) / span
        return pd.Series(area.mean(axis=0), index=list(self.feature_names)).sort_values()


def aggregate_importance(
    expl: SurvShapExplanation, obs: int | None = None
) -> pd.Series:
    """Area under ``|phi_j(t)|``, averaged over observations if ``obs`` is None."""
    phi = expl.phi if obs is None else expl.phi[[obs]]
    span = max(float(expl.times[-1] - expl.times[0]), 1e-12)
    area = np.trapezoid(np.abs(phi), expl.times, axis=2) / span
    return (
        pd.Series(area.mean(axis=0), index=list(expl.feature_names))
        .sort_values(ascending=False)
    )


def _as_shap_array(raw, n_obs: int, n_features: int, n_outputs: int) -> np.ndarray:
    """Normalise shap's several return shapes to ``(n_obs, n_features, n_outputs)``."""
    if isinstance(raw, list):
        # Older API: list of (n_obs, n_features), one per output.
        arr = np.stack([np.asarray(a, dtype=float) for a in raw], axis=-1)
    else:
        arr = np.asarray(raw, dtype=float)

    if arr.ndim == 2:
        # Single output.
        arr = arr[:, :, None]
    if arr.shape == (n_obs, n_features, n_outputs):
        return arr
    if arr.shape == (n_outputs, n_obs, n_features):
        return np.transpose(arr, (1, 2, 0))
    if arr.shape == (n_features, n_outputs) and n_obs == 1:
        return arr[None, :, :]
    raise ValueError(
        f"unexpected shap output shape {arr.shape}; expected "
        f"({n_obs}, {n_features}, {n_outputs})"
    )


SHOW_PROGRESS_ENV = "CREDITSURV_SHOW_PROGRESS"


def explain_survshap(
    model,
    X: pd.DataFrame,
    background: pd.DataFrame,
    times: np.ndarray,
    *,
    nsamples: int = 256,
    n_background: int = 50,
    seed: int = 20260921,
    silent: bool | None = None,
) -> SurvShapExplanation:
    """Compute SurvSHAP(t) attributions for the rows of ``X``.

    Parameters
    ----------
    model:
        Anything exposing ``predict_survival(X, times) -> (n, len(times))``. Both
        :class:`~creditsurv.models.cox.CoxModel` and
        :class:`~creditsurv.models.discrete_hazard.DiscreteTimeHazardModel`
        satisfy this, so the same explainer works for either.
    background:
        Reference distribution for the "feature is absent" state. Summarised to
        ``n_background`` rows internally.
    silent:
        Whether to hide shap's per-applicant progress bar. ``None`` (the default)
        hides it unless ``CREDITSURV_SHOW_PROGRESS=1`` is set, which the background
        runner does for the Model registry's evidence jobs so their logs show how
        far a long run has got. Display only: the attributions are identical
        either way (tests/test_two_phase.py checks it).
    """
    import shap

    if silent is None:
        silent = os.environ.get(SHOW_PROGRESS_ENV) != "1"

    times = np.atleast_1d(np.asarray(times, dtype=float))
    bg = summarise_background(background, n=n_background, seed=seed)
    feature_names = tuple(X.columns)
    template = X.iloc[:1].copy()

    def f(raw: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(raw, columns=list(feature_names))
        # KernelExplainer passes plain float arrays, so dtypes must be restored
        # or LightGBM will reject the categorical columns.
        for col in feature_names:
            dtype = template[col].dtype
            if str(dtype) == "category":
                codes = np.rint(frame[col].to_numpy(dtype=float)).astype(int)
                cats = template[col].cat.categories
                codes = np.clip(codes, 0, len(cats) - 1)
                frame[col] = pd.Categorical.from_codes(codes, categories=cats)
            else:
                frame[col] = frame[col].astype(dtype)
        return model.predict_survival(frame, times)

    def to_numeric(frame: pd.DataFrame) -> np.ndarray:
        out = frame.copy()
        for col in feature_names:
            if str(out[col].dtype) == "category":
                out[col] = out[col].cat.codes
        return out.to_numpy(dtype=float)

    # shap enumerates coalitions by subset size, and there are exactly p subsets
    # of size 1 and p of size p-1. So 2p is the structural floor: below it, some
    # feature never appears alone and its main effect is never identified.
    p = len(feature_names)
    hard_floor = 2 * p
    if nsamples < hard_floor:
        warnings.warn(
            f"nsamples={nsamples} is below the structural floor 2p={hard_floor} for "
            f"{p} features; raising it. Below 2p, shap cannot enumerate every "
            f"size-1 and size-(p-1) coalition, so some main effects are unidentified.",
            RuntimeWarning,
            stacklevel=2,
        )
        nsamples = hard_floor
    elif nsamples < STABLE_NSAMPLES_FLOOR:
        # Measured, not guessed: see outputs/tables/03_nsamples_study.csv. Two
        # independent coalition draws at 2p agree only at Spearman 0.88 on the
        # 73-feature Lending Club space. That noise floor is close enough to the
        # 0.80 decision threshold in compare.py that a reported rank correlation
        # could not be distinguished from sampling noise. Top-5 membership is
        # stable at every setting tested, so per-borrower notices are unaffected.
        warnings.warn(
            f"nsamples={nsamples} identifies main effects but leaves the full-ranking "
            f"noise floor around Spearman 0.88. Use >= {STABLE_NSAMPLES_FLOOR} when the "
            f"headline number is a rank correlation over all features.",
            RuntimeWarning,
            stacklevel=2,
        )

    explainer = shap.KernelExplainer(f, to_numeric(bg))
    # l1_reg=0 is deliberate. Routing through LARS feature selection
    # (l1_reg="num_features(...)" or "auto") is numerically unstable on this
    # design matrix, which contains many near-constant columns -- rare-event
    # counts such as acc_now_delinq and missing-value indicators. LARS then
    # reports degenerate active sets and returns coefficients large enough to
    # break the efficiency axiom outright (residuals of order 1e7 were observed).
    # Plain weighted least squares keeps the axiom exact.
    raw = explainer.shap_values(
        to_numeric(X), nsamples=nsamples, silent=silent, l1_reg=0
    )
    phi = _as_shap_array(raw, len(X), len(feature_names), len(times))

    base = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    if base.size != len(times):
        base = np.resize(base, len(times))

    return SurvShapExplanation(
        phi=phi,
        times=times,
        base=base,
        prediction=model.predict_survival(X, times),
        feature_names=feature_names,
        feature_values=X.copy(),
        nsamples=nsamples,
        n_background=len(bg),
    )


def check_efficiency(expl: SurvShapExplanation) -> pd.DataFrame:
    """Efficiency-axiom residual: ``S(t|x) - base(t) - sum_j phi_j(t)``.

    Should be ~0 at every time for every observation. KernelSHAP solves a
    regularised least-squares problem, so a small residual is expected; a large
    one means the implementation is wrong.
    """
    reconstructed = expl.base[None, :] + expl.phi.sum(axis=1)
    residual = expl.prediction - reconstructed
    return pd.DataFrame(
        {
            "time": expl.times,
            "max_abs_residual": np.abs(residual).max(axis=0),
            "mean_abs_residual": np.abs(residual).mean(axis=0),
            "mean_prediction": expl.prediction.mean(axis=0),
        }
    )
