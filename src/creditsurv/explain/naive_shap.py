"""Naive (time-agnostic) SHAP, as the control condition for Stage 3(a).

The comparison this supports has to be fair, so it is built to isolate exactly
one difference. Both explainers here use:

* the same fitted survival model,
* the same background set,
* the same KernelSHAP machinery and ``nsamples``.

The *only* difference is the shape of the explained output. SurvSHAP(t) explains
the vector ``S(t | x)`` over a grid of horizons; naive SHAP explains the single
scalar ``1 - S(T | x)`` -- cumulative default probability by one fixed horizon,
which is what a practitioner gets by treating a survival model as a classifier
and reaching for `shap`.

That design matters. Training a separate binary classifier and comparing its SHAP
values would confound the attribution method with the model, and any disagreement
found would be uninterpretable. Here, a disagreement can only come from the
time dimension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .survshap import _as_shap_array, summarise_background

__all__ = ["NaiveShapExplanation", "explain_naive_shap"]


@dataclass
class NaiveShapExplanation:
    """Scalar attributions for ``1 - S(at_month | x)``."""

    phi: np.ndarray            # (n_observations, n_features)
    base: float
    prediction: np.ndarray     # (n_observations,)
    feature_names: tuple[str, ...]
    feature_values: pd.DataFrame
    at_month: float
    nsamples: int = 0
    n_background: int = 0

    def importance(self, obs: int | None = None) -> pd.Series:
        """Mean absolute attribution per feature."""
        phi = self.phi if obs is None else self.phi[[obs]]
        return (
            pd.Series(np.abs(phi).mean(axis=0), index=list(self.feature_names))
            .sort_values(ascending=False)
        )

    def signed_importance(self, obs: int | None = None) -> pd.Series:
        """Mean signed attribution. Positive means the feature raises risk.

        Note the sign convention is the *opposite* of SurvSHAP(t)'s, because this
        explains a default probability while SurvSHAP(t) explains a survival
        probability. :mod:`creditsurv.explain.compare` aligns them before
        comparing, and that alignment is a real source of error if forgotten.
        """
        phi = self.phi if obs is None else self.phi[[obs]]
        return pd.Series(phi.mean(axis=0), index=list(self.feature_names)).sort_values()


def explain_naive_shap(
    model,
    X: pd.DataFrame,
    background: pd.DataFrame,
    *,
    at_month: float = 36.0,
    nsamples: int = 256,
    n_background: int = 50,
    seed: int = 20260921,
    silent: bool = True,
) -> NaiveShapExplanation:
    """KernelSHAP on the scalar ``1 - S(at_month | x)``."""
    import shap

    bg = summarise_background(background, n=n_background, seed=seed)
    feature_names = tuple(X.columns)
    template = X.iloc[:1].copy()
    times = np.array([float(at_month)])

    def f(raw: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(raw, columns=list(feature_names))
        for col in feature_names:
            dtype = template[col].dtype
            if str(dtype) == "category":
                codes = np.rint(frame[col].to_numpy(dtype=float)).astype(int)
                cats = template[col].cat.categories
                frame[col] = pd.Categorical.from_codes(
                    np.clip(codes, 0, len(cats) - 1), categories=cats
                )
            else:
                frame[col] = frame[col].astype(dtype)
        return 1.0 - model.predict_survival(frame, times)[:, 0]

    def to_numeric(frame: pd.DataFrame) -> np.ndarray:
        out = frame.copy()
        for col in feature_names:
            if str(out[col].dtype) == "category":
                out[col] = out[col].cat.codes
        return out.to_numpy(dtype=float)

    # Identical floor to survshap: if the two explainers used different coalition
    # budgets, the Stage 3(a) comparison would be measuring sampling effort.
    nsamples = max(nsamples, 2 * len(feature_names))

    explainer = shap.KernelExplainer(f, to_numeric(bg))
    # l1_reg=0 for the same reason as in survshap: LARS feature selection is
    # unstable on this design matrix's near-constant columns. Both explainers
    # must use identical settings anyway, or the Stage 3(a) comparison would be
    # measuring the regulariser rather than the time dimension.
    raw = explainer.shap_values(
        to_numeric(X), nsamples=nsamples, silent=silent, l1_reg=0
    )
    phi = _as_shap_array(raw, len(X), len(feature_names), 1)[:, :, 0]

    return NaiveShapExplanation(
        phi=phi,
        base=float(np.asarray(explainer.expected_value, dtype=float).reshape(-1)[0]),
        prediction=1.0 - model.predict_survival(X, times)[:, 0],
        feature_names=feature_names,
        feature_values=X.copy(),
        at_month=float(at_month),
        nsamples=nsamples,
        n_background=len(bg),
    )
