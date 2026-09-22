"""Exact, deterministic attributions for the discrete-time hazard model.

Why this exists
---------------
:mod:`creditsurv.explain.survshap` is KernelSHAP-based: it costs seconds per
applicant and samples coalitions, so two runs of the same applicant can disagree
on near-tied reasons. That is affordable for research and for one applicant at a
time, and hopeless for a file of a million. TreeSHAP reads the fitted trees
directly -- exact, deterministic, and microseconds per row -- but it explains what
the booster outputs, which here is a *per-period hazard margin*, not survival.

What is attributed
------------------
The model is a binary booster over person-period rows, so with
``f_t(x)`` the raw margin for period ``t`` and ``h_t = sigmoid(f_t)``:

    S(H | x) = prod_t (1 - h_t)          for the periods up to horizon H
    L(H | x) = -log S = sum_t softplus(f_t)

``L`` is the cumulative hazard: monotone in the default probability
(``PD = 1 - exp(-L)``), so any ranking of contributions to ``L`` is a ranking of
contributions to predicted default risk. TreeSHAP gives exact Shapley values
``phi_j(x, t)`` of each ``f_t``. Since ``d softplus(f)/df = sigmoid(f) = h_t``,
the first-order expansion of ``L`` around the applicant's own margins is

    L(x) ~= const + sum_t h_t * sum_j phi_j(x, t)

and the attribution used here is the inner sum reordered:

    contribution_j(x) = sum_t h_t(x) * phi_j(x, t)

Because Shapley values are linear in the value function, this *is* the exact
Shapley value of the hazard-weighted margin ``sum_t h_t * f_t`` -- it is not an
approximate Shapley value of an exact quantity, but an exact Shapley value of a
first-order approximation of one. The distinction matters and is why this is
validated against SurvSHAP(t) before being used for notices rather than assumed
equivalent (see FINDINGS section 7).

Sign convention matches :class:`SurvShapExplanation`: positive raises survival, so
the contributions above are negated. ``period`` is the model's own time index, not
an applicant characteristic, and is dropped from the attributions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["TreeShapExplanation", "explain_tree_shap"]


@dataclass
class TreeShapExplanation:
    """The subset of :class:`SurvShapExplanation`'s interface the notice code uses.

    Duck-typed deliberately: ``build_adverse_action_notice`` then works unchanged,
    so a notice built from TreeSHAP goes through exactly the same Regulation B
    filtering, direction checks and wording as one built from SurvSHAP(t).
    """

    phi: np.ndarray                      # (n_obs, n_features) in survival direction
    times: np.ndarray
    prediction: np.ndarray               # (n_obs, n_times) survival probabilities
    feature_names: tuple[str, ...]
    feature_values: pd.DataFrame
    horizon_months: float
    base: np.ndarray = field(default_factory=lambda: np.zeros(1))
    explainer: str = "treeshap"

    @property
    def n_observations(self) -> int:
        return self.phi.shape[0]

    def importance(self, obs: int | None = None) -> pd.Series:
        values = (np.abs(self.phi[obs]) if obs is not None
                  else np.abs(self.phi).mean(axis=0))
        return pd.Series(values, index=list(self.feature_names)).sort_values(
            ascending=False)

    def signed_importance(self, obs: int | None = None) -> pd.Series:
        values = self.phi[obs] if obs is not None else self.phi.mean(axis=0)
        return pd.Series(values, index=list(self.feature_names))


def explain_tree_shap(model, X: pd.DataFrame, *, horizon_months: float = 36.0,
                      times: np.ndarray | None = None,
                      batch_rows: int = 50_000) -> TreeShapExplanation:
    """Attributions for predicted default risk at ``horizon_months``.

    ``model`` is a fitted :class:`~creditsurv.models.discrete_hazard.
    DiscreteTimeHazardModel`. One TreeSHAP pass is made per period up to the
    horizon; the per-period Shapley values are combined with the applicant's own
    hazards as described in the module docstring.

    Deterministic: the same row always yields the same attributions, whatever else
    is in the batch, so a notice does not depend on how the file was divided.
    """
    if getattr(model, "booster", None) is None:
        raise RuntimeError("model is not fitted; TreeSHAP needs the booster")

    times = np.atleast_1d(np.asarray(
        times if times is not None else [horizon_months], dtype=float))
    n_periods = max(1, int(np.ceil(float(horizon_months) / model.time_bin_months)))
    grid = np.arange(1, n_periods + 1)

    columns = list(model.feature_names)
    period_at = columns.index(model.period_col)
    keep = [i for i, c in enumerate(columns) if c != model.period_col]
    names = tuple(columns[i] for i in keep)

    phi = np.zeros((len(X), len(keep)), dtype=float)
    work = X.copy()
    for start in range(0, len(X), batch_rows):        # bounded memory per pass
        block = work.iloc[start:start + batch_rows]
        acc = np.zeros((len(block), len(keep)), dtype=float)
        for p in grid:
            frame = block.copy()
            frame[model.period_col] = np.int16(p)
            frame = frame[columns]
            contrib = model.booster.predict(frame, pred_contrib=True)
            contrib = np.asarray(contrib, dtype=float)
            margin = contrib.sum(axis=1)              # raw score = sum of phi + base
            hazard = 1.0 / (1.0 + np.exp(-margin))    # d softplus / d margin
            acc += hazard[:, None] * contrib[:, keep]
        phi[start:start + batch_rows] = -acc          # + raises survival
        del acc

    survival = model.predict_survival(X, times)
    return TreeShapExplanation(
        phi=phi, times=times, prediction=survival, feature_names=names,
        feature_values=X.drop(columns=[model.period_col], errors="ignore").copy(),
        horizon_months=float(horizon_months),
        base=np.zeros(len(times)))
