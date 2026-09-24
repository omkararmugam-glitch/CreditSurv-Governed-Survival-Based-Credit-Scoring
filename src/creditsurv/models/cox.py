"""Cox proportional hazards baseline (`lifelines`).

The interpretable reference point. Its coefficients are log hazard ratios, which
a credit risk reviewer can read directly, and it is the model against which the
higher-capacity discrete-time hazard model has to justify its extra complexity.

Two things about Cox on this data are worth stating rather than glossing:

* **The PH assumption is almost certainly violated somewhere.** Credit risk is
  strongly age-dependent -- early defaults look different from late ones -- and a
  single time-invariant hazard ratio per feature cannot represent that.
  :meth:`CoxModel.check_proportional_hazards` runs the test so the violation is
  measured rather than assumed away. That violation is a large part of *why*
  SurvSHAP(t) is interesting: attributions that change with `t` can show what a
  constant hazard ratio cannot.
* **Absolute risk still varies with loan age even under PH.** The hazard *ratio*
  is constant, but the baseline hazard is not, so a fixed borrower profile has a
  different conditional default probability at month 6 than at month 30.
  :func:`conditional_default_probability` reports that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "CoxModel",
    "conditional_default_probability",
    "age_dependent_risk_profile",
]


@dataclass
class CoxModel:
    """Thin wrapper around ``lifelines.CoxPHFitter``.

    Parameters
    ----------
    penalizer:
        Ridge penalty. A non-zero default is deliberate: this design matrix has ~140
        one-hot columns including rare state and purpose levels, and an unpenalised
        fit on those is numerically fragile.

        It was 0.01, and that was measured to be too weak. lifelines adds the penalty
        to the summed partial log-likelihood rather than the mean, so its effect
        shrinks as rows are added: at 60,000 rows an 0.01 ridge held the fit together
        and at 300,000 it did not, giving |coef| up to 458 and a test concordance of
        0.497 (FINDINGS 7h). At 0.05 the same fit converges, and on the pre-registered
        holdout it reproduces the published Cox concordance of 0.6602 to four decimal
        places, so the wider ridge costs nothing measurable.
    """

    penalizer: float = 0.05
    l1_ratio: float = 0.0
    fitter: object | None = None
    feature_names: tuple[str, ...] = ()

    def fit(
        self,
        X: pd.DataFrame,
        duration: pd.Series,
        event: pd.Series,
        *,
        show_progress: bool = False,
    ) -> "CoxModel":
        from lifelines import CoxPHFitter

        df = X.copy()
        df["_duration"] = np.asarray(duration, dtype=float)
        df["_event"] = np.asarray(event, dtype=int)

        fitter = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        fitter.fit(
            df,
            duration_col="_duration",
            event_col="_event",
            show_progress=show_progress,
        )
        self.fitter = fitter
        self.feature_names = tuple(X.columns)
        return self

    # -- prediction ---------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.fitter is None:
            raise RuntimeError("CoxModel is not fitted; call .fit() first")

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Linear predictor (log partial hazard). Higher = worse."""
        self._check_fitted()
        return np.asarray(
            self.fitter.predict_log_partial_hazard(X[list(self.feature_names)]),
            dtype=float,
        )

    def predict_survival(self, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        """``S(t | x)`` as an ``(n_samples, n_times)`` array."""
        self._check_fitted()
        times = np.atleast_1d(np.asarray(times, dtype=float))
        sf = self.fitter.predict_survival_function(
            X[list(self.feature_names)], times=times
        )
        # lifelines returns (times x samples); transpose to (samples x times).
        return np.asarray(sf.to_numpy(dtype=float).T)

    # -- interpretation -----------------------------------------------------

    def coefficient_table(self) -> pd.DataFrame:
        """Coefficients as hazard ratios with confidence intervals.

        Sorted by absolute effect size, which is what a reviewer wants to read
        first.
        """
        self._check_fitted()
        s = self.fitter.summary
        out = pd.DataFrame(
            {
                "feature": s.index,
                "coef": s["coef"].to_numpy(),
                "hazard_ratio": s["exp(coef)"].to_numpy(),
                "se": s["se(coef)"].to_numpy(),
                "hr_lower_95": s["exp(coef) lower 95%"].to_numpy(),
                "hr_upper_95": s["exp(coef) upper 95%"].to_numpy(),
                "p": s["p"].to_numpy(),
            }
        )
        out["abs_coef"] = out["coef"].abs()
        return out.sort_values("abs_coef", ascending=False).drop(columns="abs_coef")

    def check_proportional_hazards(self, X: pd.DataFrame, duration, event) -> pd.DataFrame:
        """Run lifelines' PH test and return per-feature results.

        A violation is expected here and is a finding, not a bug. Returns an
        empty frame if the test cannot run.
        """
        self._check_fitted()
        df = X.copy()
        df["_duration"] = np.asarray(duration, dtype=float)
        df["_event"] = np.asarray(event, dtype=int)
        try:
            res = self.fitter.check_assumptions(
                df, p_value_threshold=0.01, show_plots=False
            )
        except Exception as exc:  # pragma: no cover - diagnostic only
            return pd.DataFrame({"error": [str(exc)]})

        rows = []
        for item in res or []:
            try:
                frame = item[0].summary if hasattr(item[0], "summary") else None
                if frame is None:
                    continue
                for name, row in frame.iterrows():
                    rows.append(
                        {
                            "feature": name[0] if isinstance(name, tuple) else name,
                            "test_statistic": float(row.get("test_statistic", np.nan)),
                            "p": float(row.get("p", np.nan)),
                        }
                    )
            except Exception:
                continue
        return pd.DataFrame(rows)


def conditional_default_probability(
    survival: np.ndarray, times: np.ndarray, given_age: float, horizon: float
) -> np.ndarray:
    """P(default within ``horizon`` | survived to ``given_age``).

    ``1 - S(age + horizon) / S(age)``. This is the number that actually matters
    operationally: a servicer holding a 24-month-old performing loan cares about
    the next 12 months, not about risk measured from origination.
    """
    survival = np.asarray(survival, dtype=float)
    times = np.atleast_1d(np.asarray(times, dtype=float))

    def _at(t: float) -> np.ndarray:
        return survival[:, int(np.argmin(np.abs(times - t)))]

    s_now = _at(given_age)
    s_later = _at(given_age + horizon)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(s_now > 0, s_later / s_now, np.nan)
    return 1.0 - ratio


def age_dependent_risk_profile(
    model,
    profile: pd.DataFrame,
    times: np.ndarray,
    *,
    ages: tuple[int, ...] = (0, 6, 12, 18, 24, 30),
    horizon: int = 12,
) -> pd.DataFrame:
    """How a *fixed* borrower profile's risk changes as the loan seasons.

    This is the spec's "report risk as explicitly age-dependent" requirement. The
    borrower is identical in every row; only the loan's age differs. A single
    static score cannot express this, which is the point.
    """
    times = np.atleast_1d(np.asarray(times, dtype=float))
    survival = model.predict_survival(profile, times)

    rows = []
    for age in ages:
        if age + horizon > times.max():
            continue
        p = conditional_default_probability(survival, times, float(age), float(horizon))
        k = int(np.argmin(np.abs(times - age)))
        rows.append(
            {
                "loan_age_months": age,
                "survival_to_date": float(np.mean(survival[:, k])),
                f"p_default_next_{horizon}m": float(np.nanmean(p)),
            }
        )
    return pd.DataFrame(rows)
