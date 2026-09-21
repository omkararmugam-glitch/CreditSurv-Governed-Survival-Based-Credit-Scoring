"""Discrete-time hazard model: LightGBM on a person-period expansion.

The higher-capacity counterpart to Cox. Instead of a partial likelihood, each
loan is expanded into one row per period it was under observation, with a binary
target marking the period in which it defaulted. A gradient-boosted classifier
then estimates the discrete hazard

    h(t | x) = P(default in period t | survived to t, x)

and the survival curve follows from the product over periods:

    S(t | x) = prod_{s <= t} (1 - h(s | x))

Why this rather than a random survival forest: it scales. A RandomSurvivalForest
on 2.26M loans is not feasible on a laptop, whereas this reduces to ordinary
binary classification that LightGBM handles at this size. It also drops the
proportional-hazards restriction entirely -- the period index is just another
feature, so the model can learn interactions between loan age and borrower
attributes, which is exactly the structure Cox cannot represent.

Memory
------
Naive monthly expansion of the full dataset is ~45M rows, which will not fit
alongside a feature matrix on a 15 GB machine. Three levers, all explicit:

* ``time_bin_months`` -- quarterly bins cut the row count roughly 3x.
* ``max_horizon_months`` -- truncate follow-up.
* ``negative_subsample`` -- keep every default period but a fraction of the
  non-event periods, with compensating ``sample_weight``. This is standard
  case-control sampling; it leaves the hazard estimate consistent up to a known
  offset and the weights correct that offset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "PersonPeriod",
    "expand_person_period",
    "DiscreteTimeHazardModel",
    "expansion_row_estimate",
]


def expansion_row_estimate(
    duration: np.ndarray, time_bin_months: int = 1, max_horizon_months: int | None = None
) -> int:
    """Rows the expansion will produce, for checking feasibility before doing it."""
    d = np.asarray(duration, dtype=float)
    if max_horizon_months is not None:
        d = np.minimum(d, max_horizon_months)
    return int(np.ceil(d / time_bin_months).sum())


@dataclass
class PersonPeriod:
    """Expanded person-period dataset."""

    X: pd.DataFrame
    y: np.ndarray
    weight: np.ndarray
    loan_index: np.ndarray
    period: np.ndarray
    n_loans: int
    n_periods: int
    time_bin_months: int
    negative_subsample: float = 1.0

    def __len__(self) -> int:
        return len(self.y)


def expand_person_period(
    X: pd.DataFrame,
    duration: np.ndarray,
    event: np.ndarray,
    *,
    time_bin_months: int = 1,
    max_horizon_months: int | None = None,
    negative_subsample: float = 1.0,
    loan_weight: np.ndarray | None = None,
    period_col: str = "period",
    seed: int = 20260921,
) -> PersonPeriod:
    """Expand one row per loan into one row per period under observation.

    A loan observed for ``d`` months contributes periods ``1..ceil(d/bin)``. The
    target is 1 only in its final period, and only if the loan defaulted;
    censored loans contribute all-zero targets, which is precisely how censoring
    enters this likelihood.

    ``loan_weight`` gives a per-*loan* weight (for example reject-inference
    inverse-propensity weights). It is broadcast to every person-period belonging
    to that loan, so a loan's influence is scaled as a whole rather than varying
    across its own periods.
    """
    if time_bin_months < 1:
        raise ValueError("time_bin_months must be >= 1")
    if not 0.0 < negative_subsample <= 1.0:
        raise ValueError("negative_subsample must be in (0, 1]")

    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)
    if len(duration) != len(X) or len(event) != len(X):
        raise ValueError("X, duration and event must have the same length")

    d = duration.copy()
    if max_horizon_months is not None:
        # Truncating follow-up also censors anything that happened later.
        truncated = d > max_horizon_months
        event = np.where(truncated, 0, event)
        d = np.minimum(d, max_horizon_months)

    n_periods_per_loan = np.maximum(np.ceil(d / time_bin_months).astype(int), 1)
    loan_index = np.repeat(np.arange(len(X)), n_periods_per_loan)

    # period = 1, 2, ... within each loan
    starts = np.concatenate([[0], np.cumsum(n_periods_per_loan)[:-1]])
    period = np.arange(len(loan_index)) - np.repeat(starts, n_periods_per_loan) + 1

    is_last = period == np.repeat(n_periods_per_loan, n_periods_per_loan)
    y = (is_last & np.repeat(event == 1, n_periods_per_loan)).astype("int8")

    if loan_weight is None:
        base_weight = np.ones(len(X), dtype="float32")
    else:
        base_weight = np.asarray(loan_weight, dtype="float32")
        if len(base_weight) != len(X):
            raise ValueError(
                f"loan_weight has {len(base_weight)} entries but X has {len(X)} rows"
            )
    weight = np.repeat(base_weight, n_periods_per_loan)

    if negative_subsample < 1.0:
        rng = np.random.default_rng(seed)
        keep = (y == 1) | (rng.random(len(y)) < negative_subsample)
        loan_index, period, y = loan_index[keep], period[keep], y[keep]
        weight = weight[keep]
        # Up-weight retained negatives so the expected hazard is unchanged.
        weight[y == 0] = weight[y == 0] / negative_subsample

    expanded = X.iloc[loan_index].reset_index(drop=True)
    expanded[period_col] = period.astype("int16")

    return PersonPeriod(
        X=expanded,
        y=np.asarray(y),
        weight=weight,
        loan_index=loan_index,
        period=period,
        n_loans=len(X),
        n_periods=int(n_periods_per_loan.max()),
        time_bin_months=time_bin_months,
        negative_subsample=negative_subsample,
    )


@dataclass
class DiscreteTimeHazardModel:
    """LightGBM discrete-time hazard model."""

    time_bin_months: int = 1
    max_horizon_months: int | None = 60
    negative_subsample: float = 1.0
    params: dict = field(default_factory=dict)
    num_boost_round: int = 400
    seed: int = 20260921

    booster: object | None = None
    feature_names: tuple[str, ...] = ()
    categorical_features: tuple[str, ...] = ()
    period_col: str = "period"
    max_period_: int = 0
    training_rows_: int = 0

    DEFAULT_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
    }

    def fit(
        self,
        X: pd.DataFrame,
        duration: np.ndarray,
        event: np.ndarray,
        *,
        valid: tuple[pd.DataFrame, np.ndarray, np.ndarray] | None = None,
        loan_weight: np.ndarray | None = None,
        early_stopping_rounds: int | None = 50,
        verbose_eval: int | bool = False,
    ) -> "DiscreteTimeHazardModel":
        import lightgbm as lgb

        pp = expand_person_period(
            X,
            duration,
            event,
            time_bin_months=self.time_bin_months,
            max_horizon_months=self.max_horizon_months,
            negative_subsample=self.negative_subsample,
            loan_weight=loan_weight,
            period_col=self.period_col,
            seed=self.seed,
        )
        self.feature_names = tuple(pp.X.columns)
        self.categorical_features = tuple(
            c for c in pp.X.columns if str(pp.X[c].dtype) == "category"
        )
        self.max_period_ = pp.n_periods
        self.training_rows_ = len(pp)

        params = {**self.DEFAULT_PARAMS, **self.params, "seed": self.seed}
        train_set = lgb.Dataset(
            pp.X,
            label=pp.y,
            weight=pp.weight,
            categorical_feature=list(self.categorical_features) or "auto",
            free_raw_data=True,
        )

        valid_sets, callbacks = [], []
        if valid is not None:
            vX, vd, ve = valid
            vpp = expand_person_period(
                vX,
                vd,
                ve,
                time_bin_months=self.time_bin_months,
                max_horizon_months=self.max_horizon_months,
                period_col=self.period_col,
                seed=self.seed,
            )
            valid_sets = [
                lgb.Dataset(
                    vpp.X[list(self.feature_names)],
                    label=vpp.y,
                    reference=train_set,
                    categorical_feature=list(self.categorical_features) or "auto",
                )
            ]
            if early_stopping_rounds:
                callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        if verbose_eval:
            callbacks.append(lgb.log_evaluation(int(verbose_eval)))

        self.booster = lgb.train(
            params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks or None,
        )
        return self

    # -- prediction ---------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.booster is None:
            raise RuntimeError("model is not fitted; call .fit() first")

    def predict_hazard(self, X: pd.DataFrame, periods: np.ndarray) -> np.ndarray:
        """``h(t | x)`` as ``(n_samples, n_periods)``.

        One booster call per period rather than one big expansion, which keeps
        peak memory at ``n_samples`` rows instead of ``n_samples * n_periods``.
        """
        self._check_fitted()
        periods = np.atleast_1d(np.asarray(periods, dtype=int))
        base = X.copy()
        out = np.empty((len(X), len(periods)), dtype=float)
        for k, p in enumerate(periods):
            base[self.period_col] = np.int16(p)
            out[:, k] = self.booster.predict(base[list(self.feature_names)])
        return np.clip(out, 1e-9, 1 - 1e-9)

    def predict_survival(self, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        """``S(t | x)`` at the requested month values.

        Hazards are estimated on the model's own period grid and the cumulative
        product is then read off at the requested months, so the caller can ask
        for any horizon without worrying about bin width.
        """
        self._check_fitted()
        times = np.atleast_1d(np.asarray(times, dtype=float))
        max_period = max(
            1, int(np.ceil(float(np.max(times)) / self.time_bin_months))
        )
        grid = np.arange(1, max_period + 1)
        hazard = self.predict_hazard(X, grid)
        surv_grid = np.cumprod(1.0 - hazard, axis=1)

        # Month t falls in period ceil(t / bin); t <= 0 means survival 1.
        idx = np.ceil(times / self.time_bin_months).astype(int) - 1
        out = np.ones((len(X), len(times)), dtype=float)
        valid = idx >= 0
        out[:, valid] = surv_grid[:, np.clip(idx[valid], 0, surv_grid.shape[1] - 1)]
        return out

    def predict_risk(self, X: pd.DataFrame, at_month: float | None = None) -> np.ndarray:
        """Cumulative default probability by ``at_month`` (default: full horizon)."""
        t = at_month if at_month is not None else (
            self.max_horizon_months or self.max_period_ * self.time_bin_months
        )
        return 1.0 - self.predict_survival(X, np.array([t]))[:, 0]

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        self._check_fitted()
        return (
            pd.DataFrame(
                {
                    "feature": self.booster.feature_name(),
                    "importance": self.booster.feature_importance(importance_type),
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
