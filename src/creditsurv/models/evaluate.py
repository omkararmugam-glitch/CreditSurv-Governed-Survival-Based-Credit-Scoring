"""Survival metrics, implemented here because `scikit-survival` is unavailable.

See FINDINGS.md section 0 for why. The three metrics that matter:

* **Harrell's concordance index** -- rank agreement between predicted risk and
  observed survival, over comparable pairs. Delegated to `lifelines`.
* **Cumulative/dynamic time-dependent AUC** (Uno et al. 2007) -- at horizon `t`,
  cases are loans that have defaulted by `t` and controls are loans still
  performing at `t`. Inverse-probability-of-censoring weights correct for the
  fact that censored loans are not missing at random with respect to time.
* **Integrated Brier score** -- a proper scoring rule, so it penalises
  miscalibration as well as mis-ranking. Concordance cannot detect a model whose
  ranking is perfect but whose probabilities are all wrong.

All three need the *censoring* distribution `G(t) = P(C > t)`, estimated by
Kaplan-Meier on the reversed event indicator. Where `G(t)` approaches zero the
weights explode, so it is floored and the floor is reported rather than hidden --
this is why time-dependent AUC is not trustworthy at horizons beyond which almost
everything is censored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "CensoringModel",
    "concordance_index",
    "cumulative_dynamic_auc",
    "brier_score",
    "integrated_brier_score",
    "calibration_table",
    "evaluate_survival",
    "EvaluationResult",
]


_G_FLOOR = 1e-3
"""Floor on the censoring survival probability used in IPCW weights.

A weight of 1/G can reach 1000x at this floor. Any horizon where the floor binds
for a meaningful share of observations is reported as unreliable.
"""


class CensoringModel:
    """Kaplan-Meier estimate of the censoring distribution ``G(t) = P(C > t)``.

    Fitted on the **training** split and applied to test data, so that the
    weights do not peek at the test outcomes.
    """

    def __init__(self, duration: np.ndarray, event: np.ndarray, g_floor: float = _G_FLOOR):
        from lifelines import KaplanMeierFitter

        self.g_floor = float(g_floor)
        # Reversed indicator: "event" for this fit is being censored.
        self._kmf = KaplanMeierFitter().fit(
            np.asarray(duration, dtype=float), 1 - np.asarray(event, dtype=int)
        )
        self.max_time = float(np.max(duration))

    def _predict(self, times: np.ndarray | float) -> np.ndarray:
        """Raw ``G(t)`` as a 1-D array.

        ``KaplanMeierFitter.predict`` collapses a single-element input to a scalar,
        which silently turns every downstream array operation into a 0-d one, so
        the result is forced back to 1-D here.
        """
        t = np.atleast_1d(np.asarray(times, dtype=float))
        return np.atleast_1d(np.asarray(self._kmf.predict(t), dtype=float))

    def survival(self, times: np.ndarray | float) -> np.ndarray:
        """``G(t)``, floored. Scalar or array in, always array out."""
        return np.clip(self._predict(times), self.g_floor, 1.0)

    def floor_binds(self, times: np.ndarray | float) -> np.ndarray:
        """Whether the floor is active at each time -- i.e. weights unreliable."""
        return self._predict(times) <= self.g_floor


def concordance_index(
    duration: np.ndarray, event: np.ndarray, risk: np.ndarray
) -> float:
    """Harrell's C. ``risk`` is higher-is-worse (larger = defaults sooner)."""
    from lifelines.utils import concordance_index as _ci

    # lifelines expects a predicted survival *time*, so risk is negated.
    return float(_ci(np.asarray(duration), -np.asarray(risk), np.asarray(event)))


def cumulative_dynamic_auc(
    survival: np.ndarray,
    times: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    censoring: CensoringModel,
) -> pd.DataFrame:
    """Time-dependent AUC at each horizon in ``times``.

    Parameters
    ----------
    survival:
        ``(n_samples, n_times)`` matrix of predicted ``S(t | x)``.
    duration, event:
        Observed outcomes for the same samples.
    censoring:
        Fitted on the training split.

    Returns
    -------
    DataFrame with one row per horizon: ``time``, ``auc``, ``n_cases``,
    ``n_controls`` and ``reliable``. ``auc`` is ``NaN`` where a horizon has no
    cases or no controls, which is reported rather than silently dropped.
    """
    survival = np.asarray(survival, dtype=float)
    times = np.atleast_1d(np.asarray(times, dtype=float))
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)

    if survival.shape != (len(duration), len(times)):
        raise ValueError(
            f"survival must be (n_samples, n_times) = "
            f"({len(duration)}, {len(times)}), got {survival.shape}"
        )

    g_at_t = censoring.survival(times)
    floor_binds = censoring.floor_binds(times)
    rows = []

    for k, t in enumerate(times):
        # Cumulative cases, dynamic controls.
        is_case = (duration <= t) & (event == 1)
        is_ctrl = duration > t
        n_case, n_ctrl = int(is_case.sum()), int(is_ctrl.sum())

        if n_case == 0 or n_ctrl == 0:
            rows.append((t, np.nan, n_case, n_ctrl, False))
            continue

        # Risk at horizon t is 1 - S(t), so higher = worse.
        risk = 1.0 - survival[:, k]
        w_case = 1.0 / censoring.survival(duration[is_case])

        ctrl_sorted = np.sort(risk[is_ctrl])
        case_risk = risk[is_case]
        n_lower = np.searchsorted(ctrl_sorted, case_risk, side="left")
        n_equal = np.searchsorted(ctrl_sorted, case_risk, side="right") - n_lower
        # Controls all share weight 1/G(t), which cancels in the ratio.
        concordant = n_lower + 0.5 * n_equal

        auc = float(np.sum(w_case * concordant) / (np.sum(w_case) * n_ctrl))
        rows.append((t, auc, n_case, n_ctrl, not bool(floor_binds[k])))

    return pd.DataFrame(
        rows, columns=["time", "auc", "n_cases", "n_controls", "reliable"]
    )


def brier_score(
    survival: np.ndarray,
    times: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    censoring: CensoringModel,
) -> pd.DataFrame:
    """IPCW Brier score at each horizon. Lower is better.

    Loans censored before ``t`` contribute nothing -- their status at ``t`` is
    genuinely unknown -- and the remaining contributions are up-weighted to
    compensate.
    """
    survival = np.asarray(survival, dtype=float)
    times = np.atleast_1d(np.asarray(times, dtype=float))
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)

    if survival.shape != (len(duration), len(times)):
        raise ValueError("survival must be (n_samples, n_times)")

    g_at_t = censoring.survival(times)
    g_at_dur = censoring.survival(duration)
    rows = []

    for k, t in enumerate(times):
        s = survival[:, k]
        had_event = (duration <= t) & (event == 1)
        still_alive = duration > t

        contrib = np.zeros(len(duration), dtype=float)
        # Observed default by t: true survival indicator is 0.
        contrib[had_event] = (s[had_event] ** 2) / g_at_dur[had_event]
        # Still performing at t: true survival indicator is 1.
        contrib[still_alive] = ((1.0 - s[still_alive]) ** 2) / g_at_t[k]

        n_used = int(had_event.sum() + still_alive.sum())
        rows.append((t, float(contrib.sum() / len(duration)), n_used))

    return pd.DataFrame(rows, columns=["time", "brier", "n_contributing"])


def integrated_brier_score(brier: pd.DataFrame) -> float:
    """Trapezoidal integral of the Brier curve, normalised by its time span."""
    valid = brier.dropna(subset=["brier"])
    if len(valid) < 2:
        return float("nan")
    t = valid["time"].to_numpy(dtype=float)
    b = valid["brier"].to_numpy(dtype=float)
    return float(np.trapezoid(b, t) / (t[-1] - t[0]))


def calibration_table(
    survival_at_t: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    t: float,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Predicted vs observed survival at horizon ``t``, by predicted decile.

    Observed survival within each bin is a Kaplan-Meier estimate, not a raw
    proportion, so censored loans inside the bin are handled correctly.
    """
    from lifelines import KaplanMeierFitter

    s = np.asarray(survival_at_t, dtype=float)
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)

    ranks = pd.qcut(pd.Series(s).rank(method="first"), n_bins, labels=False)
    rows = []
    for b in range(n_bins):
        m = (ranks == b).to_numpy()
        if m.sum() < 2:
            continue
        try:
            km = KaplanMeierFitter().fit(duration[m], event[m])
            observed = float(km.predict(t))
        except Exception:
            observed = float("nan")
        rows.append(
            {
                "bin": b,
                "n": int(m.sum()),
                "predicted_survival": float(s[m].mean()),
                "observed_survival": observed,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["error"] = out["predicted_survival"] - out["observed_survival"]
    return out


@dataclass
class EvaluationResult:
    """Everything reported for one model on one split."""

    model: str
    split: str
    concordance: float
    auc_table: pd.DataFrame
    brier_table: pd.DataFrame
    ibs: float
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    n: int = 0
    event_rate: float = float("nan")

    def summary(self) -> dict:
        aucs = {
            f"auc_{int(r.time)}m": (None if pd.isna(r.auc) else round(float(r.auc), 4))
            for r in self.auc_table.itertuples()
        }
        return {
            "model": self.model,
            "split": self.split,
            "n": self.n,
            "event_rate": round(self.event_rate, 4),
            "concordance": round(self.concordance, 4),
            "ibs": None if pd.isna(self.ibs) else round(self.ibs, 4),
            **aucs,
            "unreliable_horizons": [
                int(r.time) for r in self.auc_table.itertuples() if not r.reliable
            ],
        }

    def __str__(self) -> str:
        lines = [f"{self.model} [{self.split}]  n={self.n:,}  "
                 f"event_rate={self.event_rate:.4f}",
                 f"  concordance = {self.concordance:.4f}",
                 f"  IBS         = {self.ibs:.4f}"]
        for r in self.auc_table.itertuples():
            flag = "" if r.reliable else "   (unreliable: censoring floor binds)"
            auc = "   n/a" if pd.isna(r.auc) else f"{r.auc:.4f}"
            lines.append(
                f"  AUC({int(r.time):>2}m) = {auc}  "
                f"cases={r.n_cases:,} controls={r.n_controls:,}{flag}"
            )
        return "\n".join(lines)


def evaluate_survival(
    *,
    model_name: str,
    split_name: str,
    survival: np.ndarray,
    times: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    censoring: CensoringModel,
    risk: np.ndarray | None = None,
    calibration_at: float | None = None,
) -> EvaluationResult:
    """Run the full metric suite for one model on one split.

    ``risk`` defaults to ``1 - S(t_max)``, i.e. cumulative default probability
    over the whole evaluated horizon, which is the natural single-number risk
    ordering when no linear predictor is available.
    """
    survival = np.asarray(survival, dtype=float)
    times = np.atleast_1d(np.asarray(times, dtype=float))
    if risk is None:
        risk = 1.0 - survival[:, -1]

    auc = cumulative_dynamic_auc(survival, times, duration, event, censoring)
    brier = brier_score(survival, times, duration, event, censoring)

    calib = pd.DataFrame()
    if calibration_at is not None:
        k = int(np.argmin(np.abs(times - calibration_at)))
        calib = calibration_table(survival[:, k], duration, event, float(times[k]))

    return EvaluationResult(
        model=model_name,
        split=split_name,
        concordance=concordance_index(duration, event, risk),
        auc_table=auc,
        brier_table=brier,
        ibs=integrated_brier_score(brier),
        calibration=calib,
        n=len(duration),
        event_rate=float(np.mean(event)),
    )
