"""Construction of the ``(duration_months, event)`` survival target.

Time origin is ``issue_d``; time is measured in **integer months**. Lending Club
dates are month-granular strings such as ``"Dec-2018"``, so day-level resolution
would be fabricated precision. Integer months also feed the discrete-time hazard
model in :mod:`creditsurv.models.discrete_hazard` without further bucketing.

End-of-observation date by outcome class::

    EVENT                 -> last_pymnt_d   (last payment received)
    CENSORED_PAYOFF       -> last_pymnt_d   (payoff)
    CENSORED_DELINQUENT   -> last_pymnt_d
    CENSORED_ADMIN        -> data cutoff

For events, ``last_pymnt_d`` is used directly rather than adding Lending Club's
roughly 150-day charge-off lag. The target is therefore "time to cessation of
payment", which is *directly observed* rather than inferred from an assumed
servicing policy. ``event_lag_months`` runs the alternative as a sensitivity
check.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .status_rules import Outcome, classify_status

__all__ = [
    "LabelConfig",
    "parse_month_series",
    "parse_term_months",
    "months_between",
    "build_survival_target",
]


_MONTH_FORMATS: tuple[str, ...] = (
    "%b-%Y",      # Dec-2018  (accepted file, most vintages)
    "%Y-%m-%d",   # 2018-12-01
    "%b-%y",      # Dec-18
    "%Y-%m",      # 2018-12
    "%d-%b-%Y",   # 01-Dec-2018
    "%m/%d/%Y",   # 12/01/2018
)


@dataclass(frozen=True)
class LabelConfig:
    """Tunable parts of the labelling rule. Defaults are the primary spec."""

    late_31_120_is_event: bool = False
    include_policy_exceptions: bool = False

    min_duration_months: int = 1
    """Floor for observed duration. A loan cannot default in less than one
    scheduled payment cycle, and zero-length rows break the Cox partial
    likelihood."""

    term_overrun_grace_months: int = 12
    """Rows whose duration exceeds ``term + grace`` are treated as date errors
    and dropped, not winsorised."""

    event_lag_months: int = 0
    """Sensitivity switch: months added to event times to approximate Lending
    Club's charge-off lag. ``0`` is the primary specification."""

    data_cutoff: str | None = None
    """``"YYYY-MM"``. When ``None``, inferred as the latest observed payment
    month in the data rather than hardcoded."""


def parse_month_series(values: pd.Series) -> pd.Series:
    """Parse a column of mixed-format month strings to month-start timestamps.

    Each known Lending Club format is tried in turn on the rows still unparsed.
    This is both faster and safer than letting ``to_datetime`` infer, since
    inference can silently flip between month-first and day-first across chunks.
    """
    text = values.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    remaining = text.notna() & (text.str.len() > 0)

    for fmt in _MONTH_FORMATS:
        if not remaining.any():
            break
        idx = remaining[remaining].index
        parsed = pd.to_datetime(text.loc[idx], format=fmt, errors="coerce")
        ok = parsed.notna()
        if ok.any():
            out.loc[parsed.index[ok]] = parsed[ok]
            remaining.loc[parsed.index[ok]] = False

    # Collapse to the first of the month: day-of-month is not meaningful here.
    return out.dt.to_period("M").dt.to_timestamp()


def parse_term_months(values: pd.Series) -> pd.Series:
    """Extract an integer month count from strings like ``" 36 months"``."""
    digits = values.astype("string").str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(digits, errors="coerce").astype("Int64")


def months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Whole months from ``start`` to ``end``, ``NA`` where either is missing."""
    delta = (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)
    return delta.astype("Float64").astype("Int64")


def _infer_cutoff(last_pymnt: pd.Series, issue: pd.Series) -> pd.Timestamp:
    """Latest month the extract plausibly observed."""
    candidates = [s.max() for s in (last_pymnt, issue) if s.notna().any()]
    if not candidates:
        raise ValueError("Cannot infer data cutoff: no parsable dates found.")
    return max(candidates)


def build_survival_target(
    df: pd.DataFrame,
    config: LabelConfig | None = None,
    *,
    status_col: str = "loan_status",
    issue_col: str = "issue_d",
    last_pymnt_col: str = "last_pymnt_d",
    term_col: str = "term",
) -> tuple[pd.DataFrame, dict]:
    """Attach ``duration_months`` / ``event`` and report exactly what was dropped.

    Returns
    -------
    (labeled, audit)
        ``labeled`` contains only retained rows, with the survival target and
        provenance columns appended. ``audit`` is a plain dict of counts,
        suitable for writing straight into FINDINGS.md. Every exclusion is
        accounted for, so the row budget always reconciles.
    """
    config = config or LabelConfig()
    n_input = len(df)

    # Classify by *unique* status value rather than per row. On the full
    # dataset that is 9 calls instead of 2.26 million, and it avoids copying
    # the frame -- which matters, since the ingested frame is ~2.3 GB.
    status_raw = df[status_col]
    lookup = {
        value: classify_status(
            value,
            late_31_120_is_event=config.late_31_120_is_event,
            include_policy_exceptions=config.include_policy_exceptions,
        ).value
        for value in pd.unique(status_raw.dropna().astype(object))
    }
    outcome = pd.Categorical(
        status_raw.map(lookup).fillna(Outcome.EXCLUDED.value),
        categories=[o.value for o in Outcome],
    )
    outcome_s = pd.Series(outcome, index=df.index)

    issue_month = parse_month_series(df[issue_col])
    last_pymnt_month = parse_month_series(df[last_pymnt_col])
    term_months = parse_term_months(df[term_col])

    cutoff = (
        pd.Period(config.data_cutoff, freq="M").to_timestamp()
        if config.data_cutoff
        else _infer_cutoff(last_pymnt_month, issue_month)
    )

    # --- end-of-observation date -------------------------------------------
    uses_cutoff = outcome_s == Outcome.CENSORED_ADMIN.value
    end = last_pymnt_month.where(~uses_cutoff, cutoff)

    # A charged-off loan with no payment on record never paid at all: place the
    # event at the first missed scheduled payment rather than dropping the row.
    is_event_row = outcome_s == Outcome.EVENT.value
    never_paid = is_event_row & end.isna()
    if never_paid.any():
        fallback = issue_month + pd.DateOffset(months=config.min_duration_months)
        end = end.where(~never_paid, fallback)

    if config.event_lag_months:
        end = end.where(
            ~is_event_row, end + pd.DateOffset(months=config.event_lag_months)
        )

    # --- duration ----------------------------------------------------------
    raw_duration = months_between(issue_month, end)
    duration = raw_duration.clip(lower=config.min_duration_months)
    floored = raw_duration.notna() & (raw_duration < config.min_duration_months)

    # --- exclusions --------------------------------------------------------
    reasons = pd.Series(pd.NA, index=df.index, dtype="string")

    def mark(mask: pd.Series, reason: str) -> None:
        reasons.loc[mask.fillna(False) & reasons.isna()] = reason

    mark(outcome_s == Outcome.EXCLUDED.value, "excluded_status")
    mark(issue_month.isna(), "missing_issue_d")
    mark(term_months.isna(), "missing_term")
    mark(end.isna(), "missing_end_date")
    overrun = (
        duration.notna()
        & term_months.notna()
        & (duration > term_months + config.term_overrun_grace_months)
    )
    mark(overrun, "term_overrun")

    # Materialise only the retained rows, so peak memory stays near one copy
    # of the input rather than two-plus.
    keep = reasons.isna()
    labeled = df.loc[keep].copy()
    labeled["outcome"] = outcome_s.loc[keep]
    labeled["issue_month"] = issue_month.loc[keep]
    labeled["last_pymnt_month"] = last_pymnt_month.loc[keep]
    labeled["term_months"] = term_months.loc[keep]
    labeled["observation_end_month"] = end.loc[keep]
    labeled["never_paid"] = never_paid.loc[keep]
    labeled["duration_raw_months"] = raw_duration.loc[keep]
    labeled["duration_months"] = duration.loc[keep].astype("int32")
    labeled["duration_floored"] = floored.loc[keep]
    labeled["event"] = is_event_row.loc[keep].astype("int8")

    audit = {
        "n_input": int(n_input),
        "n_retained": int(len(labeled)),
        "data_cutoff": str(cutoff.date()),
        "cutoff_source": "config" if config.data_cutoff else "inferred",
        "status_counts": {
            str(k): int(v)
            for k, v in df[status_col].value_counts(dropna=False).items()
        },
        "outcome_counts": {
            str(k): int(v) for k, v in outcome_s.value_counts(dropna=False).items()
        },
        "drop_counts": {
            str(k): int(v) for k, v in reasons.value_counts(dropna=True).items()
        },
        "n_never_paid": int(never_paid.sum()),
        "n_duration_floored": int(floored.sum()),
        "event_rate": float(labeled["event"].mean()) if len(labeled) else float("nan"),
        "duration_months_median": (
            float(labeled["duration_months"].median()) if len(labeled) else float("nan")
        ),
        "duration_months_max": (
            int(labeled["duration_months"].max()) if len(labeled) else 0
        ),
        "config": {
            "late_31_120_is_event": config.late_31_120_is_event,
            "include_policy_exceptions": config.include_policy_exceptions,
            "min_duration_months": config.min_duration_months,
            "term_overrun_grace_months": config.term_overrun_grace_months,
            "event_lag_months": config.event_lag_months,
        },
    }
    if audit["n_retained"] + sum(audit["drop_counts"].values()) != n_input:
        raise AssertionError("row budget does not reconcile; exclusion masks overlap")
    return labeled, audit
