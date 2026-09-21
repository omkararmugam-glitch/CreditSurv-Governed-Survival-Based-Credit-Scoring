"""Feature encoders for the Lending Club origination features.

Kept separate from :mod:`creditsurv.features.build` so each transform is a pure
function that can be tested on a handful of rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "parse_emp_length",
    "income_band",
    "fico_midpoint",
    "credit_history_months",
    "winsorize",
    "EMP_LENGTH_ORDER",
    "INCOME_BAND_EDGES",
]


EMP_LENGTH_ORDER: dict[str, float] = {
    "< 1 year": 0.5,
    "1 year": 1.0,
    "2 years": 2.0,
    "3 years": 3.0,
    "4 years": 4.0,
    "5 years": 5.0,
    "6 years": 6.0,
    "7 years": 7.0,
    "8 years": 8.0,
    "9 years": 9.0,
    "10+ years": 10.0,
}
"""``emp_length`` is ordinal, not nominal, so it is mapped to years rather than
one-hot encoded. ``"< 1 year"`` becomes 0.5 and ``"10+ years"`` becomes 10.0 --
the latter is right-censored in reality, which is worth remembering when reading
any attribution for this feature.
"""


INCOME_BAND_EDGES: tuple[float, ...] = (0, 30_000, 50_000, 75_000, 100_000, 150_000, np.inf)


def parse_emp_length(values: pd.Series) -> pd.Series:
    """Map employment-length strings to years as ``float32``.

    Unparsable or missing values stay ``NaN``; roughly 6.5% of the real file is
    missing here, and that missingness is itself informative (no employment
    record), so it is never filled with zero.
    """
    text = values.astype("string").str.strip()
    return text.map(EMP_LENGTH_ORDER).astype("float32")


def income_band(values: pd.Series) -> pd.Series:
    """Bucket annual income into ordered bands, for segment analysis only.

    Used by Stage 3's segment-stability report, never as a model feature -- the
    model sees continuous income.
    """
    labels = ["<30k", "30-50k", "50-75k", "75-100k", "100-150k", "150k+"]
    return pd.cut(values, bins=list(INCOME_BAND_EDGES), labels=labels, right=False)


def fico_midpoint(low: pd.Series, high: pd.Series) -> pd.Series:
    """Collapse the FICO band to its midpoint.

    ``fico_range_low`` and ``fico_range_high`` are almost perfectly collinear
    (they differ by 4 points by construction), which breaks the Cox
    partial-likelihood Hessian. One midpoint column carries the same information.
    """
    return ((low.astype("float32") + high.astype("float32")) / 2.0).astype("float32")


def credit_history_months(earliest_cr_line: pd.Series, issue_month: pd.Series) -> pd.Series:
    """Months of credit history at origination.

    Derived rather than raw: ``earliest_cr_line`` is a date, and a model cannot
    use a date, but its distance from origination is a strong risk feature.
    """
    delta = (issue_month.dt.year - earliest_cr_line.dt.year) * 12 + (
        issue_month.dt.month - earliest_cr_line.dt.month
    )
    return delta.astype("float32")


def winsorize(
    values: pd.Series, lower: float | None = None, upper: float | None = None
) -> pd.Series:
    """Clip to the given quantiles, returning ``float32``.

    Used for the rejected file's self-reported DTI, which contains negative
    values and a maximum of 50,000,031%. The cut is always passed explicitly by
    the caller so it appears in the audit rather than being buried here.
    """
    s = values.astype("float32")
    lo = s.quantile(lower) if lower is not None else None
    hi = s.quantile(upper) if upper is not None else None
    return s.clip(lower=lo, upper=hi)
