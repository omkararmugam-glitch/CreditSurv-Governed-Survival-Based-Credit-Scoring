"""Design-matrix construction for both model families.

Two models need two shapes of input:

* **Cox** (`lifelines`) needs a fully numeric, full-rank matrix. Categoricals are
  one-hot encoded with a dropped reference level, numerics are standardised, and
  collinear or zero-variance columns are removed -- otherwise the
  partial-likelihood Hessian is singular and the fit either fails or returns
  meaningless standard errors.
* **LightGBM** handles `category` dtype natively and is invariant to monotone
  rescaling, so it gets the raw columns.

Missingness is treated as signal, not noise. ``mths_since_last_delinq`` is 51%
missing and ``mths_since_last_record`` 84% missing on the real data, but those
gaps mean "no such record exists" -- a *good* sign -- so imputing a number would
destroy the information. Numeric columns with structural missingness get an
explicit ``_missing`` indicator, and the underlying value is median-filled only
for Cox, which cannot accept NaN. LightGBM sees the NaN directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..io import schema as sch
from . import encoders as enc

__all__ = [
    "FeatureSpec",
    "DesignMatrix",
    "default_spec",
    "add_derived_features",
    "build_design_matrix",
    "train_test_split_loans",
]


# Numeric columns whose missingness carries meaning rather than being a data
# fault. Each gets a companion <name>_missing indicator.
STRUCTURAL_MISSING: tuple[str, ...] = (
    "mths_since_last_delinq",
    "mths_since_last_record",
    "mths_since_last_major_derog",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_revol_delinq",
    "mths_since_recent_inq",
    "mths_since_recent_bc",
    "mo_sin_old_il_acct",
    "emp_length_years",
    "revol_util",
    "bc_util",
    "percent_bc_gt_75",
    "pct_tl_nvr_dlq",
)

# Dropped in favour of derived replacements (see add_derived_features).
SUPERSEDED: tuple[str, ...] = (
    "fico_range_low",
    "fico_range_high",
    "emp_length",
    "funded_amnt",
)


@dataclass(frozen=True)
class FeatureSpec:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    structural_missing: tuple[str, ...] = field(default=STRUCTURAL_MISSING)

    @property
    def all_columns(self) -> tuple[str, ...]:
        return tuple(self.numeric) + tuple(self.categorical)


@dataclass
class DesignMatrix:
    """A model-ready matrix plus everything needed to reproduce it."""

    X: pd.DataFrame
    duration: pd.Series
    event: pd.Series
    spec: FeatureSpec
    flavour: str
    dropped: dict[str, str] = field(default_factory=dict)
    standardisation: dict[str, tuple[float, float]] = field(default_factory=dict)
    fill_values: dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.X)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach derived origination features. Returns a new frame.

    ``fico_midpoint`` replaces the two near-collinear FICO bounds,
    ``emp_length_years`` turns an ordinal string into a number, and
    ``installment_to_income`` is the affordability ratio a human underwriter
    would actually look at.
    """
    out = df.copy()

    if {"fico_range_low", "fico_range_high"} <= set(out.columns):
        out["fico_midpoint"] = enc.fico_midpoint(
            out["fico_range_low"], out["fico_range_high"]
        )
    if "emp_length" in out.columns:
        out["emp_length_years"] = enc.parse_emp_length(out["emp_length"])
    if {"installment", "annual_inc"} <= set(out.columns):
        monthly = out["annual_inc"].astype("float32") / 12.0
        ratio = out["installment"].astype("float32") / monthly.replace(0.0, np.nan)
        out["installment_to_income"] = ratio.replace([np.inf, -np.inf], np.nan).astype(
            "float32"
        )
    if {"loan_amnt", "annual_inc"} <= set(out.columns):
        ratio = out["loan_amnt"].astype("float32") / out["annual_inc"].replace(
            0.0, np.nan
        ).astype("float32")
        out["loan_to_income"] = ratio.replace([np.inf, -np.inf], np.nan).astype("float32")
    if "annual_inc" in out.columns:
        # Income is strongly right-skewed; the log is the sensible scale for a
        # linear model and harmless for a tree.
        out["log_annual_inc"] = np.log1p(
            out["annual_inc"].clip(lower=0).astype("float32")
        ).astype("float32")
    if "annual_inc" in out.columns:
        out["income_band"] = enc.income_band(out["annual_inc"])
    return out


DERIVED_NUMERIC: tuple[str, ...] = (
    "fico_midpoint",
    "emp_length_years",
    "installment_to_income",
    "loan_to_income",
    "log_annual_inc",
)


def default_spec(
    available: object, *, extended: bool = True, with_lc_grade: bool = False
) -> FeatureSpec:
    """Build the feature spec, intersected with the columns actually present.

    Runs ``assert_no_leakage`` before returning, so a spec can never carry a
    post-origination column into a model.
    """
    present = set(map(str, available))  # type: ignore[arg-type]

    numeric = [
        c
        for c in list(sch.CORE_NUMERIC)
        + (list(sch.EXTENDED_NUMERIC) if extended else [])
        + list(DERIVED_NUMERIC)
        if c in present and c not in SUPERSEDED
    ]
    categorical = [c for c in sch.CORE_CATEGORICAL if c in present and c not in SUPERSEDED]
    if with_lc_grade:
        numeric += [c for c in ("int_rate",) if c in present]
        categorical += [c for c in ("grade", "sub_grade") if c in present]

    # De-duplicate while preserving order.
    numeric = list(dict.fromkeys(numeric))
    categorical = list(dict.fromkeys(categorical))

    sch.assert_no_leakage(numeric + categorical)
    return FeatureSpec(
        numeric=tuple(numeric),
        categorical=tuple(categorical),
        structural_missing=tuple(c for c in STRUCTURAL_MISSING if c in present or c in numeric),
    )


def build_design_matrix(
    df: pd.DataFrame,
    spec: FeatureSpec | None = None,
    *,
    flavour: str = "gbm",
    duration_col: str = "duration_months",
    event_col: str = "event",
    max_categories: int = 60,
    standardisation: dict[str, tuple[float, float]] | None = None,
    fill_values: dict[str, float] | None = None,
    reference_columns: list[str] | None = None,
) -> DesignMatrix:
    """Build a model-ready matrix.

    Parameters
    ----------
    flavour:
        ``"gbm"`` keeps native dtypes and NaN; ``"cox"`` one-hot encodes,
        median-fills and standardises.
    standardisation, fill_values, reference_columns:
        Pass the values learned on the training split to transform a test split
        identically. Leaving them ``None`` learns them from ``df``, which is only
        correct for the training split itself.
    """
    if flavour not in {"gbm", "cox"}:
        raise ValueError(f"flavour must be 'gbm' or 'cox', got {flavour!r}")

    work = add_derived_features(df)
    spec = spec or default_spec(work.columns)
    sch.assert_no_leakage(spec.all_columns)

    dropped: dict[str, str] = {}
    frames: list[pd.DataFrame] = []

    # --- numeric ------------------------------------------------------------
    num = pd.DataFrame(index=work.index)
    for col in spec.numeric:
        if col not in work.columns:
            dropped[col] = "absent from input"
            continue
        num[col] = pd.to_numeric(work[col], errors="coerce").astype("float32")

    # Missing-as-signal indicators, before any filling.
    for col in spec.structural_missing:
        if col in num.columns:
            num[f"{col}_missing"] = num[col].isna().astype("int8")

    # --- categorical --------------------------------------------------------
    cats: list[str] = []
    for col in spec.categorical:
        if col not in work.columns:
            dropped[col] = "absent from input"
            continue
        n_levels = work[col].astype("category").cat.categories.size
        if n_levels > max_categories:
            dropped[col] = f"{n_levels} levels exceeds max_categories={max_categories}"
            continue
        cats.append(col)

    if flavour == "gbm":
        cat_df = pd.DataFrame(
            {c: work[c].astype("category") for c in cats}, index=work.index
        )
        X = pd.concat([num, cat_df], axis=1) if cats else num
        learned_std: dict[str, tuple[float, float]] = {}
        learned_fill: dict[str, float] = {}
    else:
        # One-hot with a dropped reference level to keep the matrix full rank.
        dummies = (
            pd.get_dummies(
                work[cats].astype("category"), drop_first=True, dummy_na=False, dtype="int8"
            )
            if cats
            else pd.DataFrame(index=work.index)
        )
        X = pd.concat([num, dummies], axis=1)

        learned_fill = dict(fill_values or {})
        for col in num.columns:
            if X[col].isna().any():
                if col not in learned_fill:
                    med = float(X[col].median())
                    learned_fill[col] = 0.0 if not np.isfinite(med) else med
                X[col] = X[col].fillna(learned_fill[col])

        learned_std = dict(standardisation or {})
        for col in num.columns:
            if col.endswith("_missing"):
                continue
            if col not in learned_std:
                mu, sd = float(X[col].mean()), float(X[col].std())
                learned_std[col] = (mu, sd if sd > 1e-12 else 1.0)
            mu, sd = learned_std[col]
            X[col] = ((X[col] - mu) / sd).astype("float32")

        # Zero-variance columns carry no information and make the Hessian
        # singular. Dropped here rather than left for lifelines to choke on.
        if reference_columns is None:
            nunique = X.nunique(dropna=False)
            for col in nunique[nunique <= 1].index:
                dropped[str(col)] = "zero variance"
            X = X.drop(columns=list(nunique[nunique <= 1].index))
        else:
            X = X.reindex(columns=reference_columns, fill_value=0)

    sch.assert_no_leakage(X.columns)
    return DesignMatrix(
        X=X,
        duration=work[duration_col].astype("int32"),
        event=work[event_col].astype("int8"),
        spec=spec,
        flavour=flavour,
        dropped=dropped,
        standardisation=learned_std,
        fill_values=learned_fill,
    )


def train_test_split_loans(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    seed: int = 20260921,
    scheme: str = "random",
    year_col: str = "issue_year",
    out_of_time_cutoff: int | None = None,
) -> tuple[pd.Index, pd.Index]:
    """Split loan ids into train/test.

    ``scheme="random"`` is the primary specification. ``scheme="out_of_time"``
    holds out the latest vintages, which is how a lender would really validate,
    but on this dataset it confounds model quality with censoring depth -- the
    2018 book is 86% still-current -- so it is reported as a robustness check
    rather than as the headline.
    """
    if scheme == "out_of_time":
        if year_col not in df.columns:
            raise ValueError(f"{year_col!r} required for out-of-time split")
        years = df[year_col]
        cutoff = out_of_time_cutoff
        if cutoff is None:
            cutoff = int(years.quantile(1.0 - test_size))
        test_mask = years >= cutoff
        return df.index[~test_mask], df.index[test_mask]

    if scheme != "random":
        raise ValueError(f"unknown scheme {scheme!r}")

    rng = np.random.default_rng(seed)
    n = len(df)
    perm = rng.permutation(n)
    n_test = int(round(n * test_size))
    test_pos, train_pos = perm[:n_test], perm[n_test:]
    return df.index[np.sort(train_pos)], df.index[np.sort(test_pos)]
