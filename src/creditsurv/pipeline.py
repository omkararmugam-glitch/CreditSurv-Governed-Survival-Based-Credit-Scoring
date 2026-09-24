"""Helpers shared by the stage scripts.

The main job here is fix 4 from the holdout review: a stage that loads a fitted
model must read the *data file that model was trained and split on*, and must not
infer it from the tag's spelling. Stages 3 and 4 used to pick the full labelled
dataset only when the tag started with ``"full"``, and otherwise read the 200k
development sample. With ``--tag holdout`` that would have matched the dev
sample's row numbers against the holdout's loan index -- selecting essentially
arbitrary loans, with no error.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

from .provenance import PROJECT_ROOT

__all__ = [
    "load_model_bundle",
    "resolve_data_source",
    "encode_data_source",
    "parse_year_range",
    "load_feature_frame",
    "DERIVATION_INPUTS",
    "FEATURE_EXTRAS",
]

DERIVATION_INPUTS: frozenset[str] = frozenset({
    "fico_range_low", "fico_range_high", "emp_length", "installment", "annual_inc",
    "loan_amnt", "term", "int_rate"})
"""Raw columns the derived features are computed from.

Read whenever a model's features are read, even when the spec does not name them,
because the features that depend on them do. Kept beside
:func:`creditsurv.features.build.add_derived_features`, which is the only thing that
uses them."""

FEATURE_EXTRAS: tuple[str, ...] = ("duration_months", "event")
"""The survival outcome. Every stage that evaluates or explains a model needs it."""


def load_feature_frame(src, spec, *, extras=FEATURE_EXTRAS, columns=(),
                       verbose: bool = False) -> pd.DataFrame:
    """Read exactly the columns a model needs, computing the ones the file lacks.

    A model trained with ``--with-derived`` has features the parquet never stored:
    ``fico_midpoint``, ``emp_length_years``, ``term_months`` and the income ratios are
    computed at training time, not written back. Asking pyarrow for them by name fails
    with ``No match for fico_midpoint``, which is how two separate stages broke
    (FINDINGS 7j). So the raw inputs are read and the derived columns are computed
    here, exactly as training computed them, in the one place every stage goes
    through.

    Parameters
    ----------
    spec:
        The bundle's :class:`~creditsurv.features.build.FeatureSpec`. Every one of its
        columns is guaranteed present in the result, or the call raises.
    extras, columns:
        Further columns to read: ``extras`` defaults to the survival outcome,
        ``columns`` is for anything a particular stage also wants (``grade``,
        ``issue_year``). Both are read when stored and reported when not.
    verbose:
        Print which features were computed rather than read. On for the stage
        scripts, off inside the application.

    Raises
    ------
    ValueError
        If a spec feature is neither stored in the file nor derivable from it, naming
        every such feature. Silently dropping one would change the model's inputs.
    """
    # Imported here: features.build imports nothing from this module, and keeping the
    # import local avoids making that a rule anyone has to remember.
    from .derive import derive_features
    from .features.build import add_derived_features
    import pyarrow.parquet as pq

    src = Path(src)
    wanted = list(dict.fromkeys(list(spec.all_columns) + list(extras)
                                + list(columns)))
    stored = set(pq.read_schema(src).names)
    to_read = sorted((set(wanted) | DERIVATION_INPUTS) & stored)
    frame = add_derived_features(pd.read_parquet(src, columns=to_read))
    # Anything still absent goes through the derivation rules the upload path uses,
    # so a column is derivable here exactly when it is derivable there. term_months
    # from "36 months" is the case this exists for: training builds it in Stage 1, so
    # the labelled parquet stores it, but a file that stores only `term` should not be
    # treated as missing a feature it plainly contains.
    short = [c for c in spec.all_columns if c not in frame.columns]
    if short:
        frame, _derived, _blocked = derive_features(frame, short)
    for col in [c for c in frame.columns if frame[c].dtype == object]:
        frame[col] = frame[col].astype("category")

    computed = [c for c in wanted if c not in stored and c in frame.columns]
    if verbose and computed:
        print(f"computed from raw columns rather than read: {', '.join(computed)}")

    absent = [c for c in spec.all_columns if c not in frame.columns]
    if absent:
        raise ValueError(
            f"{absent} are features of this model but are neither stored in "
            f"{src.name} nor derivable from it, so it cannot be evaluated or scored "
            f"against this file.")
    missing_extras = [c for c in list(extras) + list(columns)
                      if c not in frame.columns]
    if verbose and missing_extras:
        print(f"note: {', '.join(missing_extras)} not in {src.name}")
    return frame


def load_model_bundle(models_dir: Path, model_tag: str) -> tuple[dict, Path]:
    """Load ``02_models_<tag>.pkl``. Raises FileNotFoundError if absent."""
    path = Path(models_dir) / f"02_models_{model_tag}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/02_train_models.py first.")
    with open(path, "rb") as fh:
        return pickle.load(fh), path


def encode_data_source(path: Path) -> str:
    """Store a data path relative to the project where possible, so the saved
    model stays valid if the project folder is moved."""
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def resolve_data_source(bundle: dict, cfg, model_tag: str) -> Path:
    """The data file a model bundle was built from.

    Bundles written by the current Stage 2 record it explicitly. Bundles written
    earlier do not, so for those the old rule is applied -- but loudly, and only
    for the tags it was written for.
    """
    recorded = bundle.get("data_source")
    if recorded:
        p = Path(recorded)
        return p if p.is_absolute() else PROJECT_ROOT / p

    if model_tag.startswith("full"):
        fallback = cfg.paths.labeled_parquet
    elif model_tag.startswith("dev"):
        fallback = cfg.paths.dev_sample_parquet
    else:
        raise ValueError(
            f"model bundle for tag {model_tag!r} does not record its data source, "
            f"and the tag is not one the legacy rule covers. Re-run "
            f"02_train_models.py to produce a bundle that records it.")
    print(f"note: model bundle predates data-source recording; assuming "
          f"{fallback.name} from the tag {model_tag!r}.", file=sys.stderr)
    return fallback


def parse_year_range(text: str | None) -> list[int] | None:
    """``"2016-2018"`` -> ``[2016, 2017, 2018]``; ``"2016"`` -> ``[2016]``."""
    if not text:
        return None
    parts = str(text).split("-")
    if len(parts) == 1:
        return [int(parts[0])]
    if len(parts) != 2:
        raise ValueError(f"year range must look like 2016 or 2016-2018, got {text!r}")
    lo, hi = int(parts[0]), int(parts[1])
    if hi < lo:
        raise ValueError(f"year range is reversed: {text!r}")
    return list(range(lo, hi + 1))
