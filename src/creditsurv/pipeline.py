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

from .provenance import PROJECT_ROOT

__all__ = [
    "load_model_bundle",
    "resolve_data_source",
    "encode_data_source",
    "parse_year_range",
]


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
