"""Score an uploaded applicant file end to end, using the trained pipeline.

This module is orchestration, not modelling. Every step calls code the pipeline
already uses and tests:

===================  ====================================================
step                 existing code it calls
===================  ====================================================
checking file        :mod:`creditsurv.io.schema` (the model's own feature
                     spec, carried in the saved bundle)
cleaning data        :func:`creditsurv.features.build.build_design_matrix`
                     (which applies ``add_derived_features``)
scoring applicants   ``model.predict_survival`` -- the fitted
                     :class:`DiscreteTimeHazardModel` or :class:`CoxModel`
explaining           :func:`creditsurv.explain.survshap.explain_survshap`,
                     or :func:`...tree_shap.explain_tree_shap` for bulk runs
                     (``decision.bulk_explainer``; see FINDINGS section 7)
notices              :func:`...adverse_action.build_adverse_action_notice`
traceability         :func:`creditsurv.provenance.build_stamp`
===================  ====================================================

Nothing here retrains anything: a run loads ``02_models_<tag>.pkl`` and fails if
it is absent.

Three things are deliberate and visible rather than silent:

**The decision threshold is policy, not output.** It comes from
:class:`creditsurv.config.DecisionConfig`, is shown on the dashboard and is
written into ``run_summary.csv``.

**Partial files are accepted but flagged.** A file missing optional bureau
columns is scored with those features left missing, and the coverage is reported
on screen and in every output file. Missing one of :data:`CORE_REQUIRED` stops
the run instead.

**Cleaning values belong to the model.** They are fitted once on its whole training
split and saved with it; scoring reads them and never fits. A model without them
refuses to score and says which command adds them, rather than quietly learning
bounds from a sample or from the file being scored.
"""

from __future__ import annotations

import io
import json
import re
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import drift as drift_mod
from . import eda
from .derive import FeatureCosts, derive_features, load_costs
from .schema_match import propose_mapping
from .cleaning import (CleaningPolicy, CleaningReport, CleaningValues, clean,
                       fit_values, policy_from_config)   # fit_values: never called
# here -- imported so tests/test_cleaning_values_source.py can assert that scoring
# does not fit cleaning values.
from .config import Config
from .explain.adverse_action import MAX_PRINCIPAL_REASONS, build_adverse_action_notice
from .explain.parallel import (Ledger, explain_rows_parallel, keep_awake,
                               suggest_workers)
from .explain.survshap import explain_survshap
from .explain.tree_shap import explain_tree_shap
from .features.build import build_design_matrix
from .pipeline import load_model_bundle, resolve_data_source
from .provenance import PROJECT_ROOT, build_stamp, file_fingerprint

__all__ = ["CORE_REQUIRED", "PROVISIONAL_REQUIRED", "ALIASES", "OUTPUT_NAMES", "BatchError",
           "ValidationReport", "ScoringContext", "BatchResult", "read_upload",
           "validate", "load_context", "prepare", "profile", "run_batch",
           "load_result", "bundle_zip", "iter_chunks", "Aggregates", "CAP_NOTE",
           "choose_explainer", "RUNS_DIR"]

RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"

PARALLEL_MIN_ROWS = 32
"""Below this many applicants to explain, one process is faster: starting workers
and shipping the model to each costs more than the work saved."""

PREVIEW_ROWS = 2_000
"""Rows kept in memory for the dashboard preview table. The whole result
is in scored_applicants.csv."""


class _Reservoir:
    """A uniform random sample of up to ``size`` rows, drawn across the whole file.

    Priority sampling: every row gets an independent uniform key and the rows with
    the ``size`` smallest keys are kept, which is a uniform sample without
    replacement no matter how the file is divided into blocks. Only the current
    sample and one block's keys are ever held, so memory stays bounded, and the
    work is vectorised rather than row by row.

    Why not simply the first block: a file sorted by date -- which is how loan
    files usually arrive -- has a first block that is one vintage, not the file.
    Profiling and drift would then describe a population the file does not have,
    which is the failure the drift check exists to catch.
    """

    def __init__(self, size: int, seed: int = 0):
        self.size = int(size)
        self.rng = np.random.default_rng(seed)
        self.seen = 0
        self._rows: pd.DataFrame | None = None
        self._keys: np.ndarray = np.empty(0, dtype=float)

    def add_block(self, block: pd.DataFrame) -> None:
        n = len(block)
        if n == 0:
            return
        self.seen += n
        keys = self.rng.random(n)
        if n > self.size:                       # only the best of this block matter
            best = np.argpartition(keys, self.size)[: self.size]
            block, keys = block.iloc[best], keys[best]
        rows = block.reset_index(drop=True)
        if self._rows is None:
            self._rows, self._keys = rows, keys
        else:
            self._rows = pd.concat([self._rows, rows], ignore_index=True)
            self._keys = np.concatenate([self._keys, keys])
        if len(self._rows) > self.size:
            keep = np.argpartition(self._keys, self.size)[: self.size]
            keep.sort()
            self._rows = self._rows.iloc[keep].reset_index(drop=True)
            self._keys = self._keys[keep]

    def frame(self) -> pd.DataFrame:
        return (self._rows.reset_index(drop=True) if self._rows is not None
                else pd.DataFrame())


PROFILE_ROWS = 50_000
"""Rows the profile and the drift check are computed on -- a uniform random
sample drawn across the whole file (:class:`_Reservoir`), never its first rows.
Both describe the upload rather than decide anything, and a sample this size
settles their numbers well inside their own noise; the count used is reported
as ``profiled_rows``."""

OUTPUT_NAMES: tuple[str, ...] = (
    "scored_applicants.csv", "approved_applicants.csv", "rejected_applicants.csv",
    "adverse_action_notices.zip", "run_summary.csv", "cleaning_report.csv",
    "data_drift.csv", "data_profile_missing.csv", "data_profile_numeric.csv",
    "data_profile_categories.csv", "data_profile_outliers.csv",
    "validation_report.json", "provenance.json", "aggregates.json",
)
"""Every file a finished run writes. Used to recognise a completed run directory
and to refuse writing a second run into one."""

PROVISIONAL_REQUIRED: tuple[str, ...] = (
    "loan_amnt", "installment", "annual_inc", "dti", "open_acc", "revol_bal",
    "delinq_2yrs", "inq_last_6mths", "purpose", "home_ownership",
)
"""A stand-in, used **only** for a model with no ablation table.

These ten were picked by hand, on the reasonable-sounding but unmeasured view that
a file without them is not an applicant file. Measurement disagrees: on the full
model seven of them cost under 0.007 concordance to lose, and only three reach the
0.010 bar (FINDINGS 7d). So wherever a model has a table the measured split is the
only rule and this list plays no part; a model without one is scored under this
list and told so, with the command that replaces it."""

CORE_REQUIRED = PROVISIONAL_REQUIRED
"""Kept as the former name of :data:`PROVISIONAL_REQUIRED`."""

ALIASES: dict[str, str] = {
    # Fixed, reviewable renames -- never inferred per file. Anything not listed
    # here and not a model feature is ignored and reported, not guessed at.
    "loan_purpose": "purpose",
    "purpose_of_loan": "purpose",
    "state": "addr_state",
    "borrower_state": "addr_state",
    "open_credit_lines": "open_acc",
    "open_credit_lines_count": "open_acc",
    "inquiries_last_6m": "inq_last_6mths",
    "inquiries_last_6mths": "inq_last_6mths",
    "income": "annual_inc",
    "annual_income": "annual_inc",
    "loan_amount": "loan_amnt",
    "monthly_payment": "installment",
    "debt_to_income": "dti",
    "revolving_balance": "revol_bal",
    "revolving_utilization": "revol_util",
    "delinquencies_2y": "delinq_2yrs",
    "public_records": "pub_rec",
    "home_ownership_status": "home_ownership",
}

UNUSED_NOTE: dict[str, str] = {
    "credit_score": "no trained model uses a bureau score: the Stage 2 feature "
                    "spec drops fico_range_* before the derived fico_midpoint "
                    "exists, so it was never fitted",
    "fico_range_low": "superseded by fico_midpoint, which the trained spec does "
                      "not contain",
    "fico_range_high": "superseded by fico_midpoint, which the trained spec does "
                       "not contain",
    "emp_length": "employment length is not in the trained feature spec",
    "emp_length_years": "employment length is not in the trained feature spec",
    "term": "loan term is not a model feature",
}

ID_CANDIDATES = ("applicant_id", "application_id", "id", "loan_id", "member_id")


class BatchError(RuntimeError):
    """A failure with a plain-language message for the page and a technical detail
    for the collapsible section."""

    def __init__(self, message: str, detail: str = "", fix: str = ""):
        super().__init__(message)
        self.message = message
        self.detail = detail or message
        self.fix = fix


@dataclass
class ValidationReport:
    n_rows: int = 0
    mapped: dict[str, str] = field(default_factory=dict)
    ignored: dict[str, str] = field(default_factory=dict)
    missing_core: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    present: list[str] = field(default_factory=list)
    id_column: str | None = None
    warnings: list[str] = field(default_factory=list)
    recognised: dict[str, str] = field(default_factory=dict)
    """Uploaded column -> model feature, for columns that were renamed."""
    derived: dict[str, str] = field(default_factory=dict)
    """Feature -> how it was computed from other columns."""
    blocked: dict[str, list[str]] = field(default_factory=dict)
    """Feature -> the inputs a derivation would have needed."""
    required_missing: list[str] = field(default_factory=list)
    optional_missing: list[str] = field(default_factory=list)
    required_rule: str = "provisional"
    """"measured" when the model's ablation table decided which features are
    required, "provisional" when the hand-picked list stood in for it."""
    required_rule_note: str = ""
    unused: dict[str, str] = field(default_factory=dict)
    costs: object = None

    @property
    def n_features(self) -> int:
        return len(self.present) + len(self.missing_optional) + len(self.missing_core)

    @property
    def coverage(self) -> float:
        return len(self.present) / self.n_features if self.n_features else 0.0

    @property
    def ok(self) -> bool:
        return not self.missing_core

    def message(self) -> str:
        """What was recognised, what was derived, what is missing and why.

        Written for whoever has to fix the file, so every missing column says either
        "not in file" or which inputs a derivation would have needed.
        """
        lines = []
        if self.recognised:
            lines.append("Recognised: " + "; ".join(
                f"{k} -> {v}" for k, v in self.recognised.items()))
        if self.derived:
            lines.append("Derived: " + "; ".join(
                f"{k} ({v})" for k, v in self.derived.items()))
        missing_bits = []
        for feature in self.required_missing + self.optional_missing:
            needed = self.blocked.get(feature)
            if needed:
                missing_bits.append(f"{feature} (needs {' and '.join(needed)})")
            else:
                missing_bits.append(f"{feature} (not in file)")
        if missing_bits:
            lines.append("Missing: " + "; ".join(missing_bits[:12])
                         + (f" and {len(missing_bits) - 12} more"
                            if len(missing_bits) > 12 else ""))
        if self.unused:
            lines.append("Recognised but unused by this model: " + "; ".join(
                f"{k} ({v})" for k, v in self.unused.items()))
        if self.required_rule == "provisional" and self.required_rule_note:
            lines.append(self.required_rule_note)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"n_rows": self.n_rows, "n_features_expected": self.n_features,
                "n_features_present": len(self.present), "present": self.present,
                "recognised": self.recognised, "derived": self.derived,
                "blocked": {k: list(v) for k, v in self.blocked.items()},
                "required_missing": self.required_missing,
                "optional_missing": self.optional_missing,
                "required_rule": self.required_rule,
                "required_rule_note": self.required_rule_note,
                "recognised_but_unused": self.unused,
                "coverage": round(self.coverage, 4), "id_column": self.id_column,
                "mapped": self.mapped, "ignored": self.ignored,
                "missing_core": self.missing_core,
                "missing_optional": self.missing_optional,
                "warnings": self.warnings}


def _source(data, filename: str):
    """A path or in-memory buffer to read from, with the cheap checks done first.

    A path is streamed straight from disk, which is what keeps memory flat for a
    large file: the bytes never all exist at once.
    """
    name = Path(filename).name
    if Path(name).suffix.lower() not in (".csv", ".txt"):
        raise BatchError(
            f"{name} is not a CSV file.",
            f"suffix {Path(name).suffix!r} is not supported",
            "Save your data as .csv and upload it again.")
    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.exists():
            raise BatchError(f"{name} could not be read.", f"missing: {path}",
                             "Upload the file again.")
        if path.stat().st_size == 0 or not path.read_bytes()[:4096].strip():
            raise BatchError(f"{name} is empty.", "no content",
                             "Upload a file with a header row and at least one "
                             "applicant.")
        return path
    raw = data if isinstance(data, bytes) else data.read()
    if not raw.strip():
        raise BatchError(f"{name} is empty.", "0 bytes of content",
                         "Upload a file with a header row and at least one applicant.")
    return io.BytesIO(raw)


def iter_chunks(data, filename: str, chunk_rows: int = 50_000):
    """Yield the upload in row blocks, so peak memory does not grow with file size.

    Every failure mode :func:`read_upload` reports is reported here too, on the
    first block, so a bad file still fails before anything is scored.
    """
    src = _source(data, filename)
    name = Path(filename).name
    try:
        reader = pd.read_csv(src, low_memory=False, chunksize=max(int(chunk_rows), 1))
        empty = True
        for chunk in reader:
            if chunk.empty and empty:
                continue
            empty = False
            yield chunk
        if empty:
            raise BatchError(f"{name} has a header but no applicants.",
                             "0 data rows", "Add at least one row of applicant data.")
    except UnicodeDecodeError as exc:
        raise BatchError(f"{name} is not readable as text.", repr(exc),
                         "Export it again as UTF-8 CSV.") from exc
    except pd.errors.ParserError as exc:
        raise BatchError(f"{name} is not a valid CSV: its rows do not all have the "
                         "same number of columns.", repr(exc),
                         "Open it in a spreadsheet, check for stray commas or "
                         "quotes, and export again.") from exc
    except pd.errors.EmptyDataError as exc:
        raise BatchError(f"{name} has no columns to read.", repr(exc),
                         "The first row must be a header of column names.") from exc


def read_upload(data, filename: str) -> pd.DataFrame:
    """Read an uploaded CSV whole. Used where the file is known to be small
    (tests, and :func:`profile` on an already-loaded frame); the scoring path uses
    :func:`iter_chunks`."""
    src = _source(data, filename)
    name = Path(filename).name
    try:
        df = pd.read_csv(src, low_memory=False)
    except UnicodeDecodeError as exc:
        raise BatchError(f"{name} is not readable as text.", repr(exc),
                         "Export it again as UTF-8 CSV.") from exc
    except pd.errors.ParserError as exc:
        raise BatchError(f"{name} is not a valid CSV: its rows do not all have the "
                         "same number of columns.", repr(exc),
                         "Open it in a spreadsheet, check for stray commas or "
                         "quotes, and export again.") from exc
    except pd.errors.EmptyDataError as exc:
        raise BatchError(f"{name} has no columns to read.", repr(exc),
                         "The first row must be a header of column names.") from exc
    if df.empty:
        raise BatchError(f"{name} has a header but no applicants.",
                         f"{len(df.columns)} columns, 0 rows",
                         "Add at least one row of applicant data.")
    return df


def validate(df: pd.DataFrame, spec, *, values=None, costs=None,
             mapping: dict | None = None) -> tuple[pd.DataFrame, ValidationReport]:
    """Rename known aliases, check the model's features, report the rest.

    Returns the frame with model-feature names, and the report. A missing *core*
    feature is reported here and raised by :func:`run_batch`, so the caller can
    show every problem at once rather than one per attempt.
    """
    rep = ValidationReport(n_rows=len(df))
    rep.proposal = None
    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]

    expected = set(spec.all_columns)
    rep.id_column = next((c for c in ID_CANDIDATES if c in work.columns), None)

    # 1. Recognition. A proposal, not an application: `mapping` is what a person
    #    confirmed, and when nothing was confirmed the pre-selected high-confidence
    #    matches are used and reported rather than applied quietly.
    # Proposing a mapping costs real work (name similarity plus a content check per
    # candidate), so it happens once per file: later blocks are handed the mapping
    # the first block produced.
    proposal = None if mapping is not None else propose_mapping(work, spec,
                                                                values=values)
    renames = dict(mapping) if mapping is not None else {
        k: v for k, v in proposal.mapping().items() if k != v}
    renames = {k: v for k, v in renames.items()
               if k in work.columns and v not in work.columns}
    work = work.rename(columns=renames)
    rep.mapped = dict(renames)
    rep.recognised = dict(renames)
    rep.unused = dict(proposal.recognised_but_unused) if proposal else {}
    rep.proposal = proposal

    # 2. Derivation. Exact arithmetic from columns the file does have, never a guess.
    wanted = [c for c in spec.all_columns if c not in work.columns]
    work, derived, blocked = derive_features(work, wanted)
    rep.derived, rep.blocked = derived, blocked

    # 3. What is left, and which tier it falls in.
    rep.present = [c for c in spec.all_columns if c in work.columns]
    missing = [c for c in spec.all_columns if c not in work.columns]
    costs = costs if costs is not None else FeatureCosts()
    rep.costs = costs
    # One rule or the other, never a mixture: a measured table replaces the
    # provisional list outright rather than adding to it.
    if costs.measured:
        required, rep.required_rule = set(costs.required()), "measured"
    else:
        required, rep.required_rule = set(PROVISIONAL_REQUIRED), "provisional"
    rep.required_rule_note = costs.rule_note()
    rep.required_missing = [c for c in missing if c in required]
    rep.optional_missing = [c for c in missing if c not in required]
    rep.missing_core = list(rep.required_missing)
    rep.missing_optional = list(rep.optional_missing)
    for col in work.columns:
        if col in expected or col == rep.id_column:
            continue
        rep.ignored[col] = (rep.unused.get(col)
                            or UNUSED_NOTE.get(col, "not a feature of the trained model"))

    if renames:
        rep.warnings.append(
            f"{len(renames)} column(s) were recognised under another name: "
            + ", ".join(f"{k} -> {v}" for k, v in list(renames.items())[:6])
            + ("..." if len(renames) > 6 else ""))
    if derived:
        rep.warnings.append(
            f"{len(derived)} feature(s) were computed exactly from other columns: "
            + ", ".join(derived) + ". They are marked as derived in the outputs.")
    if rep.optional_missing:
        detail = costs.describe(rep.optional_missing)
        rep.warnings.append(
            f"{len(rep.optional_missing)} of the model's {rep.n_features} features "
            f"are not in this file and are treated as missing"
            + (f". Measured cost: {detail}" if detail else "."))
    if rep.ignored:
        rep.warnings.append(
            f"{len(rep.ignored)} column(s) in the file are not model features and "
            f"were ignored: {', '.join(list(rep.ignored)[:8])}"
            + ("..." if len(rep.ignored) > 8 else ""))
    return work, rep


@dataclass
class ScoringContext:
    """A loaded model plus everything needed to encode and explain new rows."""

    cfg: Config
    model_tag: str
    model_name: str
    model: object
    spec: object
    bundle: dict
    model_path: Path
    background: pd.DataFrame          # design-matrix rows, for SurvSHAP(t)
    reference: pd.DataFrame           # raw training rows, for the drift check
    clean_values: CleaningValues      # fitted on training rows, never on an upload
    policy: CleaningPolicy
    times: np.ndarray
    data_source: Path
    values_from_bundle: bool = True

    @property
    def flavour(self) -> str:
        return "cox" if self.model_name == "cox" else "gbm"


def available_models(models_dir: Path) -> list[str]:
    return sorted(p.stem.replace("02_models_", "")
                  for p in Path(models_dir).glob("02_models_*.pkl"))


def load_context(cfg: Config, model_tag: str | None = None,
                 model_name: str | None = None) -> ScoringContext:
    """Load a trained bundle and a background sample from its *training* split.

    The background is what SurvSHAP(t) treats as "feature absent", and the
    training ranges are what out-of-range flagging compares against, so both come
    from the split the model was fitted on -- never from the uploaded file.
    """
    d = cfg.decision
    model_tag = model_tag or d.model_tag
    model_name = model_name or d.model
    try:
        bundle, model_path = load_model_bundle(cfg.paths.models_dir, model_tag)
    except FileNotFoundError as exc:
        raise BatchError(
            f"No trained model found for '{model_tag}'.", str(exc),
            "Train one first: Advanced > Run Pipeline.") from exc

    key = "discrete_hazard" if model_name in ("discrete_hazard", "gbm") else model_name
    if key not in bundle["artefacts"]:
        raise BatchError(
            f"The '{model_tag}' bundle has no {model_name} model.",
            f"available: {sorted(bundle['artefacts'])}",
            "Pick another model, or retrain with Advanced > Run Pipeline.")

    spec = bundle["spec"]
    src = resolve_data_source(bundle, cfg, model_tag)
    if not Path(src).exists():
        raise BatchError(
            "The data this model was trained on is no longer on disk, so new "
            "applicants cannot be compared against it.", f"missing: {src}",
            "Restore it, or train a model on data that is present.")
    cols = list(spec.all_columns) + ["duration_months", "event"]
    train = pd.read_parquet(src, columns=cols)
    for c in [c for c in train.columns if train[c].dtype == object]:
        train[c] = train[c].astype("category")
    idx = bundle["train_idx"].intersection(train.index)
    train = train.loc[idx] if len(idx) else train
    if len(train) > d.background_rows:
        train = train.sample(d.background_rows, random_state=cfg.explain.seed)

    art = bundle["artefacts"]
    if key == "cox":
        dm = build_design_matrix(train, spec, flavour="cox",
                                 standardisation=art.get("cox_standardisation"),
                                 fill_values=art.get("cox_fill_values"),
                                 reference_columns=art.get("cox_columns"))
    else:
        dm = build_design_matrix(train, spec, flavour="gbm")

    # Cleaning values are the model's, fitted once on its whole training split and
    # only ever read here. Scoring never fits them: a value learned from the file
    # being scored would make each upload its own yardstick, and a value learned
    # from a 20,000-row sample is not the one the model was trained against.
    policy = policy_from_config(cfg)
    saved = bundle.get("cleaning_values")
    from_bundle = bool(saved)
    if not saved:
        sidecar = cfg.paths.models_dir / f"02_cleaning_values_{model_tag}.json"
        if sidecar.exists():
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                saved = payload.get("values") or payload
            except (OSError, ValueError) as exc:
                raise BatchError(
                    f"The cleaning values for '{model_tag}' could not be read.",
                    f"{sidecar}: {exc}",
                    f"Rebuild them: python scripts/02s_save_cleaning_values.py "
                    f"--model-tag {model_tag} --overwrite") from exc
    if not saved:
        raise BatchError(
            f"The '{model_tag}' model has no saved cleaning values, so it cannot "
            f"score a file.",
            f"neither the bundle nor {cfg.paths.models_dir.as_posix()}/"
            f"02_cleaning_values_{model_tag}.json holds them",
            f"Add them once, without retraining: python "
            f"scripts/02s_save_cleaning_values.py --model-tag {model_tag}")
    values = CleaningValues.from_dict(saved)

    return ScoringContext(
        cfg=cfg, model_tag=model_tag, model_name=key, model=art[key], spec=spec,
        bundle=bundle, model_path=model_path, background=dm.X, reference=train,
        clean_values=values, policy=policy,
        times=np.array(cfg.model.eval_horizons_months, dtype=float),
        data_source=Path(src), values_from_bundle=from_bundle)


def prepare(df: pd.DataFrame, ctx: ScoringContext
            ) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Clean and encode uploaded rows exactly as training does.

    Value cleaning is :func:`creditsurv.cleaning.clean` with the values fitted on
    the model's training split; encoding is the pipeline's own
    ``build_design_matrix``. Returns the design matrix, a per-row flag frame and
    the cleaning report.
    """
    work, report, flags = clean(df, ctx.spec, ctx.clean_values, policy=ctx.policy)

    # build_design_matrix expects the survival label columns; new applicants have
    # no outcome, so placeholders are passed and never used.
    for col, value in (("duration_months", 0), ("event", 0)):
        if col not in work.columns:
            work[col] = value

    art = ctx.bundle["artefacts"]
    if ctx.flavour == "cox":
        dm = build_design_matrix(work, ctx.spec, flavour="cox",
                                 standardisation=art.get("cox_standardisation"),
                                 fill_values=art.get("cox_fill_values"),
                                 reference_columns=art.get("cox_columns"))
    else:
        dm = build_design_matrix(work, ctx.spec, flavour="gbm")
        # Columns absent from the upload are absent from the matrix; LightGBM
        # needs the trained column set, in order, so they are restored as missing.
        trained = art.get("gbm_columns")
        if trained is not None:
            for col in trained:
                if col not in dm.X.columns:
                    # A missing-as-signal indicator for an absent column is not
                    # unknown: the value genuinely is not there, so it is 1.
                    dm.X[col] = 1 if col.endswith("_missing") else np.nan
            dm.X = dm.X[list(trained)]
            for col, levels in ctx.clean_values.categories.items():
                if col in dm.X.columns:
                    dm.X[col] = pd.Categorical(dm.X[col].astype("string").str.strip(),
                                               categories=list(levels))
    return dm.X, flags, report


def profile(df: pd.DataFrame, ctx: ScoringContext) -> dict:
    """The upload's own EDA plus its drift against the training data.

    Read-only: :mod:`creditsurv.eda` and :mod:`creditsurv.drift` never modify a
    frame, and the drift reference is the model's training split.
    """
    cols = [c for c in ctx.spec.all_columns if c in df.columns]
    numeric = [c for c in ctx.spec.numeric if c in df.columns]
    categorical = [c for c in ctx.spec.categorical if c in df.columns]
    return {
        "overview": eda.overview(df),
        "missing": eda.missing_table(df, cols),
        "numeric": eda.numeric_summary(df, numeric),
        "categorical": eda.categorical_frequencies(df, categorical),
        "outliers": eda.outlier_table(df, numeric),
        "drift": drift_mod.compare(ctx.reference, df, ctx.spec,
                                   min_rows=getattr(ctx.cfg.decision, "drift_min_rows",
                                                    drift_mod.MIN_ROWS)),
    }


@dataclass
class Aggregates:
    """Whole-file statistics accumulated block by block.

    The dashboard's charts are drawn from these rather than from a full frame in
    memory, so a 500 MB upload costs no more to display than a 25 KB one.
    """

    n_rows: int = 0
    n_approved: int = 0
    n_rejected: int = 0
    n_rejected_explained: int = 0
    pd_sum: float = 0.0
    pd_bins: int = 40
    pd_hist: list = field(default_factory=lambda: [0] * 40)
    by_group: dict = field(default_factory=dict)       # "level|decision" -> count
    reason_counts: dict = field(default_factory=dict)
    rows_out_of_range: int = 0
    group_column: str = ""

    @property
    def mean_pd(self) -> float:
        return self.pd_sum / self.n_rows if self.n_rows else 0.0

    @property
    def n_rejected_without_reasons(self) -> int:
        return self.n_rejected - self.n_rejected_explained

    def add(self, pd_h, decision, group, flags, reasons) -> None:
        self.n_rows += len(pd_h)
        self.n_approved += int((decision == "approve").sum())
        self.n_rejected += int((decision == "reject").sum())
        self.pd_sum += float(np.nansum(pd_h))
        idx = np.clip((np.nan_to_num(pd_h) * self.pd_bins).astype(int),
                      0, self.pd_bins - 1)
        for b, c in zip(*np.unique(idx, return_counts=True)):
            self.pd_hist[int(b)] += int(c)
        if group is not None:
            levels = group.astype("string").fillna("(missing)").to_numpy()
            for level, dec in zip(levels, decision):
                key = f"{level}|{dec}"
                self.by_group[key] = self.by_group.get(key, 0) + 1
        self.rows_out_of_range += int((flags["n_out_of_range"] > 0).sum())
        for row in reasons.values():
            self.n_rejected_explained += 1
            for r in row:
                self.reason_counts[r.reason] = self.reason_counts.get(r.reason, 0) + 1

    def risk_frame(self, threshold: float) -> pd.DataFrame:
        """The predicted-risk histogram, as rows a chart can draw."""
        width = 1.0 / self.pd_bins
        return pd.DataFrame({
            "bin_start": [i * width for i in range(self.pd_bins)],
            "bin_end": [(i + 1) * width for i in range(self.pd_bins)],
            "applicants": list(self.pd_hist),
            "decision": ["reject" if (i + 0.5) * width >= threshold else "approve"
                         for i in range(self.pd_bins)]})

    def group_frame(self) -> pd.DataFrame:
        rows = [{"group": k.rsplit("|", 1)[0], "decision": k.rsplit("|", 1)[1],
                 "applicants": v} for k, v in self.by_group.items()]
        return pd.DataFrame(rows)

    def reason_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"reason": k, "times cited": v} for k, v in
                             sorted(self.reason_counts.items(), key=lambda kv: -kv[1])])

    def to_dict(self) -> dict:
        return {"n_rows": self.n_rows, "n_approved": self.n_approved,
                "n_rejected": self.n_rejected,
                "n_rejected_explained": self.n_rejected_explained,
                "pd_sum": self.pd_sum, "pd_bins": self.pd_bins,
                "pd_hist": list(self.pd_hist), "by_group": self.by_group,
                "reason_counts": self.reason_counts,
                "rows_out_of_range": self.rows_out_of_range,
                "group_column": self.group_column}

    @classmethod
    def from_dict(cls, payload: dict) -> "Aggregates":
        return cls(**{k: v for k, v in payload.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class BatchResult:
    run_dir: Path
    scored: pd.DataFrame          # a preview; the whole file is in the CSV
    approved: pd.DataFrame
    rejected: pd.DataFrame
    summary: dict
    report: ValidationReport
    files: dict
    seconds: float
    clean_report: CleaningReport | None = None
    profile: dict = field(default_factory=dict)
    drift: object = None
    aggregates: Aggregates | None = None
    notices: list = field(default_factory=list)


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", Path(name).stem)[:40] or "upload"


CAP_NOTE = "reasons not generated: run stopped before this row"
"""What a rejected applicant without reasons is marked with. Every rejected
applicant is explained, so this appears only when a run was stopped before
finishing -- and because explanations are seeded per applicant and written to a
ledger as they complete, re-running continues from that row instead of starting
again. Never blank: a declined applicant with no stated reasons is a compliance
gap, so the row says so and the count is shown on the dashboard."""


def _reason_columns(reasons: dict, positions, upto: int) -> dict:
    out = {}
    for i in range(1, upto + 1):
        out[f"reason_{i}"] = [reasons[p][i - 1].reason
                              if p in reasons and len(reasons[p]) >= i else ""
                              for p in positions]
        out[f"reason_{i}_feature"] = [reasons[p][i - 1].feature
                                      if p in reasons and len(reasons[p]) >= i else ""
                                      for p in positions]
    return out


def _without_features(expl, absent: set):
    """Drop features the file never had from an explanation.

    A notice must not state a reason based on a column that was not in the file: the
    model treats it as missing, and "your revolving balance" is not a reason anyone
    can act on when no revolving balance was supplied.
    """
    import dataclasses

    keep = [i for i, name in enumerate(expl.feature_names) if name not in absent]
    if len(keep) == len(expl.feature_names):
        return expl
    names = tuple(expl.feature_names[i] for i in keep)
    phi = expl.phi[:, keep, ...]
    values = expl.feature_values[[n for n in names if n in expl.feature_values.columns]]
    return dataclasses.replace(expl, phi=phi, feature_names=names,
                               feature_values=values)


def choose_explainer(ctx: ScoringContext, requested: str | None = None) -> str:
    """Which explainer will write this run's reasons, and why it can.

    TreeSHAP reads the fitted trees, so it needs the discrete-hazard booster; for
    the Cox model there is nothing to read and SurvSHAP(t) is the only option.
    """
    requested = (requested or getattr(ctx.cfg.decision, "bulk_explainer", "survshap")
                 or "survshap").lower()
    tree_possible = (ctx.model_name == "discrete_hazard"
                     and getattr(ctx.model, "booster", None) is not None)
    if requested == "treeshap":
        if not tree_possible:
            raise BatchError(
                "TreeSHAP needs the discrete-hazard model, and this run uses "
                f"{ctx.model_name}.",
                "decision.bulk_explainer=treeshap with a non-tree model",
                "Set decision.bulk_explainer to survshap or auto, or score with "
                "the discrete-hazard model.")
        return "treeshap"
    if requested == "auto":
        return "treeshap" if tree_possible else "survshap"
    return "survshap"


def run_batch(data, filename: str, cfg: Config, *, model_tag: str | None = None,
              model_name: str | None = None, threshold: float | None = None,
              max_explained: int | None = None, runs_dir: Path | None = None,
              run_dir: Path | None = None, chunk_rows: int | None = None,
              explainer: str | None = None, mapping: dict | None = None,
              progress=None, ctx: ScoringContext | None = None) -> BatchResult:
    """Score an upload end to end, in row blocks, and write this run's files.

    The file is read, cleaned, scored and explained ``chunk_rows`` rows at a time
    and the outputs are appended as it goes, so peak memory is set by the block
    size rather than the file size.

    Explanations are spent on **rejected** applicants only, in file order, against
    one running budget, so the cap behaves exactly as it would in a single pass:
    approved rows never consume it, and a rejected row past it is marked with
    :data:`CAP_NOTE` rather than left blank.

    Scores, decisions and cleaning flags do not depend on ``chunk_rows``. The stated
    reasons can: SurvSHAP(t) draws coalitions per call, so two applicants explained
    in the same block or in different ones may swap two near-tied reasons -- the
    per-applicant instability FINDINGS already records. The block size is written
    into the run summary and the provenance stamp for that reason.

    ``progress(step, state, message)`` is called with ``state`` in
    ``{"running", "done", "failed"}`` for steps ``check``, ``clean``, ``profile``,
    ``score``, ``explain``, ``files``.
    """
    started = time.perf_counter()
    d = cfg.decision
    threshold = d.reject_at_or_above if threshold is None else float(threshold)
    max_explained = d.max_explained if max_explained is None else int(max_explained)
    chunk_rows = int(chunk_rows or getattr(d, "chunk_rows", 50_000))
    runs_dir = Path(runs_dir or RUNS_DIR)

    def say(step: str, state: str, message: str = "") -> None:
        if progress:
            progress(step, state, message)

    def fail(step: str, exc: BatchError):
        say(step, "failed", exc.message)
        raise exc

    # Where the outputs will go is decided now, but the folder is only created
    # when there is something to write into it: a file refused at the checking
    # step leaves nothing behind.
    if run_dir is not None:
        stamp_dir = Path(run_dir)
        clash = ([p.name for p in stamp_dir.iterdir()
                  if p.name in OUTPUT_NAMES and p.is_file()]
                 if stamp_dir.exists() else [])
        if clash:
            raise BatchError(
                f"{stamp_dir.name} already holds results from a finished run.",
                f"existing: {sorted(clash)}",
                "Upload the file again to start a new run.")
    else:
        base = runs_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{_slug(filename)}"
        stamp_dir, n = base, 1
        while stamp_dir.exists():
            n += 1
            stamp_dir = base.with_name(f"{base.name}_{n}")

    def ensure_dir() -> Path:
        stamp_dir.mkdir(parents=True, exist_ok=True)
        return stamp_dir

    scored_path = stamp_dir / "scored_applicants.csv"
    approved_path = stamp_dir / "approved_applicants.csv"
    rejected_path = stamp_dir / "rejected_applicants.csv"

    # A cap of 0 (the default) means no cap: every rejected applicant is explained.
    budget = None if max_explained <= 0 else max_explained
    workers = int(getattr(d, "explain_workers", 0)) or None
    explained_running = 0
    n_rejected_total = 0
    # Explained, but every adverse driver was either undisclosable or had no
    # Regulation B wording: a different gap from "not explained", and counted
    # separately rather than looking like a blank.
    no_disclosable = 0
    step_seconds: dict[str, float] = {}

    def timed(step: str, seconds: float) -> None:
        step_seconds[step] = round(step_seconds.get(step, 0.0) + seconds, 2)

    # A long explanation pass is no use if the machine sleeps halfway through it.
    awake, awake_note = keep_awake(True)

    say("check", "running")
    t_step = time.perf_counter()
    try:
        ctx = ctx or load_context(cfg, model_tag, model_name)
        timed("load_model", time.perf_counter() - t_step)
        t_step = time.perf_counter()
        costs = load_costs(ctx.model_tag, ctx.cfg.paths.tables_dir)
        chunks = iter_chunks(data, filename, chunk_rows)
        first = next(chunks)
        _, report = validate(first, ctx.spec, values=ctx.clean_values, costs=costs,
                             mapping=mapping)
        if report.required_missing:
            raise BatchError(
                "Your file is missing required column(s): "
                + ", ".join(report.required_missing) + ". "
                + report.required_rule_note,
                report.message(),
                "Add the column(s), rename an existing one to match, or supply what "
                "a derivation needs -- the Details list which inputs are missing for "
                "each. Required features are those whose absence costs at least "
                f"{0.010:.3f} concordance (FINDINGS 7d).")
    except BatchError as exc:
        fail("check", exc)
    explainer_name = choose_explainer(ctx, explainer)
    workers = 1 if explainer_name == "treeshap" else workers
    ledger = (None if explainer_name == "treeshap"
              else Ledger.load(stamp_dir / "explained_rows.jsonl"))
    timed("check_file", time.perf_counter() - t_step)
    say("check", "done",
        f"{len(report.present)} of {report.n_features} model features present, "
        f"read in blocks of {chunk_rows:,} rows")

    horizons = {int(t): i for i, t in enumerate(ctx.times)}
    k_h = horizons.get(int(d.horizon_months), len(ctx.times) - 1)
    k12 = horizons.get(12, 0)
    pd_col = f"pd_{int(d.horizon_months)}m"

    clean_report = None
    prof = None
    drift_result = None
    profile_frame = pd.DataFrame()
    reservoir = _Reservoir(PROFILE_ROWS, seed=cfg.explain.seed)
    agg = Aggregates()
    agg.group_column = "purpose" if "purpose" in ctx.spec.categorical else ""
    preview, preview_kept = [], 0
    notices = []
    row_offset = 0
    n_blocks = 0
    said_clean = said_score = said_explain = said_profile = False

    def blocks():
        yield first
        yield from chunks

    for block in blocks():
        n_blocks += 1
        block = block.reset_index(drop=True)
        block, _ = validate(block, ctx.spec, values=ctx.clean_values,
                            costs=costs, mapping=report.recognised or mapping)

        # -------------------------------------------------------- clean ----
        if not said_clean:
            say("clean", "running")
        t_step = time.perf_counter()
        try:
            X, flags, block_report = prepare(block, ctx)
        except BatchError as exc:
            fail("clean", exc)
        except Exception as exc:
            fail("clean", BatchError(
                "The file could not be prepared for the model.", repr(exc),
                "Check that numeric columns contain numbers and that there are no "
                "merged header rows."))
        timed("clean", time.perf_counter() - t_step)
        clean_report = (block_report if clean_report is None
                        else clean_report.merge(block_report))
        if not said_clean:
            said_clean = True
            say("clean", "done", "; ".join(clean_report.plain_english()[:2]))

        # Every block feeds the sample, so the profile and the drift check
        # describe the whole file rather than its opening rows. That means
        # profiling finishes with the last block, not the first.
        if not said_profile:
            said_profile = True
            say("profile", "running")
        reservoir.add_block(block)

        # -------------------------------------------------------- score ----
        if not said_score:
            say("score", "running")
        t_step = time.perf_counter()
        try:
            surv = ctx.model.predict_survival(X, ctx.times)
        except Exception as exc:
            fail("score", BatchError(
                "The model could not score these applicants.", repr(exc),
                "The file's values may be far outside anything the model was "
                "trained on. Check the Details, or try another model."))
        timed("score", time.perf_counter() - t_step)
        pd12 = 1.0 - surv[:, k12]
        pd_h = 1.0 - surv[:, k_h]
        decision = np.where(pd_h >= threshold, "reject", "approve")
        ids = (block[report.id_column].astype(str).to_numpy()
               if report.id_column and report.id_column in block.columns
               else np.array([f"APP-{row_offset + i + 1:06d}"
                              for i in range(len(block))]))

        # ------------------------------------------------------ explain ----
        if not said_explain:
            say("explain", "running")
        t_step = time.perf_counter()
        n_rejected_total += int((decision == "reject").sum())
        reject_pos = np.flatnonzero(decision == "reject")
        # Every rejected applicant is explained. A cap exists only as an explicit
        # override for a quick look; at its default of 0 it does nothing.
        explain_pos = reject_pos if budget is None else reject_pos[:max(budget, 0)]
        reasons, fair_flags, notice_file = {}, {}, {}
        explained_note = pd.Series("not applicable (approved)",
                                   index=range(len(block)))
        explained_note.iloc[reject_pos] = CAP_NOTE
        if len(explain_pos):
            try:
                if explainer_name == "treeshap":
                    expl = explain_tree_shap(
                        ctx.model, X.iloc[explain_pos],
                        horizon_months=float(d.horizon_months), times=ctx.times)
                else:
                    # Seeded per applicant from the applicant's own id, so a notice
                    # does not depend on batch membership, worker count or whether
                    # an earlier run was interrupted (FINDINGS 7.0b).
                    expl = explain_rows_parallel(
                        ctx.model, X.iloc[explain_pos], ctx.background, ctx.times,
                        row_ids=[str(ids[p]) for p in explain_pos],
                        nsamples=d.explain_nsamples,
                        n_background=d.explain_n_background, seed=cfg.explain.seed,
                        workers=(1 if len(explain_pos) < PARALLEL_MIN_ROWS
                                 else workers),
                        ledger=ledger,
                        progress=lambda done, total: say(
                            "explain", "running",
                            f"{explained_running + done:,} of {n_rejected_total or total:,} "
                            f"rejected applicants explained"))
                if report.optional_missing or report.required_missing:
                    expl = _without_features(
                        expl, set(report.optional_missing) | set(report.required_missing))
                for j, pos in enumerate(explain_pos):
                    notice = build_adverse_action_notice(
                        expl, obs=j, applicant_id=str(ids[pos]),
                        horizon_months=int(d.horizon_months),
                        model_name=f"{ctx.model_name} ({ctx.model_tag}), "
                                   f"{explainer_name}")
                    reasons[int(pos)] = notice.reasons
                    if not notice.reasons:
                        no_disclosable += 1
                    fair_flags[int(pos)] = ", ".join(notice.fair_lending_flags)
                    fname = f"notice_{_slug(str(ids[pos]))}.txt"
                    notices.append((fname, notice.render()))
                    notice_file[int(pos)] = fname
                    explained_note.iloc[pos] = "explained"
                explained_running += len(explain_pos)
                if budget is not None:
                    budget -= len(explain_pos)
            except Exception as exc:
                fail("explain", BatchError(
                    "The applicants were scored, but the reasons could not be "
                    "generated.", repr(exc),
                    "Try a smaller file, or lower decision.max_explained in "
                    "config/config.yaml."))

        timed("explain", time.perf_counter() - t_step)

        # -------------------------------------------------------- write ----
        t_step = time.perf_counter()
        front = pd.DataFrame({
            "row_id": np.arange(row_offset + 1, row_offset + len(block) + 1),
            "applicant_id": ids,
            "pd_12m": np.round(pd12, 4),
            pd_col: np.round(pd_h, 4),
            "decision": decision,
            "threshold": threshold,
            "model_tag": ctx.model_tag,
            "model": ctx.model_name,
            "explained": explained_note.to_numpy(),
            # Which explainer produced this row's reasons, so an output file always
            # says what its reasons are based on.
            "explainer": np.where(explained_note.to_numpy() == "explained",
                                  explainer_name, ""),
        })
        positions = list(range(len(block)))
        for name, values in _reason_columns(reasons, positions, 3).items():
            front[name] = values
        front["features_present"] = len(report.present)
        front["features_missing"] = len(report.missing_optional)
        scored = pd.concat(
            [front, flags.reset_index(drop=True),
             block.drop(columns=[c for c in block.columns if c in front.columns],
                        errors="ignore")], axis=1)

        group = (block[agg.group_column]
                 if agg.group_column and agg.group_column in block.columns else None)
        agg.add(pd_h, decision, group, flags, reasons)

        rejected = scored[scored["decision"] == "reject"].copy()
        rej_pos = list(rejected.index)
        rejected[f"reason_{MAX_PRINCIPAL_REASONS}"] = [
            reasons[p][3].reason if p in reasons and len(reasons[p]) > 3 else ""
            for p in rej_pos]
        for i in range(1, MAX_PRINCIPAL_REASONS + 1):
            rejected[f"reason_{i}_attribution"] = [
                round(reasons[p][i - 1].attribution, 6)
                if p in reasons and len(reasons[p]) >= i else ""
                for p in rej_pos]
        rejected["fair_lending_flag"] = [fair_flags.get(p, "") for p in rej_pos]
        rejected["direction_consistent"] = [
            all(r.direction_consistent for r in reasons[p]) if p in reasons else ""
            for p in rej_pos]
        rejected["notice_file"] = [notice_file.get(p, "") for p in rej_pos]
        approved = scored[scored["decision"] == "approve"]

        ensure_dir()
        for path, frame in ((scored_path, scored), (approved_path, approved),
                            (rejected_path, rejected)):
            header = not path.exists()
            frame.to_csv(path, index=False, mode="w" if header else "a",
                         header=header)
        if preview_kept < PREVIEW_ROWS:
            preview.append(scored.head(PREVIEW_ROWS - preview_kept))
            preview_kept += min(len(scored), PREVIEW_ROWS - preview_kept)

        timed("write_rows", time.perf_counter() - t_step)
        row_offset += len(block)
        said_score = said_explain = True
        del X, flags, scored, rejected, approved, block

    say("score", "done", f"{agg.n_approved:,} approved, {agg.n_rejected:,} rejected")
    cap_note = ""
    if agg.n_rejected_without_reasons:
        cap_note = (f"; {agg.n_rejected_without_reasons:,} rejected applicant(s) have "
                    f"NO reasons -- re-run to continue from there")
    say("explain", "done",
        f"{agg.n_rejected_explained:,} of {agg.n_rejected:,} rejected applicants "
        f"explained{cap_note}")

    # ----------------------------------------------------------- profile --
    t_step = time.perf_counter()
    profile_frame = reservoir.frame()
    try:
        prof = profile(profile_frame, ctx)
    except Exception as exc:
        fail("profile", BatchError(
            "The file was scored, but it could not be profiled.", repr(exc),
            "Try again, or check the Details for the column involved."))
    drift_result = prof["drift"]
    say("profile", "done",
        f"drift: {drift_result.status}"
        + (f" ({drift_result.n_large} large, {drift_result.n_moderate} moderate)"
           if drift_result.status in ("large", "moderate") else "")
        + (f", on a random sample of {len(profile_frame):,} of {agg.n_rows:,} rows"
           if len(profile_frame) < agg.n_rows else ""))

    timed("profile_and_drift", time.perf_counter() - t_step)

    # ----------------------------------------------------------- outputs --
    t_step = time.perf_counter()
    say("files", "running")
    files = {"scored_applicants.csv": scored_path,
             "approved_applicants.csv": approved_path,
             "rejected_applicants.csv": rejected_path}
    ensure_dir()
    input_copy = stamp_dir / f"input_{Path(filename).name}"
    if isinstance(data, (str, Path)) and Path(data).resolve() == input_copy.resolve():
        pass                                  # already saved here by the caller
    elif isinstance(data, (str, Path)):
        input_copy.write_bytes(Path(data).read_bytes())
    elif isinstance(data, bytes):
        input_copy.write_bytes(data)
    if notices:
        zpath = stamp_dir / "adverse_action_notices.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, text in notices:
                zf.writestr(fname, text)
        files["adverse_action_notices.zip"] = zpath

    for name, frame in (("data_profile_missing.csv", prof["missing"]),
                        ("data_profile_numeric.csv", prof["numeric"]),
                        ("data_profile_categories.csv", prof["categorical"]),
                        ("data_profile_outliers.csv", prof["outliers"])):
        path = stamp_dir / name
        frame.to_csv(path, index=False)
        files[name] = path
    files["cleaning_report.csv"] = stamp_dir / "cleaning_report.csv"
    clean_report.to_frame().to_csv(files["cleaning_report.csv"], index=False)
    files["data_drift.csv"] = stamp_dir / "data_drift.csv"
    drift_result.table.to_csv(files["data_drift.csv"], index=False)

    elapsed = time.perf_counter() - started
    model_fp = file_fingerprint(ctx.model_path)
    summary = {
        "run_id": stamp_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_file": Path(filename).name,
        "source_sha256": (file_fingerprint(input_copy).get("sha256")
                          if input_copy.exists() else None),
        "n_rows": agg.n_rows,
        "n_approved": agg.n_approved,
        "n_rejected": agg.n_rejected,
        "approval_rate": round(agg.n_approved / agg.n_rows, 4) if agg.n_rows else 0.0,
        f"mean_pd_{int(d.horizon_months)}m": round(agg.mean_pd, 4),
        "model_tag": ctx.model_tag,
        "model": ctx.model_name,
        "model_sha256": model_fp.get("sha256"),
        "model_trained_on": str(ctx.data_source),
        "threshold": threshold,
        "horizon_months": int(d.horizon_months),
        "explainer": explainer_name,
        "explain_nsamples": (d.explain_nsamples if explainer_name == "survshap"
                             else None),
        "explain_n_background": (d.explain_n_background if explainer_name == "survshap"
                                 else None),
        "max_explained": max_explained,
        "n_explained": agg.n_rejected_explained,
        "n_rejected_without_reasons": agg.n_rejected_without_reasons,
        "n_explained_without_disclosable_reason": no_disclosable,
        "n_notices": len(notices),
        "features_expected": report.n_features,
        "features_present": len(report.present),
        "features_missing": len(report.missing_optional),
        "feature_coverage": round(report.coverage, 4),
        "degraded_coverage": bool(report.coverage < d.min_feature_coverage),
        "required_rule": report.required_rule,
        "required_features": "; ".join(
            report.costs.required() if report.costs is not None
            and report.costs.measured else PROVISIONAL_REQUIRED),
        "mapped_columns": "; ".join(f"{k}->{v}" for k, v in report.mapped.items()),
        "features_derived": "; ".join(f"{k} ({v})" for k, v in report.derived.items()),
        "features_missing_optional": "; ".join(report.optional_missing),
        "optional_missing_cost": (report.costs.describe(report.optional_missing)
                                  if report.costs is not None else ""),
        "schema_message": report.message(),
        "ignored_columns": "; ".join(report.ignored),
        "rows_out_of_range": agg.rows_out_of_range,
        "cleaning_policy_version": ctx.policy.version,
        "cleaning_values_version": ctx.clean_values.version,
        "cleaning_values_fitted_rows": ctx.clean_values.fitted_rows,
        "cleaning_values_from_model_bundle": ctx.values_from_bundle,
        "rows_dropped_by_cleaning": clean_report.rows_in - clean_report.rows_out,
        "drift_status": drift_result.status,
        "drift_features_moderate": drift_result.n_moderate,
        "drift_features_large": drift_result.n_large,
        "explain_seeded": explainer_name == "survshap",
        "explain_workers": (workers or suggest_workers(max(n_rejected_total, 1))
                            if explainer_name == "survshap" else 1),
        "sleep_blocked": bool(awake),
        "seconds_by_step": step_seconds,
        "profiled_rows": len(profile_frame),
        "profile_sampling": "uniform random across the whole file",
        "chunk_rows": chunk_rows,
        "blocks": n_blocks,
        "seconds_total": round(elapsed, 1),
    }
    files["run_summary.csv"] = stamp_dir / "run_summary.csv"
    # The per-step timings are finished first, so run_summary.csv, provenance.json
    # and the returned result all carry the same numbers.
    timed("write_outputs", time.perf_counter() - t_step)
    step_seconds["total"] = round(time.perf_counter() - started, 2)
    summary["seconds_by_step"] = dict(step_seconds)
    summary["seconds_total"] = step_seconds["total"]
    pd.DataFrame([summary]).to_csv(files["run_summary.csv"], index=False)
    (stamp_dir / "validation_report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    (stamp_dir / "aggregates.json").write_text(
        json.dumps(agg.to_dict(), indent=2), encoding="utf-8")

    stamp = build_stamp(
        stage="batch_score",
        inputs={"upload": input_copy, "model": ctx.model_path,
                "training_data": ctx.data_source},
        outputs=dict(files),
        config_path=PROJECT_ROOT / "config" / "config.yaml",
        args={"threshold": threshold, "model_tag": ctx.model_tag,
              "model": ctx.model_name, "max_explained": max_explained,
              "explainer": explainer_name,
              "chunk_rows": chunk_rows, "cleaning_policy": ctx.policy.version,
              "cleaning_values_fitted_rows": ctx.clean_values.fitted_rows})
    (stamp_dir / "provenance.json").write_text(
        json.dumps({"provenance": stamp, "summary": summary,
                    "cleaning": clean_report.to_dict()}, indent=2),
        encoding="utf-8")
    keep_awake(False)
    say("files", "done", f"{len(files)} files written to {stamp_dir.name}")

    elapsed = step_seconds["total"]
    preview_frame = (pd.concat(preview, ignore_index=True) if preview
                     else pd.DataFrame())
    approved_preview = rejected_preview = preview_frame
    if len(preview_frame):
        approved_preview = preview_frame[preview_frame["decision"] == "approve"]
        rejected_preview = preview_frame[preview_frame["decision"] == "reject"]
    return BatchResult(
        run_dir=stamp_dir, scored=preview_frame, approved=approved_preview,
        rejected=rejected_preview, summary=summary, report=report, files=files,
        seconds=elapsed, clean_report=clean_report, profile=prof,
        drift=drift_result, aggregates=agg, notices=notices)


def load_result(run_dir: Path) -> BatchResult:
    """Rebuild a :class:`BatchResult` from a finished run directory.

    This is how the dashboard picks up a run that executed in a background
    process: everything it displays was written to disk by that run, so nothing is
    recomputed and the page cannot show numbers the files do not contain.
    """
    run_dir = Path(run_dir)
    prov_path = run_dir / "provenance.json"
    scored_path = run_dir / "scored_applicants.csv"
    if not (prov_path.exists() and scored_path.exists()):
        raise BatchError(f"{run_dir.name} does not hold a finished run.",
                         f"missing {'provenance.json' if not prov_path.exists() else 'scored_applicants.csv'}",
                         "Wait for the run to finish, or start it again.")
    payload = json.loads(prov_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    clean_report = CleaningReport(**payload["cleaning"]) if payload.get("cleaning")         else CleaningReport()

    def read(name: str) -> pd.DataFrame:
        path = run_dir / name
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, keep_default_na=True)

    report = ValidationReport()
    vr = run_dir / "validation_report.json"
    if vr.exists():
        raw = json.loads(vr.read_text(encoding="utf-8"))
        report = ValidationReport(
            n_rows=raw.get("n_rows", 0), mapped=raw.get("mapped", {}),
            ignored=raw.get("ignored", {}), missing_core=raw.get("missing_core", []),
            missing_optional=raw.get("missing_optional", []),
            present=raw.get("present", []) or [""] * raw.get("n_features_present", 0),
            id_column=raw.get("id_column"), warnings=raw.get("warnings", []))

    drift_table = read("data_drift.csv")
    drift_result = None
    if not drift_table.empty:
        status = summary.get("drift_status", "stable")
        drift_result = drift_mod.DriftResult(
            table=drift_table, status=status,
            n_moderate=int(summary.get("drift_features_moderate", 0)),
            n_large=int(summary.get("drift_features_large", 0)),
            n_unknown=int((drift_table["status"] == "unknown").sum()),
            n_rows=int(summary.get("profiled_rows", 0)),
            notes=[f"{r.feature}: {r.measure} {r.score} ({r.status})"
                   for r in drift_table.itertuples()
                   if r.status in ("large", "moderate", "unknown")][:10])

    notices: list[tuple[str, str]] = []
    zpath = run_dir / "adverse_action_notices.zip"
    if zpath.exists():
        with zipfile.ZipFile(zpath) as zf:
            notices = [(n, zf.read(n).decode("utf-8", "replace")) for n in zf.namelist()]

    agg_path = run_dir / "aggregates.json"
    aggregates = (Aggregates.from_dict(json.loads(agg_path.read_text(encoding="utf-8")))
                  if agg_path.exists() else None)
    # Only the preview is read into memory; charts come from the aggregates, which
    # is what keeps a finished 500 MB run as cheap to display as a small one.
    scored = pd.read_csv(scored_path, nrows=PREVIEW_ROWS)
    return BatchResult(
        run_dir=run_dir, scored=scored,
        approved=scored[scored["decision"] == "approve"] if len(scored) else scored,
        rejected=scored[scored["decision"] == "reject"] if len(scored) else scored,
        summary=summary, report=report, aggregates=aggregates,
        files={name: run_dir / name for name in OUTPUT_NAMES
               if (run_dir / name).exists()},
        seconds=float(summary.get("seconds_total", 0.0)),
        clean_report=clean_report,
        profile={"overview": {"rows": int(summary.get("n_rows", len(scored))),
                              "columns": int(scored.shape[1]),
                              "numeric_columns": int(scored.select_dtypes("number").shape[1]),
                              "categorical_columns": int(
                                  scored.select_dtypes(["object", "string"]).shape[1])},
                 "missing": read("data_profile_missing.csv"),
                 "numeric": read("data_profile_numeric.csv"),
                 "categorical": read("data_profile_categories.csv"),
                 "outliers": read("data_profile_outliers.csv"),
                 "drift": drift_result},
        drift=drift_result, notices=notices)


def bundle_zip(result: BatchResult) -> bytes:
    """Every output file of one run, as a single in-memory ZIP."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(result.run_dir.iterdir()):
            if path.is_file():
                zf.write(path, path.name)
    return buf.getvalue()
