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
notices              :func:`...adverse_action.build_adverse_action_notice`,
                     split into the applicant notice and the internal
                     review record
model approval       :func:`creditsurv.registry.assess`
checks               :func:`creditsurv.run_checks.verify_run`
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

**Only an approved model decides.** The model must be approved in the registry
(``config/models.yaml``); otherwise the run is refused, or -- when the caller
overrides explicitly -- finishes stamped "not for lending decisions" in every
output and on every notice.

**Applicant notices and internal records never share a file.** Notices go to
``adverse_action_notices.zip``; fair-lending flags, drivers and attributions go to
``internal/``. Every notice screens itself, and the run reads the zip back from
disk and screens it again before it is marked finished.

**A run checks itself before it is finished.** :mod:`creditsurv.run_checks` reads
the written files back and verifies counts, decisions, risk ordering, reasons,
disclosability, notice content and approval. A blocking failure withholds the
notices and leaves no ``provenance.json``, so the run is never loaded as finished.

**Cleaning values belong to the model.** They are fitted once on its whole training
split and saved with it; scoring reads them and never fits. A model without them
refuses to score and says which command adds them, rather than quietly learning
bounds from a sample or from the file being scored.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import drift as drift_mod
from . import eda
from .derive import (DERIVATIONS, FeatureCosts, derive_features,
                     expand_wanted, load_costs)
from .schema_match import propose_mapping
from .cleaning import (COLUMN_PARSERS, DUPLICATE_RULES, CleaningPolicy,
                       CleaningReport, CleaningValues, DuplicateTracker, clean,
                       coerce_column, fit_values,
                       policy_from_config)   # fit_values: never called
# here -- imported so tests/test_cleaning_values_source.py can assert that scoring
# does not fit cleaning values.
from .config import Config
from .environment import policy_block_message, policy_blocked_exception
from .explain.adverse_action import (MAX_PRINCIPAL_REASONS, NOT_FOR_LENDING,
                                     build_adverse_action_notice)
from .explain.parallel import (Ledger, explain_rows_parallel, keep_awake,
                               suggest_workers)
from .explain.survshap import explain_survshap
from .explain.tree_shap import explain_tree_shap
from .features.build import build_design_matrix
from .pipeline import (load_feature_frame, load_model_bundle,
                       resolve_data_source)
from .provenance import PROJECT_ROOT, build_stamp, file_fingerprint
from .registry import assess
from .run_checks import (NO_REASON_NOTE, REASONS_PENDING, blocking_failures,
                         verify_run, write_checks)

__all__ = ["CORE_REQUIRED", "PROVISIONAL_REQUIRED", "ALIASES", "OUTPUT_NAMES", "BatchError",
           "ValidationReport", "ScoringContext", "BatchResult", "read_upload",
           "validate", "load_context", "prepare", "profile", "run_batch",
           "load_result", "bundle_zip", "iter_chunks", "Aggregates", "CAP_NOTE",
           "choose_explainer", "RUNS_DIR", "INTERNAL_DIR", "INTERNAL_FLAGS",
           "INTERNAL_RECORDS", "NO_REASON_NOTE", "score_file", "explain_run",
           "PHASE2_DIR", "REASONS_PENDING"]

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
    "validation_checks.csv",
)
"""Every file a finished run writes at the top of its folder. Used to recognise a completed run directory
and to refuse writing a second run into one."""

INTERNAL_DIR = "internal"
"""Sub-folder for material the applicant is never given: fair-lending flags,
non-disclosable drivers, attributions, model details."""
PARTIAL_SNAPSHOT = "PARTIAL_SNAPSHOT.txt"
"""Put first in a bundle zipped while Phase 2 is unfinished."""
INTERNAL_FLAGS = "internal_review_flags.csv"
INTERNAL_RECORDS = "internal_review_records.jsonl"
INTERNAL_README = (
    "INTERNAL -- NEVER SEND ANY FILE IN THIS FOLDER TO AN APPLICANT.\n\n"
    f"{INTERNAL_FLAGS}: one row per explained rejected applicant -- fair-lending "
    "flag,\nnon-disclosable drivers, the feature and attribution behind each stated "
    "reason.\n"
    f"{INTERNAL_RECORDS}: the full internal review record per applicant, as JSON.\n\n"
    "Applicant notices are in ../adverse_action_notices.zip and contain none of "
    "this.\n")
INTERNAL_COLUMNS: tuple[str, ...] = (
    "applicant_id", "notice_status", "fair_lending_flag", "non_disclosable_drivers",
    "top_driver_not_disclosable", "top_driver", "top_driver_attribution",
    "predicted_default_probability", "horizon_months", "model",
    *[f"reason_{i}_{kind}" for i in range(1, MAX_PRINCIPAL_REASONS + 1)
      for kind in ("feature", "attribution")],
    "direction_consistent")
"""Columns of internal_review_flags.csv, fixed so an empty block still writes them."""

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

    def __init__(self, message: str, detail: str = "", fix: str = "",
                 run_dir: Path | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail or message
        self.fix = fix
        self.run_dir = run_dir
        """Set when a run wrote files before failing, e.g. validation_checks.csv."""


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
    coerced_text: dict[str, int] = field(default_factory=dict)
    """Feature -> values that were text only a parser could read. Counted here
    because the parsing happens here; folded into the cleaning report by
    :func:`run_batch`, which is where the file's report is assembled."""
    required_rule: str = "provisional"
    """"measured" when the model's ablation table decided which features are
    required, "provisional" when the hand-picked list stood in for it."""
    required_rule_note: str = ""
    unlearned_missing: list[str] = field(default_factory=list)
    """Absent features the model never saw missing in training, so its NaN route for
    them is an unlearned default rather than a degradation (FINDINGS 7o). Judged on
    the training missing rate, independently of ablation cost."""
    unlearned_rule_note: str = ""
    filled_from_training: dict[str, object] = field(default_factory=dict)
    """Feature -> the training value substituted for it, for those of
    ``unlearned_missing`` the run filled rather than refused. Named here, per
    feature, because a count or a coverage percentage cannot say which applicant
    attribute was invented."""
    unlearned_action: str = "fill"
    unused: dict[str, str] = field(default_factory=dict)
    unused_targets: dict[str, str] = field(default_factory=dict)
    """Uploaded column -> the feature it was recognised as, for columns this model
    does not score. What lets a monitoring segment find ``state`` in a file scored
    by a model with no geography (FINDINGS 7c)."""
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
        if self.filled_from_training:
            lines.append(
                "Filled with a training value, not read from the file: " + "; ".join(
                    f"{k} = {v}" for k, v in self.filled_from_training.items())
                + ". The model never saw these features missing in training, so "
                "leaving them absent would rest on an unlearned default. Do not "
                "treat these applicants' results as fully reliable for these "
                "attributes.")
        if self.unused:
            lines.append("Recognised but unused by this model: " + "; ".join(
                f"{k} ({v})" for k, v in self.unused.items()))
        if self.required_rule == "provisional" and self.required_rule_note:
            lines.append(self.required_rule_note)
        if self.unlearned_missing and self.unlearned_rule_note:
            lines.append(self.unlearned_rule_note)
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
                "unused_targets": self.unused_targets,
                "coverage": round(self.coverage, 4), "id_column": self.id_column,
                "mapped": self.mapped, "ignored": self.ignored,
                "missing_core": self.missing_core,
                "missing_optional": self.missing_optional,
                "unlearned_missing": self.unlearned_missing,
                "unlearned_rule_note": self.unlearned_rule_note,
                "unlearned_action": self.unlearned_action,
                "filled_from_training": dict(self.filled_from_training),
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
             mapping: dict | None = None, fill_values: dict | None = None,
             unlearned_action: str = "fill",
             ) -> tuple[pd.DataFrame, ValidationReport]:
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

    # A synonym is a change of name, not a reading. "term" arrives holding
    # "36 months" and is renamed to term_months, which also skips step 2 below --
    # the column now exists, so its derivation, the one place that parsed it, is
    # never reached. Everything that reads *this* frame then saw text where it
    # expected months: the drift check scored term_months as 100% missing
    # (PSI 27.631 = 2*ln(1/eps), the same number for every file, which is how it
    # was found), the numeric profile dropped the column, and the input-quality
    # gate counted it readable because a string is not missing. Scoring was right
    # throughout, because creditsurv.cleaning parses the column on its own path --
    # so the two paths disagreed about one feature.
    #
    # The parser cleaning uses is applied here instead, so there is one reading of
    # a term and every reader of this frame gets it. Applied to the column
    # whenever it is present, not only when renamed: a file whose column is
    # already called term_months skips the derivation the same way. The parsers
    # are idempotent -- a column of numbers is returned as numbers.
    for col in COLUMN_PARSERS:
        if col in work.columns:
            work[col], n_text = coerce_column(col, work[col])
            if n_text:
                rep.coerced_text[col] = n_text

    rep.mapped = dict(renames)
    rep.recognised = dict(renames)
    rep.unused = dict(proposal.recognised_but_unused) if proposal else {}
    rep.unused_targets = dict(proposal.unused_targets) if proposal else {}
    rep.proposal = proposal

    # 2. Derivation. Exact arithmetic from columns the file does have, never a guess.
    wanted = [c for c in spec.all_columns if c not in work.columns]
    work, derived, blocked = derive_features(work, expand_wanted(wanted))
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
    # The gate's second rule. A feature the training rows never saw missing has no
    # learned route for its absence, so leaving it NaN hands every row to the
    # booster's default direction -- which is why this is decided on the training
    # missing rate and not on the ablation cost that already called these optional.
    rep.unlearned_missing = costs.unlearned_missing(rep.optional_missing)
    rep.unlearned_rule_note = costs.unlearned_note(rep.optional_missing)
    rep.unlearned_action = unlearned_action
    if rep.unlearned_missing:
        if unlearned_action == "block":
            rep.required_missing = rep.required_missing + rep.unlearned_missing
            rep.optional_missing = [c for c in rep.optional_missing
                                    if c not in rep.unlearned_missing]
        else:
            rep.filled_from_training = {
                c: (fill_values or {})[c] for c in rep.unlearned_missing
                if c in (fill_values or {})}
    rep.missing_core = list(rep.required_missing)
    rep.missing_optional = list(rep.optional_missing)
    # A raw column that fed a derivation was used, whatever the fixed notes say.
    fed = {}
    for rule in DERIVATIONS:
        if rule.feature in derived:
            for c in rule.needs:
                fed.setdefault(c, []).append(rule.feature)
    for col in work.columns:
        if col in expected or col == rep.id_column:
            continue
        rep.ignored[col] = (
            f"not a model feature itself; used to compute {', '.join(fed[col])}"
            if col in fed else rep.unused.get(col)
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
    # One warning per feature, naming it and the value put in its place. Folding
    # these into the optional-missing cost line is exactly what hid this case: that
    # line reports concordance, and concordance is blind to the shift (FINDINGS 7o).
    for feature, value in rep.filled_from_training.items():
        rep.warnings.append(
            f"{feature}: absent from file, filled with the training "
            f"{'median' if not isinstance(value, str) else 'modal value'} "
            f"({value}) because the model never saw it missing in training and so "
            f"has no learned route for its absence. Do not treat these applicants' "
            f"results as fully reliable for this attribute.")
    unfilled = [c for c in rep.unlearned_missing
                if c not in rep.filled_from_training
                and c not in rep.required_missing]
    if unfilled:
        rep.warnings.append(
            f"{', '.join(unfilled)}: absent from file, the model never saw them "
            f"missing in training, and no training value was available to stand in. "
            f"They were left missing, so these rows rest on the booster's default "
            f"direction for them -- an unmeasured constant (FINDINGS 7o).")
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
    train_missing: dict[str, float] = field(default_factory=dict)
    """Feature -> share of the model's training rows where it was missing, measured
    on ``reference``. What the unlearned-default rule needs and the ablation table
    does not hold; see derive.UNLEARNED_MISSING_FLOOR."""
    train_fill: dict[str, object] = field(default_factory=dict)
    """Feature -> the value that stands in when the file omits it entirely: the
    median for a numeric feature, the modal level for a categorical one, both from
    the training rows. Only ever used for a feature the unlearned-default rule
    caught, and only when it names the feature in the output."""

    @property
    def flavour(self) -> str:
        return "cox" if self.model_name == "cox" else "gbm"

    def costs_with_training(self, costs):
        """``costs`` carrying this model's training missing rates, so the gate can
        apply both of its rules rather than only the measured one."""
        return costs.with_train_missing(
            self.train_missing, rows=len(self.reference),
            source=f"{Path(self.data_source).name} training split",
            structural=getattr(self.spec, "structural_missing", ()),
            floor=getattr(self.cfg.decision, "unlearned_missing_floor", None))

    def unlearned_fill(self, costs) -> dict[str, object]:
        """Feature -> stand-in value, for every feature the rule would catch."""
        return {f: self.train_fill[f] for f in self.spec.all_columns
                if f in self.train_fill and costs.unlearned(f)}


def available_models(models_dir: Path) -> list[str]:
    return sorted(p.stem.replace("02_models_", "")
                  for p in Path(models_dir).glob("02_models_*.pkl"))


def context_key(cfg: Config, model_tag: str | None = None,
                model_name: str | None = None) -> tuple:
    """Everything :func:`load_context` depends on, as a hashable key.

    A cache of loaded models (the dashboard's) is keyed on this, so it is reused
    across uploads and page interactions yet can never serve a stale model: replace
    the bundle, its cleaning values or its training data, or change the background
    settings, and the key changes.
    """
    d = cfg.decision
    tag = model_tag or d.model_tag
    models = Path(cfg.paths.models_dir)

    def stamp(p: Path):
        try:
            st = Path(p).stat()
            return (str(p), st.st_mtime_ns, st.st_size)
        except OSError:
            return (str(p), None, None)

    return (tag, model_name or d.model, stamp(models / f"02_models_{tag}.pkl"),
            stamp(models / f"02_cleaning_values_{tag}.json"),
            stamp(Path(cfg.paths.data_dir) / "accepted_labeled.parquet"),
            int(d.background_rows), int(cfg.explain.seed),
            tuple(cfg.model.eval_horizons_months))


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
        try:
            bundle, model_path = load_model_bundle(cfg.paths.models_dir, model_tag)
        except BaseException as exc:
            # Unpickling a bundle imports lightgbm to rebuild its booster, so a
            # blocked library surfaces here as an OSError from inside pickle.load.
            if not policy_blocked_exception(exc):
                raise
            raise BatchError(
                "This machine cannot score applicants: Windows is blocking the "
                "libraries the model needs.",
                f"{type(exc).__name__}: {exc}",
                policy_block_message()) from exc
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
    try:
        train = load_feature_frame(src, spec)
    except ValueError as exc:
        raise BatchError(
            f"The '{model_tag}' model cannot be compared against the data it was "
            f"trained on.", str(exc),
            "Retrain the model, or restore the data file it was trained on.") from exc
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
    train_missing, train_fill = _training_missingness(train, spec, values)

    return ScoringContext(
        cfg=cfg, model_tag=model_tag, model_name=key, model=art[key], spec=spec,
        bundle=bundle, model_path=model_path, background=dm.X, reference=train,
        clean_values=values, policy=policy,
        times=np.array(cfg.model.eval_horizons_months, dtype=float),
        data_source=Path(src), values_from_bundle=from_bundle,
        train_missing=train_missing, train_fill=train_fill)


def _training_missingness(train: pd.DataFrame, spec,
                          values: CleaningValues) -> tuple[dict, dict]:
    """How often each feature was missing in training, and what stands in for it.

    The rate is the input to the unlearned-default rule. The stand-in is the value
    that rule substitutes: the cleaning values' median where they hold one -- the
    same number the Cox path fills with, fitted on the whole training split rather
    than this sample -- and otherwise the sample's own median, which is all a feature
    derived after cleaning (fico_midpoint, emp_length_years) has.
    """
    rates: dict[str, float] = {}
    fills: dict[str, object] = {}
    categorical = set(getattr(spec, "categorical", ()))
    for col in spec.all_columns:
        if col not in train.columns:
            continue           # unknown, and an unknown rate is never a low one
        series = train[col]
        rates[col] = float(series.isna().mean())
        if col in categorical:
            modes = series.dropna().astype("string").mode()
            if len(modes):
                fills[col] = str(modes.iloc[0])
            continue
        if col in values.medians:
            fills[col] = float(values.medians[col])
            continue
        median = pd.to_numeric(series, errors="coerce").median()
        if pd.notna(median):
            fills[col] = float(median)
    return rates, fills


def prepare(df: pd.DataFrame, ctx: ScoringContext,
            fill_values: dict | None = None,
            ) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Clean and encode uploaded rows exactly as training does.

    Value cleaning is :func:`creditsurv.cleaning.clean` with the values fitted on
    the model's training split; encoding is the pipeline's own
    ``build_design_matrix``. Returns the design matrix, a per-row flag frame and
    the cleaning report.

    ``fill_values`` are the stand-ins for features absent from the file that the
    model never saw missing in training -- ``ValidationReport.filled_from_training``,
    which is also what the run reports. Passed in rather than read from ``ctx`` so
    that what was substituted is decided once, by the gate, and every block of a
    large file is encoded the same way.
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
            # ...except where leaving them missing would hit a route the model never
            # learned. A feature the training rows never saw blank taught the booster
            # nothing about NaN, so every row falls to its default split direction: a
            # fixed, arbitrary constant, and on the file that found this one that
            # pushed mean PD from 0.2225 to 0.3339 (FINDINGS 7o, which names the
            # file; nothing here branches on it). A median fitted on the training
            # rows is also a constant, but a defensible one the run can name -- which
            # it does, per feature, in the report and the summary.
            filled = fill_values if fill_values is not None else {}
            for col in trained:
                if col in dm.X.columns:
                    continue
                if col in filled:
                    dm.X[col] = filled[col]
                    continue
                # A missing-as-signal indicator for an absent column is not
                # unknown: the value genuinely is not there, so it is 1.
                dm.X[col] = 1 if col.endswith("_missing") else np.nan
            # A filled feature is present, so its indicator says present. Done after
            # the loop so it holds however the column was restored.
            for col in filled:
                indicator = f"{col}_missing"
                if indicator in dm.X.columns:
                    dm.X[indicator] = 0
            dm.X = dm.X[list(trained)]
            for col, levels in ctx.clean_values.categories.items():
                if col in dm.X.columns:
                    dm.X[col] = pd.Categorical(dm.X[col].astype("string").str.strip(),
                                               categories=list(levels))
    return dm.X, flags, report


def input_quality_inputs(ctx: ScoringContext, report: ValidationReport) -> dict:
    """What the input-quality check compares a run against: the limit, the model
    features the file supplied (by name or by derivation), and each one's missing
    share in the training rows the context holds."""
    ref = ctx.reference
    return {
        "max_share": float(getattr(ctx.cfg.decision, "input_quality_max_share", 0.05)),
        "features": list(report.present),
        "train_missing": {c: float(ref[c].isna().mean()) for c in report.present
                          if c in ref.columns and len(ref)},
        "structural": [c for c in getattr(ctx.spec, "structural_missing", ())
                       if c in report.present],
    }


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


MONITOR_SEGMENTS: tuple[str, ...] = ("purpose", "income_band", "addr_state")
"""Dimensions every run reports its decisions by, beyond the headline counts.

Monitoring, not modelling. ``addr_state`` is deliberately **not** a feature of the
approved model (FINDINGS 7c, 7l), and reporting outcomes by it is the whole point:
a model that cannot see geography can still decide unevenly across it, and that is
only visible if the run writes it down. ``income_band`` is the banding Stage 3's
segment report uses (``features.encoders.income_band``), not a second definition.

Accumulated here, block by block, for the same reason as every other figure on this
class: a 500 MB upload must cost no more to display than a 25 KB one, so the
dashboard never reads ``scored_applicants.csv`` back to count anything.
"""


def segment_labels(block: pd.DataFrame, X: pd.DataFrame,
                   report: "ValidationReport") -> pd.DataFrame:
    """The monitoring dimensions for one block, as columns of labels.

    Each dimension is taken from the matrix the model actually scored where it is in
    it, and from the uploaded file where it is not -- so a figure about a feature is
    about the value the model saw, and a figure about a column the model ignores is
    about the file. The two cases are distinguished in the output by
    :data:`MONITOR_SEGMENTS` membership and said out loud on the page.

    A dimension the file cannot supply is simply absent: nothing is imputed, and no
    "(unknown)" level is invented to fill a chart.
    """
    from .features import encoders as enc

    # Indexed on the matrix that was scored, not on the uploaded block: cleaning can
    # drop rows, and ``decision`` has one entry per scored row. Taking the index from
    # the block instead would silently pair a decision with another applicant's label.
    out = pd.DataFrame(index=X.index)

    def column(name: str):
        if name in X.columns:
            return X[name]
        if name in block.columns:
            return block[name].reindex(X.index)
        # A column this model does not score may still be in the file under the
        # name the file gave it.
        for uploaded, feature in (report.unused_targets or {}).items():
            if feature == name and uploaded in block.columns:
                return block[uploaded].reindex(X.index)
        return None

    purpose = column("purpose")
    if purpose is not None:
        out["purpose"] = purpose.astype("string").str.strip()

    income = column("annual_inc")
    if income is not None:
        # Left as the ordered Categorical income_band returns: the band order is
        # part of the definition, and the run records it so no reader has to restate
        # the edges to sort them.
        out["income_band"] = enc.income_band(pd.to_numeric(income, errors="coerce"))

    state = column("addr_state")
    if state is not None:
        out["addr_state"] = state.astype("string").str.strip().str.upper()

    return out


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
    n_reasons_pending: int = 0
    """Rejected rows Phase 2 has not reached yet: marked "reasons pending", which is
    a state, not a gap -- unlike rows a cap, a sample or a skip left without."""
    pd_sum: float = 0.0
    pd_bins: int = 40
    pd_hist: list = field(default_factory=lambda: [0] * 40)
    by_group: dict = field(default_factory=dict)       # "level|decision" -> count
    by_segment: dict = field(default_factory=dict)
    """"column|level|decision" -> count, over :data:`MONITOR_SEGMENTS`. A superset of
    ``by_group``, which is kept as it was so a run written before this still reads."""
    segment_columns: list = field(default_factory=list)
    """Which dimensions this run actually had the columns for, in display order. An
    empty list on a run scored before segments were recorded, which the page says
    rather than drawing an empty chart."""
    segment_order: dict = field(default_factory=dict)
    """Dimension -> its own level order, for the dimensions that have one. Income
    bands are ordinal and belong in band order on a chart, whatever their sizes, and
    the order comes from the banding that produced them rather than from a list
    restated somewhere else."""
    reason_counts: dict = field(default_factory=dict)
    rows_out_of_range: int = 0
    group_column: str = ""
    n_fair_lending_flagged: int = 0
    """Explained rejections where a non-disclosable feature was among the strongest
    adverse drivers (the notice's own flag: top 2 x MAX_PRINCIPAL_REASONS)."""
    n_top_driver_not_disclosable: int = 0
    """...and where it was the single strongest one."""
    flag_features: dict = field(default_factory=dict)

    def add_fair_lending(self, records) -> None:
        for rec in records:
            if rec.flagged:
                self.n_fair_lending_flagged += 1
                for f in rec.fair_lending_flags:
                    self.flag_features[f] = self.flag_features.get(f, 0) + 1
            if rec.top_driver_not_disclosable:
                self.n_top_driver_not_disclosable += 1

    @property
    def fair_lending_share(self) -> float:
        return (self.n_fair_lending_flagged / self.n_rejected_explained
                if self.n_rejected_explained else 0.0)

    @property
    def mean_pd(self) -> float:
        return self.pd_sum / self.n_rows if self.n_rows else 0.0

    @property
    def n_rejected_without_reasons(self) -> int:
        return max(0, self.n_rejected - self.n_rejected_explained
                   - self.n_reasons_pending)

    def add_segments(self, segments, decision) -> None:
        """Count decisions by each monitoring dimension in this block."""
        if segments is None or len(segments) == 0:
            return
        for col in MONITOR_SEGMENTS:
            if col not in segments.columns:
                continue
            if col not in self.segment_columns:
                self.segment_columns.append(col)
            values = segments[col]
            if str(values.dtype) == "category" and col not in self.segment_order:
                self.segment_order[col] = [str(c) for c in values.cat.categories]
            levels = values.astype("string").fillna("(missing)").to_numpy()
            for level, dec in zip(levels, decision):
                key = f"{col}|{level}|{dec}"
                self.by_segment[key] = self.by_segment.get(key, 0) + 1

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

    def segment_frame(self) -> pd.DataFrame:
        """``by_segment`` as rows: one per dimension, level and decision."""
        rows = []
        for key, count in self.by_segment.items():
            column, _, rest = key.partition("|")
            level, _, decision = rest.rpartition("|")
            rows.append({"column": column, "group": level, "decision": decision,
                         "applicants": count})
        return pd.DataFrame(rows, columns=["column", "group", "decision",
                                           "applicants"])

    def reason_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"reason": k, "times cited": v} for k, v in
                             sorted(self.reason_counts.items(), key=lambda kv: -kv[1])])

    def to_dict(self) -> dict:
        return {"n_rows": self.n_rows, "n_approved": self.n_approved,
                "n_rejected": self.n_rejected,
                "n_rejected_explained": self.n_rejected_explained,
                "n_reasons_pending": self.n_reasons_pending,
                "pd_sum": self.pd_sum, "pd_bins": self.pd_bins,
                "pd_hist": list(self.pd_hist), "by_group": self.by_group,
                "by_segment": self.by_segment,
                "segment_columns": list(self.segment_columns),
                "segment_order": self.segment_order,
                "reason_counts": self.reason_counts,
                "rows_out_of_range": self.rows_out_of_range,
                "group_column": self.group_column,
                "n_fair_lending_flagged": self.n_fair_lending_flagged,
                "n_top_driver_not_disclosable": self.n_top_driver_not_disclosable,
                "flag_features": self.flag_features}

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
    checks: pd.DataFrame = field(default_factory=pd.DataFrame)
    """validation_checks.csv: one row per post-run check, PASS/FAIL/OVERRIDDEN."""


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", Path(name).stem)[:40] or "upload"


RISK_DECIMALS = 4
"""Decimals of predicted risk written to the decision files. The decision is taken
on the rounded value, so a file never contradicts its own threshold."""

REJECTED_REASON_COLUMNS: tuple[str, ...] = (
    "reason_4", "reason_4_feature",
    *[f"reason_{i}_attribution" for i in range(1, 5)],
    "fair_lending_flag", "direction_consistent", "notice_file")
"""Columns rejected_applicants.csv carries beyond scored_applicants.csv; empty in
Phase 1 and filled in by Phase 2."""

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


def _save_frame(frame: pd.DataFrame, path: Path, writer=None, meta: dict | None = None):
    """Append a design-matrix block to a parquet file, exactly restorable.

    Categorical columns are stored as text and their category lists recorded in
    ``meta``, so :func:`_load_frame` rebuilds the very dtypes the model scored --
    which is what makes Phase 2's reasons the reasons for Phase 1's decisions.
    Returns the writer, to be passed back for the next block and closed at the end.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = frame.copy()
    if meta is not None and "columns" not in meta:
        meta["columns"] = [str(c) for c in frame.columns]
        meta["dtypes"] = {str(c): str(t) for c, t in frame.dtypes.items()}
        meta["categories"] = {str(c): [str(v) for v in frame[c].cat.categories]
                              for c in frame.columns
                              if isinstance(frame[c].dtype, pd.CategoricalDtype)}
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].astype("string")
    table = pa.Table.from_pandas(out, preserve_index=True,
                                 schema=writer.schema if writer is not None else None)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema)
    writer.write_table(table)
    return writer


def _load_frame(path: Path, meta: dict, rows=None) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if rows is not None:
        frame = frame.loc[frame.index.intersection(pd.Index(rows))]
    for c, levels in meta.get("categories", {}).items():
        frame[c] = pd.Categorical(frame[c].astype("string"), categories=levels)
    return frame[[c for c in meta["columns"] if c in frame.columns]]


PHASE2_DIR = "phase2"
"""What Phase 1 leaves for Phase 2 inside the run folder: the plan, the rejected
applicants' model inputs exactly as scored, the explainer's background sample, and
-- as Phase 2 runs -- its progress, results and the per-applicant ledger."""


def score_file(data, filename: str, cfg: Config, *, model_tag: str | None = None,
               model_name: str | None = None, threshold: float | None = None,
               runs_dir: Path | None = None, run_dir: Path | None = None,
               chunk_rows: int | None = None, explainer: str | None = None,
               mapping: dict | None = None, progress=None,
               ctx: ScoringContext | None = None,
               allow_unapproved_model: bool = False) -> BatchResult:
    """**Phase 1**: check, clean, score, decide and drift-check a file, and write it.

    The one function that turns an applicant file into decisions. The dashboard
    (inline and in the background), ``06_score_upload.py`` and :func:`run_batch` all
    call it; ``tests/test_two_phase.py`` fails if anything else scores applicants.

    It explains nobody. Every rejected row is written with ``explained = "reasons
    pending"`` and the model inputs Phase 2 needs are saved beside it, so decisions
    for a file of any size are on disk -- and checked -- in the time it takes to
    read, clean and score it. :func:`creditsurv.phase2.explain_run` (Phase 2) then
    fills the reasons in.

    The file is read, cleaned and scored ``chunk_rows`` rows at a time and written
    as it goes, so peak memory is set by the block size, not the file size. Scores,
    decisions and cleaning flags do not depend on ``chunk_rows``.

    ``progress(step, state, message)`` is called with ``state`` in
    ``{"running", "done", "failed"}`` for steps ``check``, ``clean``, ``profile``,
    ``score``, ``files``.

    The model must be approved in the registry. ``allow_unapproved_model=True`` is
    the explicit override: the run proceeds, and every output says it is not for
    lending decisions. The same stamp applies to a threshold other than the
    published ``decision.reject_at_or_above``.
    """
    started = time.perf_counter()
    d = cfg.decision
    threshold = d.reject_at_or_above if threshold is None else float(threshold)
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
    phase2_dir = stamp_dir / PHASE2_DIR
    step_seconds: dict[str, float] = {}

    def timed(step: str, seconds: float) -> None:
        step_seconds[step] = round(step_seconds.get(step, 0.0) + seconds, 2)

    say("check", "running")
    t_step = time.perf_counter()
    try:
        ctx = ctx or load_context(cfg, model_tag, model_name)
        timed("load_model", time.perf_counter() - t_step)
        t_step = time.perf_counter()
        # The ablation costs, plus the training missing rates the table does not
        # hold, so the gate can apply both of its rules.
        costs = ctx.costs_with_training(
            load_costs(ctx.model_tag, ctx.cfg.paths.tables_dir))
        unlearned_action = str(getattr(d, "unlearned_missing_action", "fill"))
        unlearned_fill = ctx.unlearned_fill(costs)
        chunks = iter_chunks(data, filename, chunk_rows)
        first = next(chunks)
        _, report = validate(first, ctx.spec, values=ctx.clean_values, costs=costs,
                             mapping=mapping, fill_values=unlearned_fill,
                             unlearned_action=unlearned_action)
        if report.required_missing:
            raise BatchError(
                "Your file is missing required column(s): "
                + ", ".join(report.required_missing) + ". "
                + report.required_rule_note
                + (" " + report.unlearned_rule_note
                   if report.unlearned_missing else ""),
                report.message(),
                "Add the column(s), rename an existing one to match, or supply what "
                "a derivation needs -- the Details list which inputs are missing for "
                "each. Required features are those whose absence costs at least "
                f"{0.010:.3f} concordance (FINDINGS 7d), or that the model never saw "
                f"missing in training (FINDINGS 7o).")
        # Decided once, from the first block, and reused for every block after it:
        # what stands in for an absent feature cannot vary down a file.
        fills = dict(report.filled_from_training)
    except BatchError as exc:
        fail("check", exc)
    explainer_name = choose_explainer(ctx, explainer)

    # Only an approved model makes lending decisions. Checked on every run, against
    # the model file actually loaded, so a stale tag or a swapped file cannot pass.
    approval = assess(ctx.model_tag, cfg, model_path=ctx.model_path,
                      spec_features=ctx.spec.all_columns, explainer=explainer_name,
                      nsamples=d.explain_nsamples, n_background=d.explain_n_background)
    if not approval.approved and not allow_unapproved_model:
        fail("check", BatchError(
            f"The '{ctx.model_tag}' model is not approved for lending decisions "
            f"(registry status: {approval.status}).",
            approval.message(),
            f"Score with an approved model ({Path(cfg.paths.registry).as_posix()}), "
            f"or override explicitly -- the outputs are then stamped not for lending "
            f"decisions. To see what approval needs: python "
            f"scripts/07_model_registry.py rules --model-tag {ctx.model_tag}"))
    published = float(d.reject_at_or_above)
    threshold_published = bool(np.isclose(threshold, published, rtol=0, atol=1e-12))
    not_for_lending = []
    if not approval.approved:
        not_for_lending.append(f"model {ctx.model_tag} is {approval.label}")
    if not threshold_published:
        not_for_lending.append(f"threshold {threshold:g} is not the published "
                               f"{published:g}")
    for_lending = not not_for_lending
    feature_screen = tuple(ctx.spec.all_columns) + tuple(
        (ctx.bundle.get("artefacts") or {}).get("gbm_columns") or ())
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
    design_writer, design_meta = None, {}
    row_offset = 0
    n_blocks = 0
    said_clean = said_score = said_profile = False
    # One decision, and at most one notice, per applicant: repeats are removed
    # before cleaning, across block boundaries too, and counted in the report.
    duplicates = DuplicateTracker(report.id_column)

    def blocks():
        yield first
        yield from chunks

    for block in blocks():
        n_blocks += 1
        block = block.reset_index(drop=True)
        block, block_validation = validate(
            block, ctx.spec, values=ctx.clean_values, costs=costs,
            mapping=report.recognised or mapping, fill_values=unlearned_fill,
            unlearned_action=unlearned_action)
        n_read = len(block)
        block, dropped = duplicates.drop(block)
        block = block.reset_index(drop=True)
        dropped = {k: v for k, v in dropped.items() if v}
        if block.empty:
            # Every row of this block repeated an earlier one: nothing to score.
            gone = CleaningReport(rows_in=n_read, dropped_by_rule=dropped)
            clean_report = gone if clean_report is None else clean_report.merge(gone)
            continue

        # -------------------------------------------------------- clean ----
        if not said_clean:
            say("clean", "running")
        t_step = time.perf_counter()
        try:
            X, flags, block_report = prepare(block, ctx, fill_values=fills)
        except BatchError as exc:
            fail("clean", exc)
        except Exception as exc:
            fail("clean", BatchError(
                "The file could not be prepared for the model.", repr(exc),
                "Check that numeric columns contain numbers and that there are no "
                "merged header rows."))
        timed("clean", time.perf_counter() - t_step)
        block_report.rows_in = n_read
        # A term is parsed in validate, before anything profiles the frame, so
        # cleaning sees months rather than text and no longer counts the coercion
        # itself. The count is carried over here, so cleaning_report.csv still
        # reports the column as text that was read.
        for col, n_text in block_validation.coerced_text.items():
            block_report.coerced_text[col] = (
                block_report.coerced_text.get(col, 0) + n_text)
        for rule, n in dropped.items():
            block_report.dropped_by_rule[rule] = (
                block_report.dropped_by_rule.get(rule, 0) + n)
        clean_report = (block_report if clean_report is None
                        else clean_report.merge(block_report))
        if not said_clean:
            said_clean = True
            say("clean", "done", "; ".join(clean_report.plain_english()[:2]))

        # Every block feeds the sample, so the profile and the drift check
        # describe the whole file rather than its opening rows.
        if not said_profile:
            said_profile = True
            say("profile", "running")
        t_step = time.perf_counter()
        reservoir.add_block(block)
        timed("profile_and_drift", time.perf_counter() - t_step)

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
        # The decision is taken on exactly the risk that is written. Deciding on
        # the unrounded value let an applicant at 0.29996 be approved while the file
        # showed 0.3, which the published rule rejects: 145 of 1.3M rows on a 450 MB
        # file, caught by decisions_match_threshold (FINDINGS 7m).
        pd12 = np.round(1.0 - surv[:, k12], RISK_DECIMALS)
        pd_h = np.round(1.0 - surv[:, k_h], RISK_DECIMALS)
        if not said_score:
            say("decide", "running", f"reject at {pd_col} >= {threshold:g}")
        decision = np.where(pd_h >= threshold, "reject", "approve")
        ids = (block[report.id_column].astype(str).to_numpy()
               if report.id_column and report.id_column in block.columns
               else np.array([f"APP-{row_offset + i + 1:06d}"
                              for i in range(len(block))]))
        row_ids = np.arange(row_offset + 1, row_offset + len(block) + 1)
        reject_pos = np.flatnonzero(decision == "reject")

        # -------------------------------------------------------- write ----
        t_step = time.perf_counter()
        explained_note = np.where(decision == "reject", REASONS_PENDING,
                                  "not applicable (approved)")
        front = pd.DataFrame({
            "row_id": row_ids,
            "applicant_id": ids,
            "pd_12m": pd12,
            pd_col: pd_h,
            "decision": decision,
            "threshold": threshold,
            "model_tag": ctx.model_tag,
            "model": ctx.model_name,
            "model_status": approval.label,
            "for_lending_decisions": for_lending,
            "explained": explained_note,
            # Which explainer produced this row's reasons; Phase 2 fills it in.
            "explainer": "",
        })
        for name, values in _reason_columns({}, list(range(len(block))), 3).items():
            front[name] = values
        front["features_present"] = len(report.present)
        front["features_missing"] = len(report.missing_optional)
        front["features_filled_from_training"] = (
            "; ".join(report.filled_from_training) or "")
        scored = pd.concat(
            [front, flags.reset_index(drop=True),
             block.drop(columns=[c for c in block.columns if c in front.columns],
                        errors="ignore")], axis=1)

        group = (block[agg.group_column]
                 if agg.group_column and agg.group_column in block.columns else None)
        agg.add(pd_h, decision, group, flags, {})
        agg.add_segments(segment_labels(block, X, report), decision)

        rejected = scored[scored["decision"] == "reject"].copy()
        for col in REJECTED_REASON_COLUMNS:
            rejected[col] = ""
        approved = scored[scored["decision"] == "approve"]

        ensure_dir()
        for path, frame in ((scored_path, scored), (approved_path, approved),
                            (rejected_path, rejected)):
            header = not path.exists()
            frame.to_csv(path, index=False, mode="w" if header else "a",
                         header=header)
        # What Phase 2 explains: these rows' model inputs exactly as scored, keyed
        # by row_id, so the reasons are the reasons for these decisions.
        if len(reject_pos):
            phase2_dir.mkdir(parents=True, exist_ok=True)
            design = X.iloc[reject_pos].copy()
            design.index = pd.Index(row_ids[reject_pos], name="row_id")
            design.insert(0, "__applicant_id", ids[reject_pos])
            design_writer = _save_frame(design, phase2_dir / "rejected_design.parquet",
                                        design_writer, design_meta)
        if preview_kept < PREVIEW_ROWS:
            preview.append(scored.head(PREVIEW_ROWS - preview_kept))
            preview_kept += min(len(scored), PREVIEW_ROWS - preview_kept)

        timed("write_rows", time.perf_counter() - t_step)
        row_offset += len(block)
        said_score = True
        del X, flags, scored, rejected, approved, block
    if design_writer is not None:
        design_writer.close()

    agg.n_reasons_pending = agg.n_rejected
    say("score", "done", f"{agg.n_rows:,} applicants scored")
    say("decide", "done", f"{agg.n_approved:,} approved, {agg.n_rejected:,} rejected")

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
        shutil.copyfile(data, input_copy)     # streamed; never the whole file in memory
    elif isinstance(data, bytes):
        input_copy.write_bytes(data)
    internal_dir = stamp_dir / INTERNAL_DIR
    internal_dir.mkdir(parents=True, exist_ok=True)
    (internal_dir / "README.txt").write_text(INTERNAL_README, encoding="utf-8")
    pd.DataFrame(columns=list(INTERNAL_COLUMNS)).to_csv(internal_dir / INTERNAL_FLAGS,
                                                        index=False)
    (internal_dir / INTERNAL_RECORDS).write_text("", encoding="utf-8")
    files[f"{INTERNAL_DIR}/{INTERNAL_FLAGS}"] = internal_dir / INTERNAL_FLAGS
    files[f"{INTERNAL_DIR}/{INTERNAL_RECORDS}"] = internal_dir / INTERNAL_RECORDS

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

    # ------------------------------------------------------------ checks --
    # Read back from disk, not from memory: the files are what leaves this run.
    say("files", "running", "checking the written outputs")
    say("checks", "running", "reading the written files back")
    quality_inputs = input_quality_inputs(ctx, report)
    checks = verify_run(
        stamp_dir, n_rows_read=agg.n_rows, threshold=threshold,
        published_threshold=published, horizon_months=int(d.horizon_months),
        feature_names=feature_screen, model_approved=approval.approved,
        model_label=approval.message(), allow_unapproved=allow_unapproved_model,
        cap_note=CAP_NOTE, chunk_rows=chunk_rows,
        notices_zip=stamp_dir / "__no_notices_yet__.zip",
        input_quality=quality_inputs)
    files["validation_checks.csv"] = write_checks(
        checks, stamp_dir / "validation_checks.csv")
    blocking = blocking_failures(checks)
    n_pass = sum(c.status == "PASS" for c in checks)
    say("checks", "failed" if blocking else "done",
        f"{len(blocking)} blocking check(s) failed: "
        + ", ".join(c.name for c in blocking) if blocking
        else f"{n_pass} of {len(checks)} passed")
    review_share = float(getattr(d, "fair_lending_review_share", 0.05))
    confirm_above = int(getattr(d, "explain_confirm_above", 1000))

    model_fp = file_fingerprint(ctx.model_path)
    phase2_state = "not needed" if agg.n_rejected == 0 else (
        "awaiting choice" if agg.n_rejected > confirm_above else "not started")
    summary = {
        "run_id": stamp_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_file": Path(filename).name,
        "source_sha256": (file_fingerprint(input_copy).get("sha256")
                          if input_copy.exists() else None),
        "n_rows": agg.n_rows,
        "n_rows_in_file": clean_report.rows_in,
        "n_duplicates_removed": sum(duplicates.counts.values()),
        "duplicates_removed_by_rule": "; ".join(
            f"{k}={v}" for k, v in duplicates.counts.items() if v),
        "duplicate_applicant_examples": "; ".join(duplicates.examples),
        "n_approved": agg.n_approved,
        "n_rejected": agg.n_rejected,
        "approval_rate": round(agg.n_approved / agg.n_rows, 4) if agg.n_rows else 0.0,
        f"mean_pd_{int(d.horizon_months)}m": round(agg.mean_pd, 4),
        "run_status": ("failed_checks" if blocking else
                       "finished" if agg.n_rejected == 0 else "decisions_ready"),
        "phase2_state": phase2_state,
        "phase2_mode": "",
        "explain_confirm_above": confirm_above,
        "for_lending_decisions": for_lending and not blocking,
        "not_for_lending_reasons": "; ".join(
            not_for_lending + ([f"{len(blocking)} blocking check(s) failed"]
                               if blocking else [])),
        "model_tag": ctx.model_tag,
        "model": ctx.model_name,
        "model_registry_status": approval.status,
        "model_approved": approval.approved,
        "model_approval_problems": " ".join(approval.problems),
        "model_override": bool(allow_unapproved_model and not approval.approved),
        "model_sha256": model_fp.get("sha256"),
        "model_trained_on": str(ctx.data_source),
        "threshold": threshold,
        "published_threshold": published,
        "threshold_is_published": threshold_published,
        "horizon_months": int(d.horizon_months),
        "explainer": explainer_name,
        "explain_nsamples": (d.explain_nsamples if explainer_name == "survshap"
                             else None),
        "explain_n_background": (d.explain_n_background if explainer_name == "survshap"
                                 else None),
        "max_explained": None,
        "n_explained": 0,
        "n_reasons_pending": agg.n_rejected,
        "n_rejected_without_reasons": 0,
        "n_explained_without_disclosable_reason": 0,
        "n_pending_manual_review": 0,
        "n_notices": 0,
        # Fair-lending monitoring, on every run; filled in by Phase 2.
        "n_fair_lending_flagged": 0,
        "fair_lending_flag_share": 0.0,
        "n_top_driver_not_disclosable": 0,
        "fair_lending_flag_features": "",
        "fair_lending_review_share": review_share,
        "fair_lending_review_required": False,
        "validation_checks_passed": not blocking,
        "n_checks_failed": sum(c.status == "FAIL" for c in checks),
        "n_checks_overridden": sum(c.status == "OVERRIDDEN" for c in checks),
        "failed_checks": "; ".join(c.name for c in checks if c.status == "FAIL"),
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
        # Kept out of optional_missing_cost on purpose. That column reports measured
        # concordance, and a concordance figure is what let this case through: it is
        # blind to the level shift an unlearned default produces (FINDINGS 7o).
        "n_features_filled_from_training": len(report.filled_from_training),
        "features_filled_from_training": "; ".join(
            f"{k} = {v}" for k, v in report.filled_from_training.items()),
        "features_filled_note": "; ".join(
            f"{k}: absent from file, filled with training "
            f"{'modal value' if isinstance(v, str) else 'median'} {v}, do not treat "
            f"this applicant's result as fully reliable"
            for k, v in report.filled_from_training.items()),
        "features_unlearned_missing": "; ".join(report.unlearned_missing),
        "unlearned_missing_action": report.unlearned_action,
        "unlearned_missing_floor": (report.costs.unlearned_floor
                                    if report.costs is not None else None),
        "unlearned_rule_note": report.unlearned_rule_note,
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
        "explain_workers": None,
        "seconds_by_step": step_seconds,
        "profiled_rows": len(profile_frame),
        "profile_sampling": "uniform random across the whole file",
        "chunk_rows": chunk_rows,
        "blocks": n_blocks,
        "phase1_seconds": None,
        "phase2_seconds": None,
        "seconds_total": None,
    }
    files["run_summary.csv"] = stamp_dir / "run_summary.csv"

    # Everything Phase 2 needs, so it never re-reads the upload or the training
    # data: a Phase 2 process starts in the time it takes to load the model.
    if agg.n_rejected:
        bg_meta: dict = {}
        _save_frame(ctx.background, phase2_dir / "background.parquet",
                    meta=bg_meta).close()
        plan = {
            "run_id": stamp_dir.name,
            "model_tag": ctx.model_tag, "model_name": ctx.model_name,
            "model_path": str(Path(ctx.model_path).resolve()),
            "model_sha256": model_fp.get("sha256"),
            "notice_model_label": f"{ctx.model_name} ({ctx.model_tag}), "
                                  f"{explainer_name}",
            "explainer": explainer_name, "nsamples": int(d.explain_nsamples),
            "n_background": int(d.explain_n_background),
            "seed": int(cfg.explain.seed), "times": [float(t) for t in ctx.times],
            "horizon_months": int(d.horizon_months),
            "threshold": threshold, "published_threshold": published,
            "for_lending_phase1": for_lending,
            "absent_features": sorted(set(report.optional_missing)
                                      | set(report.required_missing)),
            "feature_screen": list(feature_screen),
            "n_rejected": agg.n_rejected,
            "explain_workers": int(getattr(d, "explain_workers", 0)) or None,
            "chunk_rows": chunk_rows,
            "allow_unapproved_model": bool(allow_unapproved_model),
            # Phase 2 re-runs every check; the input-quality one needs these.
            "input_quality": quality_inputs,
            "model_approved": approval.approved,
            "model_label": approval.message(),
            "design": design_meta, "background": bg_meta,
        }
        (phase2_dir / "plan.json").write_text(json.dumps(plan, indent=2),
                                              encoding="utf-8")
        (phase2_dir / "status.json").write_text(json.dumps({
            "state": phase2_state, "mode": "", "n_rejected": agg.n_rejected,
            "target": None, "done": 0, "confirm_above": confirm_above},
            indent=2), encoding="utf-8")

    timed("write_outputs", time.perf_counter() - t_step)
    step_seconds["total"] = round(time.perf_counter() - started, 2)
    summary["seconds_by_step"] = dict(step_seconds)
    summary["phase1_seconds"] = summary["seconds_total"] = step_seconds["total"]
    pd.DataFrame([summary]).to_csv(files["run_summary.csv"], index=False)
    (stamp_dir / "validation_report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    (stamp_dir / "aggregates.json").write_text(
        json.dumps(agg.to_dict(), indent=2), encoding="utf-8")

    if blocking:
        # Not finished: no provenance.json is written, so neither the page nor
        # load_result can present this run as complete, and Phase 2 refuses it.
        lines = [f"{c.name}: {c.detail}" for c in blocking]
        (stamp_dir / "RUN_FAILED_CHECKS.txt").write_text(
            "This run failed its own checks and is NOT finished. Do not use its "
            "outputs.\nNo applicant notices will be produced for it.\n\n"
            + "\n".join(lines) + "\n", encoding="utf-8")
        say("files", "failed", f"{len(blocking)} check(s) failed")
        raise BatchError(
            f"The run failed {len(blocking)} of its own checks, so its outputs are "
            f"not finished and no notices will be produced: "
            + ", ".join(c.name for c in blocking) + ".",
            "\n".join(lines),
            "Nothing from this run may be used. The checks are in "
            "validation_checks.csv in the run folder. "
            + ("input_quality is about the file: supply the missing or unreadable "
               "values named in the Details (cleaning_report.csv shows examples) and "
               "score it again. "
               if any(c.name == "input_quality" for c in blocking) else "")
            + ("Any other failed check is a defect to fix, not a file to correct."
               if any(c.name != "input_quality" for c in blocking) else ""),
            run_dir=stamp_dir)

    stamp = build_stamp(
        stage="batch_score",
        inputs={"upload": input_copy, "model": ctx.model_path,
                "training_data": ctx.data_source},
        outputs=dict(files),
        config_path=PROJECT_ROOT / "config" / "config.yaml",
        args={"threshold": threshold, "model_tag": ctx.model_tag,
              "model": ctx.model_name, "explainer": explainer_name,
              "chunk_rows": chunk_rows, "cleaning_policy": ctx.policy.version,
              "cleaning_values_fitted_rows": ctx.clean_values.fitted_rows,
              "model_registry_status": approval.status,
              "allow_unapproved_model": bool(allow_unapproved_model),
              "for_lending_decisions": summary["for_lending_decisions"]})
    (stamp_dir / "provenance.json").write_text(
        json.dumps({"provenance": stamp, "summary": summary,
                    "cleaning": clean_report.to_dict()}, indent=2),
        encoding="utf-8")
    say("files", "done", f"{len(files)} files written to {stamp_dir.name}")
    from .wsl_sync import copy_back_soon
    copy_back_soon()                  # in the WSL copy only: decisions to Windows

    preview_frame = (pd.concat(preview, ignore_index=True) if preview
                     else pd.DataFrame())
    approved_preview = rejected_preview = preview_frame
    if len(preview_frame):
        approved_preview = preview_frame[preview_frame["decision"] == "approve"]
        rejected_preview = preview_frame[preview_frame["decision"] == "reject"]
    return BatchResult(
        run_dir=stamp_dir, scored=preview_frame, approved=approved_preview,
        rejected=rejected_preview, summary=summary, report=report, files=files,
        seconds=step_seconds["total"], clean_report=clean_report, profile=prof,
        drift=drift_result, aggregates=agg, notices=[],
        checks=pd.read_csv(files["validation_checks.csv"]))


def run_batch(data, filename: str, cfg: Config, *, model_tag: str | None = None,
              model_name: str | None = None, threshold: float | None = None,
              max_explained: int | None = None, runs_dir: Path | None = None,
              run_dir: Path | None = None, chunk_rows: int | None = None,
              explainer: str | None = None, mapping: dict | None = None,
              progress=None, ctx: ScoringContext | None = None,
              allow_unapproved_model: bool = False) -> BatchResult:
    """Both phases, one after the other, in this process: decisions, then reasons.

    Nothing of its own: :func:`score_file` then :func:`creditsurv.phase2.explain_run`,
    with the loaded model handed across so it is not loaded twice. A positive
    ``max_explained`` is the explicit cap (the first N rejected rows, in file
    order); the rest are marked and the run is stamped not for lending decisions.
    """
    from .phase2 import explain_run

    ctx = ctx or load_context(cfg, model_tag, model_name)
    result = score_file(data, filename, cfg, threshold=threshold, runs_dir=runs_dir,
                        run_dir=run_dir, chunk_rows=chunk_rows, explainer=explainer,
                        mapping=mapping, progress=progress, ctx=ctx,
                        allow_unapproved_model=allow_unapproved_model)
    if not result.summary["n_rejected"]:
        return result
    cap = cfg.decision.max_explained if max_explained is None else int(max_explained)
    phase2 = explain_run(result.run_dir, cfg, mode="all",
                         limit=cap if cap and cap > 0 else None, model=ctx.model,
                         model_path=ctx.model_path, progress=progress)
    return phase2.merged_into(result)


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
            id_column=raw.get("id_column"), warnings=raw.get("warnings", []),
            unlearned_missing=raw.get("unlearned_missing", []),
            unlearned_rule_note=raw.get("unlearned_rule_note", ""),
            unlearned_action=raw.get("unlearned_action", "fill"),
            filled_from_training=raw.get("filled_from_training", {}) or {})

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
            names = zf.namelist()
            # Up to 5,000 read into memory; beyond that the zip is the place to read
            # them, and loading them all would make opening a large run slow.
            if len(names) <= 5_000:
                notices = [(n, zf.read(n).decode("utf-8", "replace")) for n in names]

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
        files={name: run_dir / name for name in
               OUTPUT_NAMES + (f"{INTERNAL_DIR}/{INTERNAL_FLAGS}",
                               f"{INTERNAL_DIR}/{INTERNAL_RECORDS}")
               if (run_dir / name).exists()},
        checks=read("validation_checks.csv"),
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
    """Every output file of one run, as a single in-memory ZIP, for the operator.

    The internal records keep their ``internal/`` folder inside the bundle, so they
    stay separate from the applicant notices there too. Zipped while Phase 2 is
    unfinished, the bundle leads with ``PARTIAL_SNAPSHOT.txt`` saying how many
    reasons are still pending: the files alone do not make that obvious.
    """
    from .phase2 import snapshot, snapshot_note        # phase2 imports this module

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        snap = snapshot(result.run_dir)                 # read now, not at render
        if snap["partial"]:
            zf.writestr(PARTIAL_SNAPSHOT, snapshot_note(
                snap, f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
        for path in sorted(result.run_dir.iterdir()):
            if path.is_file():
                zf.write(path, path.name)
        internal = result.run_dir / INTERNAL_DIR
        if internal.is_dir():
            for path in sorted(internal.iterdir()):
                if path.is_file():
                    zf.write(path, f"{INTERNAL_DIR}/{path.name}")
    return buf.getvalue()


def __getattr__(name: str):
    """Phase 2 lives in :mod:`creditsurv.phase2` and is exported from here too, so a
    caller importing the scoring API gets both phases from one place. Resolved on
    first use -- the same function object, not a wrapper -- because phase2 imports
    this module."""
    if name == "explain_run":
        from .phase2 import explain_run
        return explain_run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
