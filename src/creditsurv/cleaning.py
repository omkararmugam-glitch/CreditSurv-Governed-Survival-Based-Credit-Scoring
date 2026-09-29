"""The single source of truth for cleaning, shared by training and scoring.

Before this module, cleaning lived in three places: the ingest coercion in
:mod:`creditsurv.io.loaders`, the label-stage parsing in
:mod:`creditsurv.labeling.survival_target`, and the encoding in
:mod:`creditsurv.features.build` -- and the dashboard's scoring path repeated
part of the first one differently, so ``$85,000`` parsed to 85000.0 in training
and to NaN when scoring. Everything that happens to a *value* between the raw
file and the design matrix is now decided here, by both callers.

Division of labour, deliberately narrow so nothing moved twice:

* **This module** owns value-level cleaning: numeric text coercion, category
  alignment, optional clipping and rare-level pooling, duplicate rows, and the
  report of what it did.
* **build_design_matrix** keeps owning *encoding*: one-hot vs native categorical,
  the Cox median fill and standardisation, missing-indicator columns. Those
  already learn on training and reapply saved values, so they were not moved.
* **Row exclusions for labelling** (unusable status, missing dates, term overrun)
  stay in the labelling stage, since they need the outcome. Their counts are read
  into the report when the Stage 1 audit is available.

Learned values -- category levels, percentile ranges, medians, clip bounds -- are
fitted on **training data only** (:func:`fit_values`), saved next to the model,
and reapplied unchanged when scoring an upload (:func:`clean`). Nothing is ever
refitted on an uploaded file; :func:`clean` has no code path that can.

Policy version
--------------
:data:`PARITY_VERSION` is what the trained models were built under: coerce
numeric text, align categories to the training levels, flag anything unusual, and
change no value. Every rule that *would* alter a model input (clipping, rare-level
pooling, dropping duplicates) exists here but defaults to off, so turning one on
is a visible config change that invalidates the models rather than a silent drift.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["PARITY_VERSION", "CleaningPolicy", "CleaningValues", "CleaningReport",
           "fit_values", "clean", "coerce_numeric", "coerce_column",
           "parse_term_months", "DuplicateTracker", "policy_from_config"]

PARITY_VERSION = "v1-parity"
"""The rule set the current trained models were produced under."""

NA_TOKENS: tuple[str, ...] = ("", "n/a", "N/A", "NA", "null", "NULL", "none", "None")
"""Same tokens the ingest reader treats as missing, so a value that was missing in
training is missing when scoring."""

DUPLICATE_RULES: dict[str, str] = {
    "duplicate_row": "identical to an earlier row",
    "duplicate_applicant_id": "repeating an earlier applicant ID with different values",
}
"""Rows removed when scoring so that no applicant is decided, or sent a notice,
twice. Keys are the ``dropped_by_rule`` names in the cleaning report."""


@dataclass(frozen=True)
class CleaningPolicy:
    """Every cleaning rule, with the reason it exists.

    Defaults reproduce the behaviour the trained models were built under. The
    three rules that would change a model input are off by default and are
    proposals until approved.
    """

    version: str = PARITY_VERSION

    coerce_numeric_text: bool = True
    """Strip thousands separators, currency symbols and percent signs before
    converting to a number. Why: Lending Club's own export writes ``$85,000`` and
    ``45.6%``; the ingest already did this, so scoring must too or the same
    applicant is cleaned two different ways."""

    align_categories: bool = True
    """Map categorical values onto the levels seen in training. Why: a model
    cannot use a level it never saw, and silently re-coding categories is how a
    scoring path starts answering a different question from the trained one."""

    unseen_category_to_other: bool = False
    """OFF = an unseen level becomes missing and is counted (current behaviour).
    ON = it is pooled into 'other'. Changing this changes model inputs."""

    rare_category_min_count: int = 0
    """0 = no pooling (current behaviour). Above 0, training levels rarer than
    this are pooled into 'other'. Changing this changes model inputs."""

    clip_numeric: bool = False
    """OFF = no value is ever altered (current behaviour); out-of-range values are
    flagged only. ON = clip to the fitted quantile bounds. Why it might be wanted:
    annual_inc runs into the millions. Changing this changes model inputs."""

    clip_quantiles: tuple[float, float] = (0.001, 0.999)
    range_quantiles: tuple[float, float] = (0.005, 0.995)
    """Bounds used for *flagging* out-of-range values. Flagging alters nothing."""

    drop_duplicate_ids: bool = False
    """OFF = duplicates are counted and kept (current behaviour). Changing this
    changes the training rows."""

    def changes_model_inputs(self) -> bool:
        return bool(self.unseen_category_to_other or self.rare_category_min_count
                    or self.clip_numeric or self.drop_duplicate_ids)


def policy_from_config(cfg) -> CleaningPolicy:
    """Read the ``cleaning:`` config section, if the config carries one."""
    section = getattr(cfg, "cleaning", None)
    if section is None:
        return CleaningPolicy()
    return CleaningPolicy(**{k: v for k, v in asdict(section).items()})


@dataclass
class CleaningValues:
    """Everything learned from the training data, saved with the model.

    Refitting these on an upload would mean scoring each file against itself, so
    :func:`clean` only ever consumes them.
    """

    version: str = PARITY_VERSION
    fitted_rows: int = 0
    fitted_on: str = ""
    categories: dict[str, list[str]] = field(default_factory=dict)
    rare_levels: dict[str, list[str]] = field(default_factory=dict)
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    clip_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    medians: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"version": self.version, "fitted_rows": self.fitted_rows,
                "fitted_on": self.fitted_on, "categories": self.categories,
                "rare_levels": self.rare_levels,
                "ranges": {k: list(v) for k, v in self.ranges.items()},
                "clip_bounds": {k: list(v) for k, v in self.clip_bounds.items()},
                "medians": self.medians}

    @classmethod
    def from_dict(cls, payload: dict) -> "CleaningValues":
        return cls(version=payload.get("version", PARITY_VERSION),
                   fitted_rows=payload.get("fitted_rows", 0),
                   fitted_on=payload.get("fitted_on", ""),
                   categories={k: list(v) for k, v in (payload.get("categories") or {}).items()},
                   rare_levels={k: list(v) for k, v in (payload.get("rare_levels") or {}).items()},
                   ranges={k: (float(v[0]), float(v[1]))
                           for k, v in (payload.get("ranges") or {}).items()},
                   clip_bounds={k: (float(v[0]), float(v[1]))
                                for k, v in (payload.get("clip_bounds") or {}).items()},
                   medians={k: float(v) for k, v in (payload.get("medians") or {}).items()})


@dataclass
class CleaningReport:
    """What cleaning did to one frame. Written as 00_cleaning_report_<tag>.json
    by the training stage and as cleaning_report.csv by the dashboard."""

    rows_in: int = 0
    rows_out: int = 0
    policy_version: str = PARITY_VERSION
    values_version: str = PARITY_VERSION
    values_fitted_rows: int = 0
    dropped_by_rule: dict[str, int] = field(default_factory=dict)
    coerced_text: dict[str, int] = field(default_factory=dict)
    unreadable_numbers: dict[str, int] = field(default_factory=dict)
    unreadable_examples: dict[str, list[str]] = field(default_factory=dict)
    """Up to three raw values per column that could not be read, so a report of
    "N unreadable" says what they looked like."""
    missing_after: dict[str, int] = field(default_factory=dict)
    """Per model feature present in the file: values missing once cleaned (blank in
    the file, unreadable, or an unseen category). Read by the input-quality check."""
    imputed: dict[str, int] = field(default_factory=dict)
    clipped: dict[str, int] = field(default_factory=dict)
    out_of_range: dict[str, int] = field(default_factory=dict)
    unseen_categories: dict[str, dict[str, int]] = field(default_factory=dict)
    pooled_to_other: dict[str, int] = field(default_factory=dict)
    missing_columns: list[str] = field(default_factory=list)
    label_stage_drops: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def merge(self, other: "CleaningReport") -> "CleaningReport":
        """Combine two reports, for a file cleaned in blocks. Counts add up, so a
        chunked run reports exactly what a single-pass run would."""
        def add(a: dict, b: dict) -> dict:
            out = dict(a)
            for k, v in b.items():
                out[k] = out.get(k, 0) + v
            return out

        merged = CleaningReport(
            rows_in=self.rows_in + other.rows_in,
            rows_out=self.rows_out + other.rows_out,
            policy_version=other.policy_version or self.policy_version,
            values_version=other.values_version or self.values_version,
            values_fitted_rows=max(self.values_fitted_rows, other.values_fitted_rows),
            dropped_by_rule=add(self.dropped_by_rule, other.dropped_by_rule),
            coerced_text=add(self.coerced_text, other.coerced_text),
            unreadable_numbers=add(self.unreadable_numbers, other.unreadable_numbers),
            unreadable_examples={
                col: list(dict.fromkeys(self.unreadable_examples.get(col, [])
                                        + other.unreadable_examples.get(col, [])))[:3]
                for col in set(self.unreadable_examples) | set(other.unreadable_examples)},
            missing_after=add(self.missing_after, other.missing_after),
            imputed=add(self.imputed, other.imputed),
            clipped=add(self.clipped, other.clipped),
            out_of_range=add(self.out_of_range, other.out_of_range),
            missing_columns=self.missing_columns or other.missing_columns,
            label_stage_drops=self.label_stage_drops or other.label_stage_drops)
        for col in set(self.unseen_categories) | set(other.unseen_categories):
            merged.unseen_categories[col] = add(
                self.unseen_categories.get(col, {}), other.unseen_categories.get(col, {}))
        merged.pooled_to_other = add(self.pooled_to_other, other.pooled_to_other)
        return merged

    def to_frame(self) -> pd.DataFrame:
        """One row per (rule, column) action, for cleaning_report.csv."""
        rows = [{"scope": "rows", "rule": "rows_in", "column": "", "count": self.rows_in},
                {"scope": "rows", "rule": "rows_out", "column": "", "count": self.rows_out}]
        for rule, counts in (("dropped", self.dropped_by_rule),
                             ("label_stage_dropped", self.label_stage_drops)):
            rows += [{"scope": "rows", "rule": rule, "column": k, "count": v}
                     for k, v in counts.items()]
        rows += [{"scope": "values", "rule": "unreadable_number", "column": k,
                  "count": v, "detail": "e.g. " + ", ".join(
                      repr(x) for x in self.unreadable_examples.get(k, []))}
                 for k, v in self.unreadable_numbers.items()]
        for rule, counts in (("text_coerced", self.coerced_text),
                             ("missing_after_cleaning",
                              {k: v for k, v in self.missing_after.items() if v}),
                             ("imputed", self.imputed),
                             ("clipped", self.clipped),
                             ("out_of_range_flagged", self.out_of_range),
                             ("pooled_to_other", self.pooled_to_other)):
            rows += [{"scope": "values", "rule": rule, "column": k, "count": v}
                     for k, v in counts.items()]
        for col, levels in self.unseen_categories.items():
            rows += [{"scope": "values", "rule": "unseen_category", "column": col,
                      "count": n, "detail": level} for level, n in levels.items()]
        rows += [{"scope": "columns", "rule": "absent_from_input", "column": c, "count": 0}
                 for c in self.missing_columns]
        return pd.DataFrame(rows)

    def plain_english(self) -> list[str]:
        """The report as sentences, for the dashboard."""
        out: list[str] = []
        if self.rows_in != self.rows_out:
            out.append(f"{self.rows_in - self.rows_out:,} of {self.rows_in:,} rows were "
                       f"removed; {self.rows_out:,} were kept.")
        else:
            out.append(f"All {self.rows_in:,} rows were kept.")
        dupes = {k: v for k, v in self.dropped_by_rule.items()
                 if k in DUPLICATE_RULES and v}
        if dupes:
            out.append(f"{sum(dupes.values()):,} duplicate applicant row(s) were removed "
                       f"({', '.join(f'{v:,} {DUPLICATE_RULES[k]}' for k, v in dupes.items())}"
                       f"); each applicant is scored and decided once.")
        for col, n in sorted(self.coerced_text.items(), key=lambda kv: -kv[1])[:5]:
            out.append(f"{n:,} values in {col} were written as text (currency, "
                       f"percentage, thousands separators or a unit such as "
                       f"'36 months') and were read as numbers.")
        for col, n in sorted(self.unreadable_numbers.items(), key=lambda kv: -kv[1])[:5]:
            eg = ", ".join(repr(x) for x in self.unreadable_examples.get(col, []))
            out.append(f"{n:,} values in {col} could not be read as numbers and are "
                       f"treated as missing" + (f" (e.g. {eg})." if eg else "."))
        for col, n in sorted(self.imputed.items(), key=lambda kv: -kv[1])[:5]:
            out.append(f"{n:,} missing values in {col} were filled with the training "
                       f"median.")
        for col, n in sorted(self.clipped.items(), key=lambda kv: -kv[1])[:5]:
            out.append(f"{n:,} values in {col} were outside the training range and "
                       f"were clipped to it.")
        for col, n in sorted(self.out_of_range.items(), key=lambda kv: -kv[1])[:5]:
            out.append(f"{n:,} values in {col} are outside the range seen in training. "
                       f"They were scored unchanged and flagged.")
        for col, levels in list(self.unseen_categories.items())[:5]:
            names = ", ".join(list(levels)[:4])
            out.append(f"{sum(levels.values()):,} rows in {col} use categories the "
                       f"model never saw ({names}).")
        if self.missing_columns:
            out.append(f"{len(self.missing_columns)} model feature(s) are absent from "
                       f"this file and are treated as missing: "
                       f"{', '.join(self.missing_columns[:6])}"
                       + ("..." if len(self.missing_columns) > 6 else ""))
        return out


def coerce_numeric(values: pd.Series, *, strip_text: bool = True,
                   dtype: str = "float32") -> tuple[pd.Series, int]:
    """Text -> float32, the same way the ingest does it.

    Returns the converted series and how many values needed the text stripping
    (they would have become missing without it). ``dtype="float64"`` is for the
    derivations, which have always computed in double precision.
    """
    if not strip_text or not (values.dtype == object or str(values.dtype) in
                              ("string", "string[python]")):
        return pd.to_numeric(values, errors="coerce").astype(dtype), 0
    text = values.astype("string").str.strip()
    text = text.mask(text.isin(NA_TOKENS))
    direct = pd.to_numeric(text, errors="coerce")
    stripped = (text.str.replace(",", "", regex=False)
                    .str.replace("%", "", regex=False)
                    .str.replace("$", "", regex=False))
    parsed = pd.to_numeric(stripped, errors="coerce")
    rescued = int((direct.isna() & parsed.notna()).sum())
    return parsed.astype(dtype), rescued


_TERM = re.compile(r"^(\d{1,3})(?:\.0+)?\s*(?:-?\s*(?:months?|mos?\.?|mths?|m))?$")
"""A loan term in months: ``36``, ``36.0``, ``36 months``, ``36-month``, ``36 mo``.
Matched against the stripped, lower-cased value, so case and padding never matter."""


def parse_term_months(values: pd.Series) -> pd.Series:
    """Loan term text -> months (float), each value read on its own.

    A value that is not a term (``"three years"``, ``"36 weeks"``) becomes missing
    by itself; it cannot take the rest of the column with it.
    """
    if not (values.dtype == object or str(values.dtype).startswith("string")):
        return pd.to_numeric(values, errors="coerce").astype("float64")
    text = values.astype("string").str.strip()
    text = text.mask(text.isin(NA_TOKENS)).str.lower()
    months = text.str.extract(_TERM, expand=False)
    return pd.to_numeric(months, errors="coerce").astype("float64")


COLUMN_PARSERS = {"term_months": parse_term_months}
"""Numeric features whose values carry a unit. Every other numeric feature goes
through :func:`coerce_numeric`."""


def coerce_column(col: str, values: pd.Series, *,
                  strip_text: bool = True) -> tuple[pd.Series, int]:
    """:func:`coerce_numeric`, or the column's own parser if it has one.

    Returns the float32 series and how many values were text that only the
    stripping or the parser made readable (reported as ``text_coerced``).
    """
    parser = COLUMN_PARSERS.get(col)
    if parser is None or not strip_text:
        return coerce_numeric(values, strip_text=strip_text)
    parsed = parser(values)
    direct = pd.to_numeric(values, errors="coerce")
    rescued = int((direct.isna() & parsed.notna()).sum())
    return parsed.astype("float32"), rescued


class DuplicateTracker:
    """Finds repeated applicants across every block of a file.

    A row is a duplicate if it is identical to an earlier row, or if it repeats an
    earlier applicant ID. The first occurrence is kept. The tracker remembers what
    it has seen, so a duplicate is caught even when the two rows fall in different
    blocks, and it holds one 64-bit hash per row rather than the rows themselves.
    """

    def __init__(self, id_column: str | None = None):
        self.id_column = id_column
        self._ids: set[str] = set()
        self._rows: set[int] = set()
        self.counts: dict[str, int] = {k: 0 for k in DUPLICATE_RULES}
        self.examples: list[str] = []

    def drop(self, block: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        """``block`` without the rows already seen, and this block's counts."""
        # Hash the text of each row, so a column read as int in one block and as
        # float in another cannot make the same row look different.
        hashes = pd.util.hash_pandas_object(
            block.astype("string").fillna("\x00"), index=False).to_numpy()
        ids = None
        if self.id_column and self.id_column in block.columns:
            ids = block[self.id_column].astype("string").str.strip()
            ids = ids.mask(ids.isin(NA_TOKENS)).to_numpy()
        keep = np.ones(len(block), dtype=bool)
        counts = {k: 0 for k in DUPLICATE_RULES}
        for i, h in enumerate(hashes):
            app = ids[i] if ids is not None else None
            has_id = app is not None and not pd.isna(app)
            if h in self._rows:
                keep[i] = False
                counts["duplicate_row"] += 1
            elif has_id and app in self._ids:
                keep[i] = False
                counts["duplicate_applicant_id"] += 1
            else:
                self._rows.add(h)
            if has_id:
                if not keep[i] and len(self.examples) < 5 and app not in self.examples:
                    self.examples.append(str(app))
                self._ids.add(app)
        for k, v in counts.items():
            self.counts[k] += v
        return block.loc[keep], counts


def fit_values(df: pd.DataFrame, spec, *, policy: CleaningPolicy | None = None,
               source: str | Path = "") -> CleaningValues:
    """Learn cleaning values from **training data**.

    Called by the training stage only. The dashboard loads what this produced.
    """
    policy = policy or CleaningPolicy()
    values = CleaningValues(version=policy.version, fitted_rows=len(df),
                            fitted_on=str(source))
    lo_r, hi_r = policy.range_quantiles
    lo_c, hi_c = policy.clip_quantiles
    for col in spec.numeric:
        if col not in df.columns:
            continue
        v, _ = coerce_column(col, df[col])
        if not v.notna().any():
            continue
        values.ranges[col] = (float(v.quantile(lo_r)), float(v.quantile(hi_r)))
        values.clip_bounds[col] = (float(v.quantile(lo_c)), float(v.quantile(hi_c)))
        values.medians[col] = float(v.median())
    for col in spec.categorical:
        if col not in df.columns:
            continue
        counts = df[col].astype("string").str.strip().value_counts()
        values.categories[col] = [str(c) for c in counts.index]
        if policy.rare_category_min_count:
            values.rare_levels[col] = [str(c) for c in
                                       counts[counts < policy.rare_category_min_count].index]
    return values


def clean(df: pd.DataFrame, spec, values: CleaningValues, *,
          policy: CleaningPolicy | None = None,
          label_audit: dict | None = None) -> tuple[pd.DataFrame, CleaningReport, pd.DataFrame]:
    """Apply the fitted values to a frame. Never fits anything.

    Returns the cleaned frame, the report, and a per-row flag frame (out-of-range,
    unreadable numbers, unseen categories) for row-level output columns.
    """
    policy = policy or CleaningPolicy()
    work = df.copy()
    rep = CleaningReport(rows_in=len(df), policy_version=policy.version,
                         values_version=values.version,
                         values_fitted_rows=values.fitted_rows)
    if label_audit:
        rep.label_stage_drops = {str(k): int(v) for k, v in
                                 (label_audit.get("drop_counts") or {}).items()}
    flags_out, flags_bad, flags_unseen = [], [], []

    # ------------------------------------------------------------- numeric --
    for col in spec.numeric:
        if col not in work.columns:
            rep.missing_columns.append(col)
            continue
        original = work[col]
        numeric, rescued = coerce_column(col, original,
                                         strip_text=policy.coerce_numeric_text)
        if rescued:
            rep.coerced_text[col] = rescued
        # Per value: a blank or NA token is missing, not unreadable.
        blank = original.astype("string").str.strip().isin(NA_TOKENS).fillna(False)
        unreadable = numeric.isna() & original.notna() & ~blank
        if unreadable.any():
            rep.unreadable_numbers[col] = int(unreadable.sum())
            rep.unreadable_examples[col] = [
                str(x) for x in original[unreadable].astype(str).unique()[:3]]
        flags_bad.append(unreadable.rename(col))

        if policy.clip_numeric and col in values.clip_bounds:
            lo, hi = values.clip_bounds[col]
            outside = ((numeric < lo) | (numeric > hi)).fillna(False)
            if outside.any():
                rep.clipped[col] = int(outside.sum())
            numeric = numeric.clip(lo, hi)

        lo, hi = values.ranges.get(col, (-np.inf, np.inf))
        outside = ((numeric < lo) | (numeric > hi)).fillna(False)
        if outside.any():
            rep.out_of_range[col] = int(outside.sum())
        flags_out.append(outside.rename(col))
        work[col] = numeric
        rep.missing_after[col] = int(numeric.isna().sum())

    # --------------------------------------------------------- categorical --
    for col in spec.categorical:
        if col not in work.columns:
            rep.missing_columns.append(col)
            continue
        text = work[col].astype("string").str.strip()
        levels = list(values.categories.get(col, []))
        if not levels or not policy.align_categories:
            work[col] = text.astype("category")
            rep.missing_after[col] = int(work[col].isna().sum())
            flags_unseen.append(pd.Series(False, index=work.index, name=col))
            continue

        if policy.rare_category_min_count and values.rare_levels.get(col):
            rare = set(values.rare_levels[col])
            hits = text.isin(rare)
            if hits.any():
                rep.pooled_to_other[col] = int(hits.sum())
            text = text.mask(hits, "other")
            levels = [lv for lv in levels if lv not in rare] + ["other"]

        unseen = text.notna() & ~text.isin(levels)
        if unseen.any():
            rep.unseen_categories[col] = {
                str(k): int(v) for k, v in text[unseen].value_counts().head(10).items()}
        if policy.unseen_category_to_other:
            if "other" not in levels:
                levels = levels + ["other"]
            text = text.mask(unseen, "other")
        flags_unseen.append(unseen.rename(col))
        # Anything still outside the training levels becomes missing, which is
        # what the model does with an unknown level anyway -- but it is counted.
        work[col] = pd.Categorical(text, categories=levels)
        rep.missing_after[col] = int(work[col].isna().sum())

    # ---------------------------------------------------------------- rows --
    if policy.drop_duplicate_ids and "id" in work.columns:
        dupes = work["id"].duplicated(keep="first")
        if dupes.any():
            rep.dropped_by_rule["duplicate_id"] = int(dupes.sum())
            work = work.loc[~dupes]
            flags_out = [f.loc[~dupes] for f in flags_out]
            flags_bad = [f.loc[~dupes] for f in flags_bad]
            flags_unseen = [f.loc[~dupes] for f in flags_unseen]
    elif "id" in work.columns:
        n_dupes = int(work["id"].duplicated().sum())
        if n_dupes:
            rep.dropped_by_rule["duplicate_id_kept"] = n_dupes

    rep.rows_out = len(work)
    flags = pd.DataFrame(index=work.index)

    def _summarise(frames, prefix):
        if not frames:
            flags[f"{prefix}_fields"] = ""
            flags[f"n_{prefix}"] = 0
            return
        m = pd.concat(frames, axis=1).fillna(False)
        flags[f"{prefix}_fields"] = m.apply(
            lambda r: ", ".join(m.columns[r.to_numpy(dtype=bool)][:5]), axis=1)
        flags[f"n_{prefix}"] = m.sum(axis=1).astype(int)

    _summarise(flags_out, "out_of_range")
    _summarise(flags_bad, "unreadable_numbers")
    _summarise(flags_unseen, "unknown_categories")
    return work, rep, flags
