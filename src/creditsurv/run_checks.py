"""Checks every scoring run passes before its outputs are marked finished.

Test 1 showed that a run can finish, write every file and look healthy while
166 of its 255 applicant notices carry internal text. So after the files are
written, and before ``provenance.json`` marks the run finished, the run reads its
own outputs back **from disk** -- not from the in-memory state that produced them --
and verifies:

=================================  ============================================
check                              what it verifies
=================================  ============================================
counts_add_up                      approved + rejected = rows read, in every file
decisions_match_threshold          every decision is (risk >= the run's threshold)
threshold_is_published             that threshold is the published policy
risk_12m_le_<H>m                   12-month risk <= decision-horizon risk, every row
rejected_have_reasons_or_pending   every rejection has reasons, or says pending
no_nondisclosable_stated_reason    no stated reason rests on geography or LC's score
applicant_notices_clean            no notice carries internal content or a feature
model_approved                     the model is approved in the registry
=================================  ============================================

A **blocking** failure stops the run: ``validation_checks.csv`` is written, the
notices are withheld (renamed ``adverse_action_notices.WITHHELD.zip``) and
``provenance.json`` is not, so no page or script can load the run as finished.

Two checks can instead end ``OVERRIDDEN``: a model that is not approved, and a
threshold other than the published one, when the operator chose that explicitly.
Those runs finish but are stamped "not for lending decisions" everywhere.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .explain.adverse_action import (MAX_PRINCIPAL_REASONS, NOT_DISCLOSABLE,
                                     REASON_TEMPLATES, SCORE_FEATURES,
                                     find_internal_content)

__all__ = ["CheckResult", "CHECK_DESCRIPTIONS", "verify_run", "write_checks",
           "blocking_failures", "NO_REASON_NOTE", "PENDING_NOTES", "REASONS_PENDING",
           "SKIPPED_NOTE", "OUTSIDE_SAMPLE_NOTE"]

NO_REASON_NOTE = "pending manual review: no disclosable adverse reason"

REASONS_PENDING = "reasons pending"
"""Phase 2 has not reached this rejected applicant yet. Decisions are written in
Phase 1, in minutes; reasons follow in Phase 2 and replace this mark row by row."""
SKIPPED_NOTE = "reasons not generated: explanation skipped (not for lending decisions)"
OUTSIDE_SAMPLE_NOTE = ("reasons not generated: outside the random sample explained "
                       "(not for lending decisions)")

PENDING_NOTES: tuple[str, ...] = (REASONS_PENDING, SKIPPED_NOTE, OUTSIDE_SAMPLE_NOTE,
                                  NO_REASON_NOTE)
"""Every mark a rejected row without stated reasons may carry, besides the cap note.
A rejected row with neither reasons nor one of these fails the checks."""
"""A rejected applicant whose every adverse driver was undisclosable or had no
Regulation B wording. No applicant notice is issued -- one reading "no individual
factor was materially adverse" would not satisfy 12 CFR 1002.9(b)(2) -- so the row
says so and the internal record carries the drivers for a person to review."""

CHECK_DESCRIPTIONS: dict[str, str] = {
    "counts_add_up": "approved + rejected = total rows, in every output file",
    "decisions_match_threshold": "every decision matches the run's threshold",
    "threshold_is_published": "the run's threshold is the published policy",
    "risk_12m_le_horizon": "12-month risk <= decision-horizon risk for every row",
    "rejected_have_reasons_or_pending": "every rejected applicant has reasons or "
                                        "is marked pending",
    "no_nondisclosable_stated_reason": "no stated reason uses a non-disclosable or "
                                       "score feature",
    "applicant_notices_clean": "no applicant notice contains internal content",
    "model_approved": "the model is approved in the registry",
}


@dataclass
class CheckResult:
    name: str
    status: str            # PASS | FAIL | OVERRIDDEN
    blocking: bool
    detail: str
    description: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


def _ok(name, detail, key=None) -> CheckResult:
    return CheckResult(name, "PASS", False, detail, CHECK_DESCRIPTIONS[key or name])


def _fail(name, detail, key=None, blocking=True) -> CheckResult:
    return CheckResult(name, "FAIL", blocking, detail, CHECK_DESCRIPTIONS[key or name])


def _examples(ids) -> str:
    ids = list(ids)
    return ", ".join(map(str, ids[:5])) + (f" and {len(ids) - 5} more" if len(ids) > 5
                                           else "")


def _is_blank(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().isin(["", "nan"])


def verify_run(run_dir: Path, *, n_rows_read: int, threshold: float,
               published_threshold: float, horizon_months: int, feature_names,
               model_approved: bool, model_label: str, allow_unapproved: bool,
               cap_note: str, chunk_rows: int = 50_000,
               notices_zip: Path | None = None) -> list[CheckResult]:
    """Every check, computed from the files in ``run_dir``."""
    run_dir = Path(run_dir)
    pd_h = f"pd_{int(horizon_months)}m"
    scored_path = run_dir / "scored_applicants.csv"
    reason_cols = [f"reason_{i}" for i in range(1, 4)]
    feat_cols = [f"reason_{i}_feature" for i in range(1, 4)]
    wanted = {"applicant_id", "decision", "threshold", "pd_12m", pd_h,
              *reason_cols, *feat_cols}

    n = n_approve = n_reject = 0
    bad_decision_value, bad_threshold, wrong_decision, missing_risk = [], [], [], []
    non_monotone = []
    stated_bad_feature, stated_unknown_reason = [], []
    allowed_reasons = {t["reason"] for t in REASON_TEMPLATES.values()}
    barred = set(NOT_DISCLOSABLE) | set(SCORE_FEATURES)

    def barred_feature(name: str) -> bool:
        return name in barred or name.startswith("grade")

    for block in pd.read_csv(scored_path, chunksize=max(int(chunk_rows), 1),
                             usecols=lambda c: c in wanted, dtype={"applicant_id": str},
                             keep_default_na=True):
        ids = block["applicant_id"].astype(str).to_numpy()
        n += len(block)
        dec = block["decision"].astype(str)
        n_approve += int((dec == "approve").sum())
        n_reject += int((dec == "reject").sum())
        bad_decision_value += list(ids[~dec.isin(["approve", "reject"]).to_numpy()])
        thr = pd.to_numeric(block["threshold"], errors="coerce").to_numpy()
        bad_threshold += list(ids[~np.isclose(thr, threshold, rtol=0, atol=1e-12)])
        risk = pd.to_numeric(block[pd_h], errors="coerce").to_numpy()
        missing_risk += list(ids[~np.isfinite(risk)])
        expected = np.where(risk >= threshold, "reject", "approve")
        wrong_decision += list(ids[np.isfinite(risk) & (expected != dec.to_numpy())])
        if horizon_months >= 12:
            r12 = pd.to_numeric(block["pd_12m"], errors="coerce").to_numpy()
            non_monotone += list(ids[np.isfinite(r12) & np.isfinite(risk) & (r12 > risk)])
        for col in feat_cols:
            if col in block.columns:
                vals = block[col].fillna("").astype(str)
                hit = vals.map(barred_feature).to_numpy()
                stated_bad_feature += [f"{i} ({v})" for i, v in
                                       zip(ids[hit], vals.to_numpy()[hit])]
        for col in reason_cols:
            if col in block.columns:
                vals = block[col].fillna("").astype(str)
                hit = ((vals != "") & ~vals.isin(allowed_reasons)).to_numpy()
                stated_unknown_reason += list(ids[hit])

    approved_rows = sum(len(b) for b in pd.read_csv(run_dir / "approved_applicants.csv",
                                                    chunksize=max(int(chunk_rows), 1),
                                                    usecols=["decision"]))
    rejected = pd.read_csv(run_dir / "rejected_applicants.csv",
                           dtype={"applicant_id": str}, keep_default_na=True)

    out: list[CheckResult] = []

    # 1 ------------------------------------------------------------------
    problems = []
    if n != n_rows_read:
        problems.append(f"scored_applicants.csv has {n} rows, {n_rows_read} were read")
    if n_approve + n_reject != n:
        problems.append(f"{n_approve} approved + {n_reject} rejected != {n}")
    if bad_decision_value:
        problems.append(f"decision not approve/reject: {_examples(bad_decision_value)}")
    if approved_rows != n_approve:
        problems.append(f"approved_applicants.csv has {approved_rows} rows, "
                        f"expected {n_approve}")
    if len(rejected) != n_reject:
        problems.append(f"rejected_applicants.csv has {len(rejected)} rows, "
                        f"expected {n_reject}")
    out.append(_fail("counts_add_up", "; ".join(problems)) if problems else
               _ok("counts_add_up", f"{n_approve} approved + {n_reject} rejected = "
                                    f"{n} rows, in all three files"))

    # 2 ------------------------------------------------------------------
    problems = []
    if missing_risk:
        problems.append(f"{len(missing_risk)} row(s) have no {pd_h} risk: "
                        f"{_examples(missing_risk)}")
    if bad_threshold:
        problems.append(f"{len(bad_threshold)} row(s) record another threshold")
    if wrong_decision:
        problems.append(f"{len(wrong_decision)} decision(s) disagree with "
                        f"{pd_h} >= {threshold}: {_examples(wrong_decision)}")
    out.append(_fail("decisions_match_threshold", "; ".join(problems)) if problems else
               _ok("decisions_match_threshold",
                   f"all {n} decisions equal ({pd_h} >= {threshold:g})"))

    # 3 ------------------------------------------------------------------
    if np.isclose(threshold, published_threshold, rtol=0, atol=1e-12):
        out.append(_ok("threshold_is_published",
                       f"{threshold:g} is decision.reject_at_or_above"))
    else:
        out.append(CheckResult(
            "threshold_is_published", "OVERRIDDEN", False,
            f"run used {threshold:g}, the published policy is {published_threshold:g}; "
            f"stamped not for lending decisions",
            CHECK_DESCRIPTIONS["threshold_is_published"]))

    # 4 ------------------------------------------------------------------
    name = f"risk_12m_le_{int(horizon_months)}m"
    if horizon_months < 12:
        out.append(_ok(name, f"horizon {horizon_months}m is shorter than 12m; "
                             f"nothing to compare", "risk_12m_le_horizon"))
    elif non_monotone:
        out.append(_fail(name, f"{len(non_monotone)} row(s) have pd_12m > {pd_h}: "
                               f"{_examples(non_monotone)}", "risk_12m_le_horizon"))
    else:
        out.append(_ok(name, f"pd_12m <= {pd_h} for all {n} rows",
                       "risk_12m_le_horizon"))

    # 5 ------------------------------------------------------------------
    explained = rejected.get("explained", pd.Series(dtype=str)).fillna("").astype(str)
    has_reason = ~_is_blank(rejected.get("reason_1", pd.Series("", index=rejected.index)))
    pending = explained.isin([cap_note, *PENDING_NOTES])
    good = (explained.eq("explained") & has_reason) | pending
    ids = rejected.get("applicant_id", pd.Series(dtype=str)).astype(str)
    if bool(good.all()):
        out.append(_ok("rejected_have_reasons_or_pending",
                       f"{int((explained.eq('explained') & has_reason).sum())} with "
                       f"reasons, {int(pending.sum())} marked pending, of "
                       f"{len(rejected)} rejected"))
    else:
        out.append(_fail("rejected_have_reasons_or_pending",
                         f"{int((~good).sum())} rejected row(s) have neither reasons "
                         f"nor a pending mark: {_examples(ids[~good])}"))

    # 6 ------------------------------------------------------------------
    for i in range(1, MAX_PRINCIPAL_REASONS + 1):
        col = f"reason_{i}_feature"
        if col in rejected.columns:
            vals = rejected[col].fillna("").astype(str)
            hit = vals.map(barred_feature)
            stated_bad_feature += [f"{a} ({v})" for a, v in zip(ids[hit], vals[hit])]
        col = f"reason_{i}"
        if col in rejected.columns:
            vals = rejected[col].fillna("").astype(str)
            stated_unknown_reason += list(ids[(vals != "") & ~vals.isin(allowed_reasons)])
    problems = []
    if stated_bad_feature:
        problems.append(f"{len(stated_bad_feature)} stated reason(s) rest on a barred "
                        f"feature: {_examples(sorted(set(stated_bad_feature)))}")
    if stated_unknown_reason:
        problems.append(f"{len(set(stated_unknown_reason))} applicant(s) have a reason "
                        f"outside the Regulation B templates")
    out.append(_fail("no_nondisclosable_stated_reason", "; ".join(problems)) if problems
               else _ok("no_nondisclosable_stated_reason",
                        "every stated reason is template wording on a disclosable "
                        "feature"))

    # 7 ------------------------------------------------------------------
    zpath = Path(notices_zip or run_dir / "adverse_action_notices.zip")
    expected_files = set(rejected.get("notice_file", pd.Series(dtype=str))
                         .fillna("").astype(str)) - {""}
    problems = []
    names: list[str] = []
    if zpath.exists():
        with zipfile.ZipFile(zpath) as zf:
            names = zf.namelist()
            for fname in names:
                if "/" in fname or "internal" in fname.lower():
                    problems.append(f"{fname}: not an applicant notice")
                    continue
                text = zf.read(fname).decode("utf-8", "replace")
                m = re.search(r"^Applicant ID:\s+(.+)$", text, re.M)
                found = find_internal_content(text, feature_names=feature_names,
                                              applicant_id=m.group(1).strip() if m else None)
                if found:
                    problems.append(f"{fname}: {'; '.join(found)}")
    if set(names) != expected_files:
        missing = sorted(expected_files - set(names))
        extra = sorted(set(names) - expected_files)
        problems.append(f"notices do not match the rejected file (missing "
                        f"{_examples(missing) or 'none'}; unexpected "
                        f"{_examples(extra) or 'none'})")
    out.append(_fail("applicant_notices_clean",
                     f"{len(problems)} problem(s): {_examples(problems)}") if problems
               else _ok("applicant_notices_clean",
                        f"{len(names)} notice(s) read back from {zpath.name}: no "
                        f"internal wording, no feature names, no attributions"))

    # 8 ------------------------------------------------------------------
    if model_approved:
        out.append(_ok("model_approved", model_label))
    elif allow_unapproved:
        out.append(CheckResult("model_approved", "OVERRIDDEN", False,
                               f"{model_label}; explicitly overridden, stamped not for "
                               f"lending decisions", CHECK_DESCRIPTIONS["model_approved"]))
    else:
        out.append(_fail("model_approved", model_label))
    return out


def blocking_failures(results: list[CheckResult]) -> list[CheckResult]:
    return [r for r in results if r.status == "FAIL" and r.blocking]


def write_checks(results: list[CheckResult], path: Path) -> Path:
    pd.DataFrame([{"check": r.name, "status": r.status, "blocking": r.blocking,
                   "what_it_verifies": r.description, "detail": r.detail}
                  for r in results]).to_csv(path, index=False)
    return path
