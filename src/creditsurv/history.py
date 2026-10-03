"""The history of scoring runs, read from the run folders themselves.

Every figure the Overview and the Results viewer show about past runs comes from
here, and everything here comes from files a run wrote: ``provenance.json`` (whose
summary Phase 2 keeps current), ``run_summary.csv`` (written even by a run that
failed its checks), and ``stages.json`` (where a run stopped, if it stopped
early). Nothing is estimated and nothing is kept anywhere else, so deleting a run
folder removes it from the history, and a run copied in from elsewhere joins it.

One read of one small file per run, cached against that file's modification time:
listing a few hundred runs costs a directory scan.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .stages import read_stages

__all__ = ["RunRecord", "list_runs", "get_run", "overview", "run_state",
           "processing_history", "FINISHED_STATES"]

FINISHED_STATES = ("finished", "reasons pending", "explaining", "phase 2 stopped",
                   "awaiting phase 2 choice")
"""Run states whose decisions are final and pass every check: the runs whose
approval rates and flag shares are history rather than work in progress."""

_FIELDS = ("run_id", "created_at", "source_file", "model_tag", "model", "n_rows",
           "n_rows_in_file", "n_duplicates_removed", "n_approved", "n_rejected",
           "approval_rate", "n_explained", "n_notices", "n_fair_lending_flagged",
           "fair_lending_flag_share", "fair_lending_review_required",
           "for_lending_decisions", "not_for_lending_reasons", "model_approved",
           "threshold", "horizon_months", "failed_checks", "n_checks_failed",
           "drift_status", "phase2_state", "phase2_mode", "phase1_seconds",
           "phase2_seconds", "run_status", "n_reasons_pending")


@dataclass
class RunRecord:
    """One scoring run as the history sees it."""

    run_id: str
    path: Path
    state: str
    """finished, reasons pending, explaining, awaiting phase 2 choice, phase 2
    stopped, failed checks, refused, running, interrupted."""
    summary: dict
    error: dict | None = None

    def to_dict(self) -> dict:
        out = {k: self.summary.get(k) for k in _FIELDS}
        out.update(run_id=self.run_id, state=self.state, error=self.error,
                   created_at=self.summary.get("created_at") or _stamp(self.run_id))
        return out


def _stamp(run_id: str) -> str | None:
    """``20260928_173635_name`` -> ``2026-09-28T17:36:35``."""
    try:
        d, t = run_id.split("_")[:2]
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}"
    except (ValueError, IndexError):
        return None


def _number(v):
    if v in (None, ""):
        return None
    if v in ("True", "False"):
        return v == "True"
    if isinstance(v, str) and v[:1] in "{[":        # a dict or list the CSV wrote as text
        import ast
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return int(f) if f.is_integer() and "." not in str(v) else f


_CACHE: dict[str, tuple[int, dict]] = {}


def _summary(run_dir: Path) -> dict:
    """The run's summary: provenance.json's when the run finished (Phase 2 keeps it
    current), else run_summary.csv's (a run that failed its checks writes only
    that). Empty when the run wrote neither."""
    for name in ("provenance.json", "run_summary.csv"):
        path = run_dir / name
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            continue
        key = str(path)
        hit = _CACHE.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
        try:
            if name == "provenance.json":
                s = json.loads(path.read_text(encoding="utf-8")).get("summary") or {}
            else:
                with path.open(encoding="utf-8", newline="") as fh:
                    row = next(csv.DictReader(fh), {}) or {}
                s = {k: _number(v) for k, v in row.items()}
        except (OSError, ValueError, StopIteration):
            continue
        _CACHE[key] = (mtime, s)
        return s
    return {}


def run_state(run_dir: Path, summary: dict, *, live: bool = False) -> tuple[str, dict | None]:
    """The run's state in one word or two, and the error that stopped it (if any)."""
    run_dir = Path(run_dir)
    stages = read_stages(run_dir) or {}
    error = stages.get("error")
    if (run_dir / "RUN_FAILED_CHECKS.txt").exists() or \
            summary.get("run_status") == "failed_checks":
        return "failed checks", error
    if (run_dir / "provenance.json").exists() and summary:
        p2 = _phase2_state(run_dir, summary)
        if p2 in ("running", "finishing"):
            return "explaining", error
        if p2 == "awaiting choice":
            return "awaiting phase 2 choice", error
        if p2 in ("stopped", "failed"):
            return "phase 2 stopped", error
        if int(summary.get("n_reasons_pending") or 0) and p2 not in ("skipped",):
            return "reasons pending", error
        return "finished", error
    if error:
        return "refused", error
    if live:
        return "running", None
    return ("interrupted" if stages else "incomplete"), None


def _phase2_state(run_dir: Path, summary: dict) -> str:
    from .phase2 import read_progress
    try:
        return read_progress(run_dir).get("state") or summary.get("phase2_state") or ""
    except Exception:                                  # unreadable status: use summary
        return summary.get("phase2_state") or ""


def _is_run_dir(p: Path) -> bool:
    return p.is_dir() and not p.name.startswith(("_", ".")) and any(
        (p / n).exists() for n in ("provenance.json", "run_summary.csv", "stages.json"))


def get_run(runs_dir: Path, run_id: str, *, live: set | None = None,
            jobs_dir: Path | None = None) -> RunRecord | None:
    run_dir = Path(runs_dir) / run_id
    if "/" in run_id or "\\" in run_id or run_id.startswith(".") or not _is_run_dir(run_dir):
        return None
    summary = _summary(run_dir)
    state, error = run_state(run_dir, summary, live=run_id in (live or set())
                             or _runner_holds(run_id, jobs_dir))
    return RunRecord(run_id, run_dir, state, summary, error)


def _runner_holds(run_id: str, jobs_dir: Path | None = None) -> bool:
    """Whether a background runner process is working on this run now."""
    from .runner import RUNS_DIR, active_lock
    try:
        return bool(active_lock(run_id, Path(jobs_dir or RUNS_DIR)))
    except Exception:
        return False


def list_runs(runs_dir: Path, *, live: set | None = None,
              limit: int | None = None, jobs_dir: Path | None = None) -> list[RunRecord]:
    """Every scoring run, newest first."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    out = []
    for p in runs_dir.iterdir():
        if _is_run_dir(p):
            rec = get_run(runs_dir, p.name, live=live, jobs_dir=jobs_dir)
            if rec is not None:
                out.append(rec)
    # By when the run was made, not by folder name: a folder not named by the
    # usual timestamp would otherwise sort to the top whatever its age.
    out.sort(key=lambda r: (r.summary.get("created_at") or _stamp(r.run_id) or "",
                            r.run_id), reverse=True)
    return out[:limit] if limit else out


def overview(runs: list[RunRecord], registry=None, *, trend_n: int = 20,
             for_lending_only: bool = True) -> dict:
    """The Overview page's figures, from the run history and the registry.

    Trends use finished runs only (decisions final, every check passed), newest
    ``trend_n``, oldest first. ``for_lending_only`` leaves out runs stamped not for
    lending decisions (an unapproved model or an unpublished threshold), whose
    rates answer a different question.
    """
    counts: dict[str, int] = {}
    for r in runs:
        counts[r.state] = counts.get(r.state, 0) + 1
    usable = [r for r in runs if r.state in FINISHED_STATES
              and (not for_lending_only or r.summary.get("for_lending_decisions") in
                   (True, "True"))]
    trend = [r for r in usable][:trend_n][::-1]
    total_rows = sum(int(r.summary.get("n_rows") or 0) for r in usable)
    total_ok = sum(int(r.summary.get("n_approved") or 0) for r in usable)
    flagged = [r for r in trend if int(r.summary.get("n_explained") or 0) > 0]

    models: dict = {"approved": [], "awaiting_approval": 0, "by_status": {}}
    if registry is not None:
        for tag, rec in sorted(registry.models.items()):
            models["by_status"][rec.status] = models["by_status"].get(rec.status, 0) + 1
            if rec.status == "approved":
                models["approved"].append({
                    "tag": tag, "summary": rec.summary,
                    "validation_metrics": rec.validation_metrics,
                    "approved_by": rec.approval.get("approved_by"),
                    "approved_on": rec.approval.get("approved_at")
                    or rec.approval.get("approved_on"),
                    "findings_section": rec.approval.get("findings_section"),
                    "n_train": (rec.training or {}).get("n_train"),
                    "split_scheme": (rec.training or {}).get("split_scheme")})
        models["awaiting_approval"] = models["by_status"].get("candidate", 0)

    return {
        "total_runs": len(runs),
        "runs_by_state": counts,
        "finished_runs": len(usable),
        "applicants_decided": total_rows,
        "approval_rate_overall": (total_ok / total_rows) if total_rows else None,
        "trend": [{"run_id": r.run_id,
                   "created_at": r.summary.get("created_at") or _stamp(r.run_id),
                   "source_file": r.summary.get("source_file"),
                   "n_rows": r.summary.get("n_rows"),
                   "approval_rate": r.summary.get("approval_rate"),
                   "n_explained": r.summary.get("n_explained"),
                   "fair_lending_flag_share": (r.summary.get("fair_lending_flag_share")
                                               if r in flagged else None)}
                  for r in trend],
        "recent": [r.to_dict() for r in runs[:5]],
        "models": models,
        "for_lending_only": for_lending_only,
    }


# ------------------------------------------------------- processing history --

def _day(stamp) -> str:
    """The date part of a run's timestamp, or "" when it has none."""
    return str(stamp or "")[:10]


def processing_history(runs: list[RunRecord], *, date_from: str | None = None,
                       date_to: str | None = None, model_tag: str | None = None,
                       for_lending_only: bool = True, today: str | None = None) -> dict:
    """Volume and outcomes across every run, filtered, for the history section.

    Totals are over *finished* runs only (:data:`FINISHED_STATES`): a run that failed
    its checks has no decisions to count, and a run still scoring would have its
    volume counted twice as it grows. ``total_runs`` is every run in the filter,
    whatever its state, so the two numbers answer the two different questions a
    reader has -- how many runs happened, and how many applicants were decided.

    Every figure is a sum over ``run_summary.csv`` fields the runs already wrote.
    Nothing is recomputed from a decision file and nothing is cached outside the
    per-file cache :func:`_summary` already keeps, so deleting a run folder removes
    it from these totals.
    """
    from datetime import date, timedelta

    def created(r: RunRecord) -> str:
        return r.summary.get("created_at") or _stamp(r.run_id) or ""

    kept = list(runs)
    if model_tag:
        kept = [r for r in kept if r.summary.get("model_tag") == model_tag]
    if date_from:
        kept = [r for r in kept if _day(created(r)) >= str(date_from)[:10]]
    if date_to:
        kept = [r for r in kept if _day(created(r)) <= str(date_to)[:10]]

    usable = [r for r in kept if r.state in FINISHED_STATES
              and (not for_lending_only or r.summary.get("for_lending_decisions") in
                   (True, "True"))]

    def total(field: str) -> int:
        return sum(int(r.summary.get(field) or 0) for r in usable)

    rows = total("n_rows")
    approved = total("n_approved")
    rejected = total("n_rejected")
    # Phase 1 and Phase 2 finish separately, and FINISHED_STATES deliberately covers
    # four states where the decisions are final but the reasons are not written yet
    # ("reasons pending", "explaining", "phase 2 stopped", "awaiting phase 2
    # choice"). So "processed" means decided, not explained, and the difference is
    # reported rather than left for a reader to infer from a run count.
    explained = total("n_explained")
    pending = total("n_reasons_pending")
    unexplained = total("n_rejected_without_reasons")
    incomplete = [r for r in usable
                  if int(r.summary.get("n_explained") or 0)
                  < int(r.summary.get("n_rejected") or 0)]

    # "this week" and "this month" are the trailing 7 and 30 days from today, not
    # calendar periods: a run on the 1st of a month should not make the month look
    # empty. Measured against the real date so the figure ages by itself.
    anchor = date.fromisoformat(str(today)[:10]) if today else date.today()
    def since(days: int) -> int:
        cut = (anchor - timedelta(days=days)).isoformat()
        return sum(int(r.summary.get("n_rows") or 0) for r in usable
                   if _day(created(r)) >= cut)

    series = [{"run_id": r.run_id, "created_at": created(r), "day": _day(created(r)),
               "source_file": r.summary.get("source_file"),
               "model_tag": r.summary.get("model_tag"),
               "n_rows": int(r.summary.get("n_rows") or 0),
               "n_approved": int(r.summary.get("n_approved") or 0),
               "n_rejected": int(r.summary.get("n_rejected") or 0),
               "approval_rate": r.summary.get("approval_rate"),
               "n_explained": int(r.summary.get("n_explained") or 0),
               # None rather than 0 where nothing was explained: a run with no
               # reasons yet has no flag share, and plotting it as zero would read
               # as "measured, and clean".
               "fair_lending_flag_share": (
                   r.summary.get("fair_lending_flag_share")
                   if int(r.summary.get("n_explained") or 0) else None),
               "state": r.state}
              for r in sorted(usable, key=created)]

    days = sorted({_day(created(r)) for r in runs if _day(created(r))})
    return {
        "total_runs": len(kept),
        "finished_runs": len(usable),
        "applicants_processed": rows,
        "total_approved": approved,
        "total_rejected": rejected,
        "approval_rate_overall": (approved / rows) if rows else None,
        "rejected_explained": explained,
        "reasons_pending": pending,
        "rejected_without_reasons": unexplained,
        "runs_awaiting_reasons": len(incomplete),
        "applicants_awaiting_reasons": sum(int(r.summary.get("n_rows") or 0)
                                           for r in incomplete),
        "applicants_last_7_days": since(7),
        "applicants_last_30_days": since(30),
        "files_processed": len({r.summary.get("source_file") for r in usable
                                if r.summary.get("source_file")}),
        "series": series,
        "model_tags": sorted({r.summary.get("model_tag") for r in runs
                              if r.summary.get("model_tag")}),
        "first_day": days[0] if days else None,
        "last_day": days[-1] if days else None,
        "filters": {"date_from": date_from, "date_to": date_to,
                    "model_tag": model_tag, "for_lending_only": for_lending_only},
    }
