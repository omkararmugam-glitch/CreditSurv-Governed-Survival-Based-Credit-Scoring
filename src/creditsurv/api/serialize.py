"""Turning what a run wrote into JSON, and nothing else.

Every value here was computed by the pipeline and read back from the run folder
(:func:`creditsurv.batch.load_result`); this module only changes its shape. Frames
travel as pandas' ``split`` orientation, so the dashboard rebuilds them exactly.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from .. import phase2
from ..batch import (INTERNAL_DIR, INTERNAL_FLAGS, INTERNAL_RECORDS, OUTPUT_NAMES,
                     load_result)
from ..cleaning import CleaningReport
from ..stages import pipeline_view

__all__ = ["frame", "run_detail", "run_status", "list_files", "FILE_LABELS",
           "download_name", "STILL_WORKING", "cached_result"]


def frame(df: pd.DataFrame | None) -> dict:
    if df is None:
        df = pd.DataFrame()
    return json.loads(df.to_json(orient="split", index=False, date_format="iso"))


FILE_LABELS = {
    "scored_applicants.csv": "Every applicant, with risk, decision and top reasons",
    "approved_applicants.csv": "Approved applicants only",
    "rejected_applicants.csv": "Rejected applicants only, with Regulation B reasons "
                               "and fair-lending flags",
    "adverse_action_notices.zip": "Applicant notices: only what the applicant is given",
    f"{INTERNAL_DIR}/{INTERNAL_FLAGS}": "INTERNAL - never send to applicants: "
                                        "fair-lending flags, drivers, attributions",
    f"{INTERNAL_DIR}/{INTERNAL_RECORDS}": "INTERNAL - never send to applicants: the "
                                          "review record for every explained rejection",
    "validation_checks.csv": "Pass/fail of every post-run check",
    "cleaning_report.csv": "What cleaning did: per rule and per column",
    "data_drift.csv": "Per-feature drift of this file against the training data",
    "run_summary.csv": "One row describing this run, for traceability",
    "validation_report.json": "What the file's columns were read as",
    "provenance.json": "SHA-256 of the upload, the model and every output",
    "aggregates.json": "The counts behind the charts",
    "data_profile_missing.csv": "Missing values by column",
    "data_profile_numeric.csv": "Numeric summary",
    "data_profile_categories.csv": "Category counts",
    "data_profile_outliers.csv": "Outlier counts (nothing altered)",
    "RUN_FAILED_CHECKS.txt": "Why this run is not finished",
}

DOWNLOADABLE = tuple(OUTPUT_NAMES) + (f"{INTERNAL_DIR}/{INTERNAL_FLAGS}",
                                      f"{INTERNAL_DIR}/{INTERNAL_RECORDS}",
                                      "RUN_FAILED_CHECKS.txt")
"""What a client may fetch from a run folder: the run's outputs by name. Never the
Phase 2 working files (model inputs, background sample) and never an arbitrary
path."""

STILL_WORKING = ("running", "not started", "awaiting choice", "finishing")
"""Phase 2 states in which downloads are held back, as the dashboard always did:
nobody saves a half-finished file while it is still being written."""


def list_files(run_dir: Path) -> list[dict]:
    run_dir = Path(run_dir)
    out = []
    for name in DOWNLOADABLE:
        p = run_dir / name
        if p.is_file():
            out.append({"name": name, "size_bytes": p.stat().st_size,
                        "internal": name.startswith(f"{INTERNAL_DIR}/"),
                        "label": FILE_LABELS.get(name, "")})
    return out


def download_name(name: str, snap: dict) -> str:
    """``rejected_applicants.csv`` -> ``rejected_applicants_PARTIAL_351_pending.csv``
    while Phase 2 is unfinished, so a saved file says what it is."""
    name = name.rsplit("/", 1)[-1]
    if not snap.get("partial"):
        return name
    stem, dot, ext = name.rpartition(".")
    return f"{stem}_PARTIAL_{snap['pending']}_pending{dot}{ext}"


_RESULTS: "OrderedDict[tuple, object]" = OrderedDict()
_RESULTS_LOCK = threading.Lock()


def cached_result(run_dir: Path):
    """:func:`load_result`, re-read only when the run writes: Phase 2 rewrites
    provenance.json at every checkpoint, so its modification time is the key."""
    prov = Path(run_dir) / "provenance.json"
    key = (str(Path(run_dir).resolve()), prov.stat().st_mtime_ns)
    with _RESULTS_LOCK:
        if key in _RESULTS:
            _RESULTS.move_to_end(key)
            return _RESULTS[key]
    result = load_result(run_dir)
    with _RESULTS_LOCK:
        _RESULTS[key] = result
        while len(_RESULTS) > 4:
            _RESULTS.popitem(last=False)
    return result


def _read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def run_status(run_dir: Path, record) -> dict:
    """What the pipeline view polls: cheap, and read fresh every time."""
    run_dir = Path(run_dir)
    summary = record.summary if record else {}
    prog = phase2.read_progress(run_dir)
    snap = phase2.snapshot(run_dir, summary) if summary else {"state": "", "partial": False}
    recent = phase2.recent_results(run_dir, 5)
    return {
        "run_id": run_dir.name,
        "state": record.state if record else "unknown",
        "error": record.error if record else None,
        "stages": pipeline_view(run_dir, phase2_progress=prog, summary=summary),
        "phase2": {**{k: prog.get(k) for k in ("state", "done", "target", "mode",
                                                "rate_per_min", "eta_seconds", "error")},
                   "snapshot": snap,
                   "recent": [{"row_id": r.get("row_id"),
                               "applicant_id": r.get("applicant_id"),
                               "first_reason": (r["reasons"][0]["reason"]
                                                if r.get("reasons") else r.get("status"))}
                              for r in recent]},
        "finished": (run_dir / "provenance.json").exists(),
    }


def run_detail(run_dir: Path, record, *, preview_rows: int = 25) -> dict:
    """Everything the result view draws, for a run in any state.

    A finished run is read through :func:`creditsurv.batch.load_result`, the same
    function the dashboard used; a run that failed its checks (no provenance.json)
    still has its checks, cleaning report and summary, and those are returned.
    """
    run_dir = Path(run_dir)
    out = {"run_id": run_dir.name, "state": record.state, "error": record.error,
           "summary": record.summary, "files": list_files(run_dir),
           "status": run_status(run_dir, record)}
    vr = _read_json(run_dir / "validation_report.json", {}) or {}
    out["validation"] = vr
    checks_path = run_dir / "validation_checks.csv"
    out["checks"] = frame(pd.read_csv(checks_path) if checks_path.exists() else None)

    if not (run_dir / "provenance.json").exists():
        clean_path = run_dir / "cleaning_report.csv"
        out["cleaning"] = {"sentences": [], "table": frame(
            pd.read_csv(clean_path) if clean_path.exists() else None)}
        return out

    result = cached_result(run_dir)
    cr = result.clean_report or CleaningReport()
    out["cleaning"] = {"sentences": cr.plain_english(), "table": frame(cr.to_frame())}
    out["warnings"] = list(result.report.warnings)
    dr = result.drift
    out["drift"] = None if dr is None else {
        "status": dr.status, "colour": dr.colour, "headline": dr.headline(),
        "table": frame(dr.table)}
    agg = result.aggregates
    s = result.summary
    out["aggregates"] = None if agg is None else {
        "group_column": agg.group_column,
        "risk": frame(agg.risk_frame(s["threshold"])),
        "groups": frame(agg.group_frame()),
        "reasons": frame(agg.reason_frame()),
        # Empty on a run scored before segments were recorded; the page says so
        # rather than drawing a chart with nothing in it.
        "segment_columns": list(agg.segment_columns),
        "segment_order": dict(agg.segment_order),
        "segments": frame(agg.segment_frame())}
    prof = result.profile or {}
    out["profile"] = {"overview": prof.get("overview", {}),
                      **{k: frame(prof.get(k)) for k in
                         ("missing", "numeric", "categorical", "outliers")}}
    out["preview"] = frame(result.scored.head(preview_rows))
    return out
