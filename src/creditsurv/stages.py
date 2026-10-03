"""Where a scoring run is, stage by stage, written to the run folder as it goes.

Phase 1 reports its steps through the ``progress(step, state, message)`` callback
of :func:`creditsurv.batch.score_file`. A :class:`StageLog` is that callback, and
keeps what it hears in ``<run>/stages.json``: every caller that scores a file into
a known run folder -- the API's worker, and ``06_score_upload.py --run-dir`` when
the API hands a large file to the background runner -- records the same thing in
the same place. Phase 2's own ``phase2/status.json`` supplies the last two stages.

:func:`pipeline_view` merges the two into the eight boxes the dashboard's pipeline
page draws: check, clean, score, decide, drift, checks, explain, notices. Nothing
here decides anything; it only reports.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import fileio

__all__ = ["STAGES_FILE", "STAGES_LOG", "PIPELINE", "StageLog", "read_stages",
           "pipeline_view"]

STAGES_FILE = "stages.json"
STAGES_LOG = "stages.log"

PIPELINE: tuple[tuple[str, str], ...] = (
    ("check", "Check file"),
    ("clean", "Clean"),
    ("score", "Score"),
    ("decide", "Decide"),
    ("drift", "Drift"),
    ("checks", "Run checks"),
    ("explain", "Explain"),
    ("notices", "Notices"),
)
"""The stages a scoring run passes through, in order, with their labels."""

# score_file's step names -> the pipeline's. "profile" is the drift check (the
# profile and the drift comparison are one step there); "files" is writing the
# outputs, which the pipeline view does not show as a stage of its own.
_ALIASES = {"profile": "drift"}


def _write_json(path: Path, obj) -> None:
    """Atomic where the platform allows. On Windows a replace fails while a reader
    (the status endpoint, polling) has the file open, so it is retried briefly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    try:
        fileio.replace(tmp, path)
    except PermissionError:
        tmp.unlink(missing_ok=True)
        raise


class StageLog:
    """A ``progress`` callback that also records every step in ``stages.json``.

    ``also`` is another callback to pass each event on to (the CLI's printer), so
    recording never replaces what a caller already shows.
    """

    def __init__(self, run_dir: Path, *, also=None):
        self.path = Path(run_dir) / STAGES_FILE
        self.also = also
        self.state = read_stages(run_dir) or {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "finished": None,
            "error": None, "stages": {}}

    def __call__(self, step: str, state: str, message: str = "") -> None:
        key = _ALIASES.get(step, step)
        entry = self.state["stages"].setdefault(key, {"state": "pending"})
        now = time.time()
        if state == "running" and entry.get("state") != "running":
            entry["started_at"] = now
        if state in ("done", "failed"):
            entry["ended_at"] = now
            if entry.get("started_at"):
                entry["seconds"] = round(now - entry["started_at"], 2)
        entry["state"] = state
        if message:
            entry["message"] = message
        # Recording is reporting: a write that fails (a locked file) must never stop
        # the run it reports on. The next event writes the whole state again.
        try:
            _write_json(self.path, self.state)
            self._line(f"[{dict(PIPELINE).get(key, key)}] {state}"
                       + (f"  {message}" if message else ""))
        except OSError:
            pass
        if self.also:
            self.also(step, state, message)

    def _line(self, text: str) -> None:
        """One line per event in ``stages.log``: the live log of a run scored in
        a process that keeps no log of its own (the API's worker)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_name(STAGES_LOG), "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {text}\n")

    def fail(self, message: str, detail: str = "", fix: str = "") -> None:
        """The run stopped: the running stage (or the first pending one) failed."""
        stages = self.state["stages"]
        running = [k for k, _ in PIPELINE if stages.get(k, {}).get("state") == "running"]
        key = running[0] if running else next(
            (k for k, _ in PIPELINE if stages.get(k, {}).get("state") != "done"), "check")
        entry = stages.setdefault(key, {})
        entry.update(state="failed", message=message, ended_at=time.time())
        self.state["error"] = {"message": message, "detail": detail, "fix": fix,
                               "stage": key}
        self._line(f"[{dict(PIPELINE).get(key, key)}] FAILED  {message}")
        self.finish()

    def finish(self) -> None:
        # Not swallowed: this is the record that the run ended, and how it ended.
        self.state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _write_json(self.path, self.state)


def read_stages(run_dir: Path) -> dict | None:
    try:
        return json.loads((Path(run_dir) / STAGES_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def pipeline_view(run_dir: Path, *, phase2_progress: dict | None = None,
                  summary: dict | None = None) -> list[dict]:
    """The eight stages with a state each: pending, running, done, failed, skipped.

    Phase 1's stages come from ``stages.json``; a run scored before it existed (or
    by a caller that kept no stage log) is read from what it wrote instead, so an
    old finished run still shows as done rather than as pending. ``explain`` and
    ``notices`` come from Phase 2's progress.
    """
    run_dir = Path(run_dir)
    recorded = (read_stages(run_dir) or {}).get("stages", {})
    finished = (run_dir / "provenance.json").exists()
    failed_checks = (run_dir / "RUN_FAILED_CHECKS.txt").exists()
    out = []
    for key, label in PIPELINE[:6]:
        entry = dict(recorded.get(key) or {})
        if not entry and (finished or failed_checks):
            entry = {"state": "done"}
        if key == "checks" and failed_checks and entry.get("state") != "running":
            entry["state"] = "failed"
        entry.setdefault("state", "pending")
        out.append({"key": key, "label": label, **entry})

    prog = phase2_progress or {}
    s = summary or {}
    p2 = prog.get("state", "")
    n_rejected = int(s.get("n_rejected", 0) or 0)
    phase1_ok = finished and not failed_checks
    if not phase1_ok or (not n_rejected and finished):
        explain = notices = "skipped" if finished and not n_rejected else "pending"
    elif p2 in ("running", "finishing"):
        explain, notices = "running", "pending"
    elif p2 == "completed":
        explain, notices = "done", "done"
    elif p2 == "skipped":
        explain, notices = "skipped", "skipped"
    elif p2 == "failed_checks":
        explain, notices = "done", "failed"
    elif p2 in ("stopped", "failed"):
        explain, notices = "failed", "pending"
    else:                                   # not started, awaiting choice
        explain, notices = "pending", "pending"
    done = int(prog.get("done") or 0)
    target = int(prog.get("target") or n_rejected or 0)
    out.append({"key": "explain", "label": "Explain", "state": explain,
                "message": (f"{done:,} of {target:,} rejected applicants explained"
                            if target else ""),
                "done": done, "target": target})
    out.append({"key": "notices", "label": "Notices", "state": notices,
                "message": (f"{int(s.get('n_notices', 0) or 0):,} notices"
                            if notices == "done" else
                            "withheld: Phase 2 failed its checks"
                            if notices == "failed" else "")})
    return out
