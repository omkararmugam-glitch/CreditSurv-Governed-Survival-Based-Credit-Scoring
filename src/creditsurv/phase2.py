"""Phase 2 of a scoring run: reasons and adverse-action notices, after the decisions.

:func:`creditsurv.batch.score_file` (Phase 1) writes every decision within minutes
and marks each rejected row ``"reasons pending"``. :func:`explain_run` is the one
function that explains them. The dashboard, background runs, ``06_score_upload.py``
and :func:`creditsurv.batch.run_batch` all call it; ``tests/test_two_phase.py``
fails if anything else explains applicants or builds notices for a scoring run.

What it does, and what it keeps:

* **Streams.** One worker pool for the whole run (``iter_explanations``); each
  applicant's result is appended to ``phase2/results.jsonl`` as it finishes, and
  progress (done / target, rate, ETA) to ``phase2/status.json`` about once a
  second. ``rejected_applicants.csv``, the notice zip and the internal files are
  rebuilt from the results at checkpoints, so reasons appear in the files while it
  runs. Rows not yet reached keep ``"reasons pending"``.
* **Resumes.** Results and the per-applicant attribution ledger survive an
  interruption; running it again continues where it stopped, with identical
  reasons, because every applicant is seeded from its own id.
* **Explains what Phase 1 decided.** The model inputs are the ones Phase 1 scored,
  saved beside the decisions, and the model file must hash to the one Phase 1
  used, or it refuses.
* **Keeps the notice / internal split.** Every notice is built by
  ``build_adverse_action_notice`` and screened as it renders; each checkpoint
  screens every notice again before it enters ``adverse_action_notices.zip``;
  internal records go only to ``internal/``. The final step re-runs every
  post-run check from the files on disk before the run is marked finished.
* **Choices, stamped.** ``mode="all"`` explains every rejected applicant;
  ``"sample"`` a seeded random sample; ``limit`` the first N (the old cap);
  ``"skip"`` nobody. Anything short of all is stamped not for lending decisions.
  ``only=[row_id, ...]`` explains named applicants on demand, in seconds.
"""

from __future__ import annotations

import json
import os
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .batch import (CAP_NOTE, INTERNAL_COLUMNS, INTERNAL_DIR, INTERNAL_FLAGS,
                    INTERNAL_README, INTERNAL_RECORDS, MAX_PRINCIPAL_REASONS,
                    PARALLEL_MIN_ROWS, PHASE2_DIR, PREVIEW_ROWS, Aggregates,
                    BatchError, BatchResult, _load_frame, _slug, _without_features)
from .config import load_config
from .environment import policy_block_message, policy_blocked_exception
from .explain.adverse_action import (NOT_FOR_LENDING, build_adverse_action_notice,
                                     find_internal_content)
from .explain.parallel import (Ledger, assemble, iter_explanations, keep_awake,
                               suggest_workers)
from .explain.survshap import summarise_background
from .explain.tree_shap import explain_tree_shap
from .provenance import PROJECT_ROOT, build_stamp, file_fingerprint
from .run_checks import (NO_REASON_NOTE, OUTSIDE_SAMPLE_NOTE, REASONS_PENDING,
                         SKIPPED_NOTE, blocking_failures, verify_run, write_checks)

__all__ = ["explain_run", "Phase2Result", "read_progress", "MODES", "estimate_seconds",
           "materialize", "pending_rows", "result_for", "recent_results",
           "launch_background"]

MODES = ("all", "sample", "skip")
TREESHAP_BATCH = 512


# --------------------------------------------------------------- the files --

def _p(run_dir: Path, name: str) -> Path:
    return Path(run_dir) / PHASE2_DIR / name


def _read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)                  # readers never see a half-written file


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except ImportError:                    # pragma: no cover
        return True


class _Lock:
    """An exclusive lock file for the run's Phase 2 files.

    A background Phase 2 and an on-demand "explain this applicant" may write the
    same run at once; appends and rebuilds happen under this lock. A lock whose
    process has died is broken rather than waited on forever.
    """

    def __init__(self, run_dir: Path, timeout: float = 120.0):
        self.path = _p(run_dir, "files.lock")
        self.timeout = timeout

    def __enter__(self):
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    holder = int(self.path.read_text() or 0)
                except (OSError, ValueError):
                    holder = 0
                if holder and not _pid_alive(holder):
                    self.path.unlink(missing_ok=True)
                    continue
                if time.monotonic() > deadline:
                    raise BatchError("Another process is writing this run's reasons.",
                                     f"lock {self.path} held by pid {holder}",
                                     "Wait a moment and try again.")
                time.sleep(0.05)

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def _read_lines(path: Path) -> list[dict]:
    out = []
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:             # a half-written final line
                continue
    return out


def _results(run_dir: Path) -> dict[int, dict]:
    done: dict[int, dict] = {}
    for rec in _read_lines(_p(run_dir, "results.jsonl")):
        done[int(rec["row_id"])] = rec     # the latest result for a row wins
    return done


def _notices(run_dir: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for rec in _read_lines(_p(run_dir, "notices.jsonl")):
        out[int(rec["row_id"])] = rec
    return out


def _append(run_dir: Path, results: list[dict], notices: list[dict]) -> None:
    with _Lock(run_dir):
        for name, rows in (("results.jsonl", results), ("notices.jsonl", notices)):
            if rows:
                with _p(run_dir, name).open("a", encoding="utf-8") as fh:
                    fh.write("".join(json.dumps(r, default=str) + "\n" for r in rows))


def _plan(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    if not (run_dir / "provenance.json").exists():
        raise BatchError(
            f"{run_dir.name} has no finished decisions, so there is nothing to explain.",
            "provenance.json is missing: Phase 1 did not finish, or failed its checks",
            "Score the file first; Phase 2 only explains decisions that passed.")
    plan = _read_json(_p(run_dir, "plan.json"))
    if plan is None:
        raise BatchError(f"{run_dir.name} has no rejected applicants to explain.",
                         f"{_p(run_dir, 'plan.json')} is missing", "")
    return plan


def read_progress(run_dir: Path) -> dict:
    """Phase 2's progress, as the page and the CLI show it.

    ``alive`` says whether the process recorded as running still exists, so a run
    killed by a sleep or a closed terminal shows as stopped rather than as running
    forever.
    """
    st = _read_json(_p(run_dir, "status.json"), {}) or {}
    st.setdefault("state", "not needed")
    st["alive"] = st.get("state") == "running" and _pid_alive(st.get("pid"))
    if st.get("state") == "running" and not st["alive"]:
        st["state"] = "stopped"
    return st


def estimate_seconds(n: int, cfg) -> float:
    """Rough wall time to explain ``n`` applicants with SurvSHAP(t) here."""
    return n * float(getattr(cfg.decision, "explain_seconds_each", 2.1))


UNFINISHED = ("running", "not started", "awaiting choice", "stopped", "failed",
              "finishing")
"""Phase 2 states in which the run's files are a partial snapshot: some rejected
applicants still say "reasons pending"."""


def snapshot(run_dir: Path, summary: dict | None = None) -> dict:
    """Whether a run's files are final or a partial snapshot, read live.

    One answer for the page's banner, the download names and the marker inside
    the zip, so they cannot disagree about whether Phase 2 is done. ``pending`` is
    target minus done while Phase 2 runs (the summary lags until a checkpoint),
    otherwise the summary's own count.
    """
    prog = read_progress(Path(run_dir))
    if summary is None:
        prov = _read_json(Path(run_dir) / "provenance.json", {}) or {}
        summary = prov.get("summary", {})
    state = prog.get("state", "")
    # explain_run marks itself completed, then rewrites the final files and the
    # summary (which records that state). Until the summary agrees, the files on
    # disk are the last checkpoint's: "finishing", not done. Only while the Phase 2
    # process lives -- one that died mid-write is not left finishing forever.
    if (state in ("completed", "skipped")
            and summary.get("phase2_state") not in (None, state)
            and _pid_alive(prog.get("pid"))):
        state = "finishing"
    n_rejected = int(summary.get("n_rejected", 0) or 0)
    target = int(prog.get("target") or n_rejected)
    done = int(prog.get("done") or 0)
    pending = int(summary.get("n_reasons_pending", 0) or 0)
    if state == "running":
        pending = max(target - done, 0)
    elif state in ("not started", "awaiting choice"):
        pending = max(pending, n_rejected - done)
    return {"state": state, "partial": bool(n_rejected) and (
                state in UNFINISHED or pending > 0),
            "done": done, "target": target, "pending": pending,
            "rate_per_min": prog.get("rate_per_min"),
            "eta_seconds": prog.get("eta_seconds") if state == "running" else None}


def snapshot_note(snap: dict, when: str) -> str:
    """The plain-text marker put in a partial download, so the file says it."""
    return (f"PARTIAL SNAPSHOT -- NOT THE FINISHED RUN\n\n"
            f"Taken {when} while Phase 2 (reasons and notices) was "
            f"'{snap['state']}': {snap['done']:,} of {snap['target']:,} rejected "
            f"applicants explained, {snap['pending']:,} still marked "
            f"\"reasons pending\".\n\n"
            f"Decisions (approve / reject) are final. Reasons, adverse-action notices, "
            f"fair-lending flags and the run checks cover only the applicants "
            f"explained so far. Download again once Phase 2 has finished for the "
            f"complete result.\n")


def pending_rows(run_dir: Path, limit: int = 500) -> pd.DataFrame:
    """Rejected rows still marked "reasons pending", for the page's picker."""
    path = Path(run_dir) / "rejected_applicants.csv"
    out = []
    for chunk in pd.read_csv(path, dtype=str, keep_default_na=False,
                             usecols=lambda c: c in ("row_id", "applicant_id",
                                                     "explained", "pd_12m") or
                             c.startswith("pd_"), chunksize=50_000):
        out.append(chunk[chunk["explained"] == REASONS_PENDING])
        if sum(len(o) for o in out) >= limit:
            break
    frame = pd.concat(out) if out else pd.DataFrame()
    return frame.head(limit)


def result_for(run_dir: Path, row_id: int) -> tuple[dict | None, str]:
    """One applicant's Phase 2 result and notice text ("" when no notice)."""
    res = _results(run_dir).get(int(row_id))
    notice = _notices(run_dir).get(int(row_id))
    text = notice["text"] if (notice and res and res["status"] == "explained") else ""
    return res, text


def recent_results(run_dir: Path, n: int = 5) -> list[dict]:
    """The last ``n`` applicants explained, read from the end of the results file
    so a long run's progress view stays cheap."""
    path = _p(run_dir, "results.jsonl")
    if not path.exists():
        return []
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        fh.seek(max(0, fh.tell() - 256_000))
        lines = fh.read().decode("utf-8", "replace").splitlines()[-n - 1:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out[-n:]


def launch_background(run_dir: Path, mode: str = "all", sample_n: int | None = None,
                      *, runs_dir: Path | None = None, config: str | None = None) -> Path:
    """Start Phase 2 for a run as a detached background process.

    The same :mod:`creditsurv.runner` machinery large uploads use, running
    ``06_score_upload.py --explain``, which calls :func:`explain_run`. It outlives
    the browser tab, and one per run: a second launch while one runs is refused.
    """
    from .runner import RUNS_DIR, launch

    run_dir = Path(run_dir)
    args = ["scripts/06_score_upload.py", "--explain", str(run_dir), "--phase2", mode]
    if mode == "sample":
        args += ["--sample-n", str(int(sample_n or 0))]
    if config:
        args += ["--config", str(config)]
    return launch([{"key": "explain", "name": f"Reasons for {run_dir.name}",
                    "tag": run_dir.name, "args": args}],
                  lock_tag=f"{run_dir.name}_explain", runs_dir=runs_dir or RUNS_DIR,
                  meta={"source": "ui-phase2", "run": run_dir.name, "mode": mode,
                        "keep_awake": True})


# ----------------------------------------------------------- the one function --

@dataclass
class Phase2Result:
    run_dir: Path
    summary: dict
    checks: pd.DataFrame
    aggregates: Aggregates
    files: dict
    notices: list = field(default_factory=list)
    state: str = ""

    def merged_into(self, result: BatchResult) -> BatchResult:
        """The Phase 1 result, brought up to date with what Phase 2 wrote."""
        result.summary = self.summary
        result.checks = self.checks
        result.aggregates = self.aggregates
        result.notices = self.notices
        result.files.update(self.files)
        result.seconds = float(self.summary.get("seconds_total") or result.seconds)
        scored = pd.read_csv(result.run_dir / "scored_applicants.csv", nrows=PREVIEW_ROWS)
        result.scored = scored
        result.approved = scored[scored["decision"] == "approve"] if len(scored) else scored
        result.rejected = scored[scored["decision"] == "reject"] if len(scored) else scored
        return result


def explain_run(run_dir: Path, cfg=None, *, mode: str = "all",
                sample_n: int | None = None, limit: int | None = None,
                only=None, model=None, model_path: Path | None = None,
                workers: int | None = None, progress=None,
                checkpoint_seconds: float | None = None) -> Phase2Result:
    """**Phase 2**: explain a scored run's rejected applicants and write notices.

    ``mode`` is ``"all"``, ``"sample"`` (with ``sample_n``) or ``"skip"``; ``limit``
    explains only the first N rejected rows; ``only`` explains the named row_ids on
    demand and leaves the run's mode alone. ``model`` may be passed by a caller that
    already holds it (the dashboard's cache, :func:`run_batch`), together with
    ``model_path``; either way the file must hash to the one Phase 1 scored with.
    """
    run_dir = Path(run_dir)
    started = time.perf_counter()
    plan = _plan(run_dir)
    cfg = cfg or load_config(PROJECT_ROOT / "config" / "config.yaml")
    status_path = _p(run_dir, "status.json")
    status = _read_json(status_path, {}) or {}
    target_path = _p(run_dir, "target.json")

    def say(state: str, message: str = "") -> None:
        if progress:
            progress("explain", state, message)

    if mode not in MODES:
        raise BatchError(f"Unknown Phase 2 choice {mode!r}.", f"use one of {MODES}", "")

    # One background Phase 2 per run; on-demand explanations may run beside it.
    live = read_progress(run_dir)
    if only is None and live.get("alive") and int(live.get("pid") or 0) != os.getpid():
        raise BatchError("Reasons are already being generated for this run.",
                         f"pid {live.get('pid')} is running Phase 2", "Watch its progress.")

    design_meta, bg_meta = plan["design"], plan["background"]
    ids_frame = pd.read_parquet(_p(run_dir, "rejected_design.parquet"),
                                columns=["__applicant_id"])
    all_rows = [int(r) for r in ids_frame.index]
    applicant_of = dict(zip(all_rows, ids_frame["__applicant_id"].astype(str)))

    # ---------------------------------------------------------- targets ----
    if only is not None:
        only = [int(r) for r in only]
        unknown = [r for r in only if r not in applicant_of]
        if unknown:
            raise BatchError("That row is not a rejected applicant of this run.",
                             f"row_id(s) {unknown}", "Pick a row marked reasons pending.")
        target = _read_json(target_path, None)
        wanted = only
    else:
        if mode == "skip":
            wanted = []
        elif mode == "sample":
            n = int(sample_n or 0)
            if n <= 0:
                raise BatchError("A sample needs a size.", "sample_n <= 0", "")
            rng = np.random.default_rng(int(plan["seed"]))
            wanted = sorted(int(r) for r in rng.choice(all_rows, min(n, len(all_rows)),
                                                       replace=False))
        elif limit:
            wanted = all_rows[:int(limit)]
        else:
            wanted = list(all_rows)
        full = mode == "all" and not limit
        target = {"mode": mode, "limit": int(limit) if limit else None,
                  "sample_n": int(sample_n) if mode == "sample" else None,
                  "rows": None if full else wanted, "decided_at": _now()}
        _write_json(target_path, target)
    specimen = _specimen(plan, target)

    done = _results(run_dir)
    # A result rendered under a different lending status is redone -- from the
    # attribution ledger, so it costs a notice build, not a SurvSHAP run.
    todo = [r for r in wanted if r not in done or done[r].get("specimen") != specimen]

    if only is None:
        status.update(state="skipped" if mode == "skip" else "running",
                      mode=_mode_label(target), target=len(wanted),
                      done=len(wanted) - len(todo), n_rejected=len(all_rows),
                      pid=os.getpid(), started_at=time.time(), updated_at=time.time(),
                      rate_per_min=None, eta_seconds=None)
        _write_json(status_path, status)

    if todo:
        model = _verified_model(plan, model, model_path)
        _explain(run_dir, plan, cfg, model, todo, applicant_of, specimen,
                 status_path if only is None else None, workers, say,
                 checkpoint_seconds, design_meta, bg_meta, wanted_n=len(wanted))

    # ------------------------------------------------------------ finish ----
    remaining = [r for r in _targeted(target, all_rows) if r not in _results(run_dir)]
    background_busy = only is not None and read_progress(run_dir).get("alive")
    final = not remaining and not background_busy
    if only is None:
        status = _read_json(status_path, {}) or {}
        status.update(state="skipped" if mode == "skip" else "completed",
                      done=len(wanted), updated_at=time.time(), eta_seconds=0,
                      seconds=round(time.perf_counter() - started, 1))
        _write_json(status_path, status)
    out = materialize(run_dir, final=final, phase2_seconds=time.perf_counter() - started)
    say("done", f"{out.summary['n_explained']:,} of {out.summary['n_rejected']:,} "
                f"rejected applicants explained")
    return out


# ------------------------------------------------------------ explanation --

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _targeted(target: dict | None, all_rows: list[int]) -> list[int]:
    if target is None:
        return list(all_rows)
    if target.get("mode") == "skip":
        return []
    return list(all_rows) if target.get("rows") is None else list(target["rows"])


def _mode_label(target: dict | None) -> str:
    if target is None:
        return ""
    if target.get("mode") == "sample":
        return f"sample of {target.get('sample_n')}"
    if target.get("limit"):
        return f"first {target['limit']}"
    return target.get("mode", "")


def _specimen(plan: dict, target: dict | None) -> str | None:
    """Notices are for lending only when the decisions are, and every rejected
    applicant gets reasons. An undecided run is stamped until it is decided."""
    full = target is not None and target.get("mode") == "all" and target.get("rows") is None
    return None if (plan.get("for_lending_phase1") and full) else NOT_FOR_LENDING


def _verified_model(plan: dict, model, model_path):
    path = Path(model_path or plan["model_path"])
    sha = file_fingerprint(path).get("sha256")
    if sha != plan["model_sha256"]:
        raise BatchError(
            "The model file has changed since these decisions were made, so its "
            "reasons would not explain them.",
            f"{path}: sha256 {str(sha)[:12]}, decisions made with "
            f"{str(plan['model_sha256'])[:12]}",
            "Score the file again with the current model.")
    if model is not None:
        return model
    from .pipeline import load_model_bundle
    tag = path.stem.replace("02_models_", "")
    try:
        bundle, _ = load_model_bundle(path.parent, tag)
    except BaseException as exc:
        if policy_blocked_exception(exc):
            raise BatchError("This machine cannot explain applicants: Windows is "
                             "blocking the libraries the model needs.",
                             f"{type(exc).__name__}: {exc}", policy_block_message()) from exc
        raise
    return bundle["artefacts"][plan["model_name"]]


def _explain(run_dir, plan, cfg, model, todo, applicant_of, specimen, status_path,
             workers, say, checkpoint_seconds, design_meta, bg_meta, *, wanted_n):
    times = np.asarray(plan["times"], dtype=float)
    horizon = int(plan["horizon_months"])
    absent = set(plan.get("absent_features") or [])
    X_all = _load_frame(_p(run_dir, "rejected_design.parquet"), design_meta, rows=todo)
    X_all = X_all.drop(columns=["__applicant_id"], errors="ignore").loc[todo]
    seed_ids = [applicant_of[r] for r in todo]
    checkpoint = float(checkpoint_seconds if checkpoint_seconds is not None else
                       getattr(cfg.decision, "phase2_checkpoint_seconds", 20))
    awake, _ = keep_awake(True)
    buffer_r, buffer_n = [], []
    state = {"done_session": 0, "t0": time.time(), "last_write": 0.0,
             "last_checkpoint": time.time(), "checkpoint_cost": 0.0,
             "first_flush": True}
    base_done = wanted_n - len(todo)

    def flush(force=False):
        now = time.time()
        if buffer_r and (force or now - state["last_write"] >= 1.0):
            _append(run_dir, buffer_r, buffer_n)
            buffer_r.clear()
            buffer_n.clear()
            state["last_write"] = now
            if status_path is not None:
                el = max(now - state["t0"], 1e-6)
                rate = state["done_session"] / el
                left = wanted_n - base_done - state["done_session"]
                st = _read_json(status_path, {}) or {}
                st.update(done=base_done + state["done_session"], updated_at=now,
                          rate_per_min=round(rate * 60, 2),
                          eta_seconds=round(left / rate) if rate > 0 else None)
                _write_json(status_path, st)
            # Reasons reach the files at checkpoints: the first straight away, so the
            # page shows real reasons within seconds, then spaced so that rebuilding
            # a large rejected file never costs more than a fifth of the time. The
            # files are written before progress is announced, so whoever reacts to
            # "N explained" finds those N in the files.
            gap = max(checkpoint, 5 * state["checkpoint_cost"])
            if state["first_flush"] or now - state["last_checkpoint"] >= gap:
                t = time.perf_counter()
                materialize(run_dir, final=False)
                state["checkpoint_cost"] = time.perf_counter() - t
                state["last_checkpoint"] = time.time()
                state["first_flush"] = False
            say("running", f"{base_done + state['done_session']:,} of {wanted_n:,} "
                           f"rejected applicants explained")

    def record(expl, j, row_id):
        aid = applicant_of[row_id]
        notice = build_adverse_action_notice(
            expl, obs=j, applicant_id=aid, horizon_months=horizon,
            model_name=plan["notice_model_label"], specimen=specimen)
        internal = notice.internal_record()
        fname = f"notice_{_slug(aid)}.txt" if notice.reasons else ""
        buffer_r.append({
            "row_id": int(row_id), "applicant_id": aid,
            "status": "explained" if notice.reasons else NO_REASON_NOTE,
            "explainer": plan["explainer"], "specimen": specimen,
            "reasons": [{"feature": r.feature, "reason": r.reason,
                         "attribution": round(float(r.attribution), 6),
                         "direction_consistent": bool(r.direction_consistent)}
                        for r in notice.reasons],
            "fair_lending_flag": ", ".join(notice.fair_lending_flags),
            "flagged": internal.flagged,
            "fair_lending_flags": list(internal.fair_lending_flags),
            "top_driver_not_disclosable": internal.top_driver_not_disclosable,
            "internal_row": internal.to_row(), "internal": internal.to_dict(),
            "notice_file": fname, "at": _now()})
        if notice.reasons:
            # The applicant's text, kept apart from the internal record even here.
            buffer_n.append({"row_id": int(row_id), "notice_file": fname,
                             "applicant_id": aid, "text": notice.render()})
        state["done_session"] += 1

    try:
        if plan["explainer"] == "treeshap":
            for start in range(0, len(todo), TREESHAP_BATCH):
                rows = todo[start:start + TREESHAP_BATCH]
                expl = explain_tree_shap(model, X_all.loc[rows],
                                         horizon_months=float(horizon), times=times)
                expl = _without_features(expl, absent) if absent else expl
                for j, row_id in enumerate(rows):
                    record(expl, j, row_id)
                flush()
        else:
            background = _load_frame(_p(run_dir, "background.parquet"), bg_meta)
            bg = summarise_background(background, n=int(plan["n_background"]),
                                      seed=int(plan["seed"]))
            n_workers = (1 if len(todo) < PARALLEL_MIN_ROWS else
                         workers or plan.get("explain_workers")
                         or suggest_workers(len(todo)))
            ledger = Ledger.load(Path(run_dir) / "explained_rows.jsonl")
            base = model.predict_survival(bg, times).mean(axis=0)   # once per run
            for position, _, phi in iter_explanations(
                    model, X_all, bg, times, row_ids=seed_ids,
                    nsamples=int(plan["nsamples"]), seed=int(plan["seed"]),
                    workers=n_workers, ledger=ledger):
                row_id = todo[position]
                one = X_all.iloc[[position]]
                expl = assemble(model, one, phi[None, ...], times, bg,
                                int(plan["nsamples"]), base=base)
                expl = _without_features(expl, absent) if absent else expl
                record(expl, 0, row_id)
                flush()
        flush(force=True)
    except BatchError:
        raise
    except Exception as exc:
        flush(force=True)                           # keep what was finished
        fix = (policy_block_message() if policy_blocked_exception(exc)
               else "Run Phase 2 again: it resumes from the last applicant finished.")
        if status_path is not None:
            st = _read_json(status_path, {}) or {}
            st.update(state="failed", error=repr(exc), updated_at=time.time())
            _write_json(status_path, st)
        raise BatchError("The applicants were scored, but the reasons could not all "
                         "be generated.", repr(exc), fix) from exc
    finally:
        if awake:
            keep_awake(False)


# ------------------------------------------------------ rebuilding the files --

def _note(row_id: int, results: dict, target: dict | None, targeted: set) -> str:
    if row_id in results:
        return results[row_id]["status"]
    if target is not None and target.get("mode") == "skip":
        return SKIPPED_NOTE
    if row_id not in targeted:
        return CAP_NOTE if target and target.get("limit") else OUTSIDE_SAMPLE_NOTE
    return REASONS_PENDING


def _rewrite(path: Path, fill, chunk_rows: int = 50_000) -> None:
    """Rewrite a decision file in blocks, changing only the reason columns.

    Every value is read and written as text, so nothing else in the file changes.
    """
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    first = True
    for chunk in pd.read_csv(path, dtype=str, keep_default_na=False,
                             chunksize=chunk_rows):
        fill(chunk)
        chunk.to_csv(tmp, index=False, mode="w" if first else "a", header=first)
        first = False
    if first:                                        # empty file: keep it as it is
        return
    os.replace(tmp, path)


COPY_BACK_EVERY = 60.0
"""Seconds between copy-backs from Phase 2 checkpoints; the final one always goes."""
_LAST_COPY_BACK: dict[str, float] = {}


def materialize(run_dir: Path, *, final: bool = False,
                phase2_seconds: float | None = None) -> Phase2Result:
    """Write Phase 2's results into the run's files.

    At a checkpoint: ``rejected_applicants.csv``, the notice zip, the internal
    files, the aggregates and the summary. At the end (``final``) also
    ``scored_applicants.csv``, then every post-run check from disk, then the
    provenance stamp. A notice that fails the screen withholds the zip and fails
    the run loudly, whichever of the two it is.
    """
    run_dir = Path(run_dir)
    plan = _plan(run_dir)
    with _Lock(run_dir):
        results = _results(run_dir)
        notices = _notices(run_dir)
        target = _read_json(_p(run_dir, "target.json"), None)
        ids_frame = pd.read_parquet(_p(run_dir, "rejected_design.parquet"),
                                    columns=["__applicant_id"])
        all_rows = [int(r) for r in ids_frame.index]
        targeted = set(_targeted(target, all_rows))
        chunk_rows = int(plan.get("chunk_rows") or 50_000)

        def fill_rejected(chunk: pd.DataFrame) -> None:
            rows = chunk["row_id"].astype(int).tolist()
            cols = {c: [] for c in ("explained", "explainer", "fair_lending_flag",
                                    "direction_consistent", "notice_file")}
            reason_cols = {f"reason_{i}{s}": [] for i in range(1, MAX_PRINCIPAL_REASONS + 1)
                           for s in ("", "_feature", "_attribution")}
            for r in rows:
                res = results.get(r)
                reasons = res["reasons"] if res else []
                cols["explained"].append(_note(r, results, target, targeted))
                cols["explainer"].append(res["explainer"] if res else "")
                cols["fair_lending_flag"].append(res["fair_lending_flag"] if res else "")
                cols["direction_consistent"].append(
                    str(all(x["direction_consistent"] for x in reasons)) if res else "")
                cols["notice_file"].append(res["notice_file"] if res else "")
                for i in range(1, MAX_PRINCIPAL_REASONS + 1):
                    x = reasons[i - 1] if len(reasons) >= i else None
                    reason_cols[f"reason_{i}"].append(x["reason"] if x else "")
                    reason_cols[f"reason_{i}_feature"].append(x["feature"] if x else "")
                    reason_cols[f"reason_{i}_attribution"].append(
                        str(x["attribution"]) if x else "")
            for c, v in {**cols, **reason_cols}.items():
                if c in chunk.columns:
                    chunk[c] = v

        _rewrite(run_dir / "rejected_applicants.csv", fill_rejected, chunk_rows)

        # -- notices: screened again before they enter the zip --------------
        issued = {r: n for r, n in notices.items()
                  if r in results and results[r]["status"] == "explained"
                  and results[r]["notice_file"] == n["notice_file"]}
        zpath = run_dir / "adverse_action_notices.zip"
        problems = []
        for r, n in issued.items():
            found = find_internal_content(n["text"], feature_names=plan["feature_screen"],
                                          applicant_id=n["applicant_id"])
            if found:
                problems.append(f"{n['notice_file']}: {'; '.join(found)}")
        if issued:
            tmp = zpath.with_name(zpath.name + f".{os.getpid()}.tmp")
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
                for r in sorted(issued):
                    zf.writestr(issued[r]["notice_file"], issued[r]["text"])
            os.replace(tmp, zpath if not problems else
                       run_dir / "adverse_action_notices.WITHHELD.zip")
            if problems:
                zpath.unlink(missing_ok=True)

        # -- internal records: only ever in internal/ -----------------------
        internal = run_dir / INTERNAL_DIR
        internal.mkdir(parents=True, exist_ok=True)
        (internal / "README.txt").write_text(INTERNAL_README, encoding="utf-8")
        ordered = [results[r] for r in sorted(results)]
        pd.DataFrame([x["internal_row"] for x in ordered],
                     columns=list(INTERNAL_COLUMNS)).to_csv(internal / INTERNAL_FLAGS,
                                                            index=False)
        (internal / INTERNAL_RECORDS).write_text(
            "".join(json.dumps(x["internal"], default=str) + "\n" for x in ordered),
            encoding="utf-8")

        # -- aggregates and summary ----------------------------------------
        agg = Aggregates.from_dict(_read_json(run_dir / "aggregates.json", {}))
        agg.reason_counts, agg.flag_features = {}, {}
        agg.n_fair_lending_flagged = agg.n_top_driver_not_disclosable = 0
        agg.n_rejected_explained = len(results)
        for x in ordered:
            for reason in x["reasons"]:
                agg.reason_counts[reason["reason"]] = \
                    agg.reason_counts.get(reason["reason"], 0) + 1
        agg.add_fair_lending([SimpleNamespace(**{k: x[k] for k in (
            "flagged", "fair_lending_flags", "top_driver_not_disclosable")})
            for x in ordered])
        agg.n_reasons_pending = sum(1 for r in targeted if r not in results)
        (run_dir / "aggregates.json").write_text(json.dumps(agg.to_dict(), indent=2),
                                                 encoding="utf-8")

        payload = _read_json(run_dir / "provenance.json")
        s = dict(payload["summary"])
        status = read_progress(run_dir)
        no_reason = sum(1 for x in ordered if x["status"] == NO_REASON_NOTE)
        mode = _mode_label(target)
        stamps = [x for x in str(s.get("not_for_lending_reasons") or "").split("; ")
                  if x and not x.startswith(("reasons ", "explanation skipped",
                                             "a notice failed"))]
        if target and target.get("mode") == "skip":
            stamps.append("explanation skipped: no reasons or notices were produced")
        elif target and target.get("mode") == "sample":
            stamps.append(f"reasons generated for a random sample of "
                          f"{len(targeted):,} of {len(all_rows):,} rejected applicants")
        elif target and target.get("limit"):
            stamps.append(f"reasons capped at the first {len(targeted):,} of "
                          f"{len(all_rows):,} rejected applicants")
        if problems:
            stamps.append("a notice failed the content screen")
        review = float(s.get("fair_lending_review_share", 0.05))
        s.update({
            "phase2_state": status.get("state", ""), "phase2_mode": mode,
            "max_explained": target.get("limit") if target else None,
            "n_explained": len(results),
            "n_reasons_pending": agg.n_reasons_pending,
            "n_rejected_without_reasons": agg.n_rejected_without_reasons,
            "n_explained_without_disclosable_reason": no_reason,
            "n_pending_manual_review": no_reason,
            "n_notices": 0 if problems else len(issued),
            "n_fair_lending_flagged": agg.n_fair_lending_flagged,
            "fair_lending_flag_share": round(agg.fair_lending_share, 4),
            "n_top_driver_not_disclosable": agg.n_top_driver_not_disclosable,
            "fair_lending_flag_features": "; ".join(
                f"{k} ({v})" for k, v in sorted(agg.flag_features.items(),
                                                key=lambda kv: -kv[1])),
            "fair_lending_review_required": bool(
                agg.n_fair_lending_flagged and agg.fair_lending_share > review),
            "explain_workers": plan.get("explain_workers"),
            "not_for_lending_reasons": "; ".join(stamps),
        })
        # Undecided (Phase 2 not chosen yet): the decisions keep their own status.
        full = target is None or (target.get("mode") == "all" and not target.get("limit"))
        s["for_lending_decisions"] = bool(plan.get("for_lending_phase1") and full
                                          and s.get("validation_checks_passed", True)
                                          and not problems)
        if phase2_seconds is not None:
            steps = dict(s.get("seconds_by_step") or {})
            steps["explain"] = round(float(steps.get("explain", 0.0)) + phase2_seconds, 2)
            steps["total"] = round(float(s.get("phase1_seconds") or 0) + steps["explain"], 2)
            s["seconds_by_step"] = steps
            s["phase2_seconds"] = round(phase2_seconds, 1)
            s["seconds_total"] = round(float(s.get("phase1_seconds") or 0)
                                       + phase2_seconds, 1)

        files = {"rejected_applicants.csv": run_dir / "rejected_applicants.csv",
                 f"{INTERNAL_DIR}/{INTERNAL_FLAGS}": internal / INTERNAL_FLAGS,
                 f"{INTERNAL_DIR}/{INTERNAL_RECORDS}": internal / INTERNAL_RECORDS}
        if zpath.exists():
            files["adverse_action_notices.zip"] = zpath
        checks_path = run_dir / "validation_checks.csv"

        if final and not problems:
            answer = {r: results[r] for r in results}

            def fill_scored(chunk: pd.DataFrame) -> None:
                rej = chunk["decision"] == "reject"
                rows = chunk["row_id"].astype(int)
                for idx in chunk.index[rej]:
                    r = int(rows[idx])
                    res = answer.get(r)
                    reasons = res["reasons"] if res else []
                    chunk.at[idx, "explained"] = _note(r, results, target, targeted)
                    chunk.at[idx, "explainer"] = res["explainer"] if res else ""
                    for i in range(1, 4):
                        x = reasons[i - 1] if len(reasons) >= i else None
                        chunk.at[idx, f"reason_{i}"] = x["reason"] if x else ""
                        chunk.at[idx, f"reason_{i}_feature"] = x["feature"] if x else ""

            _rewrite(run_dir / "scored_applicants.csv", fill_scored, chunk_rows)
            checks = verify_run(
                run_dir, n_rows_read=int(s["n_rows"]), threshold=float(plan["threshold"]),
                published_threshold=float(plan["published_threshold"]),
                horizon_months=int(plan["horizon_months"]),
                feature_names=plan["feature_screen"],
                model_approved=bool(plan["model_approved"]),
                model_label=plan["model_label"],
                allow_unapproved=bool(plan["allow_unapproved_model"]),
                cap_note=CAP_NOTE, chunk_rows=chunk_rows,
                notices_zip=zpath if zpath.exists() else run_dir / "__none__.zip",
                # Absent from plans written before the check existed.
                input_quality=plan.get("input_quality"))
            write_checks(checks, checks_path)
            blocking = blocking_failures(checks)
            if blocking and zpath.exists():
                zpath.replace(run_dir / "adverse_action_notices.WITHHELD.zip")
                files.pop("adverse_action_notices.zip", None)
            s.update({
                "validation_checks_passed": not blocking,
                "n_checks_failed": sum(c.status == "FAIL" for c in checks),
                "n_checks_overridden": sum(c.status == "OVERRIDDEN" for c in checks),
                "failed_checks": "; ".join(c.name for c in checks if c.status == "FAIL"),
                "run_status": "failed_checks" if blocking else "finished",
            })
            if blocking:
                s["for_lending_decisions"] = False
                s["not_for_lending_reasons"] = "; ".join(
                    [x for x in stamps] + [f"{len(blocking)} blocking check(s) failed"])
                s["n_notices"] = 0
            payload["phase2"] = build_stamp(
                stage="batch_explain",
                inputs={"model": Path(plan["model_path"]),
                        "decisions": run_dir / "scored_applicants.csv"},
                outputs=dict(files), config_path=PROJECT_ROOT / "config" / "config.yaml",
                args={"mode": mode, "explainer": plan["explainer"],
                      "nsamples": plan["nsamples"], "n_background": plan["n_background"],
                      "seed": plan["seed"],
                      "for_lending_decisions": s["for_lending_decisions"]})
        elif problems:
            s["run_status"] = "failed_checks"
            s["for_lending_decisions"] = False
        payload["summary"] = s
        _write_json(run_dir / "provenance.json", payload)
        pd.DataFrame([s]).to_csv(run_dir / "run_summary.csv", index=False)
        checks = pd.read_csv(checks_path) if checks_path.exists() else pd.DataFrame()

    if problems or (final and s.get("run_status") == "failed_checks"):
        st = _read_json(_p(run_dir, "status.json"), {}) or {}
        st.update(state="failed_checks", updated_at=time.time())
        _write_json(_p(run_dir, "status.json"), st)
        lines = problems or [f"{r['check']}: {r['detail']}" for _, r in
                             checks[checks["status"] == "FAIL"].iterrows()]
        (run_dir / "RUN_FAILED_CHECKS.txt").write_text(
            "Phase 2 of this run failed its checks. The decisions stand; the applicant "
            "notices were withheld and must not be sent.\n\n" + "\n".join(lines) + "\n",
            encoding="utf-8")
        raise BatchError(
            "The reasons were generated, but the notices failed their checks "
            "(applicant_notices_clean) and were withheld."
            if problems else
            f"Phase 2 failed its checks: {s.get('failed_checks')}; the notices were "
            f"withheld.", "\n".join(lines),
            "The notices must not be sent. This is a defect to fix, not a file to "
            "correct.", run_dir=run_dir)

    # In the WSL copy only: reasons to Windows. At the end, and at checkpoints on the
    # way, so the Windows files say "running, N explained" instead of keeping Phase
    # 1's "not started, 0 explained" until the last applicant is done.
    if final or time.time() - _LAST_COPY_BACK.get(str(run_dir), 0.0) >= COPY_BACK_EVERY:
        from .wsl_sync import copy_back_soon
        copy_back_soon()
        _LAST_COPY_BACK[str(run_dir)] = time.time()
    small = len(issued) <= 5_000
    return Phase2Result(
        run_dir=run_dir, summary=s, checks=checks, aggregates=agg, files=files,
        notices=[(issued[r]["notice_file"], issued[r]["text"]) for r in sorted(issued)]
        if small else [], state=s.get("phase2_state", ""))
