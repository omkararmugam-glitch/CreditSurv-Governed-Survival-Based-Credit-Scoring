"""The creditsurv REST API: the scoring pipeline, the registry and the run history
over HTTP.

Every route calls a function the command line and the tests already call:

=====================================  ==============================================
route                                  wraps
=====================================  ==============================================
POST /uploads, POST /uploads/{id}/check  batch.read_upload, schema_match.propose_mapping,
                                       batch.validate (the check scoring itself runs)
POST /runs                             batch.score_file (in this process, shared model)
                                       or runner.launch of 06_score_upload.py (large)
GET  /runs, /runs/{id}, /status, /log  history, batch.load_result, stages, phase2
POST /runs/{id}/phase2                 phase2.launch_background (the runner)
POST /runs/{id}/explain-rows           phase2.explain_run(only=...)
GET  /runs/{id}/files/..., /bundle     the run's own files; batch.bundle_zip
GET  /models, /models/{tag}            registry.load_registry, assess, evaluate_rules
POST /models/{tag}/approve             registry.approve_model
POST /models/{tag}/evidence/{kind}     evidence_jobs.refusal, evidence_jobs.launch_job
GET  /jobs, /jobs/{id}                 runner.list_runs, read_status, tail_log
GET/POST /retrain/...                  plan.holdout_plan, status.preflight, runner.launch
GET  /research/..., /findings          status.overview, stage_status, findings_diff
GET  /overview                         history.overview
=====================================  ==============================================

No scoring, cleaning, explanation or governance rule is written here. A refusal is
the pipeline's own: a :class:`creditsurv.batch.BatchError` becomes HTTP 422 with its
``message``, ``detail`` and ``fix``, word for word what the command line prints.

Start it (from the project root; in WSL where Smart App Control blocks lightgbm):

    python -m uvicorn creditsurv.api.app:app --host 127.0.0.1 --port 8000

It listens on localhost only and has no authentication.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from .. import evidence_jobs, history, phase2, runner
from ..batch import (CAP_NOTE, PARTIAL_SNAPSHOT, RUNS_DIR, BatchError,
                     available_models, bundle_zip, read_upload, validate)
from ..config import load_config
from ..derive import load_costs
from ..environment import blocked_imports, policy_block_message, runtime_label
from ..provenance import PROJECT_ROOT
from ..registry import (APPROVAL_RULES, RegistryError, approve_model, assess,
                        evaluate_rules, evidence_fingerprint, load_registry,
                        registry_path, synced_copy_source)
from ..run_checks import REASONS_PENDING
from ..schema_match import propose_mapping
from ..stages import PIPELINE, STAGES_LOG, StageLog
from . import serialize
from .state import (ContextCache, RunRequest, Scorer, UploadStore, new_run_id,
                    place_input)

__all__ = ["create_app"]

API_VERSION = "1"
DECISION_FILES = ("scored_applicants.csv", "approved_applicants.csv",
                  "rejected_applicants.csv", "adverse_action_notices.zip",
                  "internal/internal_review_flags.csv",
                  "internal/internal_review_records.jsonl")
"""Files that carry decisions. A run that failed its own checks is not finished and
"nothing from this run may be used", so these are not served for it."""


class ApiError(Exception):
    def __init__(self, status: int, message: str, detail: str = "", fix: str = "",
                 **extra):
        self.status, self.message, self.detail, self.fix = status, message, detail, fix
        self.extra = extra


# ------------------------------------------------------------------ bodies --

class CheckBody(BaseModel):
    model_tag: str | None = None
    model: str | None = None
    mapping: dict[str, str] | None = None
    """None = the proposal's pre-selected matches, as scoring would use them."""


class RunBody(BaseModel):
    upload_id: str
    model_tag: str | None = None
    model: str | None = None
    threshold: float | None = None
    mapping: dict[str, str] | None = None
    allow_unapproved_model: bool = False
    background: bool | None = None
    phase2: str = Field("auto", pattern="^(auto|defer)$")


class Phase2Body(BaseModel):
    mode: str = Field("all", pattern="^(all|sample|skip)$")
    sample_n: int | None = None


class ExplainBody(BaseModel):
    row_ids: list[int]


class ApproveBody(BaseModel):
    by: str
    findings: str
    note: str = ""


class RetrainBody(BaseModel):
    size: str = "Small"
    tag: str | None = None
    overwrite: bool = False
    stages: list[str] | None = None
    confirm_replace: bool = False
    confirm_full: bool = False


# -------------------------------------------------------------------- app --

def create_app(cfg=None, *, config_path: str | Path | None = None,
               runs_dir: Path | None = None, jobs_dir: Path | None = None,
               context_loader=None, context_key_fn=None, phase2_launcher=None,
               background_launcher=None, job_launcher=None, retrain_launcher=None,
               root: Path = PROJECT_ROOT, watch_code: bool = True,
               evidence_via_wsl: bool | None = None) -> FastAPI:
    """Build the API. Every argument after ``cfg`` exists so a test can point it at
    its own folders and a stub model; the defaults are the project's."""
    config_path = Path(config_path or PROJECT_ROOT / "config" / "config.yaml")
    cfg = cfg or load_config(config_path)
    runs_dir = Path(runs_dir or RUNS_DIR)
    jobs_dir = Path(jobs_dir or runner.RUNS_DIR)
    d = cfg.decision

    contexts = ContextCache(cfg, loader=context_loader, key_fn=context_key_fn)
    uploads = UploadStore(runs_dir / "_uploads")

    def _phase2(run_dir, mode="all", sample_n=None):
        return phase2.launch_background(Path(run_dir), mode, sample_n,
                                        runs_dir=jobs_dir, config=str(config_path))
    phase2_launcher = phase2_launcher or _phase2
    scorer = Scorer(cfg, contexts, phase2_launcher=phase2_launcher)

    def _background(run_id: str, run_dir: Path, input_path: Path, req: RunRequest,
                    tag: str, model: str) -> Path:
        """A large file goes to the background runner, as the dashboard always sent
        it: 06_score_upload.py in its own process, which outlives any request."""
        args = ["scripts/06_score_upload.py", "--file", str(input_path),
                "--run-dir", str(run_dir), "--model-tag", tag, "--model", model,
                "--config", str(config_path),
                # auto: Phase 2 follows in the same process, with the model it has
                # already loaded, unless the run is large enough to ask first.
                "--phase2", "auto" if req.phase2 == "auto" else "defer"]
        if req.threshold is not None:
            args += ["--threshold", f"{req.threshold}"]
        for uploaded, feature in (req.mapping or {}).items():
            args += ["--map", f"{uploaded}={feature}"]
        if req.allow_unapproved_model:
            args.append("--allow-unapproved-model")
        return runner.launch([{"key": "score", "name": f"Decisions for {input_path.name}",
                               "tag": run_id, "args": args}],
                             lock_tag=run_id, runs_dir=jobs_dir,
                             meta={"source": "api-upload", "run": run_id,
                                   "file": input_path.name, "model_tag": tag,
                                   "threshold": req.threshold, "keep_awake": True})
    background_launcher = background_launcher or _background
    job_launcher = job_launcher or (
        lambda kind, tag: evidence_jobs.launch_job(kind, tag, runs_dir=jobs_dir))
    retrain_launcher = retrain_launcher or (
        lambda stages, lock_tag, meta: runner.launch(stages, lock_tag=lock_tag,
                                                     meta=meta, runs_dir=jobs_dir))

    app = FastAPI(title="creditsurv", version=API_VERSION,
                  description="Survival-based credit scoring: the pipeline over HTTP.")
    app.state.cfg, app.state.contexts, app.state.scorer = cfg, contexts, scorer
    app.state.runs_dir, app.state.jobs_dir, app.state.uploads = runs_dir, jobs_dir, uploads
    started = time.time()

    # -------------------------------------------------------- errors ----
    @app.exception_handler(BatchError)
    def _batch_error(_: Request, exc: BatchError):
        return JSONResponse(status_code=422, content={
            "message": exc.message, "detail": exc.detail, "fix": exc.fix,
            "run_id": Path(exc.run_dir).name if exc.run_dir else None})

    @app.exception_handler(ApiError)
    def _api_error(_: Request, exc: ApiError):
        return JSONResponse(status_code=exc.status, content={
            "message": exc.message, "detail": exc.detail, "fix": exc.fix, **exc.extra})

    @app.exception_handler(RegistryError)
    def _registry_error(_: Request, exc: RegistryError):
        return JSONResponse(status_code=503, content={
            "message": "The model registry could not be read.", "detail": str(exc),
            "fix": "Check config/models.yaml: python scripts/07_model_registry.py check"})

    def record_or_404(run_id: str):
        rec = history.get_run(runs_dir, run_id, live=scorer.active, jobs_dir=jobs_dir)
        if rec is None:
            raise ApiError(404, f"No run called {run_id!r}.")
        return rec

    # --------------------------------------------------------- about ----
    @app.get("/health")
    def health():
        return {"ok": True, "api_version": API_VERSION, "runtime": runtime_label(),
                "blocked_imports": [c.name for c in blocked_imports()],
                "models": contexts.describe(), "scoring_now": sorted(scorer.active),
                "uptime_seconds": round(time.time() - started, 1)}

    @app.get("/meta")
    def meta():
        """The settings and fixed wording the dashboard shows, so it never reads
        the config or imports the pipeline to get them."""
        return {
            "decision": {k: getattr(d, k, None) for k in (
                "model_tag", "model", "horizon_months", "reject_at_or_above",
                "background_above_mb", "explain_confirm_above", "explain_seconds_each",
                "min_feature_coverage", "input_quality_max_share",
                "fair_lending_review_share")},
            "text": {"reasons_pending": REASONS_PENDING, "cap_note": CAP_NOTE,
                     "partial_snapshot": PARTIAL_SNAPSHOT},
            "pipeline": [{"key": k, "label": v} for k, v in PIPELINE],
            # Which run states have final decisions. Published so a page can default
            # to the newest finished run without importing the history module.
            "finished_states": list(history.FINISHED_STATES),
            "runtime": runtime_label(),
            "blocked_imports": [c.name for c in blocked_imports()],
            # Why, in the words the command line prints: Smart App Control, and
            # what to do instead (serve the API from WSL).
            "blocked_message": policy_block_message(blocked_imports()),
            "synced_from": synced_copy_source(registry_path(cfg).parent.parent),
            "api_version": API_VERSION,
        }

    # ------------------------------------------------------- uploads ----
    @app.post("/uploads", status_code=201)
    def upload(file: UploadFile = File(...)):
        up = uploads.save(file.filename or "upload.csv", file.file)
        try:
            with open(up.path, "rb") as fh:
                head = read_upload(fh.read(2_000_000), up.path.name)
        except BatchError:
            uploads.delete(up.upload_id)
            raise
        return {"upload_id": up.upload_id, "name": up.path.name,
                "size_bytes": up.size, "columns": [str(c) for c in head.columns]}

    def upload_or_404(upload_id: str):
        up = uploads.get(upload_id)
        if up is None:
            raise ApiError(404, "That upload is no longer held.",
                           fix="Upload the file again.")
        return up

    @app.get("/uploads/{upload_id}")
    def upload_info(upload_id: str):
        up = upload_or_404(upload_id)
        return {"upload_id": up.upload_id, "name": up.path.name, "size_bytes": up.size}

    @app.delete("/uploads/{upload_id}")
    def upload_delete(upload_id: str):
        return {"deleted": uploads.delete(upload_id)}

    @app.post("/uploads/{upload_id}/check")
    def upload_check(upload_id: str, body: CheckBody):
        """How the file's columns will be read by this model, before anything is
        scored: the proposal to confirm, then :func:`batch.validate` -- the check
        scoring itself makes -- on the mapping given (or the proposed one)."""
        up = upload_or_404(upload_id)
        tag, model = contexts.resolve(body.model_tag, body.model)
        with open(up.path, "rb") as fh:
            head = read_upload(fh.read(2_000_000), up.path.name).head(500)
        ctx = contexts.get(tag, model)
        proposal = propose_mapping(head, ctx.spec, values=ctx.clean_values)
        mapping = (dict(body.mapping) if body.mapping is not None else
                   {k: v for k, v in proposal.mapping().items() if k != v})
        chosen = list(mapping.values())
        duplicates = sorted({f for f in chosen if chosen.count(f) > 1})
        # The same two-rule gate scoring applies, so /check cannot report a file as
        # fine and then have the run fill or refuse a column behind it.
        costs = ctx.costs_with_training(
            load_costs(ctx.model_tag, cfg.paths.tables_dir))
        action = str(getattr(cfg.decision, "unlearned_missing_action", "fill"))
        _, report = validate(head, ctx.spec, values=ctx.clean_values, costs=costs,
                             mapping=mapping, fill_values=ctx.unlearned_fill(costs),
                             unlearned_action=action)
        return {
            "upload_id": up.upload_id, "model_tag": tag, "model": model,
            "proposal": serialize.frame(proposal.to_frame()),
            "conflicts": proposal.conflicts,
            "recognised_but_unused": proposal.recognised_but_unused,
            "features": list(ctx.spec.all_columns),
            "mapping": mapping, "duplicate_targets": duplicates,
            "derived": report.derived, "blocked": report.blocked,
            "required_missing": report.required_missing,
            "optional_missing": report.optional_missing,
            "optional_cost": costs.describe(report.optional_missing),
            "required_rule": report.required_rule,
            "rule_note": report.required_rule_note,
            "unlearned_missing": report.unlearned_missing,
            "unlearned_rule_note": report.unlearned_rule_note,
            "unlearned_action": report.unlearned_action,
            "filled_from_training": dict(report.filled_from_training),
            "ok": not report.required_missing and not duplicates,
        }

    # ---------------------------------------------------------- runs ----
    @app.post("/runs", status_code=202)
    def start_run(body: RunBody):
        up = upload_or_404(body.upload_id)
        tag, model = contexts.resolve(body.model_tag, body.model)
        req = RunRequest(upload_id=up.upload_id, model_tag=tag, model=model,
                         threshold=body.threshold, mapping=body.mapping,
                         allow_unapproved_model=body.allow_unapproved_model,
                         background=body.background, phase2=body.phase2)
        run_id = new_run_id(up.path.name, runs_dir)
        run_dir = runs_dir / run_id
        input_path = place_input(up, run_dir)
        background = (body.background if body.background is not None
                      else up.size / 1e6 >= float(d.background_above_mb))
        if background:
            # Recorded before the process exists, so the run is visible (and shown
            # as starting) from the moment this request returns.
            StageLog(run_dir)("check", "running", "starting the background process")
            try:
                job = background_launcher(run_id, run_dir, input_path, req, tag, model)
            except runner.LockHeld as exc:
                raise ApiError(409, str(exc))
            return {"run_id": run_id, "background": True, "job_id": Path(job).name}
        scorer.submit(run_id, run_dir, input_path, up.path.name, req)
        return {"run_id": run_id, "background": False, "job_id": None}

    @app.get("/runs")
    def runs(limit: int = Query(200, ge=1, le=5000), state: str | None = None):
        out = [r.to_dict() for r in history.list_runs(runs_dir, live=scorer.active,
                                                      jobs_dir=jobs_dir)]
        if state:
            out = [r for r in out if r["state"] == state]
        return {"runs": out[:limit], "total": len(out)}

    @app.get("/runs/{run_id}")
    def run(run_id: str, preview_rows: int = Query(25, ge=0, le=2000)):
        rec = record_or_404(run_id)
        return serialize.run_detail(rec.path, rec, preview_rows=preview_rows)

    @app.get("/runs/{run_id}/status")
    def run_status(run_id: str, log_lines: int = Query(0, ge=0, le=5000)):
        rec = record_or_404(run_id)
        out = serialize.run_status(rec.path, rec)
        if log_lines:
            out["log"] = _run_log(rec.path, log_lines)
        return out

    @app.get("/runs/{run_id}/log")
    def run_log(run_id: str, lines: int = Query(200, ge=1, le=100_000)):
        rec = record_or_404(run_id)
        return {"log": _run_log(rec.path, lines)}

    def _run_log(run_dir: Path, lines: int) -> str:
        """The run's stage log, then the log of every background process that
        worked on it (Phase 1 of a large file, Phase 2)."""
        parts = []
        p = run_dir / STAGES_LOG
        if p.exists():
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
        for job in _jobs_for(run_dir.name):
            text = runner.tail_log(job, lines)
            if text:
                parts.append(f"--- {job.name} ---\n{text}")
        return "\n".join("\n".join(parts).splitlines()[-lines:])

    def _jobs_for(run_id: str) -> list[Path]:
        out = []
        for job in runner.list_runs(jobs_dir):
            try:
                plan = json.loads((job / "plan.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if plan.get("lock_tag") in (run_id, f"{run_id}_explain") or \
                    (plan.get("meta") or {}).get("run") == run_id:
                out.append(job)
        return sorted(out)

    def _finished_or_409(rec) -> None:
        if rec.state == "failed checks":
            raise ApiError(409, "This run failed its own checks, so its outputs are "
                                "not finished and no notices will be produced.",
                           fix="See validation_checks.csv for which check failed.")
        if not (rec.path / "provenance.json").exists():
            raise ApiError(409, f"The decisions for {rec.run_id} are not ready "
                                f"({rec.state}).")

    @app.post("/runs/{run_id}/phase2", status_code=202)
    def run_phase2(run_id: str, body: Phase2Body):
        rec = record_or_404(run_id)
        _finished_or_409(rec)
        if body.mode == "sample" and not body.sample_n:
            raise ApiError(422, "A sample needs a size.", fix="Pass sample_n.")
        try:
            job = phase2_launcher(rec.path, body.mode, body.sample_n)
        except runner.LockHeld as exc:
            raise ApiError(409, "Phase 2 is already running for this run.", str(exc))
        return {"run_id": run_id, "job_id": Path(job).name if job else None,
                "mode": body.mode}

    @app.get("/runs/{run_id}/pending-rows")
    def pending(run_id: str, limit: int = Query(500, ge=1, le=5000)):
        rec = record_or_404(run_id)
        _finished_or_409(rec)
        return {"rows": serialize.frame(phase2.pending_rows(rec.path, limit))}

    @app.post("/runs/{run_id}/explain-rows")
    def explain_rows(run_id: str, body: ExplainBody):
        """Phase 2 for named applicants now, with the model this API already holds
        -- the dashboard's "explain one applicant" button."""
        rec = record_or_404(run_id)
        _finished_or_409(rec)
        s = rec.summary
        ctx = contexts.get(s.get("model_tag"), s.get("model"))
        phase2.explain_run(rec.path, cfg, only=list(body.row_ids), model=ctx.model,
                           model_path=ctx.model_path, workers=1)
        return {"results": [_result(rec.path, r) for r in body.row_ids]}

    def _result(run_dir: Path, row_id: int) -> dict:
        res, text = phase2.result_for(run_dir, int(row_id))
        return {"row_id": int(row_id), "result": res, "notice_text": text}

    @app.get("/runs/{run_id}/results/{row_id}")
    def row_result(run_id: str, row_id: int):
        return _result(record_or_404(run_id).path, row_id)

    @app.get("/runs/{run_id}/files")
    def files(run_id: str):
        rec = record_or_404(run_id)
        snap = (phase2.snapshot(rec.path, rec.summary)
                if (rec.path / "provenance.json").exists() else {"partial": False})
        return {"files": serialize.list_files(rec.path), "snapshot": snap,
                "locked": _locked(snap)}

    def _locked(snap: dict) -> bool:
        return bool(snap.get("partial")) and snap.get("state") in serialize.STILL_WORKING

    @app.get("/runs/{run_id}/files/{name:path}")
    def file(run_id: str, name: str):
        rec = record_or_404(run_id)
        allowed = {f["name"] for f in serialize.list_files(rec.path)}
        if name not in allowed:
            raise ApiError(404, f"{name} is not an output of {run_id}.")
        if rec.state == "failed checks" and name in DECISION_FILES:
            raise ApiError(409, "This run failed its own checks: its decisions and "
                                "notices may not be used, so they are not served.",
                           fix="validation_checks.csv and RUN_FAILED_CHECKS.txt say why.")
        snap = {"partial": False}
        if (rec.path / "provenance.json").exists():
            snap = phase2.snapshot(rec.path, rec.summary)
            if _locked(snap):
                raise ApiError(409, "Phase 2 is still writing this run's files; "
                                    "downloads open when it finishes.",
                               snapshot=snap)
        media = ("application/zip" if name.endswith(".zip") else
                 "application/json" if name.endswith(".json") else "text/csv"
                 if name.endswith(".csv") else "text/plain")
        return FileResponse(rec.path / name, media_type=media,
                            filename=serialize.download_name(name, snap))

    @app.get("/runs/{run_id}/bundle")
    def bundle(run_id: str):
        rec = record_or_404(run_id)
        _finished_or_409(rec)
        snap = phase2.snapshot(rec.path, rec.summary)
        if _locked(snap):
            raise ApiError(409, "Phase 2 is still writing this run's files; the "
                                "bundle opens when it finishes.", snapshot=snap)
        data = bundle_zip(serialize.cached_result(rec.path))
        fname = (f"{run_id}_PARTIAL_{snap['pending']}_reasons_pending.zip"
                 if snap.get("partial") else f"{run_id}.zip")
        return Response(data, media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{fname}"'})

    # -------------------------------------------------------- models ----
    rules_cache: dict = {}

    def all_rules(reg) -> dict:
        key = (str(reg.path), evidence_fingerprint(cfg, reg.path))
        if key not in rules_cache:
            rules_cache.clear()
            rules_cache[key] = {tag: [{"rule": r.rule, "passed": r.passed,
                                       "detail": r.detail}
                                      for r in evaluate_rules(cfg, rec)]
                                for tag, rec in reg.models.items()}
        return rules_cache[key]

    def _model(tag, rec, rules, reg) -> dict:
        a = assess(tag, cfg, registry=reg)
        return {"tag": tag, "status": rec.status, "summary": rec.summary,
                "status_reason": rec.status_reason, "label": a.label,
                "approved": a.approved, "problems": a.problems,
                "rules": rules.get(tag, []),
                "all_rules_pass": all(r["passed"] for r in rules.get(tag, [])),
                "approval": rec.approval, "validation_metrics": rec.validation_metrics,
                "training": rec.training, "known_defects": rec.known_defects,
                "non_disclosable_features": rec.non_disclosable_features,
                "n_features": len(rec.all_features)}

    @app.get("/models")
    def models():
        reg = load_registry(cfg)
        rules = all_rules(reg)
        return {"models": [_model(t, r, rules, reg) for t, r in sorted(reg.models.items())],
                "available": available_models(cfg.paths.models_dir),
                "default": d.model_tag, "registry": str(reg.path)}

    @app.get("/models/rules")
    def model_rules():
        return {"rules": [{"rule": r, "short": r.split("_", 1)[0], "requires": t}
                          for r, t in APPROVAL_RULES.items()]}

    @app.get("/models/{tag}")
    def model(tag: str):
        reg = load_registry(cfg)
        rec = reg.get(tag)
        if rec is None:
            raise ApiError(404, f"{tag} is not in the registry.")
        rules = all_rules(reg)
        out = _model(tag, rec, rules, reg)
        out["evidence_jobs"] = _evidence(tag, rules)
        return out

    def _evidence(tag: str, rules: dict) -> list[dict]:
        passed = {r["rule"]: r["passed"] for r in rules.get(tag, [])}
        out = []
        for kind, job in evidence_jobs.JOBS.items():
            out.append({"kind": kind, "label": job.label, "rule": job.rule,
                        "minutes": job.minutes,
                        "command": f"python {job.script} {' '.join(job.args(tag))}",
                        "refusal": evidence_jobs.refusal(
                            kind, tag, rule_passed=passed.get(job.rule, False),
                            tables_dir=cfg.paths.tables_dir, runs_dir=jobs_dir,
                            via_wsl=evidence_via_wsl)})
        return out

    @app.post("/models/{tag}/evidence/{kind}", status_code=202)
    def run_evidence(tag: str, kind: str):
        reg = load_registry(cfg)
        if reg.get(tag) is None:
            raise ApiError(404, f"{tag} is not in the registry.")
        job = next((j for j in _evidence(tag, all_rules(reg)) if j["kind"] == kind), None)
        if job is None:
            raise ApiError(404, f"No evidence job called {kind!r}.")
        if job["refusal"]:
            raise ApiError(409, job["refusal"])
        try:
            run_dir = job_launcher(kind, tag)
        except runner.LockHeld as exc:
            raise ApiError(409, str(exc))
        return {"job_id": Path(run_dir).name}

    @app.post("/models/{tag}/approve")
    def approve(tag: str, body: ApproveBody):
        outcome = approve_model(cfg, tag, by=body.by, findings=body.findings,
                                note=body.note)
        results = [{"rule": r.rule, "passed": r.passed, "detail": r.detail}
                   for r in outcome.results]
        if not outcome.approved:
            raise ApiError(409, f"Not approved. {outcome.refusal}", results=results)
        rules_cache.clear()
        return {"approved": True, "tag": tag, "results": results}

    @app.get("/registry/context")
    def registry_context():
        path = registry_path(cfg)
        return {"registry": str(path), "runtime": runtime_label(),
                "synced_from": synced_copy_source(path.parent.parent),
                "jobs_via_wsl": (evidence_jobs.runs_in_wsl() if evidence_via_wsl is None
                                 else evidence_via_wsl)}

    # ---------------------------------------------------------- jobs ----
    def _job(job: Path, lines: int = 0) -> dict:
        status = runner.read_status(job)
        try:
            plan = json.loads((job / "plan.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            plan = {}
        out = {"job_id": job.name, "state": runner.effective_state(job),
               "meta": plan.get("meta") or {}, "lock_tag": plan.get("lock_tag"),
               "created": status.get("created"), "finished": status.get("finished"),
               "failed_stage": status.get("failed_stage"), "stages": status.get("stages"),
               "commands": ["python " + " ".join(s.get("args") or s.get("argv") or [])
                            for s in plan.get("stages") or []]}
        if lines:
            log = runner.tail_log(job, lines)
            out["log"] = log
            out["progress"] = evidence_jobs.parse_progress(log)
        return out

    @app.get("/jobs")
    def jobs(source: str | None = None, limit: int = Query(50, ge=1, le=1000)):
        out = []
        for job in runner.list_runs(jobs_dir):
            try:
                j = _job(job)
            except (OSError, ValueError, KeyError):
                continue
            if source and j["meta"].get("source") not in source.split(","):
                continue
            out.append(j)
            if len(out) >= limit:
                break
        return {"jobs": out}

    @app.get("/jobs/{job_id}")
    def job(job_id: str, lines: int = Query(200, ge=0, le=200_000)):
        path = jobs_dir / job_id
        if "/" in job_id or "\\" in job_id or not (path / "status.json").exists():
            raise ApiError(404, f"No job called {job_id!r}.")
        return _job(path, lines)

    # ------------------------------------------------------- retrain ----
    from ..plan import SIZES, STAGE_KEYS, TagError, holdout_plan
    from ..status import preflight

    def _plan(size: str, tag: str | None, overwrite: bool, stages) -> dict:
        if size not in SIZES:
            raise ApiError(422, f"size must be one of {list(SIZES)}")
        try:
            plan = holdout_plan(size, overwrite=overwrite, tag=tag or None,
                                only=tuple(stages) if stages is not None else None)
        except TagError as exc:
            raise ApiError(422, f"Tag refused: {exc}")
        hits = preflight(plan, cfg.paths)

        def rel(p):
            try:
                return Path(p).resolve().relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                return str(p)
        files = [rel(p) for v in hits.values() for p in v]
        return {"plan": plan, "out": {
            "size": size, "tag": plan.tag, "strat_tag": plan.strat_tag,
            "overwrite": overwrite, "write_findings": plan.write_findings,
            "stages": [{"key": s.key, "name": s.name, "tag": s.tag,
                        "command": "python " + " ".join(s.args)} for s in plan.stages],
            "preflight": {k: [rel(p) for p in v] for k, v in hits.items()},
            "n_existing": len(files),
            "n_untracked": sum(f.startswith(("outputs/models/", "outputs/data/"))
                               for f in files),
            "lock": runner.active_lock(plan.tag, jobs_dir),
            "blocked_imports": [c.name for c in blocked_imports()]}}

    def _free_tag(size: str) -> str:
        base = SIZES[size].tag
        if size == "Full":
            return base
        for tag in [base] + [f"{base}_{n}" for n in range(2, 100)]:
            if not preflight(holdout_plan(size, tag=tag), cfg.paths):
                return tag
        return base

    @app.get("/retrain/options")
    def retrain_options():
        return {"sizes": {k: {"tag": v.tag, "full_data": v.full_data,
                              "free_tag": _free_tag(k)} for k, v in SIZES.items()},
                "stage_keys": list(STAGE_KEYS)}

    @app.get("/retrain/plan")
    def retrain_plan(size: str = "Small", tag: str | None = None,
                     overwrite: bool = False, stages: str | None = None):
        keys = [s for s in stages.split(",") if s] if stages is not None else None
        return _plan(size, tag, overwrite, keys)["out"]

    @app.post("/retrain", status_code=202)
    def retrain(body: RetrainBody):
        """Start the holdout sequence. Every refusal the page shows is enforced
        here too, so no client can skip one."""
        built = _plan(body.size, body.tag, body.overwrite, body.stages)
        plan, out = built["plan"], built["out"]
        if not plan.stages:
            raise ApiError(422, "No stages selected.")
        if out["n_existing"] and not body.overwrite:
            raise ApiError(409, f"{out['n_existing']} output file(s) already exist for "
                                f"this tag; the scripts would refuse to run.",
                           fix="Use a new tag, or set overwrite.", preflight=out["preflight"])
        if out["n_existing"] and not body.confirm_replace:
            raise ApiError(409, f"Overwrite would replace {out['n_existing']} file(s) "
                                f"({out['n_untracked']} not in git).",
                           fix="Confirm the replacement (confirm_replace).")
        if body.size == "Full" and not body.confirm_full:
            raise ApiError(409, "A Full run takes 35-55 minutes and most of the "
                                "machine's memory.", fix="Confirm it (confirm_full).")
        if out["lock"]:
            raise ApiError(409, f"Tag {plan.tag!r} is in use by run "
                                f"{out['lock'].get('run_id')}.")
        if out["blocked_imports"]:
            raise ApiError(409, "This server cannot run the pipeline: "
                                + ", ".join(out["blocked_imports"]) + " cannot be loaded.",
                           fix="Start the API in WSL (run_linux.ps1).")
        try:
            job = retrain_launcher(
                [{"key": s.key, "name": s.name, "tag": s.tag, "args": list(s.args)}
                 for s in plan.stages], plan.tag,
                {"size": body.size, "tag": plan.tag, "overwrite": body.overwrite,
                 "writes_findings": plan.write_findings, "source": "ui",
                 "heartbeat_seconds": 30, "show_progress": True})
        except runner.LockHeld as exc:
            raise ApiError(409, str(exc))
        return {"job_id": Path(job).name, "tag": plan.tag}

    # ------------------------------------------------------ research ----
    from .. import status as research
    research_cache: dict = {}

    @app.get("/research/overview")
    def research_overview():
        """Provenance of every research result (hashes inputs, so cached 60 s)."""
        hit = research_cache.get("overview")
        if hit and time.time() - hit[0] < 60:
            return hit[1]
        ov = research.overview(cfg.paths.tables_dir)
        out = {"tags": {t: [asdict(r) for r in rows] for t, rows in ov.items()}}
        research_cache["overview"] = (time.time(), out)
        return out

    @app.get("/research/tags")
    def research_tags():
        return {"tags": research.discover_tags(cfg.paths.tables_dir)}

    @app.get("/research/tags/{variant}")
    def research_tag(variant: str):
        base = variant[: -len("_strat")] if variant.endswith("_strat") else variant
        if base not in research.discover_tags(cfg.paths.tables_dir):
            raise ApiError(404, f"No research results for {variant!r}.")
        fp = research.fingerprint_cache()
        rows = [research.stage_status(cfg.paths.tables_dir, base, k, fp)
                for k in research.RESULT_STAGES]
        tables = sorted(p for p in Path(cfg.paths.tables_dir).iterdir()
                        if p.is_file() and p.stem.endswith(f"_{variant}"))
        return {"variant": variant,
                "stages": [asdict(r) for r in rows if r.tag == variant],
                "files": [{"name": p.name, "kind": p.suffix.lstrip(".")} for p in tables],
                "figures": [p.name for p in research.figures_for_tag(
                    cfg.paths.figures_dir, variant)]}

    @app.get("/research/files/{name}")
    def research_file(name: str):
        if "/" in name or "\\" in name or name.startswith("."):
            raise ApiError(404, "No such file.")
        for folder in (cfg.paths.tables_dir, cfg.paths.figures_dir):
            p = Path(folder) / name
            if p.is_file():
                return FileResponse(p, filename=name)
        raise ApiError(404, f"{name} not found.")

    @app.get("/findings")
    def findings():
        path = root / "FINDINGS.md"
        if not path.exists():
            raise ApiError(404, "FINDINGS.md not found.")
        return {"text": path.read_text(encoding="utf-8"),
                "diff": research.findings_diff(root)}

    # ------------------------------------------------------ overview ----
    @app.get("/overview")
    def overview(for_lending_only: bool = True, trend_n: int = Query(20, ge=2, le=500),
                 date_from: str | None = None, date_to: str | None = None,
                 model_tag: str | None = None):
        """The Overview page, from the run history and the registry.

        ``date_from``, ``date_to`` (ISO dates) and ``model_tag`` filter the
        ``history`` block only -- the processing history the page's filter panel
        drives. The rest of the payload is unfiltered, so the registry panel and the
        recent-runs list do not change as the history window moves.
        """
        try:
            reg = load_registry(cfg)
        except RegistryError:
            reg = None
        runs_ = history.list_runs(runs_dir, live=scorer.active, jobs_dir=jobs_dir)
        out = history.overview(runs_, reg, trend_n=trend_n,
                               for_lending_only=for_lending_only)
        out["history"] = history.processing_history(
            runs_, date_from=date_from, date_to=date_to, model_tag=model_tag,
            for_lending_only=for_lending_only)
        return out

    # -------------------------------------------------- code in WSL ----
    if watch_code:
        from ..wsl_sync import start_restart_watcher
        start_restart_watcher(root, busy=scorer.busy, name="api-server")
    return app


_APP: dict = {}


def __getattr__(name: str):
    """``uvicorn creditsurv.api.app:app`` builds the project's app on first use, so
    importing this module (as the tests do, to build their own) costs nothing.
    ``CREDITSURV_CONFIG`` points it at another config file."""
    if name == "app":
        if "app" not in _APP:
            _APP["app"] = create_app(config_path=os.environ.get("CREDITSURV_CONFIG")
                                     or None)
        return _APP["app"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
