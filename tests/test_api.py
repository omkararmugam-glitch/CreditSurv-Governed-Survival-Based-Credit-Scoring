"""The REST API, end to end on the stub model from test_batch.

Two kinds of test. The first kind drives the API as the dashboard does: upload, check
the columns, start a run, poll its status, read the result, download files, list and
approve models. The second pins the safeguards: every one that holds when
score_file is called directly holds, identically, when the API calls it -- approval,
the input-quality gate, duplicates, the notice / internal split, and the decisions
themselves.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import test_batch as tb  # noqa: E402
from creditsurv import phase2  # noqa: E402
from creditsurv.api.app import create_app  # noqa: E402
from creditsurv.batch import score_file  # noqa: E402
from creditsurv.explain.adverse_action import INTERNAL_MARKERS  # noqa: E402

TERMINAL = {"finished", "failed checks", "refused", "reasons pending",
            "awaiting phase 2 choice", "phase 2 stopped", "interrupted"}


# ------------------------------------------------------------------ fixtures --

@pytest.fixture
def cfg(tmp_path):
    """test_batch's config: everything under tmp_path."""
    return tb.Config(paths=tb.Paths(data_dir=tmp_path / "data",
                                    models_dir=tmp_path / "models",
                                    figures_dir=tmp_path / "figures",
                                    tables_dir=tmp_path / "tables",
                                    registry=tmp_path / "models.yaml"),
                     decision=tb.DecisionConfig(
        model_tag="stub", horizon_months=36, reject_at_or_above=0.45,
        explain_nsamples=2 * len(tb.NUMERIC + tb.CATEGORICAL), explain_n_background=8,
        max_explained=3, background_rows=200, explain_workers=1))


@pytest.fixture
def ctx(cfg):
    """test_batch's stub model, approved in a registry of its own."""
    train = tb._training_frame()
    dm = tb.build_design_matrix(train, tb.SPEC, flavour="gbm")
    model_path = tb._write_dummy_model(cfg)
    tb.write_registry(cfg, model_path)
    return tb.ScoringContext(
        cfg=cfg, model_tag="stub", model_name="discrete_hazard", model=tb.StubModel(),
        spec=tb.SPEC, bundle={"artefacts": {"gbm_columns": list(dm.X.columns)}},
        model_path=model_path, background=dm.X, reference=train,
        clean_values=tb.fit_values(train, tb.SPEC, source="stub_training.parquet"),
        policy=tb.policy_from_config(cfg),
        times=np.array([6.0, 12.0, 24.0, 36.0]), data_source=tb._dummy_source(cfg))


def _make(cfg, ctx, tmp_path, *, auto_phase2=True, **kw):
    loads = {"n": 0}

    def loader(_cfg, tag, model):
        loads["n"] += 1
        return ctx

    def explain_now(run_dir, mode="all", sample_n=None):
        # Phase 2 in this process, so a test can wait for it; the real launcher
        # runs the same explain_run in a background process.
        phase2.explain_run(Path(run_dir), cfg, mode=mode, sample_n=sample_n,
                           model=ctx.model, model_path=ctx.model_path, workers=1)
        return None

    app = create_app(cfg, runs_dir=tmp_path / "runs", jobs_dir=tmp_path / "jobs",
                     context_loader=loader, context_key_fn=lambda c, t, m: (t, m),
                     phase2_launcher=explain_now if auto_phase2 else
                     kw.pop("phase2_launcher", None) or explain_now,
                     watch_code=False, root=tmp_path, **kw)
    client = TestClient(app)
    client.loads = loads
    client.app_ = app
    return client


@pytest.fixture
def api(cfg, ctx, tmp_path):
    return _make(cfg, ctx, tmp_path)


def upload(client, data: bytes | None = None, name="applicants.csv") -> str:
    r = client.post("/uploads", files={"file": (name, data or tb._upload(), "text/csv")})
    assert r.status_code == 201, r.text
    return r.json()["upload_id"]


def wait(client, run_id, until=TERMINAL, timeout=120) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        s = client.get(f"/runs/{run_id}/status").json()
        if s["state"] in until and not client.app_.state.scorer.active:
            return s
        time.sleep(0.1)
    raise AssertionError(f"run {run_id} still {s['state']} after {timeout}s")


def run_file(client, data=None, **body) -> tuple[str, dict]:
    uid = upload(client, data)
    r = client.post("/runs", json={"upload_id": uid, **body})
    assert r.status_code == 202, r.text
    run_id = r.json()["run_id"]
    return run_id, wait(client, run_id)


# ------------------------------------------------------------ the happy path --

def test_upload_check_run_poll_and_read_the_result(api):
    uid = upload(api)
    check = api.post(f"/uploads/{uid}/check", json={}).json()
    # The aliased columns in _upload are proposed, not applied quietly.
    assert check["mapping"] == {"inquiries_last_6m": "inq_last_6mths",
                                "loan_purpose": "purpose"}
    assert check["ok"] and not check["required_missing"]

    r = api.post("/runs", json={"upload_id": uid, "mapping": check["mapping"]})
    assert r.status_code == 202 and r.json()["background"] is False
    run_id = r.json()["run_id"]
    status = wait(api, run_id)
    assert status["state"] == "finished"
    # Every stage of the pipeline view ended done, notices included.
    assert {s["key"]: s["state"] for s in status["stages"]} == {
        "check": "done", "clean": "done", "score": "done", "decide": "done",
        "drift": "done", "checks": "done", "explain": "done", "notices": "done"}

    detail = api.get(f"/runs/{run_id}").json()
    s = detail["summary"]
    assert s["n_rows"] == 12 and s["n_approved"] + s["n_rejected"] == 12
    # Every rejected applicant has a notice, or is held for manual review because no
    # reason could be stated in Regulation B wording (then no notice is issued).
    assert s["n_notices"] + s["n_pending_manual_review"] == s["n_rejected"] > 0
    assert s["n_notices"] > 0
    checks = pd.DataFrame(**{k: detail["checks"][k] for k in ("columns", "data")})
    assert (checks["status"] == "PASS").all() and "input_quality" in set(checks["check"])
    assert detail["drift"]["status"] and detail["aggregates"]["risk"]["data"]
    assert len(detail["preview"]["data"]) == 12

    log = api.get(f"/runs/{run_id}/log").json()["log"]
    assert "[Score] done" in log and "[Run checks] done" in log


def test_history_lists_the_run_and_the_overview_counts_it(api):
    run_id, _ = run_file(api)
    runs = api.get("/runs").json()["runs"]
    assert runs[0]["run_id"] == run_id and runs[0]["state"] == "finished"
    ov = api.get("/overview").json()
    assert ov["total_runs"] == 1 and ov["finished_runs"] == 1
    assert ov["trend"][0]["approval_rate"] == runs[0]["approval_rate"]
    assert ov["models"]["approved"][0]["tag"] == "stub"


def test_every_output_can_be_downloaded_and_the_bundle_zips_them(api):
    run_id, st = run_file(api)
    listing = api.get(f"/runs/{run_id}/files").json()
    names = {f["name"] for f in listing["files"]}
    assert {"scored_applicants.csv", "adverse_action_notices.zip",
            "internal/internal_review_flags.csv", "validation_checks.csv"} <= names, \
        (st["state"], st["error"], run_id)
    assert not listing["locked"]
    for name in names:
        r = api.get(f"/runs/{run_id}/files/{name}")
        assert r.status_code == 200, name
    bundle = zipfile.ZipFile(io.BytesIO(api.get(f"/runs/{run_id}/bundle").content))
    assert "scored_applicants.csv" in bundle.namelist()
    # Only the run's outputs by name: never its Phase 2 working files or a path.
    assert api.get(f"/runs/{run_id}/files/phase2/plan.json").status_code == 404
    assert api.get(f"/runs/{run_id}/files/../../x").status_code == 404


def test_the_model_is_loaded_once_and_shared_across_requests(api):
    run_file(api)
    run_file(api)
    uid = upload(api)
    api.post(f"/uploads/{uid}/check", json={})
    assert api.loads["n"] == 1
    assert api.get("/health").json()["models"]["hits"] >= 2


def test_phase2_status_while_deferred_then_started_on_request(cfg, ctx, tmp_path):
    client = _make(cfg, ctx, tmp_path)
    run_id, status = run_file(client, phase2="defer")
    assert status["state"] == "reasons pending"
    assert {s["key"]: s["state"] for s in status["stages"]}["explain"] == "pending"
    pending = client.get(f"/runs/{run_id}/pending-rows").json()["rows"]["data"]
    assert pending

    r = client.post(f"/runs/{run_id}/phase2", json={"mode": "all"})
    assert r.status_code == 202
    status = client.get(f"/runs/{run_id}/status").json()
    assert status["state"] == "finished"
    assert status["phase2"]["done"] == status["phase2"]["target"] > 0


def test_one_applicant_explained_on_demand(cfg, ctx, tmp_path):
    client = _make(cfg, ctx, tmp_path)
    run_id, _ = run_file(client, phase2="defer")
    row = client.get(f"/runs/{run_id}/pending-rows").json()["rows"]
    row_id = int(pd.DataFrame(row["data"], columns=row["columns"])["row_id"].iloc[0])
    out = client.post(f"/runs/{run_id}/explain-rows", json={"row_ids": [row_id]}).json()
    res = out["results"][0]
    assert res["row_id"] == row_id and res["result"]["status"] in (
        "explained", "pending manual review")


def test_large_files_go_to_the_background_runner_with_the_cli(cfg, ctx, tmp_path,
                                                              monkeypatch):
    """The API's background path is the command the dashboard always used:
    06_score_upload.py with --run-dir, in a runner process."""
    from creditsurv import runner
    launched = {}

    def fake_launch(stages, *, lock_tag, meta=None, runs_dir=None, **_):
        launched.update(stages=stages, lock_tag=lock_tag, meta=meta)
        d = Path(runs_dir) / f"job_{lock_tag}"
        d.mkdir(parents=True)
        return d
    monkeypatch.setattr(runner, "launch", fake_launch)
    client = _make(cfg, ctx, tmp_path)
    uid = upload(client)
    r = client.post("/runs", json={"upload_id": uid, "background": True,
                                   "mapping": {"loan_purpose": "purpose"},
                                   "allow_unapproved_model": True, "threshold": 0.4})
    assert r.json()["background"] is True
    args = launched["stages"][0]["args"]
    assert args[0] == "scripts/06_score_upload.py"
    run_id = r.json()["run_id"]
    assert args[args.index("--run-dir") + 1].endswith(run_id)
    assert args[args.index("--phase2") + 1] == "auto"
    assert "--allow-unapproved-model" in args and "loan_purpose=purpose" in args
    assert launched["lock_tag"] == run_id


# ------------------------------------------- safeguards, identical via the API --

def test_decisions_are_identical_to_calling_score_file_directly(api, cfg, ctx, tmp_path):
    data = tb._upload(30)
    direct = score_file(data, "applicants.csv", cfg, ctx=ctx, runs_dir=tmp_path / "direct")
    run_id, _ = run_file(api, data, phase2="defer")
    via_api = api.app_.state.runs_dir / run_id
    cols = ["applicant_id", "pd_12m", "pd_36m", "decision"]
    a = pd.read_csv(direct.run_dir / "scored_applicants.csv", dtype={"applicant_id": str})
    b = pd.read_csv(via_api / "scored_applicants.csv", dtype={"applicant_id": str})
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_an_unapproved_model_is_refused_with_the_pipelines_own_message(cfg, ctx,
                                                                       tmp_path):
    tb.write_registry(cfg, ctx.model_path, status="candidate")
    client = _make(cfg, ctx, tmp_path)
    run_id, status = run_file(client)
    assert status["state"] == "refused"
    err = status["error"]
    assert "is not approved for lending decisions" in err["message"]
    assert "07_model_registry.py rules" in err["fix"]
    assert not (client.app_.state.runs_dir / run_id / "scored_applicants.csv").exists()


def test_the_override_finishes_stamped_not_for_lending(cfg, ctx, tmp_path):
    tb.write_registry(cfg, ctx.model_path, status="candidate")
    client = _make(cfg, ctx, tmp_path)
    run_id, _ = run_file(client, allow_unapproved_model=True)
    s = client.get(f"/runs/{run_id}").json()["summary"]
    assert s["for_lending_decisions"] is False
    assert "not approved" in s["not_for_lending_reasons"]


def test_the_input_quality_gate_fails_the_run_and_withholds_its_decisions(api):
    df = pd.read_csv(io.BytesIO(tb._upload()))
    df["dti"] = df["dti"].astype(object)
    df.loc[1:, "dti"] = "see attached"
    run_id, status = run_file(api, df.to_csv(index=False).encode())
    assert status["state"] == "failed checks"
    stages = {s["key"]: s for s in status["stages"]}
    assert stages["checks"]["state"] == "failed"
    assert "input_quality" in stages["checks"]["message"]
    assert stages["notices"]["state"] == "pending"
    detail = api.get(f"/runs/{run_id}").json()
    checks = pd.DataFrame(detail["checks"]["data"], columns=detail["checks"]["columns"])
    assert checks.set_index("check").loc["input_quality", "status"] == "FAIL"
    # Nothing from the run may be used, so its decision files are not served ...
    for name in ("scored_applicants.csv", "rejected_applicants.csv"):
        assert api.get(f"/runs/{run_id}/files/{name}").status_code == 409
    # ... but the checks that say why are.
    assert api.get(f"/runs/{run_id}/files/validation_checks.csv").status_code == 200
    assert api.post(f"/runs/{run_id}/phase2", json={"mode": "all"}).status_code == 409


def test_duplicates_are_scored_once_through_the_api(api):
    df = pd.read_csv(io.BytesIO(tb._upload()))
    messy = pd.concat([df, df.iloc[[0, 5]]], ignore_index=True)
    run_id, _ = run_file(api, messy.to_csv(index=False).encode())
    s = api.get(f"/runs/{run_id}").json()["summary"]
    assert s["n_rows"] == 12 and s["n_duplicates_removed"] == 2
    scored = pd.read_csv(io.BytesIO(api.get(
        f"/runs/{run_id}/files/scored_applicants.csv").content), dtype={"applicant_id": str})
    assert scored["applicant_id"].is_unique


def test_notices_carry_nothing_internal_and_internal_files_are_marked(api):
    run_id, _ = run_file(api)
    files = {f["name"]: f for f in api.get(f"/runs/{run_id}/files").json()["files"]}
    assert files["internal/internal_review_flags.csv"]["internal"] is True
    assert files["adverse_action_notices.zip"]["internal"] is False
    z = zipfile.ZipFile(io.BytesIO(api.get(
        f"/runs/{run_id}/files/adverse_action_notices.zip").content))
    assert z.namelist()
    for name in z.namelist():
        assert not name.startswith("internal")
        text = z.read(name).decode("utf-8").lower()
        # The run's own applicant_notices_clean check, read again from what the API
        # served: no internal marker in any notice.
        assert "internal review record" not in text


def test_downloads_are_held_while_phase2_is_still_writing(cfg, ctx, tmp_path):
    client = _make(cfg, ctx, tmp_path)
    run_id, _ = run_file(client, phase2="defer")
    run_dir = client.app_.state.runs_dir / run_id
    status = json.loads((run_dir / "phase2" / "status.json").read_text())
    status.update(state="running", pid=__import__("os").getpid(), target=5, done=1)
    (run_dir / "phase2" / "status.json").write_text(json.dumps(status))
    assert client.get(f"/runs/{run_id}/files").json()["locked"] is True
    assert client.get(f"/runs/{run_id}/files/scored_applicants.csv").status_code == 409
    assert client.get(f"/runs/{run_id}/bundle").status_code == 409


def test_a_refused_file_says_why(api):
    r = api.post("/uploads", files={"file": ("a.csv", b"", "text/csv")})
    assert r.status_code == 422 and "is empty" in r.json()["message"]
    r = api.post("/uploads", files={"file": ("a.xlsx", b"x,y\n1,2\n", "text/csv")})
    assert r.status_code == 422 and "not a CSV" in r.json()["message"]
    assert api.post("/runs", json={"upload_id": "0" * 16}).status_code == 404


def test_a_missing_required_column_is_refused_before_scoring(api):
    run_id, status = run_file(api, tb._upload(drop=["dti"]))
    assert status["state"] == "refused"
    assert "missing required column" in status["error"]["message"]


# -------------------------------------------------------------------- models --

def test_models_list_rules_and_assessment(api):
    body = api.get("/models").json()
    stub = next(m for m in body["models"] if m["tag"] == "stub")
    assert stub["approved"] is True and stub["label"] == "approved"
    assert [r["rule"] for r in stub["rules"]][:1] == ["A1_model_file"]
    rules = api.get("/models/rules").json()["rules"]
    assert len(rules) == 7 and rules[0]["short"] == "A1"
    one = api.get("/models/stub").json()
    assert {j["kind"] for j in one["evidence_jobs"]} == {"ablation",
                                                         "explainer_validation"}
    assert api.get("/models/nope").status_code == 404


def test_approval_is_refused_when_rules_fail_and_nothing_is_written(api, cfg, ctx):
    tb.write_registry(cfg, ctx.model_path, status="candidate")
    before = Path(cfg.paths.registry).read_text()
    r = api.post("/models/stub/approve", json={"by": "me", "findings": "7l"})
    assert r.status_code == 409 and r.json()["message"].startswith("Not approved.")
    assert r.json()["results"]            # which rules failed, and why
    assert Path(cfg.paths.registry).read_text() == before


def test_approval_needs_a_name_and_a_findings_section(api, cfg, ctx):
    tb.write_registry(cfg, ctx.model_path, status="candidate")
    r = api.post("/models/stub/approve", json={"by": " ", "findings": "7l"})
    assert r.status_code == 409 and "names who made it" in r.json()["message"]


def test_evidence_jobs_are_refused_by_the_same_rules_as_the_page(api, cfg, monkeypatch):
    from creditsurv import evidence_jobs
    tables = Path(cfg.paths.tables_dir)
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "03e_ablation_stub.json").write_text(json.dumps({"model_tag": "stub"}))
    monkeypatch.setattr(evidence_jobs, "wsl_available", lambda: True)
    r = api.post("/models/stub/evidence/ablation")
    assert r.status_code == 409 and "already exists" in r.json()["message"]
    assert api.post("/models/stub/evidence/nope").status_code == 404


# ------------------------------------------------------------------- retrain --

def test_retrain_refuses_a_primary_tag_and_existing_outputs(api, cfg):
    r = api.get("/retrain/plan", params={"size": "Small", "tag": "full"})
    assert r.status_code == 422 and "Tag refused" in r.json()["message"]
    plan = api.get("/retrain/plan", params={"size": "Small", "tag": "t_api"}).json()
    assert plan["stages"] and plan["write_findings"] is False
    tables = Path(cfg.paths.tables_dir)
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "02_metrics_t_api.json").write_text("{}")
    r = api.post("/retrain", json={"size": "Small", "tag": "t_api", "stages": ["02"]})
    assert r.status_code == 409 and "already exist" in r.json()["message"]
    r = api.post("/retrain", json={"size": "Small", "tag": "t_api", "stages": ["02"],
                                   "overwrite": True})
    assert r.status_code == 409 and "confirm" in r.json()["fix"].lower()
