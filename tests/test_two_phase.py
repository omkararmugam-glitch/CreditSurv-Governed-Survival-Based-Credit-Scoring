"""Decisions first, reasons after: the two-phase pipeline, on small fixtures.

Phase 1 is creditsurv.batch.score_file and Phase 2 is creditsurv.phase2.explain_run:
exactly one function each, used by every caller (dashboard inline and background,
06_score_upload.py, run_batch). The last section fails if a new caller scores or
explains applicants any other way -- the test that makes this a pipeline fix and
not a fix for one file.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import pickle
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import test_batch as tb
from creditsurv.history import processing_history
from creditsurv import batch, phase2
from creditsurv.batch import BatchError, score_file
from creditsurv.explain.adverse_action import NOT_FOR_LENDING, find_internal_content
from creditsurv.phase2 import explain_run, read_progress
from creditsurv.provenance import PROJECT_ROOT
from creditsurv.run_checks import (NO_REASON_NOTE, OUTSIDE_SAMPLE_NOTE,
                                   REASONS_PENDING, SKIPPED_NOTE)
from test_governance import GEO_SPEC, GeoModel, _cfg, _ctx, _geo_training, _geo_upload


def _phase1(tmp_path, n=24, **cfg_kw):
    cfg = _cfg(tmp_path, **cfg_kw)
    ctx = _ctx(cfg)
    res = score_file(tb._upload(n), "a.csv", cfg, ctx=ctx, runs_dir=tmp_path / "runs")
    return cfg, ctx, res


def _rejected(run_dir) -> pd.DataFrame:
    return pd.read_csv(Path(run_dir) / "rejected_applicants.csv",
                       dtype={"applicant_id": str})


def _notice_texts(run_dir) -> dict:
    z = Path(run_dir) / "adverse_action_notices.zip"
    if not z.exists():
        return {}
    with zipfile.ZipFile(z) as zf:
        return {n: zf.read(n).decode("utf-8") for n in zf.namelist()}


# ============================================================== Phase 1 ==

class TestPhase1:
    def test_decisions_are_complete_and_checked_with_nobody_explained(self, tmp_path):
        cfg, ctx, res = _phase1(tmp_path)
        s = res.summary
        assert s["n_approved"] + s["n_rejected"] == s["n_rows"] == 24
        assert s["n_rejected"] > 0
        assert s["run_status"] == "decisions_ready"
        assert s["phase2_state"] == "not started"
        assert s["n_reasons_pending"] == s["n_rejected"]
        assert s["n_rejected_without_reasons"] == 0          # pending is not missing
        rejected = _rejected(res.run_dir)
        assert (rejected["explained"] == REASONS_PENDING).all()   # never blank
        assert not (res.run_dir / "adverse_action_notices.zip").exists()
        checks = pd.read_csv(res.files["validation_checks.csv"])
        assert (checks["status"] == "PASS").all(), checks
        assert (res.run_dir / "provenance.json").exists()
        # What Phase 2 needs is beside the decisions, so it never re-reads the
        # upload or the training data.
        for name in ("plan.json", "rejected_design.parquet", "background.parquet",
                     "status.json"):
            assert (res.run_dir / "phase2" / name).exists(), name

    def test_phase1_explains_nobody_even_if_explaining_would_hang(self, tmp_path,
                                                                  monkeypatch):
        from creditsurv.explain import parallel

        def boom(*a, **k):
            raise AssertionError("Phase 1 must not explain anyone")
        for name in ("iter_explanations", "explain_rows_parallel", "explain_rows"):
            monkeypatch.setattr(parallel, name, boom)
        monkeypatch.setattr(phase2, "iter_explanations", boom)
        monkeypatch.setattr(phase2, "build_adverse_action_notice", boom)
        _, _, res = _phase1(tmp_path)
        assert res.summary["n_rejected"] > 0

    def test_saved_inputs_are_exactly_what_was_scored(self, tmp_path):
        cfg, ctx, res = _phase1(tmp_path)
        plan = json.loads((res.run_dir / "phase2" / "plan.json").read_text())
        X = batch._load_frame(res.run_dir / "phase2" / "rejected_design.parquet",
                              plan["design"]).drop(columns="__applicant_id")
        rejected = _rejected(res.run_dir).set_index("row_id")
        again = ctx.model.predict_survival(X, ctx.times)
        pd_h = np.round(1 - again[:, list(ctx.times).index(36.0)], 4)
        assert np.allclose(pd_h, rejected.loc[X.index, "pd_36m"].to_numpy())
        for c, levels in plan["design"]["categories"].items():
            assert list(X[c].cat.categories) == levels

    def test_a_risk_on_the_rounding_boundary_is_decided_as_written(self, tmp_path):
        """0.29996 is written as 0.3, and 0.3 is a rejection under the published
        rule; deciding on the unrounded value approved it. Found on a 1.3M-row
        file (145 rows), caught by decisions_match_threshold."""
        class Boundary(tb.StubModel):
            def predict_survival(self, X, times):
                times = np.atleast_1d(np.asarray(times, dtype=float))
                pd36 = np.resize([0.29996, 0.29994, 0.30004, 0.1], len(X))
                return np.exp(np.outer(np.log(1 - pd36), times / 36.0))
        cfg = _cfg(tmp_path, reject_at_or_above=0.30)
        res = score_file(tb._upload(8), "a.csv", cfg, ctx=_ctx(cfg, model=Boundary()),
                         runs_dir=tmp_path / "runs", threshold=0.30)
        scored = pd.read_csv(res.files["scored_applicants.csv"])
        assert list(scored["pd_36m"][:4]) == [0.3, 0.2999, 0.3, 0.1]
        assert list(scored["decision"][:4]) == ["reject", "approve", "reject", "approve"]
        checks = pd.read_csv(res.files["validation_checks.csv"]).set_index("check")
        assert checks.loc["decisions_match_threshold", "status"] == "PASS"

    @pytest.mark.parametrize("n", [12, 60, 240])
    def test_phase1_time_does_not_depend_on_how_many_are_rejected(self, tmp_path, n):
        """Phase 1 costs the same per row whatever the reject rate: no explanation
        is hidden in it. Here, a threshold that rejects nearly everyone."""
        cfg = _cfg(tmp_path, reject_at_or_above=0.01)
        t = time.perf_counter()
        res = score_file(tb._upload(n), "a.csv", cfg, ctx=_ctx(cfg, status="approved"),
                         runs_dir=tmp_path / "runs", threshold=0.01,
                         allow_unapproved_model=True)
        assert res.summary["n_rejected"] >= n - 1
        assert time.perf_counter() - t < 20


# ============================================================== Phase 2 ==

class TestPhase2:
    def test_results_are_copied_back_while_it_runs_not_only_at_the_end(
            self, tmp_path, monkeypatch):
        """Copied back only at the end, the Windows files kept Phase 1's "not started,
        0 explained" for the whole of Phase 2 -- on a 1k file, fifty minutes of what
        looked like Phase 2 never starting."""
        from creditsurv import wsl_sync
        cfg, ctx, res = _phase1(tmp_path)
        states = []
        monkeypatch.setattr(wsl_sync, "copy_back_soon",
                            lambda *a, **k: states.append(
                                read_progress(res.run_dir)["state"]))
        monkeypatch.setattr(phase2, "_LAST_COPY_BACK", {})
        explain_run(res.run_dir, cfg, model=ctx.model, model_path=ctx.model_path,
                    checkpoint_seconds=0)
        assert states[0] == "running", states           # a copy mid-run
        assert len(states) >= 2                          # and the final one
        # throttled: one per COPY_BACK_EVERY, not one per checkpoint
        assert len(states) <= 3, states

    def test_progress_is_reported_and_reasons_fill_in_while_it_runs(self, tmp_path):
        cfg, ctx, res = _phase1(tmp_path)
        seen = []

        def progress(step, state, message=""):
            if state == "running" and not seen:
                rej = _rejected(res.run_dir)
                st = read_progress(res.run_dir)
                seen.append((int((rej["explained"] == "explained").sum()),
                             int((rej["explained"] == REASONS_PENDING).sum()),
                             st["state"], st["done"], st["target"]))

        out = explain_run(res.run_dir, cfg, model=ctx.model, model_path=ctx.model_path,
                          progress=progress, checkpoint_seconds=0)
        explained, pending, state, done, target = seen[0]
        assert explained >= 1 and pending >= 1          # part-way: both at once
        assert state == "running" and 1 <= done < target == res.summary["n_rejected"]
        st = read_progress(res.run_dir)
        assert st["state"] == "completed" and st["done"] == st["target"]
        assert st["rate_per_min"] is not None and st["eta_seconds"] == 0
        assert out.summary["run_status"] == "finished"
        assert out.summary["n_reasons_pending"] == 0
        assert not _rejected(res.run_dir)["explained"].eq(REASONS_PENDING).any()
        assert (pd.read_csv(res.run_dir / "validation_checks.csv")["status"]
                == "PASS").all()
        scored = pd.read_csv(res.run_dir / "scored_applicants.csv")
        assert not scored["explained"].eq(REASONS_PENDING).any()
        prov = json.loads((res.run_dir / "provenance.json").read_text())
        assert prov["phase2"]["stage"] == "batch_explain"

    def test_interrupted_phase2_resumes_with_identical_reasons(self, tmp_path):
        cfg, ctx, res = _phase1(tmp_path, n=30)
        calls = {"n": 0}

        def stop_after_three(step, state, message=""):
            # The first progress report comes after the first result is on disk:
            # stopping there is an interruption part-way through.
            if state == "running":
                calls["n"] += 1
                raise KeyboardInterrupt
        with pytest.raises((BatchError, KeyboardInterrupt)):
            explain_run(res.run_dir, cfg, model=ctx.model, model_path=ctx.model_path,
                        progress=stop_after_three, checkpoint_seconds=0)
        partial = phase2._results(res.run_dir)
        assert 0 < len(partial) < res.summary["n_rejected"]
        explain_run(res.run_dir, cfg, model=ctx.model, model_path=ctx.model_path)

        cfg2, ctx2, fresh = _phase1(tmp_path / "again", n=30)
        explain_run(fresh.run_dir, cfg2, model=ctx2.model, model_path=ctx2.model_path)
        a = _rejected(res.run_dir).set_index("row_id")
        b = _rejected(fresh.run_dir).set_index("row_id")
        for col in ("explained", "reason_1", "reason_2", "reason_3", "reason_4"):
            assert a[col].fillna("").equals(b[col].fillna("")), col

    def test_the_model_must_be_the_one_that_made_the_decisions(self, tmp_path):
        cfg, ctx, res = _phase1(tmp_path)
        Path(ctx.model_path).write_bytes(b"retrained since")
        with pytest.raises(BatchError, match="model file has changed"):
            explain_run(res.run_dir, cfg, model=ctx.model, model_path=ctx.model_path)

    def test_a_run_that_failed_phase1_checks_is_never_explained(self, tmp_path):
        from test_governance import BackwardsModel
        cfg = _cfg(tmp_path)
        with pytest.raises(BatchError) as exc:
            score_file(tb._upload(), "a.csv", cfg, ctx=_ctx(cfg, model=BackwardsModel()),
                       runs_dir=tmp_path / "runs")
        with pytest.raises(BatchError, match="no finished decisions"):
            explain_run(exc.value.run_dir, cfg)


# ======================================================== A3. the choice ==

class TestFinishing:
    def test_completed_is_finishing_until_the_final_files_are_written(self, tmp_path):
        """explain_run marks itself completed before it rewrites the final files. In
        that gap the page must not unlock the downloads on the last checkpoint's
        files -- but a process that died there must not stay 'finishing' forever."""
        import os
        cfg, ctx, res = _phase1(tmp_path)
        summary = dict(res.summary, phase2_state="running", n_reasons_pending=3)
        status = res.run_dir / "phase2" / "status.json"
        st = json.loads(status.read_text())
        st.update(state="completed", pid=os.getpid(), done=st.get("n_rejected", 3))
        status.write_text(json.dumps(st))
        snap = phase2.snapshot(res.run_dir, summary)
        assert snap["state"] == "finishing" and snap["partial"]

        st.update(pid=2 ** 22 + 12345)                    # no such process
        status.write_text(json.dumps(st))
        assert phase2.snapshot(res.run_dir, summary)["state"] == "completed"

        # Once the summary agrees, it is simply completed.
        explain_run(res.run_dir, cfg, model=ctx.model, model_path=ctx.model_path)
        snap = phase2.snapshot(res.run_dir)
        assert snap["state"] == "completed" and not snap["partial"]


class TestTheChoice:
    def test_above_the_threshold_phase2_waits_for_a_choice(self, tmp_path):
        cfg, ctx, res = _phase1(tmp_path, explain_confirm_above=2)
        assert res.summary["n_rejected"] > 2
        assert res.summary["phase2_state"] == "awaiting choice"
        assert read_progress(res.run_dir)["state"] == "awaiting choice"

    def test_the_threshold_is_config_applied_to_any_file(self, tmp_path):
        for i, upload in enumerate((tb._upload(24), tb._upload(24, alias=False))):
            cfg = _cfg(tmp_path / str(i), explain_confirm_above=10_000)
            res = score_file(upload, f"file_{i}.csv", cfg, ctx=_ctx(cfg),
                             runs_dir=tmp_path / str(i) / "runs")
            assert res.summary["phase2_state"] == "not started"
            assert res.summary["explain_confirm_above"] == 10_000

    def test_sample_explains_exactly_n_and_stamps_the_run(self, tmp_path):
        cfg, ctx, res = _phase1(tmp_path, n=30)
        n = 3
        out = explain_run(res.run_dir, cfg, mode="sample", sample_n=n, model=ctx.model,
                          model_path=ctx.model_path)
        rej = _rejected(res.run_dir)
        done = rej["explained"].isin(["explained", NO_REASON_NOTE])
        assert done.sum() == n
        assert (rej.loc[~done, "explained"] == OUTSIDE_SAMPLE_NOTE).all()
        assert out.summary["for_lending_decisions"] is False
        assert "random sample" in out.summary["not_for_lending_reasons"]
        assert all(NOT_FOR_LENDING in t for t in _notice_texts(res.run_dir).values())
        assert out.summary["run_status"] == "finished"          # checks passed

    def test_skip_writes_no_notices_and_stamps_the_run(self, tmp_path):
        cfg, ctx, res = _phase1(tmp_path)
        out = explain_run(res.run_dir, cfg, mode="skip")
        assert (_rejected(res.run_dir)["explained"] == SKIPPED_NOTE).all()
        assert not (res.run_dir / "adverse_action_notices.zip").exists()
        assert out.summary["for_lending_decisions"] is False
        assert "skipped" in out.summary["not_for_lending_reasons"]
        assert out.summary["n_rejected_without_reasons"] == out.summary["n_rejected"]
        assert read_progress(res.run_dir)["state"] == "skipped"

    def test_the_cli_stops_after_phase1_and_prints_the_choices(self, tmp_path,
                                                             monkeypatch, capsys):
        from test_governance import _cli_config
        spec = importlib.util.spec_from_file_location(
            "cli2", PROJECT_ROOT / "scripts" / "06_score_upload.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        cli.blocked_imports = lambda: []
        cfg = _cfg(tmp_path, explain_confirm_above=2)
        ctx = _ctx(cfg)
        monkeypatch.setattr(batch, "load_context", lambda *a, **k: ctx)
        cfg_file = _cli_config(tmp_path, cfg)
        raw = yaml.safe_load(cfg_file.read_text())
        raw["decision"]["explain_confirm_above"] = 2
        cfg_file.write_text(yaml.safe_dump(raw))
        upload = tmp_path / "input_a.csv"
        upload.write_bytes(tb._upload(24))
        code = cli.main(["--file", str(upload), "--config", str(cfg_file),
                         "--run-dir", str(tmp_path / "run")])
        out = capsys.readouterr().out
        assert code == 0 and "PHASE 2 NOT STARTED" in out
        assert "--phase2 sample --sample-n" in out and "--phase2 skip" in out
        assert not (tmp_path / "run" / "adverse_action_notices.zip").exists()
        assert (_rejected(tmp_path / "run")["explained"] == REASONS_PENDING).all()


# ================================================ A4. one applicant now ==

class TestOnDemand:
    def test_one_applicant_is_explained_in_seconds_and_nothing_else(self, tmp_path):
        cfg, ctx, res = _phase1(tmp_path)
        row = int(_rejected(res.run_dir)["row_id"].iloc[1])
        t = time.perf_counter()
        explain_run(res.run_dir, cfg, only=[row], model=ctx.model,
                    model_path=ctx.model_path, workers=1)
        assert time.perf_counter() - t < 15
        rej = _rejected(res.run_dir).set_index("row_id")
        assert rej.loc[row, "explained"] in ("explained", NO_REASON_NOTE)
        others = rej.drop(index=row)
        assert (others["explained"] == REASONS_PENDING).all()
        res1, text = phase2.result_for(res.run_dir, row)
        assert res1["row_id"] == row
        if text:
            assert rej.loc[row, "notice_file"] in _notice_texts(res.run_dir)
        # the run's choice is untouched: Phase 2 has not been started or decided
        assert read_progress(res.run_dir)["state"] == "not started"

    def test_a_row_that_is_not_rejected_is_refused(self, tmp_path):
        cfg, ctx, res = _phase1(tmp_path)
        approved_row = int(pd.read_csv(res.run_dir / "approved_applicants.csv")
                           ["row_id"].iloc[0])
        with pytest.raises(BatchError, match="not a rejected applicant"):
            explain_run(res.run_dir, cfg, only=[approved_row], model=ctx.model,
                        model_path=ctx.model_path)

    def test_pending_rows_are_listed_for_the_picker(self, tmp_path):
        cfg, ctx, res = _phase1(tmp_path)
        pending = phase2.pending_rows(res.run_dir)
        assert len(pending) == res.summary["n_rejected"]


# ================================== C1. the split survives two phases ==

class TestSplitInTwoPhases:
    def _geo(self, tmp_path):
        cfg = _cfg(tmp_path)
        ctx = _ctx(cfg, spec=GEO_SPEC, model=GeoModel(), train=_geo_training())
        res = score_file(_geo_upload(), "geo.csv", cfg, ctx=ctx,
                         runs_dir=tmp_path / "runs")
        return cfg, ctx, res

    def test_phase2_notices_carry_no_internal_content(self, tmp_path):
        cfg, ctx, res = self._geo(tmp_path)
        out = explain_run(res.run_dir, cfg, model=ctx.model, model_path=ctx.model_path)
        assert out.summary["n_fair_lending_flagged"] > 0       # geography drove it
        texts = _notice_texts(res.run_dir)
        assert texts
        for name, text in texts.items():
            assert "addr_state" not in text and "INTERNAL" not in text.upper(), name
            assert not find_internal_content(text, feature_names=GEO_SPEC.all_columns,
                                             applicant_id=text.split("Applicant ID:")[1]
                                             .split()[0])
        flags = pd.read_csv(res.run_dir / "internal" / "internal_review_flags.csv")
        assert flags["non_disclosable_drivers"].fillna("").str.contains(
            "addr_state").sum() == out.summary["n_fair_lending_flagged"]
        # and the working files keep the two apart as well
        for line in (res.run_dir / "phase2" / "notices.jsonl").read_text().splitlines():
            assert "top_adverse_drivers" not in line

    def test_the_on_demand_path_enforces_it_too(self, tmp_path, monkeypatch):
        cfg, ctx, res = _phase1(tmp_path)                 # applicants with notices
        from creditsurv.explain import adverse_action as aa
        original = aa.AdverseActionNotice.render
        monkeypatch.setattr(aa.AdverseActionNotice, "render",
                            lambda self: original(self) + "\nfair-lending review\n")
        row = int(_rejected(res.run_dir)["row_id"].iloc[0])
        with pytest.raises(BatchError, match="applicant_notices_clean"):
            explain_run(res.run_dir, cfg, only=[row], model=ctx.model,
                        model_path=ctx.model_path)
        assert not (res.run_dir / "adverse_action_notices.zip").exists()


# ================================================== B1. the model cache ==

class _Pickled:
    def predict_survival(self, X, times):
        return tb.StubModel().predict_survival(X, times)


def test_a_second_model_load_is_fast_and_the_same(tmp_path):
    """The API's model cache: once per model, shared by every request, never stale."""
    from creditsurv.api.state import ContextCache
    from creditsurv.cleaning import fit_values
    from creditsurv.config import load_config

    train = tb._training_frame(4000)
    data = tmp_path / "data"
    data.mkdir()
    train.to_parquet(data / "accepted_labeled.parquet")
    models = tmp_path / "models"
    models.mkdir()
    dm = tb.build_design_matrix(train, tb.SPEC, flavour="gbm")
    bundle = {"artefacts": {"discrete_hazard": _Pickled(),
                            "gbm_columns": list(dm.X.columns)},
              "spec": tb.SPEC, "train_idx": train.index[:3000],
              "test_idx": train.index[3000:],
              "data_source": str(data / "accepted_labeled.parquet"),
              "cleaning_values": fit_values(train, tb.SPEC, source="t").to_dict()}
    with open(models / "02_models_c.pkl", "wb") as fh:
        pickle.dump(bundle, fh)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "paths": {"data_dir": data.as_posix(), "models_dir": models.as_posix(),
                  "tables_dir": (tmp_path / "t").as_posix(),
                  "figures_dir": (tmp_path / "f").as_posix()},
        "decision": {"model_tag": "c", "background_rows": 500}}))

    cache = ContextCache(load_config(cfg_path))
    t = time.perf_counter()
    first = cache.get("c", "discrete_hazard")
    cold = time.perf_counter() - t
    t = time.perf_counter()
    second = cache.get("c", "discrete_hazard")
    warm = time.perf_counter() - t
    assert second is first                                   # identical, not reloaded
    assert warm < cold / 5, (cold, warm)
    assert cache.loads == 1 and cache.hits == 1
    pd.testing.assert_frame_equal(second.background, first.background)
    # Replace the model file: the key changes and it is loaded afresh.
    time.sleep(0.05)
    (models / "02_models_c.pkl").write_bytes((models / "02_models_c.pkl").read_bytes())
    third = cache.get("c", "discrete_hazard")
    assert third is not first and cache.loads == 2


# ============================================ the dashboard, both phases ==

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

VIEWS = PROJECT_ROOT / "app" / "views"


def _api(cfg, run_dir, ctx=None, launched=None):
    """An API over this run's folder; Phase 2 launches are recorded, not started."""
    import creditsurv.api.app as api_app
    from fastapi.testclient import TestClient
    kw = {}
    if ctx is not None:
        kw.update(context_loader=lambda c, t, m: ctx, context_key_fn=lambda c, t, m: (t, m))
    return TestClient(api_app.create_app(
        cfg, runs_dir=Path(run_dir).parent, jobs_dir=Path(run_dir).parent / "_jobs",
        phase2_launcher=lambda rd, mode, n: (launched if launched is not None
                                             else []).append((rd, mode, n)),
        watch_code=False, **kw))


def _draw(run_dir, cfg, ctx, launched: list):
    """The result view with its Phase 2 panel, drawn against an API over this run's
    folder. The bundle may not be built while the page draws (downloads are lazy)."""
    import sys
    if str(VIEWS) not in sys.path:
        sys.path.insert(0, str(VIEWS))
    import _client

    import creditsurv.api.app as api_app

    def no_zip(result):
        raise AssertionError("the page zipped the run while drawing it")
    api_app.bundle_zip = no_zip
    _client.use(_api(cfg, run_dir, ctx, launched))

    def script(views, run_id):
        import sys
        sys.path[:0] = [views]
        import _dashboard
        from _common import meta
        from _phase2_view import render_phase2
        m = meta()
        view = _dashboard.load(run_id)
        _dashboard.render(view, m, middle=lambda: render_phase2(
            run_id, m, summary=view.summary))

    try:
        at = AppTest.from_function(script, default_timeout=120, kwargs={
            "views": str(VIEWS), "run_id": Path(run_dir).name})
        at.run()
    finally:
        _client.use(None)
        api_app.bundle_zip = batch.bundle_zip
    assert not at.exception, [e.value for e in at.exception]
    return at


def _yaml(tmp_path, cfg) -> Path:
    from test_governance import _cli_config
    return _cli_config(tmp_path, cfg)


class TestDashboardPhases:
    def test_decisions_show_before_any_reason_and_phase2_starts_itself(self, tmp_path):
        cfg, ctx, res = _phase1(tmp_path)
        launched = []
        at = _draw(res.run_dir, cfg, ctx, launched)
        text = " ".join(m.value for m in list(at.success) + list(at.info))
        assert "Decisions ready in" in text
        assert f"Reasons pending for {res.summary['n_rejected']}" in text
        assert at.metric[0].value == f"{res.summary['n_rows']:,}"
        assert launched and launched[0][1] == "all"

    def test_above_the_threshold_the_page_asks(self, tmp_path):
        cfg, ctx, res = _phase1(tmp_path, explain_confirm_above=2)
        launched = []
        at = _draw(res.run_dir, cfg, ctx, launched)
        assert not launched                                     # nothing started
        radio = at.radio[0]
        assert radio.options == ["Explain all", "Explain a random sample", "Skip"]
        # The choice goes to the API, which starts Phase 2 in the chosen mode.
        import _client
        _client.use(_api(cfg, res.run_dir, ctx, launched))
        try:
            radio.set_value("Skip")
            next(b for b in at.button if b.label == "Start Phase 2").click().run()
        finally:
            _client.use(None)
        assert launched[0][1] == "skip"

    def test_progress_shows_while_phase2_runs(self, tmp_path):
        cfg, ctx, res = _phase1(tmp_path)
        status = res.run_dir / "phase2" / "status.json"
        import os
        st = json.loads(status.read_text())
        st.update(state="running", pid=os.getpid(), done=3, target=9,
                  rate_per_min=12.5, eta_seconds=29)
        status.write_text(json.dumps(st))
        at = _draw(res.run_dir, cfg, ctx, [])
        assert any("3 of 9" in (getattr(p, "text", "") or "") for p in at.get("progress")) \
            or any("3 of 9" in str(p.proto) for p in at.get("progress"))
        assert any("12.5 per minute" in m.value for m in at.markdown)

    def test_a_finished_run_draws_everything(self, tmp_path):
        cfg, ctx, res = _phase1(tmp_path)
        explain_run(res.run_dir, cfg, model=ctx.model, model_path=ctx.model_path)
        at = _draw(res.run_dir, cfg, ctx, [])
        assert any("Phase 2 finished" in s.value for s in at.success)

    def test_downloads_taken_mid_phase2_say_they_are_partial(self, tmp_path):
        """A zip downloaded while Phase 2 ran looked finished. The page, the button
        and the zip itself must all say it is a snapshot, and the same download
        after Phase 2 finishes must be the full result with no such marker."""
        import os
        cfg, ctx, res = _phase1(tmp_path)
        n = res.summary["n_rejected"]
        status = res.run_dir / "phase2" / "status.json"
        st = json.loads(status.read_text())
        st.update(state="running", pid=os.getpid(), done=1, target=n,
                  rate_per_min=6.0, eta_seconds=240)
        status.write_text(json.dumps(st))

        # While Phase 2 runs, nothing can be downloaded: every button is disabled.
        at = _draw(res.run_dir, cfg, ctx, [])
        banner = " ".join(w.value for w in at.warning)
        assert f"Phase 2 running: 1 of {n} explained, ~4 min remaining" in banner
        assert "Downloads are disabled until it finishes" in banner
        buttons = at.get("download_button")
        assert buttons and all(b.proto.disabled for b in buttons)

        # Stopped part-way: it will not finish on its own, so the partial files are
        # downloadable again, and say what they are.
        st.update(state="stopped")
        status.write_text(json.dumps(st))
        at = _draw(res.run_dir, cfg, ctx, [])
        assert "partial snapshot" in " ".join(w.value for w in at.warning)
        buttons = at.get("download_button")
        assert not any(b.proto.disabled for b in buttons)
        labels = [b.label for b in buttons]
        assert "⬇ rejected_applicants.csv (partial)" in labels
        assert any(lb.startswith("⬇ Download all (ZIP) — PARTIAL") for lb in labels)
        st.update(state="running")
        status.write_text(json.dumps(st))

        with zipfile.ZipFile(io.BytesIO(batch.bundle_zip(res))) as zf:
            note = zf.read(batch.PARTIAL_SNAPSHOT).decode("utf-8")
        assert "PARTIAL SNAPSHOT" in note and f"1 of {n:,}" in note
        assert f"{n - 1:,} still marked" in note

        # Phase 2 finishes: the banner goes, the buttons and the zip are the full run.
        explain_run(res.run_dir, cfg, model=ctx.model, model_path=ctx.model_path)
        assert not phase2.snapshot(res.run_dir)["partial"]
        at = _draw(res.run_dir, cfg, ctx, [])
        assert not any("partial snapshot" in w.value for w in at.warning)
        labels = [b.label for b in at.get("download_button")]
        assert "⬇ Download all (ZIP)" in labels
        assert not any("partial" in b.lower() for b in labels)
        with zipfile.ZipFile(io.BytesIO(batch.bundle_zip(batch.load_result(
                res.run_dir)))) as zf:
            assert batch.PARTIAL_SNAPSHOT not in zf.namelist()
            rejected = pd.read_csv(io.BytesIO(zf.read("rejected_applicants.csv")),
                                   dtype=str, keep_default_na=False)
        assert not (rejected["explained"] == REASONS_PENDING).any()
        assert len(rejected) == n


# ================================ the pipeline guard: one path, no bypass ==

def _py_files():
    for folder in ("src", "app", "scripts"):
        yield from (PROJECT_ROOT / folder).rglob("*.py")


SCORE = {"predict_survival"}
EXPLAIN = {"explain_rows_parallel", "iter_explanations", "explain_rows",
           "explain_survshap", "explain_tree_shap", "explain_naive_shap",
           "build_adverse_action_notice"}
ENGINE = ("src/creditsurv/explain/", "src/creditsurv/models/")
"""The implementations themselves: models predict, explainers explain."""
RESEARCH = {
    "scripts/02_train_models.py": "evaluates a trained model on its test split",
    "scripts/02r_recompute_metrics.py": "recomputes evaluation metrics",
    "scripts/03_explain.py": "Stage 3 research explanations and the sample notice",
    "scripts/03d_explainer_validation.py": "explainer comparison (FINDINGS 7)",
    "scripts/03e_feature_ablation.py": "measured cost of absent columns (7d)",
    "scripts/03f_settings_comparison.py": "cheaper-settings comparison (7a)",
    "scripts/04_reject_inference.py": "Stage 4 selection-bias diagnostic",
}
"""Research stages: they evaluate models on training data and never score an
uploaded applicant file. Adding a file here is a decision to review."""


def _calls(tree) -> list[tuple[str, str]]:
    """(called name, enclosing function) for every call in a module."""
    out = []

    def visit(node, fn):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name if fn == "<module>" else fn)
                continue
            if isinstance(child, ast.Call):
                f = child.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                out.append((name, fn))
            visit(child, fn)
    visit(tree, "<module>")
    return out


def test_no_caller_scores_or_explains_applicants_outside_the_two_phases():
    """The guard that makes this a pipeline fix. Scoring applicants means
    batch.score_file; explaining them means phase2.explain_run (via its private
    helper). Anything else that calls a model or an explainer directly fails here
    -- a new page, a new script, a faster-looking second path."""
    offenders = []
    for path in _py_files():
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if rel.startswith(ENGINE) or rel in RESEARCH:
            continue
        for name, fn in _calls(ast.parse(path.read_text(encoding="utf-8"))):
            # phase2._explain predicts once per run on the SHAP background, for the
            # explanation's baseline -- not a decision about anyone.
            if name in SCORE and (rel, fn) not in {("src/creditsurv/batch.py", "score_file"),
                                                   ("src/creditsurv/phase2.py", "_explain")}:
                offenders.append(f"{rel}:{fn} scores with {name}")
            if name in EXPLAIN and not (rel == "src/creditsurv/phase2.py"
                                        and fn == "_explain"):
                offenders.append(f"{rel}:{fn} explains with {name}")
    assert not offenders, offenders


def test_there_is_exactly_one_phase1_and_one_phase2_function():
    defs = {}
    for path in _py_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef) and node.name in (
                    "score_file", "explain_run", "_explain"):
                defs.setdefault(node.name, []).append(
                    path.relative_to(PROJECT_ROOT).as_posix())
    assert defs == {"score_file": ["src/creditsurv/batch.py"],
                    "explain_run": ["src/creditsurv/phase2.py"],
                    "_explain": ["src/creditsurv/phase2.py"]}
    # _explain is reached only through explain_run
    tree = ast.parse((PROJECT_ROOT / "src/creditsurv/phase2.py").read_text(encoding="utf-8"))
    assert {fn for name, fn in _calls(tree) if name == "_explain"} == {"explain_run"}
    assert batch.explain_run is phase2.explain_run         # re-exported, not copied


def test_run_batch_is_only_the_two_phases():
    tree = ast.parse((PROJECT_ROOT / "src/creditsurv/batch.py").read_text(encoding="utf-8"))
    called = {name for name, fn in _calls(tree) if fn == "run_batch"}
    assert {"score_file", "explain_run"} <= called
    assert not called & (SCORE | EXPLAIN)


@pytest.mark.parametrize("paths", [("src/creditsurv/api/state.py",
                                    "src/creditsurv/api/app.py"),
                                   ("scripts/06_score_upload.py",)])
def test_every_scoring_caller_goes_through_both_functions(paths):
    """The two ways in -- the API (the dashboard's) and the command line -- both
    score with score_file and explain with explain_run."""
    names = set()
    for path in paths:
        names |= {name for name, _ in _calls(ast.parse(
            (PROJECT_ROOT / path).read_text(encoding="utf-8")))}
    assert {"score_file", "explain_run"} <= names, paths


def test_background_phase2_runs_the_same_function():
    """The page's background Phase 2 is 06_score_upload.py --explain, which calls
    explain_run: no separate background implementation."""
    import inspect
    src = inspect.getsource(phase2.launch_background)
    assert '"scripts/06_score_upload.py", "--explain"' in src
    api = (PROJECT_ROOT / "src/creditsurv/api/app.py").read_text(encoding="utf-8")
    assert "phase2.launch_background" in api
    view = (VIEWS / "_phase2_view.py").read_text(encoding="utf-8")
    assert "/phase2" in view                                   # asks the API
    assert "explain_run(" not in view                          # the page never explains


def test_nothing_branches_on_a_file_name_or_fixture():
    """No special case for one dataset: the scoring and explaining code never
    mentions a test file's name."""
    suspicious = ("test_1_baseline", "test_2_renamed", "test_3_messy",
                  "test_4_drifted", "baseline_1k", "synthetic_check", "450mb",
                  "sample_applicants")
    for rel in ("src/creditsurv/batch.py", "src/creditsurv/phase2.py",
                "src/creditsurv/explain/parallel.py", "app/views/home.py",
                "app/views/_phase2_view.py", "app/views/_dashboard.py",
                "src/creditsurv/api/app.py", "src/creditsurv/api/state.py",
                "src/creditsurv/history.py", "src/creditsurv/stages.py",
                "scripts/06_score_upload.py"):
        text = (PROJECT_ROOT / rel).read_text(encoding="utf-8").lower()
        for word in suspicious:
            assert word not in text, f"{rel} mentions {word}"


# ======================================================= C2, C3 plumbing ==

def test_the_runner_reports_progress_for_a_silent_stage(tmp_path):
    import sys
    from creditsurv.runner import launch, tail_log
    run = launch([{"key": "q", "name": "quiet", "tag": "t", "args": [],
                   "argv": [sys.executable, "-c", "import time; time.sleep(2.6)"]}],
                 lock_tag="t", runs_dir=tmp_path, detach=False,
                 meta={"heartbeat_seconds": 1})
    assert tail_log(run).count("still running") >= 2


class TestWslSync:
    def test_nothing_happens_outside_the_wsl_copy(self, tmp_path):
        from creditsurv import wsl_sync
        assert wsl_sync.windows_source(tmp_path) is None
        assert wsl_sync.copy_back_soon(tmp_path) is False

    def test_newer_windows_files_are_found(self, tmp_path):
        import os
        from creditsurv import wsl_sync
        src, dst = tmp_path / "win", tmp_path / "wsl"
        for base in (src, dst):
            (base / "src" / "pkg").mkdir(parents=True)
            (base / "src" / "pkg" / "a.py").write_text("x = 1")
        (src / "src" / "pkg" / "new.py").write_text("y = 2")          # missing in WSL
        now = time.time()
        os.utime(dst / "src" / "pkg" / "a.py", (now - 100, now - 100))
        os.utime(src / "src" / "pkg" / "a.py", (now, now))            # newer on Windows
        (src / "src" / "pkg" / "__pycache__").mkdir()
        (src / "src" / "pkg" / "__pycache__" / "a.cpython.pyc").write_text("")
        assert wsl_sync.stale_files(dst, src) == ["src/pkg/a.py", "src/pkg/new.py"]
        os.utime(src / "src" / "pkg" / "a.py", (now - 100.5, now - 100.5))  # within slack
        assert wsl_sync.stale_files(dst, src) == ["src/pkg/new.py"]

    def test_a_running_job_or_phase2_blocks_a_sync(self, tmp_path, monkeypatch):
        import os
        from creditsurv import wsl_sync
        # this test process stands in for a job of ours (see test_part_c for the
        # check that a reused pid of a stranger does not count)
        monkeypatch.setattr(wsl_sync, "_pid_alive", lambda pid: pid == os.getpid())
        assert wsl_sync.sync_blockers(tmp_path) == []
        locks = tmp_path / "outputs" / "logs" / "runs" / "locks"
        locks.mkdir(parents=True)
        (locks / "job.lock").write_text(json.dumps({"pid": os.getpid()}))
        assert wsl_sync.sync_blockers(tmp_path) == ["background job job"]
        (locks / "job.lock").unlink()
        p2 = tmp_path / "outputs" / "runs" / "r1" / "phase2"
        p2.mkdir(parents=True)
        (p2 / "status.json").write_text(json.dumps({"state": "running",
                                                    "pid": os.getpid()}))
        assert wsl_sync.sync_blockers(tmp_path) == ["Phase 2 of r1"]

    def test_every_page_checks_code_freshness(self):
        common = (VIEWS / "_common.py").read_text(encoding="utf-8")
        header = common.split("def page_header")[1].split("\ndef ")[0]
        assert "code_freshness()" in header


# ---------------------------------------------- the processing history (7p) --

def _rec(run_id, created, *, rows=100, approved=80, model="m", lending=True,
         state="finished", explained=0, flag=None):
    from creditsurv.history import RunRecord
    return RunRecord(run_id, Path(run_id), state, {
        "created_at": created, "n_rows": rows, "n_approved": approved,
        "n_rejected": rows - approved, "approval_rate": approved / rows,
        "model_tag": model, "source_file": run_id.split("_", 2)[-1] + ".csv",
        "for_lending_decisions": lending, "n_explained": explained,
        "fair_lending_flag_share": flag})


class TestProcessingHistory:
    """The history section's figures: sums over run_summary.csv, nothing else."""

    def runs(self):
        return [_rec("20260901_120000_a", "2026-09-01T12:00:00", rows=100, approved=90),
                _rec("20260915_120000_b", "2026-09-15T12:00:00", rows=200, approved=150,
                     model="other"),
                _rec("20260915_130000_c", "2026-09-15T13:00:00", rows=300, approved=210),
                _rec("20260916_120000_d", "2026-09-16T12:00:00", rows=400, approved=0,
                     state="failed checks"),
                _rec("20260916_130000_e", "2026-09-16T13:00:00", rows=500, approved=250,
                     lending=False)]

    def test_it_sums_finished_runs_across_dates_and_files(self):
        h = processing_history(self.runs(), today="2026-09-16")
        # a + b + c; d failed its checks and e is not for lending decisions
        assert h["applicants_processed"] == 600
        assert h["total_approved"] == 450
        assert h["total_rejected"] == 150
        assert h["approval_rate_overall"] == 450 / 600
        assert h["total_runs"] == 5 and h["finished_runs"] == 3
        assert h["files_processed"] == 3
        assert [r["day"] for r in h["series"]] == ["2026-09-01", "2026-09-15",
                                                   "2026-09-15"]
        assert h["first_day"] == "2026-09-01" and h["last_day"] == "2026-09-16"

    def test_a_failed_run_counts_as_a_run_and_not_as_volume(self):
        h = processing_history(self.runs(), today="2026-09-16")
        assert "20260916_120000_d" not in {r["run_id"] for r in h["series"]}
        assert h["total_runs"] > h["finished_runs"]

    def test_the_lending_filter_changes_what_is_counted(self):
        h = processing_history(self.runs(), for_lending_only=False,
                               today="2026-09-16")
        assert h["applicants_processed"] == 1100          # e joins
        assert h["finished_runs"] == 4

    def test_dates_and_model_narrow_it(self):
        h = processing_history(self.runs(), date_from="2026-09-15",
                               date_to="2026-09-15", today="2026-09-16")
        assert h["applicants_processed"] == 500
        assert h["total_runs"] == 2
        one = processing_history(self.runs(), model_tag="other", today="2026-09-16")
        assert one["applicants_processed"] == 200
        assert one["model_tags"] == ["m", "other"]        # the offered list is unfiltered

    def test_the_week_and_month_windows_trail_the_date(self):
        runs = self.runs()
        week = processing_history(runs, today="2026-09-16")
        assert week["applicants_last_7_days"] == 500      # b and c, not a
        assert week["applicants_last_30_days"] == 600     # a as well
        later = processing_history(runs, today="2026-11-01")
        assert later["applicants_last_7_days"] == 0
        assert later["applicants_processed"] == 600       # totals do not age

    def test_it_reports_how_much_of_the_volume_is_still_unexplained(self):
        """Counted volume is decisions, not notices: four of FINISHED_STATES have
        final decisions and unwritten reasons, so the gap is reported, not implied."""
        runs = [_rec("20260901_120000_a", "2026-09-01T12:00:00", rows=100, approved=60,
                     state="finished", explained=40),
                _rec("20260902_120000_b", "2026-09-02T12:00:00", rows=100, approved=60,
                     state="reasons pending", explained=0)]
        runs[0].summary["n_reasons_pending"] = 0
        runs[1].summary["n_reasons_pending"] = 40
        h = processing_history(runs, today="2026-09-02")
        assert h["applicants_processed"] == 200        # both runs decided
        assert h["total_rejected"] == 80
        assert h["rejected_explained"] == 40
        assert h["reasons_pending"] == 40
        assert h["runs_awaiting_reasons"] == 1
        assert h["applicants_awaiting_reasons"] == 100

    def test_a_run_with_no_reasons_has_no_flag_share(self):
        """None, not zero: nothing measured must not plot as 'measured, and clean'."""
        runs = [_rec("20260901_120000_a", "2026-09-01T12:00:00", explained=0, flag=0.0),
                _rec("20260902_120000_b", "2026-09-02T12:00:00", explained=10, flag=0.2)]
        series = processing_history(runs, today="2026-09-02")["series"]
        assert series[0]["fair_lending_flag_share"] is None
        assert series[1]["fair_lending_flag_share"] == 0.2
