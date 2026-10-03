"""The Model registry page: it moves the registry commands into the UI and
shortcuts nothing.

Pinned here: the page and the CLI run the same rule check and the same approval;
the evidence jobs are the FINDINGS 7l commands unchanged (no --overwrite), run in
the background by the runner that large uploads use, through WSL when the page is
served from Windows; one job at a time; and approval is never written into the WSL
copy, whose config/ is replaced from Windows on every sync.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import yaml

from creditsurv import evidence_jobs as jobs
from creditsurv import registry
from creditsurv.config import load_config
from creditsurv.provenance import PROJECT_ROOT
from creditsurv.runner import LockHeld, launch, read_status, tail_log
from test_governance import _evidence

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

VIEWS = PROJECT_ROOT / "app" / "views"


def _registry_cli():
    spec = importlib.util.spec_from_file_location(
        "registry_cli_page", PROJECT_ROOT / "scripts" / "07_model_registry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _candidate(tmp_path, **kw) -> Path:
    """A registered candidate model 'm' with complete evidence; returns the config."""
    cfg_path, _ = _evidence(tmp_path, **kw)
    assert _registry_cli().main(["register", "--model-tag", "m", "--status",
                                 "candidate", "--config", str(cfg_path)]) == 0
    return cfg_path


def _status(cfg_path) -> registry.ModelRecord:
    return registry.load_registry(load_config(cfg_path)).get("m")


# ------------------------------------------------------ the jobs themselves --

class TestJobs:
    def test_through_wsl_it_syncs_runs_and_copies_back(self):
        stages = jobs.job_stages("explainer_validation", "m", wsl_root="/mnt/c/p q")
        assert [s["key"] for s in stages] == ["sync", "explainer_validation",
                                              "copy-back"]
        launcher = ["wsl.exe", "-e", "bash", "/mnt/c/p q/scripts/wsl_launch.sh"]
        assert stages[0]["argv"] == [*launcher, "check", "/mnt/c/p q"]
        assert stages[1]["argv"][:6] == [*launcher, "run", "/mnt/c/p q"]
        assert stages[1]["argv"][6:] == ["--", "scripts/03d_explainer_validation.py",
                                         "--model-tag", "m", "--tag", "explainer_m",
                                         "--n-explain", "1000"]
        assert stages[2]["argv"] == [*launcher, "copy-back", "/mnt/c/p q"]

    def test_on_linux_the_script_runs_directly(self):
        (stage,) = jobs.job_stages("ablation", "m")
        assert "argv" not in stage
        assert stage["args"] == ["scripts/03e_feature_ablation.py", "--model-tag", "m",
                                 "--sample", "50000"]

    @pytest.mark.parametrize("kind", list(jobs.JOBS))
    def test_no_job_shortcuts_the_evidence(self, kind):
        """The FINDINGS 7l commands: no --overwrite, no smaller sample, no fewer
        applicants than the registered 1,000."""
        args = jobs.JOBS[kind].args("m")
        assert "--overwrite" not in args
        assert "--secondary-only" not in args and "--groups-only" not in args
        if kind == "explainer_validation":
            assert args[args.index("--n-explain") + 1] == "1000"
        for stage in jobs.job_stages(kind, "m", wsl_root="/w"):
            assert "--overwrite" not in stage.get("argv", [])

    def test_each_job_feeds_the_rule_it_is_for(self, tmp_path):
        assert jobs.JOBS["ablation"].rule == "A5_ablation"
        assert jobs.JOBS["explainer_validation"].rule == "A6_explainer_validation"
        # and the file the job writes is one the rule looks for
        assert jobs.JOBS["ablation"].outputs("m", tmp_path)[0].name == "03e_ablation_m.json"
        assert jobs.JOBS["explainer_validation"].outputs("m", tmp_path)[0].name \
            .startswith("03d_explainer_validation_")

    def test_launch_is_the_background_runner_with_one_lock(self, monkeypatch, tmp_path):
        seen = {}
        monkeypatch.setattr(jobs, "launch", lambda stages, **kw: seen.update(
            stages=stages, **kw) or tmp_path / "run")
        jobs.launch_job("ablation", "m", runs_dir=tmp_path, via_wsl=False)
        assert seen["lock_tag"] == jobs.LOCK_TAG
        assert seen["meta"]["source"] == jobs.SOURCE
        assert seen["meta"]["keep_awake"] is True
        assert seen["meta"]["rule"] == "A5_ablation"

    def test_one_evidence_job_at_a_time(self, tmp_path):
        locks = tmp_path / "locks"
        locks.mkdir()
        (locks / f"{jobs.LOCK_TAG}.lock").write_text(
            f'{{"run_id": "r", "pid": {os.getpid()}, "started": "now"}}')
        assert jobs.job_running(tmp_path)
        with pytest.raises(LockHeld):
            jobs.launch_job("ablation", "m", runs_dir=tmp_path, via_wsl=False)

    def test_the_runner_runs_a_full_command_stage_and_stops_at_a_failure(self, tmp_path):
        """How a Windows page runs a stage in WSL: an argv, not python args."""
        py = sys.executable
        run = launch([
            {"key": "a", "name": "a", "tag": "t", "args": [],
             "argv": [py, "-c", "print('stage one ran')"]},
            {"key": "b", "name": "b", "tag": "t", "args": [],
             "argv": [py, "-c", "raise SystemExit(4)"]},
            {"key": "c", "name": "c", "tag": "t", "args": [],
             "argv": [py, "-c", "print('WRONG')"]}],
            lock_tag="t", runs_dir=tmp_path, detach=False,
            meta={"source": jobs.SOURCE, "keep_awake": True})
        status = read_status(run)
        assert [s["state"] for s in status["stages"]] == ["completed", "failed", "not run"]
        assert status["stages"][1]["exit_code"] == 4
        log = tail_log(run)
        assert "stage one ran" in log and "WRONG" not in log
        assert "refused to overwrite" in log              # exit 4 is explained
        assert jobs.evidence_runs(tmp_path) == [run]

    def test_the_launcher_has_a_run_mode_and_marks_its_copy(self):
        text = (PROJECT_ROOT / "scripts" / "wsl_launch.sh").read_text(encoding="utf-8")
        assert "\n    run)" in text
        assert 'exec env PYTHONUNBUFFERED=1 PYTHONPATH="$DEST/src" "$PY" -u "$@"' in text
        assert f'> "$DEST/{registry.SYNC_MARKER}"' in text
        assert "\r\n" not in text


# ----------------------------------------------------------------- approval --

class TestApproval:
    def test_it_approves_only_when_every_rule_passes_now(self, tmp_path):
        cfg_path = _candidate(tmp_path)
        cfg = load_config(cfg_path)
        (cfg.paths.tables_dir / "03e_ablation_m.json").unlink()      # A5 now fails
        before = Path(cfg.paths.registry).read_bytes()
        out = registry.approve_model(cfg, "m", by="T", findings="7l")
        assert not out.approved and "A5_ablation" in out.refusal
        assert Path(cfg.paths.registry).read_bytes() == before        # nothing written

    def test_it_records_who_and_when(self, tmp_path):
        cfg = load_config(_candidate(tmp_path))
        out = registry.approve_model(cfg, "m", by="  A. Reviewer ", findings="7l",
                                     note="first approval")
        assert out.approved and len(out.results) == 7
        rec = registry.load_registry(cfg).get("m")
        assert rec.status == "approved"
        assert rec.approval["approved_by"] == "A. Reviewer"
        assert rec.approval["approved_at"][:4].isdigit() and "T" in rec.approval["approved_at"]
        assert rec.approval["rules_passed"] == sorted(registry.APPROVAL_RULES)

    @pytest.mark.parametrize("by, findings", [("", "7l"), ("T", ""), ("  ", " ")])
    def test_it_names_who_and_where(self, tmp_path, by, findings):
        cfg = load_config(_candidate(tmp_path))
        assert not registry.approve_model(cfg, "m", by=by, findings=findings).approved

    def test_a_deprecated_model_is_not_approved_by_this_route(self, tmp_path):
        cfg_path = _candidate(tmp_path)
        _registry_cli().main(["set-status", "--model-tag", "m", "--status",
                              "deprecated", "--reason", "withdrawn",
                              "--config", str(cfg_path)])
        out = registry.approve_model(load_config(cfg_path), "m", by="T", findings="7l")
        assert not out.approved and "deprecated" in out.refusal

    def test_the_wsl_copy_never_records_an_approval(self, tmp_path):
        """Its config/ is replaced from Windows on every sync: an approval written
        there would vanish, and scoring would silently go back to refusing."""
        cfg_path = _candidate(tmp_path)
        cfg = load_config(cfg_path)
        root = tmp_path / "wslcopy"
        (root / "config").mkdir(parents=True)
        (root / registry.SYNC_MARKER).write_text("/mnt/c/project\n")
        copy = root / "config" / "models.yaml"
        copy.write_bytes(Path(cfg.paths.registry).read_bytes())
        reg = registry.load_registry(path=copy)
        out = registry.approve_model(cfg, "m", by="T", findings="7l", registry=reg)
        assert not out.approved and "synced from /mnt/c/project" in out.refusal
        assert registry.load_registry(path=copy).get("m").status == "candidate"

    def test_a_passing_rerun_does_not_cancel_a_failing_validation(self, tmp_path):
        """Re-running 03d under a new output tag until the noise falls the right
        way must not satisfy A6: every validation of the model file counts."""
        cfg_path = _candidate(tmp_path, ceiling=(0.80, 0.70))           # fails
        cfg = load_config(cfg_path)
        failing = cfg.paths.tables_dir / "03d_explainer_validation_v_m.json"
        passing = failing.with_name("03d_explainer_validation_z_m.json")  # sorts last
        payload = yaml.safe_load(failing.read_text())
        payload["reference_ceiling_survshap_vs_itself"].update(
            top1_agreement=0.97, mean_top4_overlap=0.95)
        passing.write_text(__import__("json").dumps(payload))
        results = {r.rule: r for r in registry.evaluate_rules(cfg, _status(cfg_path))}
        a6 = results["A6_explainer_validation"]
        assert not a6.passed and "1 below the bar" in a6.detail
        assert not registry.approve_model(cfg, "m", by="T", findings="7l").approved
        failing.unlink()                                  # only the passing one left
        assert registry.approve_model(cfg, "m", by="T", findings="7l").approved

    def test_the_cli_and_the_api_share_one_check_and_one_approval(self):
        """The CLI and the API call the same two functions; the page calls neither
        -- it asks the API, so it cannot check or approve anything itself."""
        cli = (PROJECT_ROOT / "scripts" / "07_model_registry.py").read_text(encoding="utf-8")
        api = (PROJECT_ROOT / "src" / "creditsurv" / "api" / "app.py").read_text(
            encoding="utf-8")
        page = (VIEWS / "_registry_view.py").read_text(encoding="utf-8")
        for src in (cli, api):
            assert "evaluate_rules(" in src and "approve_model(" in src
            assert "check_approval_rules(" not in src      # no private rule list
            assert "approval_block(" not in src            # no approval of its own
        for src in (api, page):                           # never writes the registry
            assert 'status = "approved"' not in src and "reg.save(" not in src
            assert "Registry(" not in src and "models.yaml\", \"w" not in src
        assert "evaluate_rules(" not in page and "approve_model(" not in page


# ---------------------------------------------------------------- the page --

def _page(cfg_path, runs_dir, marker):
    """The page, drawn against an API over this test's registry. The job launcher
    is replaced so a click is recorded instead of starting a real job; every rule
    that refuses a job is still the API's."""
    def script(views, src, cfg_path, runs_dir, marker):
        import sys
        sys.path[:0] = [views, src]
        from pathlib import Path

        from fastapi.testclient import TestClient

        import _client
        import _registry_view as view
        from creditsurv.api.app import create_app
        from creditsurv.config import load_config

        def fake_launcher(kind, tag):
            Path(marker).write_text(f"{kind} {tag}")
            return Path(runs_dir) / "never_created"

        app = create_app(load_config(cfg_path), config_path=cfg_path,
                         jobs_dir=Path(runs_dir), job_launcher=fake_launcher,
                         watch_code=False, evidence_via_wsl=False)
        _client.use(TestClient(app))
        try:
            view.render()
        finally:
            _client.use(None)

    at = AppTest.from_function(script, default_timeout=120, kwargs={
        "views": str(VIEWS), "src": str(PROJECT_ROOT / "src"),
        "cfg_path": str(cfg_path), "runs_dir": str(runs_dir), "marker": str(marker)})
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _button(at, label):
    return next(b for b in at.button if b.label == label)


class TestPage:
    def test_the_table_shows_every_model_and_every_rule_with_its_reason(self, tmp_path):
        cfg_path = _candidate(tmp_path)
        _evidence(tmp_path, tag="m2")
        _registry_cli().main(["register", "--model-tag", "m2", "--status", "benchmark",
                              "--config", str(cfg_path)])
        at = _page(cfg_path, tmp_path / "runs", tmp_path / "marker")
        table = at.dataframe[0].value
        assert list(table["model"]) == ["m", "m2"]
        assert set(table["status"]) == {"candidate", "benchmark"}
        for rule in ("A1", "A2", "A3", "A4", "A5", "A6", "A7"):
            assert table[rule].str.match(r"^(✅ PASS|❌ FAIL) — .+").all(), rule

    def test_missing_evidence_offers_its_job_and_launches_it(self, tmp_path):
        cfg_path = _candidate(tmp_path)
        cfg = load_config(cfg_path)
        (cfg.paths.tables_dir / "03e_ablation_m.json").unlink()
        at = _page(cfg_path, tmp_path / "runs", tmp_path / "marker")
        assert not _button(at, "Run ablation").disabled
        assert _button(at, "Run explainer validation").disabled     # A6 passes
        assert not any(b.label.startswith("Approve") for b in at.button)
        _button(at, "Run ablation").click().run()
        assert (tmp_path / "marker").read_text() == "ablation m"

    def test_existing_output_is_never_overwritten_from_the_page(self, tmp_path):
        """A6 can fail with its file present (reproducibility below the bar); the
        page then refuses to rerun rather than pass --overwrite."""
        cfg_path = _candidate(tmp_path, ceiling=(0.80, 0.70))
        at = _page(cfg_path, tmp_path / "runs", tmp_path / "marker")
        b = _button(at, "Run explainer validation")
        assert b.disabled
        assert any("needs --overwrite" in c.value for c in at.caption)

    def test_jobs_wait_for_a_running_one(self, tmp_path):
        cfg_path = _candidate(tmp_path)
        cfg = load_config(cfg_path)
        (cfg.paths.tables_dir / "03e_ablation_m.json").unlink()
        runs = tmp_path / "runs"
        (runs / "locks").mkdir(parents=True)
        (runs / "locks" / f"{jobs.LOCK_TAG}.lock").write_text(
            f'{{"run_id": "busy", "pid": {os.getpid()}, "started": "now"}}')
        at = _page(cfg_path, runs, tmp_path / "marker")
        assert _button(at, "Run ablation").disabled
        assert any("Another evidence job is running" in c.value for c in at.caption)

    def test_approve_appears_when_all_seven_pass_and_rechecks(self, tmp_path):
        cfg_path = _candidate(tmp_path)
        at = _page(cfg_path, tmp_path / "runs", tmp_path / "marker")
        approve = _button(at, "Approve m")
        # No name: refused, nothing written.
        approve.click().run()
        assert any("Not approved" in e.value for e in at.error)
        assert _status(cfg_path).status == "candidate"
        # Evidence gone between drawing the page and pressing the button: the rerun
        # re-evaluates every rule from the files, so the form is no longer offered
        # and nothing is recorded. (approve_model's own re-check at the moment of
        # approval is pinned in TestApproval.)
        cfg = load_config(cfg_path)
        ablation = cfg.paths.tables_dir / "03e_ablation_m.json"
        saved = ablation.read_bytes()
        ablation.unlink()
        at.text_input(key="by_m").input("A. Reviewer")
        _button(at, "Approve m").click().run()
        assert _status(cfg_path).status == "candidate"
        assert not any(b.label == "Approve m" for b in at.button)
        assert at.dataframe[0].value.iloc[0]["A5"].startswith("❌ FAIL")
        # Evidence back: approved, with who and when.
        ablation.write_bytes(saved)
        at = _page(cfg_path, tmp_path / "runs", tmp_path / "marker")
        at.text_input(key="by_m").input("A. Reviewer")
        _button(at, "Approve m").click().run()
        rec = _status(cfg_path)
        assert rec.status == "approved" and rec.approval["approved_by"] == "A. Reviewer"
        assert rec.approval["approved_at"]
        assert any("is approved by A. Reviewer" in s.value for s in at.success)

    def test_the_wsl_copy_shows_approve_disabled(self, tmp_path):
        """The registry laid out as wsl_launch.sh leaves it: <copy>/config/models.yaml
        beside the marker it writes at <copy>/.synced_from_windows."""
        cfg_path = _candidate(tmp_path)
        root = tmp_path / "wslcopy"
        (root / "config").mkdir(parents=True)
        (root / registry.SYNC_MARKER).write_text("/mnt/c/project\n")
        raw = yaml.safe_load(cfg_path.read_text())
        copy = root / "config" / "models.yaml"
        copy.write_bytes(Path(raw["paths"]["registry"]).read_bytes())
        raw["paths"]["registry"] = copy.as_posix()
        cfg_path.write_text(yaml.safe_dump(raw))
        at = _page(cfg_path, tmp_path / "runs", tmp_path / "marker")
        assert _button(at, "Approve m").disabled
        assert any("would be lost" in w.value for w in at.warning)

    def test_the_log_of_a_job_can_be_followed(self, tmp_path):
        cfg_path = _candidate(tmp_path)
        runs = tmp_path / "runs"
        launch([{"key": "a", "name": "Run ablation for m", "tag": "m", "args": [],
                 "argv": [sys.executable, "-c", "print('ablation progress 3/64')"]}],
               lock_tag=jobs.LOCK_TAG, runs_dir=runs, detach=False,
               meta={"source": jobs.SOURCE})
        at = _page(cfg_path, runs, tmp_path / "marker")
        assert any("ablation progress 3/64" in c.value for c in at.code)
        assert any("completed" in m.value for m in at.markdown)


def test_the_page_is_in_the_app():
    app = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    assert 'VIEWS / "model_registry.py"' in app


def test_the_real_page_renders_against_the_real_registry():
    if str(VIEWS) not in sys.path:
        sys.path.insert(0, str(VIEWS))
    import _client
    from fastapi.testclient import TestClient

    from creditsurv.api.app import create_app
    _client.use(TestClient(create_app(watch_code=False)))
    try:
        at = AppTest.from_file(str(VIEWS / "model_registry.py"), default_timeout=240)
        at.run()
    finally:
        _client.use(None)
    assert not at.exception, [e.value for e in at.exception]
    models = set(at.dataframe[0].value["model"])
    assert set(yaml.safe_load((PROJECT_ROOT / "config" / "models.yaml")
                              .read_text(encoding="utf-8"))["models"]) == models
