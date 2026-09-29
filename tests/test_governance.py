"""What test 1 found, made impossible to repeat.

Test 1 (test_1_baseline_1k.csv, the ``full`` model) produced 255 applicant notices,
166 of which carried an internal fair-lending flag naming ``addr_state``; it ran on
a model with a known defect because nothing but a config tag chose it; and nothing
checked the outputs before calling the run finished. Each section below pins one
of the fixes, on small synthetic fixtures, through every scoring path the project
has: the dashboard's inline run (``run_batch``) and its result view, the background
run and the CLI (``06_score_upload.py``), and Stage 3's notice writer.

(There is no separate "Live Scoring" path in this codebase: the dashboard scores
through ``run_batch``, as the background run and the CLI do. The structural tests at
the end fail if a new caller of the notice builder appears without going through
one of the guarded paths.)
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import test_batch as tb
from creditsurv import batch, registry
from creditsurv.batch import BatchError, ScoringContext, run_batch
from creditsurv.config import Config, DecisionConfig, Paths, load_config
from creditsurv.explain.adverse_action import (NOT_FOR_LENDING, ApplicantNotice,
                                               NoticeContentError,
                                               build_adverse_action_notice,
                                               find_internal_content,
                                               write_notice_pair)
from creditsurv.explain.survshap import explain_survshap
from creditsurv.features.build import FeatureSpec, build_design_matrix
from creditsurv.provenance import PROJECT_ROOT
from creditsurv.run_checks import NO_REASON_NOTE, verify_run

GEO_SPEC = FeatureSpec(numeric=tb.NUMERIC,
                       categorical=tb.CATEGORICAL + ("addr_state",),
                       structural_missing=())
STATES = ("CA", "NY", "TX")


class GeoModel:
    """Risk driven by geography: every CA applicant is declined, mostly because of
    addr_state -- the situation test 1 found in 65% of its declines."""

    def predict_survival(self, X, times):
        times = np.atleast_1d(np.asarray(times, dtype=float))
        ca = (X["addr_state"].astype("string").fillna("") == "CA").to_numpy()
        dti = pd.to_numeric(X["dti"], errors="coerce").fillna(20.0).to_numpy()
        rate = 0.0003 * dti + 0.02 * ca
        return np.exp(-np.outer(rate, times))


class BackwardsModel(tb.StubModel):
    """Survival that rises with time, so 12-month risk exceeds 36-month risk."""

    def predict_survival(self, X, times):
        return super().predict_survival(X, times)[:, ::-1]


def _cfg(tmp_path, **decision) -> Config:
    base = dict(model_tag="stub", horizon_months=36, reject_at_or_above=0.45,
                explain_nsamples=2 * len(tb.NUMERIC + tb.CATEGORICAL) + 2,
                explain_n_background=8, max_explained=0, background_rows=200,
                explain_workers=1)
    base.update(decision)
    return Config(paths=Paths(data_dir=tmp_path / "data", models_dir=tmp_path / "models",
                              figures_dir=tmp_path / "figures",
                              tables_dir=tmp_path / "tables",
                              registry=tmp_path / "models.yaml"),
                  decision=DecisionConfig(**base))


def _geo_training(n=400) -> pd.DataFrame:
    df = tb._training_frame(n)
    df["addr_state"] = pd.Categorical(np.resize(STATES, n))
    return df


def _geo_upload(n=18) -> bytes:
    df = pd.read_csv(io.BytesIO(tb._upload(n, alias=False)))
    df["addr_state"] = np.resize(STATES, n)
    return df.to_csv(index=False).encode()


def _ctx(cfg, *, spec=tb.SPEC, model=None, train=None, status="approved",
         register=True, **registry_kw) -> ScoringContext:
    train = train if train is not None else tb._training_frame()
    dm = build_design_matrix(train, spec, flavour="gbm")
    model_path = tb._write_dummy_model(cfg)
    if register:
        tb.write_registry(cfg, model_path, status=status, spec=spec, **registry_kw)
    return ScoringContext(
        cfg=cfg, model_tag="stub", model_name="discrete_hazard",
        model=model or tb.StubModel(), spec=spec,
        bundle={"artefacts": {"gbm_columns": list(dm.X.columns)}},
        model_path=model_path, background=dm.X, reference=train,
        clean_values=tb.fit_values(train, spec, source="stub.parquet"),
        policy=tb.policy_from_config(cfg), times=np.array([6.0, 12.0, 24.0, 36.0]),
        data_source=tb._dummy_source(cfg))


def _notices(run_dir: Path) -> dict[str, str]:
    zpath = Path(run_dir) / "adverse_action_notices.zip"
    if not zpath.exists():
        return {}
    with zipfile.ZipFile(zpath) as zf:
        return {n: zf.read(n).decode("utf-8") for n in zf.namelist()}


@pytest.fixture
def geo_run(tmp_path):
    cfg = _cfg(tmp_path)
    ctx = _ctx(cfg, spec=GEO_SPEC, model=GeoModel(), train=_geo_training())
    return run_batch(_geo_upload(), "geo.csv", cfg, ctx=ctx, runs_dir=tmp_path / "runs")


# =========================================== 1. applicant notices are clean ==

class TestNoticeSplit:
    def test_geography_driven_notices_carry_no_internal_content(self, geo_run):
        """Test 1 in miniature: geography drives the declines. The notices must
        say nothing about it; the internal record must say all of it."""
        s = geo_run.summary
        assert s["n_fair_lending_flagged"] > 0                  # the flag fired
        notices = _notices(geo_run.run_dir)
        assert notices, "the fixture should produce notices"
        for name, text in notices.items():
            assert "INTERNAL" not in text.upper(), name
            assert "addr_state" not in text, name
            assert "fair-lending" not in text.lower(), name
            applicant = text.split("Applicant ID:")[1].split("\n")[0].strip()
            assert find_internal_content(text, feature_names=GEO_SPEC.all_columns,
                                         applicant_id=applicant) == [], name
        flags = pd.read_csv(geo_run.run_dir / "internal" / "internal_review_flags.csv")
        assert flags["fair_lending_flag"].sum() == s["n_fair_lending_flagged"]
        assert flags["non_disclosable_drivers"].fillna("").str.contains(
            "addr_state").sum() == s["n_fair_lending_flagged"]
        records = [json.loads(line) for line in
                   (geo_run.run_dir / "internal" / "internal_review_records.jsonl")
                   .read_text(encoding="utf-8").splitlines()]
        assert any(r["top_adverse_drivers"][0]["feature"] == "addr_state"
                   for r in records if r["top_adverse_drivers"])
        assert (geo_run.run_dir / "internal" / "README.txt").exists()

    def test_internal_files_never_enter_the_notice_zip(self, geo_run):
        names = list(_notices(geo_run.run_dir))
        assert all(n.startswith("notice_") and n.endswith(".txt") for n in names)
        assert not any("internal" in n.lower() for n in names)

    def test_every_notice_is_screened_on_disk_by_the_run(self, geo_run):
        checks = pd.read_csv(geo_run.run_dir / "validation_checks.csv")
        row = checks.set_index("check").loc["applicant_notices_clean"]
        assert row["status"] == "PASS"
        assert f"{len(_notices(geo_run.run_dir))} notice(s)" in row["detail"]

    @pytest.mark.parametrize("leak, kind", [
        ("[INTERNAL REVIEW FLAG -- NOT PART OF THE APPLICANT NOTICE]", "internal"),
        ("drivers for this applicant: addr_state.", "addr_state"),
        ("requires fair-lending review before deployment", "fair-lending"),
        ("Your dti was high.", "dti"),
        ("The monthly installment is high.", "installment"),
        ("attribution -0.031245", "attribution"),
        ("Predicted default probability 55.7%", "probability"),
    ])
    def test_the_screen_catches_each_kind_of_leak(self, leak, kind):
        problems = find_internal_content(f"Some notice text.\n{leak}\n",
                                         feature_names=("dti", "installment"))
        assert problems and any(kind in p for p in problems), problems

    def test_the_screen_allows_template_words_that_are_also_feature_names(self):
        # "purpose" is a feature and also the Regulation B wording of its reason.
        text = "1. Purpose of the credit requested\n   The stated purpose of the " \
               "requested credit contributed to this decision."
        assert find_internal_content(text, feature_names=("purpose",)) == []

    def test_the_applicant_id_from_the_file_is_not_mistaken_for_a_leak(self):
        assert find_internal_content("Applicant ID:  dti_0001\n",
                                     feature_names=("dti",),
                                     applicant_id="dti_0001") == []

    def test_an_applicant_notice_refuses_to_render_internal_content(self):
        bad = ApplicantNotice(
            applicant_id="A1", decision="declined",
            reasons=(("Excessive obligations", "Your dti is 41.2."),),
            creditor_name="X", creditor_address="Y", enforcement_agency="Z",
            notice_date=pd.Timestamp("2026-09-25").date(), screen_features=("dti",))
        with pytest.raises(NoticeContentError, match="dti"):
            bad.render()

    def test_the_old_combined_render_can_no_longer_emit_the_flag(self, toy_explanation):
        notice = build_adverse_action_notice(toy_explanation, obs=0)
        notice.fair_lending_flags = ("addr_state",)
        assert "addr_state" not in notice.render()
        assert "addr_state" in notice.render_internal()

    def test_stage3_writes_the_two_documents_to_two_folders(self, tmp_path,
                                                          toy_explanation):
        notice = build_adverse_action_notice(toy_explanation, obs=0,
                                             specimen=NOT_FOR_LENDING)
        notice.fair_lending_flags = ("addr_state",)
        applicant, internal_txt, internal_json = write_notice_pair(
            notice, tmp_path / "tables" / "03_adverse_action_notice_t.txt",
            tmp_path / "tables" / "internal")
        text = applicant.read_text(encoding="utf-8")
        assert NOT_FOR_LENDING in text                      # research, never sent
        assert "addr_state" not in text and "INTERNAL" not in text.upper()
        assert internal_txt.parent.name == "internal"
        assert "addr_state" in internal_txt.read_text(encoding="utf-8")
        assert json.loads(internal_json.read_text())["fair_lending_flags"] == ["addr_state"]
        with pytest.raises(ValueError, match="must not share a folder"):
            write_notice_pair(notice, tmp_path / "same" / "n.txt", tmp_path / "same")

    def test_stage3_uses_the_shared_writer(self):
        src = (PROJECT_ROOT / "scripts" / "03_explain.py").read_text(encoding="utf-8")
        assert "write_notice_pair(" in src
        assert ".render()" not in src.split("(b) adverse action")[1].split("(c)")[0]


@pytest.fixture
def toy_explanation():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"dti": rng.uniform(1, 40, 80), "revol_util": rng.uniform(0, 100, 80),
                       "inq_last_6mths": rng.integers(0, 6, 80).astype(float)})

    class Lin:
        def predict_survival(self, X, times):
            r = 0.001 * X["dti"].to_numpy() + 0.0005 * X["revol_util"].to_numpy() \
                + 0.004 * X["inq_last_6mths"].to_numpy()
            return np.exp(-np.outer(r, np.atleast_1d(times)))

    top = df.sort_values("dti").iloc[-3:]
    return explain_survshap(Lin(), top, df, np.array([12.0, 36.0]), nsamples=40,
                            n_background=10, seed=1)


# ============================================================ 2. registry ==

def _evidence(tmp_path, *, tag="m", features=None, ceiling=(0.95, 0.93),
              verdict="FAIL"):
    """Everything check_approval_rules reads, for a fake model file."""
    models, tables = tmp_path / "models", tmp_path / "tables"
    models.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    model = models / f"02_models_{tag}.pkl"
    model.write_bytes(b"model bytes")
    sha = hashlib.sha256(b"model bytes").hexdigest()
    prov = {"inputs": {"model": {"sha256": sha}}}
    features = features or {"numeric": ["dti", "fico_midpoint"], "categorical": ["purpose"]}
    (tables / f"02_metrics_{tag}.json").write_text(json.dumps({
        "features": features, "n_train": 100, "n_test": 20, "split_scheme": "random",
        "results": [{"model": "discrete_hazard", "split": "test", "n": 20,
                     "concordance": 0.7, "ibs": 0.1, "auc_12m": 0.71}]}))
    (models / f"02_cleaning_values_{tag}.json").write_text(json.dumps(
        {"model_tag": tag, "fitted_rows": 100, "provenance": prov}))
    (tables / f"03e_ablation_{tag}.json").write_text(json.dumps(
        {"model_tag": tag, "features": [{"name": "dti", "concordance_drop": 0.02}],
         "provenance": prov}))
    (tables / f"03d_explainer_validation_v_{tag}.json").write_text(json.dumps(
        {"model_tag": tag, "nsamples": 600, "n_background": 100, "verdict": verdict,
         "reference_ceiling_survshap_vs_itself": {"n": 300, "top1_agreement": ceiling[0],
                                                  "mean_top4_overlap": ceiling[1]},
         "provenance": prov}))
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "paths": {"models_dir": models.as_posix(), "tables_dir": tables.as_posix(),
                  "data_dir": (tmp_path / "data").as_posix(),
                  "figures_dir": (tmp_path / "figures").as_posix(),
                  "registry": (tmp_path / "models.yaml").as_posix()},
        "decision": {"model_tag": tag}}))
    return cfg_path, model


@pytest.fixture(scope="module")
def registry_cli():
    spec = importlib.util.spec_from_file_location(
        "registry_cli", PROJECT_ROOT / "scripts" / "07_model_registry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestRegistry:
    def test_every_status_is_one_of_four(self, tmp_path):
        path = tmp_path / "r.yaml"
        path.write_text(yaml.safe_dump({"models": {"x": {"status": "production"}}}))
        with pytest.raises(registry.RegistryError, match="not one of"):
            registry.load_registry(path=path)

    def test_the_project_registry_records_the_decision(self):
        reg = registry.load_registry(path=PROJECT_ROOT / "config" / "models.yaml")
        full = reg.get("full")
        assert full.status == "deprecated"
        assert {d["id"] for d in full.known_defects} >= {"7c"}
        assert all(d["blocking"] for d in full.known_defects)
        assert full.non_disclosable_features == ["addr_state"]
        new = reg.get("full_applicant_nogeo")
        assert new.status in ("candidate", "approved")
        assert "fico_midpoint" in new.all_features
        assert not new.uses_non_disclosable
        assert load_config(PROJECT_ROOT / "config" / "config.yaml").decision.model_tag \
            == "full_applicant_nogeo"
        for rec in reg.models.values():
            assert rec.status in registry.STATUSES
            assert rec.status != "approved" or rec.approval.get("rules_passed")

    def test_deprecated_full_cannot_score(self):
        a = registry.assess("full", load_config(PROJECT_ROOT / "config" / "config.yaml"))
        assert not a.approved and a.status == "deprecated"
        assert "7c" in a.message()

    def test_an_unregistered_model_is_not_approved(self, tmp_path):
        cfg = _cfg(tmp_path)
        tb.write_registry(cfg, tb._write_dummy_model(cfg), tag="other")
        a = registry.assess("stub", cfg)
        assert not a.approved and a.status == "unregistered"

    @pytest.mark.parametrize("status", ["candidate", "benchmark", "deprecated"])
    def test_only_approved_is_approved(self, tmp_path, status):
        cfg = _cfg(tmp_path)
        tb.write_registry(cfg, tb._write_dummy_model(cfg), status=status)
        assert not registry.assess("stub", cfg).approved

    def test_approved_without_the_approval_block_is_refused(self, tmp_path):
        cfg = _cfg(tmp_path)
        path = tb._write_dummy_model(cfg)
        tb.write_registry(cfg, path, rules_passed=False)
        a = registry.assess("stub", cfg, model_path=path)
        assert not a.approved and "without an approval block" in a.message()

    def test_a_swapped_model_file_is_refused(self, tmp_path):
        cfg = _cfg(tmp_path)
        path = tb._write_dummy_model(cfg)
        tb.write_registry(cfg, path)
        path.write_bytes(b"a different model")
        a = registry.assess("stub", cfg, model_path=path)
        assert not a.approved and "not the approved one" in a.message()

    def test_different_features_are_refused(self, tmp_path):
        cfg = _cfg(tmp_path)
        path = tb._write_dummy_model(cfg)
        tb.write_registry(cfg, path)
        a = registry.assess("stub", cfg, model_path=path,
                            spec_features=list(tb.SPEC.all_columns) + ["addr_state"])
        assert not a.approved and "features differ" in a.message()

    def test_changed_evidence_withdraws_approval(self, tmp_path):
        cfg = _cfg(tmp_path)
        path = tb._write_dummy_model(cfg)
        ev = tmp_path / "evidence.json"
        ev.write_text("{}")
        tb.write_registry(cfg, path,
                          evidence={ev.as_posix(): hashlib.sha256(b"{}").hexdigest()})
        assert registry.assess("stub", cfg, model_path=path).approved
        ev.write_text('{"edited": true}')
        a = registry.assess("stub", cfg, model_path=path)
        assert not a.approved and "changed since approval" in a.message()

    def test_explainer_and_settings_must_be_covered(self, tmp_path):
        cfg = _cfg(tmp_path)
        path = tb._write_dummy_model(cfg)
        tb.write_registry(cfg, path, settings=[[600, 100]])
        assert not registry.assess("stub", cfg, explainer="treeshap").approved
        a = registry.assess("stub", cfg, explainer="survshap", nsamples=300,
                            n_background=50)
        assert not a.approved and "03f" in a.message()
        assert registry.assess("stub", cfg, explainer="survshap", nsamples=600,
                               n_background=100).approved


class TestApprovalRules:
    def test_all_rules_pass_on_complete_evidence(self, tmp_path, registry_cli):
        cfg_path, _ = _evidence(tmp_path)
        cfg = load_config(cfg_path)
        rec = registry_cli.record_from_metrics("m", cfg, "candidate")
        results = registry.check_approval_rules(
            rec, models_dir=cfg.paths.models_dir, tables_dir=cfg.paths.tables_dir,
            explain_settings=(600, 100))
        assert [r.rule for r in results] == list(registry.APPROVAL_RULES)
        assert all(r.passed for r in results), [(r.rule, r.detail) for r in results]

    @pytest.mark.parametrize("breaks, rule", [
        ("ablation", "A5_ablation"),
        ("validation", "A6_explainer_validation"),
        ("reproducibility", "A6_explainer_validation"),
        ("cleaning", "A4_cleaning_values"),
        ("defect", "A3_no_blocking_defect"),
        ("settings", "A7_explain_settings"),
        ("model", "A1_model_file"),
    ])
    def test_each_rule_fails_on_its_own_gap(self, tmp_path, registry_cli, breaks, rule):
        cfg_path, model = _evidence(
            tmp_path, ceiling=(0.80, 0.70) if breaks == "reproducibility" else (0.95, 0.93),
            features=({"numeric": ["dti"], "categorical": []} if breaks == "defect"
                      else None))
        cfg = load_config(cfg_path)
        rec = registry_cli.record_from_metrics("m", cfg, "candidate")
        tables, models = cfg.paths.tables_dir, cfg.paths.models_dir
        if breaks == "ablation":
            (tables / "03e_ablation_m.json").unlink()
        elif breaks == "validation":
            (tables / "03d_explainer_validation_v_m.json").unlink()
        elif breaks == "cleaning":
            (models / "02_cleaning_values_m.json").write_text(json.dumps(
                {"fitted_rows": 5, "provenance": {"inputs": {"model": {
                    "sha256": rec.model_sha256}}}}))
        elif breaks == "model":
            model.write_bytes(b"retrained")
        settings = (300, 50) if breaks == "settings" else (600, 100)
        results = {r.rule: r for r in registry.check_approval_rules(
            rec, models_dir=models, tables_dir=tables, explain_settings=settings)}
        assert not results[rule].passed, results[rule].detail

    def test_the_7c_defect_is_recorded_automatically(self, tmp_path, registry_cli):
        cfg_path, _ = _evidence(tmp_path, features={"numeric": ["dti"],
                                                    "categorical": ["addr_state"]})
        rec = registry_cli.record_from_metrics("m", load_config(cfg_path), "benchmark")
        assert rec.known_defects[0]["id"] == "7c" and rec.known_defects[0]["blocking"]
        assert rec.non_disclosable_features == ["addr_state"]

    def test_approve_refuses_then_accepts_and_scoring_re_checks(self, tmp_path,
                                                               registry_cli):
        cfg_path, model = _evidence(tmp_path)
        run = lambda *a: registry_cli.main([*a, "--config", str(cfg_path)])  # noqa: E731
        assert run("register", "--model-tag", "m", "--status", "candidate") == 0
        # approval names who and where, and never comes from set-status
        assert run("approve", "--model-tag", "m") == 2
        assert run("set-status", "--model-tag", "m", "--status", "approved",
                   "--reason", "x") == 2
        (load_config(cfg_path).paths.tables_dir / "03e_ablation_m.json").unlink()
        assert run("approve", "--model-tag", "m", "--by", "t", "--findings", "7l") == 4
        assert registry.load_registry(load_config(cfg_path)).get("m").status == "candidate"

        cfg_path, model = _evidence(tmp_path)                 # restore the evidence
        assert run("approve", "--model-tag", "m", "--by", "t", "--findings", "7l") == 0
        cfg = load_config(cfg_path)
        rec = registry.load_registry(cfg).get("m")
        assert rec.status == "approved" and rec.approval["approved_by"] == "t"
        assert rec.approval["explainers"] == ["survshap"]      # TreeSHAP failed 03d
        assert rec.approval["explain_settings"] == [[600, 100]]
        spec = rec.all_features
        assert registry.assess("m", cfg, model_path=model, spec_features=spec,
                               nsamples=600, n_background=100).approved
        # evidence edited after approval -> no longer approved
        (cfg.paths.tables_dir / "03e_ablation_m.json").write_text('{"model_tag": "m"}')
        assert not registry.assess("m", cfg, model_path=model, spec_features=spec,
                                   nsamples=600, n_background=100).approved


# =============================================== 2b. scoring obeys approval ==

class TestScoringObeysTheRegistry:
    @pytest.mark.parametrize("status", ["candidate", "deprecated", "benchmark"])
    def test_a_non_approved_model_is_refused_before_anything_is_written(
            self, tmp_path, status):
        cfg = _cfg(tmp_path)
        ctx = _ctx(cfg, status=status)
        steps = []
        with pytest.raises(BatchError, match="not approved for lending decisions"):
            run_batch(tb._upload(), "a.csv", cfg, ctx=ctx, runs_dir=tmp_path / "runs",
                      progress=lambda s, st, m="": steps.append((s, st)))
        assert ("check", "failed") in steps
        assert not (tmp_path / "runs").exists()

    def test_an_unregistered_model_is_refused(self, tmp_path):
        cfg = _cfg(tmp_path)
        ctx = _ctx(cfg, register=False)
        with pytest.raises(BatchError, match="unregistered"):
            run_batch(tb._upload(), "a.csv", cfg, ctx=ctx, runs_dir=tmp_path / "runs")

    def test_an_override_finishes_but_is_stamped_everywhere(self, tmp_path):
        cfg = _cfg(tmp_path)
        ctx = _ctx(cfg, status="deprecated")
        res = run_batch(tb._upload(), "a.csv", cfg, ctx=ctx, runs_dir=tmp_path / "runs",
                        allow_unapproved_model=True)
        s = res.summary
        assert s["for_lending_decisions"] is False
        assert s["model_registry_status"] == "deprecated" and not s["model_approved"]
        assert s["model_override"] is True
        assert "not approved" in s["not_for_lending_reasons"]
        summary_csv = pd.read_csv(res.files["run_summary.csv"])
        assert not summary_csv["for_lending_decisions"].iloc[0]
        scored = pd.read_csv(res.files["scored_applicants.csv"])
        assert not scored["for_lending_decisions"].any()
        assert scored["model_status"].str.contains("not approved").all()
        assert all(NOT_FOR_LENDING in t for t in _notices(res.run_dir).values())
        checks = res.checks.set_index("check")
        assert checks.loc["model_approved", "status"] == "OVERRIDDEN"
        prov = json.loads((res.run_dir / "provenance.json").read_text())
        assert prov["provenance"]["args"]["for_lending_decisions"] is False

    def test_an_approved_model_scores_for_lending(self, tmp_path):
        cfg = _cfg(tmp_path)
        res = run_batch(tb._upload(), "a.csv", cfg, ctx=_ctx(cfg),
                        runs_dir=tmp_path / "runs")
        assert res.summary["for_lending_decisions"] is True
        assert res.summary["model_registry_status"] == "approved"
        assert not any(NOT_FOR_LENDING in t for t in _notices(res.run_dir).values())

    def test_a_threshold_off_policy_is_stamped_not_for_lending(self, tmp_path):
        cfg = _cfg(tmp_path)
        res = run_batch(tb._upload(), "a.csv", cfg, ctx=_ctx(cfg),
                        runs_dir=tmp_path / "runs", threshold=0.40)
        assert res.summary["threshold_is_published"] is False
        assert res.summary["for_lending_decisions"] is False
        assert res.checks.set_index("check").loc["threshold_is_published",
                                                 "status"] == "OVERRIDDEN"


# ==================================================== 3. checks on every run ==

class TestRunChecks:
    @pytest.fixture
    def finished(self, tmp_path):
        cfg = _cfg(tmp_path)
        res = run_batch(tb._upload(20), "a.csv", cfg, ctx=_ctx(cfg),
                        runs_dir=tmp_path / "runs")
        return res

    def _verify(self, res, **kw):
        args = dict(n_rows_read=res.summary["n_rows"], threshold=0.45,
                    published_threshold=0.45, horizon_months=36,
                    feature_names=tb.SPEC.all_columns, model_approved=True,
                    model_label="approved", allow_unapproved=False,
                    cap_note=batch.CAP_NOTE)
        args.update(kw)
        return {c.name: c for c in verify_run(res.run_dir, **args)}

    def test_a_clean_run_passes_every_check_and_writes_them(self, finished):
        checks = pd.read_csv(finished.files["validation_checks.csv"])
        assert set(checks["check"]) == {
            "counts_add_up", "decisions_match_threshold", "threshold_is_published",
            "risk_12m_le_36m", "rejected_have_reasons_or_pending",
            "no_nondisclosable_stated_reason", "applicant_notices_clean",
            "model_approved", "input_quality"}
        assert (checks["status"] == "PASS").all()
        assert finished.summary["validation_checks_passed"] is True
        assert finished.summary["run_status"] == "finished"
        # and a background run's page reads the same table back
        assert batch.load_result(finished.run_dir).checks.equals(checks)

    def _edit(self, path, fn):
        df = pd.read_csv(path, dtype={"applicant_id": str})
        fn(df)
        df.to_csv(path, index=False)

    def test_a_flipped_decision_fails(self, finished):
        self._edit(finished.files["scored_applicants.csv"],
                   lambda df: df.__setitem__("decision", df["decision"].replace(
                       {"approve": "reject", "reject": "approve"})))
        assert self._verify(finished)["decisions_match_threshold"].status == "FAIL"

    def test_a_missing_row_fails_the_counts(self, finished):
        path = finished.files["approved_applicants.csv"]
        df = pd.read_csv(path)
        df.iloc[1:].to_csv(path, index=False)
        assert self._verify(finished)["counts_add_up"].status == "FAIL"

    def test_twelve_month_risk_above_thirty_six_fails(self, finished):
        self._edit(finished.files["scored_applicants.csv"],
                   lambda df: df.__setitem__("pd_12m", df["pd_36m"] + 0.01))
        assert self._verify(finished)["risk_12m_le_36m"].status == "FAIL"

    def test_a_rejection_without_reasons_or_mark_fails(self, finished):
        def blank(df):
            i = df.index[df["explained"] == "explained"][0]
            df.loc[i, ["reason_1", "explained"]] = ["", ""]
        self._edit(finished.files["rejected_applicants.csv"], blank)
        assert self._verify(finished)["rejected_have_reasons_or_pending"].status == "FAIL"

    def test_pending_rows_pass_because_they_say_so(self, finished):
        def pend(df):
            i = df.index[df["explained"] == "explained"][0]
            df.loc[i, ["reason_1", "explained"]] = ["", NO_REASON_NOTE]
        self._edit(finished.files["rejected_applicants.csv"], pend)
        assert self._verify(finished)["rejected_have_reasons_or_pending"].status == "PASS"

    def test_a_non_disclosable_stated_reason_fails(self, finished):
        def geo(df):
            i = df.index[df["explained"] == "explained"][0]
            df.loc[i, "reason_1_feature"] = "addr_state"
        self._edit(finished.files["rejected_applicants.csv"], geo)
        assert self._verify(finished)["no_nondisclosable_stated_reason"].status == "FAIL"

    def test_a_leaked_notice_on_disk_fails(self, finished):
        zpath = finished.run_dir / "adverse_action_notices.zip"
        with zipfile.ZipFile(zpath) as zf:
            items = {n: zf.read(n).decode() for n in zf.namelist()}
        first = next(iter(items))
        items[first] += "\n[INTERNAL REVIEW FLAG] drivers: addr_state\n"
        with zipfile.ZipFile(zpath, "w") as zf:
            for n, t in items.items():
                zf.writestr(n, t)
        assert self._verify(finished)["applicant_notices_clean"].status == "FAIL"

    def test_an_unapproved_model_fails_unless_overridden(self, finished):
        assert self._verify(finished, model_approved=False)["model_approved"].status \
            == "FAIL"
        assert self._verify(finished, model_approved=False, allow_unapproved=True)[
            "model_approved"].status == "OVERRIDDEN"

    def test_a_failing_run_is_loud_and_never_marked_finished(self, tmp_path):
        """Non-monotone risk from a broken model: the run raises, withholds its
        notices, and leaves nothing that marks it finished."""
        cfg = _cfg(tmp_path)
        ctx = _ctx(cfg, model=BackwardsModel())
        with pytest.raises(BatchError, match="failed 1 of its own checks") as exc:
            run_batch(tb._upload(), "a.csv", cfg, ctx=ctx, runs_dir=tmp_path / "runs")
        run_dir = exc.value.run_dir
        checks = pd.read_csv(run_dir / "validation_checks.csv").set_index("check")
        assert checks.loc["risk_12m_le_36m", "status"] == "FAIL"
        assert not (run_dir / "provenance.json").exists()
        assert not (run_dir / "adverse_action_notices.zip").exists()
        assert (run_dir / "RUN_FAILED_CHECKS.txt").exists()
        assert pd.read_csv(run_dir / "run_summary.csv")["run_status"].iloc[0] \
            == "failed_checks"
        with pytest.raises(BatchError, match="does not hold a finished run"):
            batch.load_result(run_dir)

    def test_a_notice_leak_inside_the_run_is_caught_on_disk(self, tmp_path, monkeypatch):
        """Even if a future change bypassed the notice's own screen, the run's
        read-back of the zip catches it and withholds every notice."""
        from creditsurv.explain import adverse_action as aa

        original = aa.AdverseActionNotice.render
        monkeypatch.setattr(aa.AdverseActionNotice, "render",
                            lambda self: original(self) + "\nfair-lending review\n")
        cfg = _cfg(tmp_path)
        with pytest.raises(BatchError, match="applicant_notices_clean") as exc:
            run_batch(tb._upload(), "a.csv", cfg, ctx=_ctx(cfg),
                      runs_dir=tmp_path / "runs")
        run_dir = exc.value.run_dir
        assert (run_dir / "adverse_action_notices.WITHHELD.zip").exists()
        assert not (run_dir / "adverse_action_notices.zip").exists()
        assert (run_dir / "RUN_FAILED_CHECKS.txt").exists()
        # Two phases: the decisions passed their own checks in Phase 1 and stand;
        # the failed Phase 2 is recorded on the run and nothing may be sent.
        s = batch.load_result(run_dir).summary
        assert s["run_status"] == "failed_checks"
        assert s["for_lending_decisions"] is False and s["n_notices"] == 0
        assert "content screen" in s["not_for_lending_reasons"]


# =========================================== 4. fair-lending monitoring ==

class TestFairLendingMonitor:
    def test_every_run_reports_the_share(self, tmp_path):
        cfg = _cfg(tmp_path)
        res = run_batch(tb._upload(), "a.csv", cfg, ctx=_ctx(cfg),
                        runs_dir=tmp_path / "runs")
        s = res.summary
        for key in ("n_fair_lending_flagged", "fair_lending_flag_share",
                    "n_top_driver_not_disclosable", "fair_lending_review_share",
                    "fair_lending_review_required"):
            assert key in s, key
        assert s["n_fair_lending_flagged"] == 0            # no geography in the model
        assert s["fair_lending_review_required"] is False
        assert s["fair_lending_review_share"] == 0.05

    def test_geography_above_the_threshold_requires_review(self, geo_run):
        s = geo_run.summary
        assert s["fair_lending_flag_share"] == round(
            s["n_fair_lending_flagged"] / s["n_explained"], 4)
        assert s["fair_lending_flag_share"] > s["fair_lending_review_share"]
        assert s["fair_lending_review_required"] is True
        assert "addr_state" in s["fair_lending_flag_features"]

    def test_the_threshold_is_configurable(self, tmp_path):
        # Every decline in this fixture is flagged (share 1.0); a share can never
        # be above 1.0, so this setting means "never ask for review".
        cfg = _cfg(tmp_path, fair_lending_review_share=1.0)
        ctx = _ctx(cfg, spec=GEO_SPEC, model=GeoModel(), train=_geo_training())
        res = run_batch(_geo_upload(), "geo.csv", cfg, ctx=ctx,
                        runs_dir=tmp_path / "runs")
        assert res.summary["n_fair_lending_flagged"] > 0
        assert res.summary["fair_lending_review_required"] is False


# ========================================== 5. the dashboard and the CLI ==

VIEWS = PROJECT_ROOT / "app" / "views"


def _render(run_dir):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    def script(views, src, run):
        import sys
        sys.path.insert(0, views)
        sys.path.insert(0, src)
        from creditsurv.batch import load_result
        from _dashboard import render
        render(load_result(run))

    at = AppTest.from_function(script, default_timeout=180,
                               kwargs={"views": str(VIEWS),
                                       "src": str(PROJECT_ROOT / "src"),
                                       "run": str(run_dir)})
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


class TestDashboard:
    def test_shows_registry_status_checks_and_fair_lending_review(self, geo_run):
        at = _render(geo_run.run_dir)
        text = " ".join(m.value for m in list(at.success) + list(at.error)
                        + list(at.info) + list(at.warning))
        assert "Run checks: 9 of 9 passed" in text
        assert "registry status **approved**" in text
        assert "Fair-lending review required" in text
        assert "NOT FOR LENDING" not in text

    def test_an_override_run_is_stamped_on_the_page(self, tmp_path):
        cfg = _cfg(tmp_path)
        res = run_batch(tb._upload(), "a.csv", cfg, ctx=_ctx(cfg, status="candidate"),
                        runs_dir=tmp_path / "runs", allow_unapproved_model=True)
        at = _render(res.run_dir)
        errors = " ".join(e.value for e in at.error)
        assert "NOT FOR LENDING DECISIONS" in errors
        assert any("not approved for lending decisions" in i.value for i in at.info)


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location(
        "score_cli", PROJECT_ROOT / "scripts" / "06_score_upload.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.blocked_imports = lambda: []          # the stub model needs no native library
    return mod


def _cli_config(tmp_path, cfg) -> Path:
    d = cfg.decision
    path = tmp_path / "cli_config.yaml"
    path.write_text(yaml.safe_dump({
        "paths": {k: Path(getattr(cfg.paths, k)).as_posix() for k in
                  ("data_dir", "models_dir", "figures_dir", "tables_dir", "registry")},
        "decision": {"model_tag": "stub", "horizon_months": d.horizon_months,
                     "reject_at_or_above": d.reject_at_or_above,
                     "explain_nsamples": d.explain_nsamples,
                     "explain_n_background": d.explain_n_background,
                     "max_explained": d.max_explained,
                     "background_rows": d.background_rows, "explain_workers": 1}}))
    return path


class TestCommandLine:
    """06_score_upload.py -- also the process a background run executes."""

    def _run(self, cli, tmp_path, monkeypatch, status, *extra):
        cfg = _cfg(tmp_path)
        ctx = _ctx(cfg, status=status)
        monkeypatch.setattr(batch, "load_context", lambda *a, **k: ctx)
        upload = tmp_path / "input_a.csv"
        upload.write_bytes(tb._upload())
        run_dir = tmp_path / "run"
        code = cli.main(["--file", str(upload), "--config",
                         str(_cli_config(tmp_path, cfg)), "--run-dir", str(run_dir),
                         *extra])
        return code, run_dir

    def test_refuses_an_unapproved_model(self, cli, tmp_path, monkeypatch, capsys):
        code, run_dir = self._run(cli, tmp_path, monkeypatch, "candidate")
        assert code == 3
        assert "not approved for lending decisions" in capsys.readouterr().err
        assert not (run_dir / "provenance.json").exists()

    def test_override_flag_scores_and_stamps(self, cli, tmp_path, monkeypatch, capsys):
        code, run_dir = self._run(cli, tmp_path, monkeypatch, "candidate",
                                  "--allow-unapproved-model")
        out = capsys.readouterr().out
        assert code == 0
        assert "NOT FOR LENDING DECISIONS" in out
        assert "OVERRIDDEN" in out and "applicant_notices_clean" in out
        assert "Fair-lending monitor" in out
        assert (run_dir / "internal" / "internal_review_flags.csv").exists()

    def test_approved_model_prints_passing_checks(self, cli, tmp_path, monkeypatch,
                                                  capsys):
        code, _ = self._run(cli, tmp_path, monkeypatch, "approved")
        out = capsys.readouterr().out
        assert code == 0 and "NOT FOR LENDING" not in out
        assert out.count("PASS ") == 9


# ============================================ every path, structurally ==

def _py_files():
    for folder in ("src", "app", "scripts"):
        yield from (PROJECT_ROOT / folder).rglob("*.py")


def test_the_combined_internal_block_is_gone_everywhere():
    for path in _py_files():
        assert "[INTERNAL REVIEW FLAG" not in path.read_text(encoding="utf-8"), path


def test_every_caller_of_the_notice_builder_is_a_guarded_path():
    """A new scoring path -- a live-scoring page, say -- must be a deliberate
    addition here, and must go through run_batch or write_notice_pair."""
    callers = {p.relative_to(PROJECT_ROOT).as_posix() for p in _py_files()
               if "build_adverse_action_notice(" in p.read_text(encoding="utf-8")
               and p.name != "adverse_action.py"}
    assert callers == {
        "src/creditsurv/phase2.py",                    # every scoring run (Phase 2)
        "scripts/03_explain.py",                       # Stage 3, via write_notice_pair
        "scripts/03d_explainer_validation.py",         # reasons only, no notice written
        "scripts/03f_settings_comparison.py",          # reasons only, no notice written
    }, callers
    for path in ("scripts/03d_explainer_validation.py",
                 "scripts/03f_settings_comparison.py"):
        assert ".render(" not in (PROJECT_ROOT / path).read_text(encoding="utf-8")
    assert not any("build_adverse_action_notice" in p.read_text(encoding="utf-8")
                   for p in (PROJECT_ROOT / "app").rglob("*.py"))


def test_both_dashboard_paths_and_the_cli_pass_the_override_explicitly():
    home = (VIEWS / "home.py").read_text(encoding="utf-8")
    assert "allow_unapproved_model=allow_unapproved" in home      # inline run
    assert '"--allow-unapproved-model"' in home                   # background run
    cli_src = (PROJECT_ROOT / "scripts" / "06_score_upload.py").read_text(encoding="utf-8")
    assert "allow_unapproved_model=args.allow_unapproved_model" in cli_src
    # run_batch is the only place scoring decides; it defaults to refusing.
    import inspect
    assert inspect.signature(run_batch).parameters[
        "allow_unapproved_model"].default is False
