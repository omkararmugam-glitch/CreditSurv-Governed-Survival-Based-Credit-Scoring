"""The Score Applicants flow, end to end on a synthetic fixture.

No real data and no trained model: the context carries a stub model with the same
``predict_survival`` interface the pipeline's models expose, so validation,
cleaning, scoring, explanation, notices and file writing are all exercised.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

from creditsurv import batch
from creditsurv import drift as drift_mod
from creditsurv.batch import BatchError, ScoringContext, run_batch, validate
from creditsurv.cleaning import fit_values, policy_from_config
from creditsurv.config import Config, DecisionConfig, Paths
from creditsurv.features.build import FeatureSpec, build_design_matrix

NUMERIC = ("loan_amnt", "installment", "annual_inc", "dti", "open_acc", "revol_bal",
           "delinq_2yrs", "inq_last_6mths", "revol_util")   # revol_util: optional
CATEGORICAL = ("purpose", "home_ownership")
SPEC = FeatureSpec(numeric=NUMERIC, categorical=CATEGORICAL, structural_missing=())


class StubModel:
    """Survival falls with debt burden and delinquencies. Same interface as
    DiscreteTimeHazardModel/CoxModel: predict_survival(X, times) -> (n, t)."""

    def predict_survival(self, X: pd.DataFrame, times) -> np.ndarray:
        times = np.atleast_1d(np.asarray(times, dtype=float))
        dti = pd.to_numeric(X["dti"], errors="coerce").fillna(20.0).to_numpy()
        delinq = pd.to_numeric(X["delinq_2yrs"], errors="coerce").fillna(0).to_numpy()
        inq = pd.to_numeric(X["inq_last_6mths"], errors="coerce").fillna(0).to_numpy()
        rate = 0.0006 * dti + 0.004 * delinq + 0.003 * inq
        return np.exp(-np.outer(rate, times))


def _training_frame(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "loan_amnt": rng.integers(1000, 35000, n).astype(float),
        "installment": rng.uniform(30, 1200, n),
        "annual_inc": rng.lognormal(11, 0.5, n),
        "dti": rng.uniform(1, 38, n),
        "open_acc": rng.integers(1, 25, n).astype(float),
        "revol_bal": rng.uniform(0, 60000, n),
        "revol_util": rng.uniform(0, 100, n),
        "delinq_2yrs": rng.integers(0, 4, n).astype(float),
        "inq_last_6mths": rng.integers(0, 6, n).astype(float),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "car"], n),
        "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n),
        "duration_months": rng.integers(1, 36, n),
        "event": rng.integers(0, 2, n),
    })
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


@pytest.fixture
def cfg(tmp_path) -> Config:
    """Everything under tmp_path: a test never writes into the project's outputs."""
    return Config(paths=Paths(data_dir=tmp_path / "data", models_dir=tmp_path / "models",
                              figures_dir=tmp_path / "figures",
                              tables_dir=tmp_path / "tables"),
                  decision=DecisionConfig(
        model_tag="stub", horizon_months=36, reject_at_or_above=0.45,
        explain_nsamples=2 * len(NUMERIC + CATEGORICAL), explain_n_background=8,
        max_explained=3, background_rows=200))


@pytest.fixture
def runs(tmp_path):
    return tmp_path / "runs"


@pytest.fixture
def ctx(cfg) -> ScoringContext:
    train = _training_frame()
    dm = build_design_matrix(train, SPEC, flavour="gbm")
    # Cleaning values are fitted on the training frame, exactly as Stage 2 does,
    # and the context only ever reapplies them.
    return ScoringContext(
        cfg=cfg, model_tag="stub", model_name="discrete_hazard", model=StubModel(),
        spec=SPEC, bundle={"artefacts": {"gbm_columns": list(dm.X.columns)}},
        model_path=_write_dummy_model(cfg), background=dm.X, reference=train,
        clean_values=fit_values(train, SPEC, source="stub_training.parquet"),
        policy=policy_from_config(cfg),
        times=np.array([6.0, 12.0, 24.0, 36.0]), data_source=_dummy_source(cfg))


def _write_dummy_model(cfg) -> "object":
    p = cfg.paths.models_dir / "02_models_stub.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"stub")
    return p


def _dummy_source(cfg):
    p = cfg.paths.data_dir / "stub_training.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"stub")
    return p


def _upload(n=12, *, drop=None, alias=True) -> bytes:
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "applicant_id": [f"A{i:03d}" for i in range(n)],
        "loan_amnt": np.linspace(2000, 30000, n),
        "installment": np.linspace(60, 900, n),
        "annual_inc": np.linspace(25000, 150000, n),
        # Spread across the threshold: low dti approves, high dti rejects.
        "dti": np.linspace(2, 38, n),
        "open_acc": rng.integers(2, 20, n),
        "revol_bal": rng.uniform(100, 40000, n),
        "delinq_2yrs": rng.integers(0, 3, n),
        "inquiries_last_6m" if alias else "inq_last_6mths": rng.integers(0, 5, n),
        "loan_purpose" if alias else "purpose":
            [("debt_consolidation", "car")[i % 2] for i in range(n)],
        "home_ownership": [("RENT", "OWN", "MORTGAGE")[i % 3] for i in range(n)],
        "credit_score": rng.integers(580, 800, n),     # extra column: ignored
    })
    if drop:
        df = df.drop(columns=list(drop))
    return df.to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------- happy path --

def test_end_to_end_produces_every_output(runs, cfg, ctx):
    steps = []
    res = run_batch(_upload(), "applicants.csv", cfg, ctx=ctx, runs_dir=runs,
                    progress=lambda s, st, m="": steps.append((s, st)))

    assert [s for s, st in steps if st == "done"] == \
        ["check", "clean", "score", "explain", "profile", "files"]
    # Profiling samples the whole file, so it finishes with the last block --
    # after scoring, though the page lists it earlier and shows it in progress.
    assert ("profile", "running") in steps

    for name in ("scored_applicants.csv", "approved_applicants.csv",
                 "rejected_applicants.csv", "run_summary.csv",
                 "adverse_action_notices.zip", "cleaning_report.csv",
                 "data_drift.csv"):
        assert res.files[name].exists(), name
    assert (res.run_dir / "provenance.json").exists()
    assert (res.run_dir / "input_applicants.csv").exists()

    scored = pd.read_csv(res.files["scored_applicants.csv"])
    approved = pd.read_csv(res.files["approved_applicants.csv"])
    rejected = pd.read_csv(res.files["rejected_applicants.csv"])
    assert len(scored) == 12
    assert len(approved) + len(rejected) == len(scored)      # they add up
    assert len(approved) and len(rejected)                   # both sides exercised
    for col in ("row_id", "applicant_id", "pd_12m", "pd_36m", "decision", "threshold",
                "reason_1", "reason_2", "reason_3", "explained"):
        assert col in scored.columns, col
    for col in ("reason_4", "fair_lending_flag", "notice_file", "direction_consistent"):
        assert col in rejected.columns, col
    assert set(scored["decision"]) <= {"approve", "reject"}
    assert ((scored["pd_36m"] >= cfg.decision.reject_at_or_above)
            == (scored["decision"] == "reject")).all()
    assert scored["pd_12m"].between(0, 1).all()


def test_run_summary_records_the_config_threshold(runs, cfg, ctx):
    res = run_batch(_upload(), "a.csv", cfg, ctx=ctx, runs_dir=runs)
    summary = pd.read_csv(res.files["run_summary.csv"])
    assert len(summary) == 1
    assert summary["threshold"].iloc[0] == cfg.decision.reject_at_or_above
    assert summary["horizon_months"].iloc[0] == cfg.decision.horizon_months
    assert summary["n_approved"].iloc[0] + summary["n_rejected"].iloc[0] == \
        summary["n_rows"].iloc[0]
    assert summary["model_tag"].iloc[0] == "stub"
    assert summary["n_notices"].iloc[0] <= cfg.decision.max_explained


def test_notices_one_per_explained_rejected_applicant(runs, cfg, ctx):
    res = run_batch(_upload(), "a.csv", cfg, ctx=ctx, runs_dir=runs)
    rejected = pd.read_csv(res.files["rejected_applicants.csv"])
    with zipfile.ZipFile(res.files["adverse_action_notices.zip"]) as zf:
        names = zf.namelist()
        text = zf.read(names[0]).decode("utf-8")
    explained = rejected[rejected["explained"] == "explained"]
    assert len(names) == len(explained) == min(len(rejected), cfg.decision.max_explained)
    assert set(explained["notice_file"]) == set(names)
    assert "STATEMENT OF ADVERSE ACTION" in text
    if len(rejected) > cfg.decision.max_explained:
        capped = rejected[rejected["explained"] != "explained"]
        assert capped["reason_1"].fillna("").eq("").all()
        assert capped["explained"].str.contains("explanation cap reached").all()


def test_runs_never_overwrite_each_other(runs, cfg, ctx):
    a = run_batch(_upload(), "a.csv", cfg, ctx=ctx, runs_dir=runs)
    b = run_batch(_upload(), "a.csv", cfg, ctx=ctx, runs_dir=runs)
    assert a.run_dir != b.run_dir and a.run_dir.exists() and b.run_dir.exists()


def test_aliases_mapped_and_extra_columns_ignored(runs, cfg, ctx):
    res = run_batch(_upload(), "a.csv", cfg, ctx=ctx, runs_dir=runs)
    assert res.report.mapped == {"inquiries_last_6m": "inq_last_6mths",
                                 "loan_purpose": "purpose"}
    assert "credit_score" in res.report.ignored
    assert "no trained model uses a bureau score" in res.report.ignored["credit_score"]
    assert res.report.id_column == "applicant_id"
    assert res.summary["ignored_columns"] == "credit_score"


def test_out_of_range_values_are_scored_but_flagged(runs, cfg, ctx):
    df = pd.read_csv(__import__("io").BytesIO(_upload()))
    df.loc[0, "annual_inc"] = 50_000_000                      # far outside training
    res = run_batch(df.to_csv(index=False).encode(), "a.csv", cfg, ctx=ctx,
                    runs_dir=runs)
    scored = pd.read_csv(res.files["scored_applicants.csv"])
    assert scored.loc[0, "n_out_of_range"] >= 1
    assert "annual_inc" in scored.loc[0, "out_of_range_fields"]
    assert scored.loc[0, "decision"] in ("approve", "reject")   # still scored
    assert res.summary["rows_out_of_range"] >= 1


def test_missing_optional_feature_is_accepted_and_reported(runs, cfg, ctx):
    """revol_util is in the model's spec but not in the upload: scored anyway,
    with the gap reported rather than hidden."""
    res = run_batch(_upload(), "a.csv", cfg, ctx=ctx, runs_dir=runs)
    assert res.report.missing_optional == ["revol_util"]
    assert res.summary["features_present"] == res.summary["features_expected"] - 1
    assert 0 < res.summary["feature_coverage"] < 1
    scored = pd.read_csv(res.files["scored_applicants.csv"])
    assert (scored["features_missing"] == 1).all()
    assert any("treated as missing" in w for w in res.report.warnings)


# ------------------------------------------------------------ failure paths --

def test_missing_required_column_stops_with_a_plain_message(runs, cfg, ctx):
    steps = []
    with pytest.raises(BatchError) as exc:
        run_batch(_upload(drop=["annual_inc"]), "a.csv", cfg, ctx=ctx,
                  runs_dir=runs, progress=lambda s, st, m="": steps.append((s, st)))
    assert "missing required column(s): annual_inc" in exc.value.message
    assert exc.value.fix
    assert ("check", "failed") in steps
    assert not any(st == "done" and s != "check" for s, st in steps)
    assert not runs.exists() or list(runs.iterdir()) == []            # nothing written on failure


def test_empty_file_is_refused(runs, cfg, ctx):
    with pytest.raises(BatchError, match="empty"):
        run_batch(b"   ", "a.csv", cfg, ctx=ctx, runs_dir=runs)


def test_header_only_file_is_refused(runs, cfg, ctx):
    with pytest.raises(BatchError, match="no applicants"):
        run_batch(b"annual_inc,dti\n", "a.csv", cfg, ctx=ctx, runs_dir=runs)


def test_non_csv_is_refused(runs, cfg, ctx):
    with pytest.raises(BatchError, match="not a CSV file"):
        run_batch(b"%PDF-1.4 ...", "applicants.pdf", cfg, ctx=ctx, runs_dir=runs)


def test_ragged_csv_is_refused(runs, cfg, ctx):
    bad = b"a,b,c\n1,2,3\n1,2,3,4,5\n"
    with pytest.raises(BatchError, match="not a valid CSV"):
        run_batch(bad, "a.csv", cfg, ctx=ctx, runs_dir=runs)


def test_missing_model_reports_where_to_train_one(runs, cfg):
    with pytest.raises(BatchError) as exc:
        batch.load_context(cfg, model_tag="does_not_exist")
    assert "No trained model found" in exc.value.message
    assert "Run Pipeline" in exc.value.fix


def test_validate_reports_every_missing_core_column_at_once():
    df = pd.DataFrame({"loan_amnt": [1000.0], "dti": [10.0]})
    _, rep = validate(df, SPEC)
    assert set(rep.missing_core) == set(batch.CORE_REQUIRED) - {"loan_amnt", "dti"}
    assert not rep.ok


# ---------------------------------------------- cleaning, profiling and drift --

def test_cleaning_report_and_drift_are_written_and_summarised(runs, cfg, ctx):
    res = run_batch(_upload(), "a.csv", cfg, ctx=ctx, runs_dir=runs)
    cleaning = pd.read_csv(res.files["cleaning_report.csv"])
    assert {"scope", "rule", "column", "count"} <= set(cleaning.columns)
    assert cleaning.loc[cleaning["rule"] == "rows_in", "count"].iloc[0] == 12
    assert res.clean_report.rows_in == res.clean_report.rows_out == 12

    drift = pd.read_csv(res.files["data_drift.csv"])
    assert {"feature", "measure", "score", "status"} <= set(drift.columns)
    assert set(drift["feature"]) == set(NUMERIC + CATEGORICAL)
    # 12 rows is far below drift.MIN_ROWS, so drift is not assessed at all.
    assert res.summary["drift_status"] == "insufficient"
    assert res.drift.colour == "grey"
    assert "Too few rows to assess drift" in res.drift.headline()
    assert (drift["status"] == "insufficient").all()
    assert res.summary["cleaning_policy_version"] == "v1-parity"
    assert res.summary["cleaning_values_fitted_rows"] == 400   # the training frame
    assert res.summary["rows_dropped_by_cleaning"] == 0        # parity: nothing dropped


def test_profile_is_computed_on_the_upload(runs, cfg, ctx):
    res = run_batch(_upload(), "a.csv", cfg, ctx=ctx, runs_dir=runs)
    prof = res.profile
    assert prof["overview"]["rows"] == 12
    assert not prof["missing"].empty and not prof["numeric"].empty
    assert set(prof["numeric"]["column"]) <= set(NUMERIC)
    assert prof["drift"] is res.drift


def test_lending_club_formatting_is_read_not_dropped(runs, cfg, ctx):
    """A file written the way Lending Club writes one: currency and percentages.
    Before cleaning.py these became missing on the scoring path only."""
    df = pd.read_csv(__import__("io").BytesIO(_upload()))
    df["annual_inc"] = ["$" + f"{v:,.0f}" for v in df["annual_inc"]]
    df["revol_util"] = [f"{v:.1f}%" for v in np.linspace(5, 95, len(df))]
    res = run_batch(df.to_csv(index=False).encode(), "a.csv", cfg, ctx=ctx,
                    runs_dir=runs)
    scored = pd.read_csv(res.files["scored_applicants.csv"])
    assert (scored["n_unreadable_numbers"] == 0).all()
    assert res.clean_report.coerced_text["annual_inc"] == 12
    assert res.clean_report.coerced_text["revol_util"] == 12


def test_uploaded_values_never_change_the_learned_values(runs, cfg, ctx):
    before = ctx.clean_values.to_dict()
    extreme = pd.read_csv(__import__("io").BytesIO(_upload()))
    extreme["annual_inc"] = 50_000_000.0
    run_batch(extreme.to_csv(index=False).encode(), "a.csv", cfg, ctx=ctx, runs_dir=runs)
    assert ctx.clean_values.to_dict() == before


# ------------------------------------------- background runs and reading them back --

def test_explicit_run_dir_is_used_and_not_reused(runs, cfg, ctx):
    """How a background run and the page that launched it agree on a location."""
    target = runs / "20260101_000000_chosen"
    target.mkdir(parents=True)
    (target / "input_a.csv").write_bytes(_upload())
    res = run_batch(target / "input_a.csv", "a.csv", cfg, ctx=ctx, run_dir=target)
    assert res.run_dir == target
    assert res.files["scored_applicants.csv"].parent == target
    # The caller had already saved the upload there; it is not duplicated.
    assert sorted(p.name for p in target.glob("input_*")) == ["input_a.csv"]

    with pytest.raises(BatchError, match="already holds results"):
        run_batch(_upload(), "a.csv", cfg, ctx=ctx, run_dir=target)


def test_load_result_round_trips_a_finished_run(runs, cfg, ctx):
    original = run_batch(_upload(), "a.csv", cfg, ctx=ctx, runs_dir=runs)
    loaded = batch.load_result(original.run_dir)

    assert loaded.summary == original.summary
    assert loaded.seconds == original.summary["seconds_total"]
    pd.testing.assert_frame_equal(
        loaded.scored.reset_index(drop=True),
        pd.read_csv(original.files["scored_applicants.csv"]).reset_index(drop=True))
    assert len(loaded.approved) == len(original.approved)
    assert len(loaded.rejected) == len(original.rejected)
    assert loaded.clean_report.to_dict() == original.clean_report.to_dict()
    assert loaded.drift.status == original.drift.status
    assert loaded.drift.n_large == original.drift.n_large
    assert len(loaded.notices) == len(original.notices)
    # The profile tables the page shows are read from the run's own files.
    assert not loaded.profile["missing"].empty
    assert set(loaded.profile["numeric"]["column"]) == \
        set(original.profile["numeric"]["column"])


def test_load_result_refuses_an_unfinished_directory(runs):
    half = runs / "20260101_000000_half"
    half.mkdir(parents=True)
    with pytest.raises(BatchError, match="does not hold a finished run"):
        batch.load_result(half)


def test_background_script_reports_a_missing_model_through_the_runner(tmp_path):
    """The page's background path end to end, minus a trained model: the runner
    launches the CLI, the CLI exits 3, and the log carries the plain message."""
    import json
    import subprocess
    import sys
    import time

    from creditsurv.provenance import PROJECT_ROOT
    from creditsurv.runner import effective_state, launch, read_status, tail_log

    cfg_text = (PROJECT_ROOT / "config" / "config.yaml").read_text(encoding="utf-8")
    for d in ("data", "models", "figures", "tables"):
        (tmp_path / d).mkdir()
        cfg_text = cfg_text.replace(f"{d}_dir: outputs/{d}",
                                    f"{d}_dir: {(tmp_path / d).as_posix()}")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    upload = tmp_path / "input_a.csv"
    upload.write_bytes(_upload())

    run_dir = tmp_path / "run"
    log_dir = launch([{"key": "score", "name": "Score a.csv", "tag": "t",
                       "args": ["scripts/06_score_upload.py", "--file", str(upload),
                                "--run-dir", str(run_dir), "--config", str(cfg_path)]}],
                     lock_tag="t", runs_dir=tmp_path / "logs", detach=False)
    status = read_status(log_dir)
    assert status["state"] == "failed"
    assert status["stages"][0]["exit_code"] == 3        # refused, not crashed
    log = tail_log(log_dir)
    assert "No trained model found" in log
    assert "Run Pipeline" in log
    assert effective_state(log_dir) == "failed"


def test_background_script_rejects_a_missing_file(tmp_path):
    import subprocess
    import sys

    from creditsurv.provenance import PROJECT_ROOT
    out = subprocess.run([sys.executable, "scripts/06_score_upload.py",
                          "--file", str(tmp_path / "nope.csv")],
                         cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=180)
    assert out.returncode == 2
    assert "not found" in out.stderr


# ------------------------------------------- explanation cap and chunking --

def test_rejected_rows_are_explained_first_and_the_cap_is_marked(runs, cfg, ctx):
    """Approved rows never consume the cap, and a rejected row past it says so."""
    res = run_batch(_upload(), "a.csv", cfg, ctx=ctx, runs_dir=runs)
    scored = pd.read_csv(res.files["scored_applicants.csv"])
    approved = scored[scored["decision"] == "approve"]
    rejected = scored[scored["decision"] == "reject"]
    cap = cfg.decision.max_explained

    # No approved applicant was ever explained.
    assert (approved["explained"] == "not applicable (approved)").all()
    assert approved[["reason_1", "reason_2", "reason_3"]].fillna("").eq("").all().all()

    explained = rejected[rejected["explained"] == "explained"]
    capped = rejected[rejected["explained"] != "explained"]
    assert len(explained) == min(len(rejected), cap) == res.summary["n_explained"]
    assert len(capped) == res.summary["n_rejected_without_reasons"]
    assert len(capped) > 0                                   # the cap really bit
    assert capped["explained"].str.startswith(
        "reasons not generated: explanation cap reached").all()
    assert capped["explained"].str.contains("max_explained").all()
    # The explained ones are the FIRST rejected rows in the file, not a sample.
    assert list(explained["row_id"]) == list(rejected["row_id"])[:cap]
    assert res.aggregates.n_rejected_without_reasons == len(capped)


def test_cap_of_zero_leaves_every_rejected_row_marked(runs, cfg, ctx):
    res = run_batch(_upload(), "a.csv", cfg, ctx=ctx, runs_dir=runs, max_explained=0)
    rejected = pd.read_csv(res.files["rejected_applicants.csv"])
    assert len(rejected) == res.summary["n_rejected"] > 0
    assert res.summary["n_explained"] == 0
    assert res.summary["n_rejected_without_reasons"] == len(rejected)
    assert rejected["explained"].str.contains("explanation cap reached").all()
    assert "adverse_action_notices.zip" not in res.files


def test_chunked_and_single_pass_agree_on_every_decision(runs, cfg, ctx):
    """Streaming must not change what the model decides. Scores, decisions, cleaning
    flags and which rows got explained are identical whatever the block size.

    The stated *reasons* are not asserted identical: SurvSHAP(t) draws coalitions per
    call, so rows explained together in one block and apart in another can swap two
    near-tied reasons. That is the same per-applicant instability measured in
    FINDINGS, not an effect of streaming, and the block size used is recorded in the
    run summary and the provenance stamp so any output can be reproduced.
    """
    whole = run_batch(_upload(40), "a.csv", cfg, ctx=ctx, runs_dir=runs / "whole",
                      chunk_rows=1000)
    chunked = run_batch(_upload(40), "a.csv", cfg, ctx=ctx, runs_dir=runs / "chunked",
                        chunk_rows=4)
    assert whole.summary["blocks"] == 1 and chunked.summary["blocks"] == 10

    a = pd.read_csv(whole.files["scored_applicants.csv"])
    b = pd.read_csv(chunked.files["scored_applicants.csv"])
    deterministic = ["row_id", "applicant_id", "pd_12m", "pd_36m", "decision",
                     "threshold", "explained", "n_out_of_range",
                     "n_unreadable_numbers", "n_unknown_categories",
                     "features_present", "features_missing"]
    pd.testing.assert_frame_equal(a[deterministic], b[deterministic])

    for key in ("n_rows", "n_approved", "n_rejected", "approval_rate",
                "n_explained", "n_rejected_without_reasons", "n_notices",
                "rows_out_of_range", "drift_status", "features_present",
                "mean_pd_36m", "rows_dropped_by_cleaning"):
        assert whole.summary[key] == chunked.summary[key], key
    assert whole.clean_report.to_dict() == chunked.clean_report.to_dict()
    assert sorted(n for n, _ in whole.notices) == sorted(n for n, _ in chunked.notices)
    # Same counts of rows and decisions in the aggregates; reason tallies may differ
    # by which near-tied reason won.
    for key in ("n_rows", "n_approved", "n_rejected", "n_rejected_explained",
                "pd_hist", "by_group", "rows_out_of_range"):
        assert whole.aggregates.to_dict()[key] == chunked.aggregates.to_dict()[key], key


def test_aggregates_match_the_written_rows(runs, cfg, ctx):
    res = run_batch(_upload(40), "a.csv", cfg, ctx=ctx, runs_dir=runs, chunk_rows=7)
    scored = pd.read_csv(res.files["scored_applicants.csv"])
    agg = res.aggregates
    assert agg.n_rows == len(scored)
    assert agg.n_approved == int((scored["decision"] == "approve").sum())
    assert agg.n_rejected == int((scored["decision"] == "reject").sum())
    assert agg.mean_pd == pytest.approx(scored["pd_36m"].mean(), abs=1e-3)
    assert sum(agg.pd_hist) == len(scored)
    risk = agg.risk_frame(res.summary["threshold"])
    assert risk["applicants"].sum() == len(scored)
    group = agg.group_frame()
    assert group["applicants"].sum() == len(scored)
    # Reason counts add up to what the rejected file actually states.
    rejected = pd.read_csv(res.files["rejected_applicants.csv"])
    cited = sum(int(rejected[f"reason_{i}"].fillna("").ne("").sum())
                for i in (1, 2, 3, 4))
    assert agg.reason_frame()["times cited"].sum() == cited


def test_preview_is_bounded_but_the_file_is_whole(runs, cfg, ctx):
    from creditsurv.batch import PREVIEW_ROWS
    res = run_batch(_upload(40), "a.csv", cfg, ctx=ctx, runs_dir=runs, chunk_rows=5)
    assert len(res.scored) <= PREVIEW_ROWS
    assert len(pd.read_csv(res.files["scored_applicants.csv"])) == 40 == \
        res.summary["n_rows"]


def test_nothing_is_written_when_the_file_is_refused(runs, cfg, ctx):
    with pytest.raises(BatchError):
        run_batch(_upload(drop=["annual_inc"]), "a.csv", cfg, ctx=ctx, runs_dir=runs)
    assert not runs.exists() or list(runs.iterdir()) == []


# ------------------------------------------------ whole-file profile sample --

def test_reservoir_samples_uniformly_across_the_file():
    from creditsurv.batch import _Reservoir

    counts = np.zeros(10)
    for trial in range(60):
        r = _Reservoir(50, seed=trial)
        for b in range(50):                       # 50 blocks of 100 rows
            r.add_block(pd.DataFrame({"i": np.arange(b * 100, (b + 1) * 100)}))
        counts += np.histogram(r.frame()["i"].to_numpy(), bins=10,
                               range=(0, 5000))[0]
    share = counts / counts.sum()
    assert abs(share - 0.10).max() < 0.02        # flat across the file, not front-loaded


def test_reservoir_keeps_everything_when_the_file_is_small():
    from creditsurv.batch import _Reservoir

    r = _Reservoir(50, seed=1)
    r.add_block(pd.DataFrame({"i": np.arange(20)}))
    assert len(r.frame()) == 20 and r.seen == 20


def test_reservoir_is_bounded_and_repeatable():
    from creditsurv.batch import _Reservoir

    def run():
        r = _Reservoir(25, seed=7)
        for b in range(20):
            r.add_block(pd.DataFrame({"i": np.arange(b * 50, (b + 1) * 50)}))
            assert len(r.frame()) <= 25          # never grows with the file
        return r.frame()["i"].tolist()

    assert run() == run()                        # same seed, same sample


def test_drift_uses_the_whole_file_not_the_first_block(runs, cfg, ctx):
    """A file sorted so its opening rows are unrepresentative -- the shape a
    date-sorted loan file has. The first block alone reads as drifted on
    loan_amnt; the file as a whole does not, and the report must follow the file.
    """
    rng = np.random.default_rng(5)
    n, first = 2000, 100
    df = pd.read_csv(__import__("io").BytesIO(_upload(n)))
    # Opening block: tiny loans only. Remainder: the training distribution.
    df["loan_amnt"] = np.concatenate([rng.uniform(500, 1200, first),
                                      rng.uniform(1000, 35000, n - first)])

    res = run_batch(df.to_csv(index=False).encode(), "sorted.csv", cfg, ctx=ctx,
                    runs_dir=runs, chunk_rows=first, max_explained=0)
    assert res.summary["blocks"] == 20
    assert res.summary["profiled_rows"] == n
    assert res.summary["profile_sampling"].startswith("uniform random")

    def status_of(result, feature):
        row = result.table.loc[result.table["feature"] == feature]
        return row["status"].iloc[0]

    first_only = drift_mod.compare(ctx.reference, df.head(first), ctx.spec,
                                   min_rows=100)
    assert status_of(first_only, "loan_amnt") == "large"      # the opening rows
    assert status_of(res.drift, "loan_amnt") != "large"       # the actual file

    sample_mean = res.profile["numeric"].set_index("column").loc["loan_amnt", "mean"]
    assert abs(sample_mean - df["loan_amnt"].mean()) < abs(
        sample_mean - df["loan_amnt"].head(first).mean())


def test_profile_sample_is_capped_on_a_large_file(runs, cfg, ctx, monkeypatch):
    from creditsurv import batch as batch_mod

    monkeypatch.setattr(batch_mod, "PROFILE_ROWS", 100)
    res = run_batch(_upload(400), "a.csv", cfg, ctx=ctx, runs_dir=runs,
                    chunk_rows=60, max_explained=0)
    assert res.summary["n_rows"] == 400
    assert res.summary["profiled_rows"] == 100
    assert len(res.profile["missing"]) > 0


# --------------------------------------------------- which explainer is used --

def test_explainer_defaults_to_survshap_and_is_recorded(runs, cfg, ctx):
    """The default is unchanged until FINDINGS section 7 says otherwise, and every
    explained row records what produced its reasons."""
    assert cfg.decision.bulk_explainer == "survshap"
    res = run_batch(_upload(), "a.csv", cfg, ctx=ctx, runs_dir=runs)
    scored = pd.read_csv(res.files["scored_applicants.csv"])
    assert res.summary["explainer"] == "survshap"
    explained = scored[scored["explained"] == "explained"]
    assert (explained["explainer"] == "survshap").all()
    # Rows with no reasons claim no explainer.
    assert scored.loc[scored["explained"] != "explained", "explainer"].fillna("")\
        .eq("").all()


def test_auto_falls_back_to_survshap_without_a_booster(ctx):
    """The stub model has no booster, so 'auto' must not claim TreeSHAP."""
    assert batch.choose_explainer(ctx, "auto") == "survshap"
    assert batch.choose_explainer(ctx, None) == "survshap"


def test_treeshap_requested_without_a_tree_model_is_refused(ctx):
    with pytest.raises(BatchError, match="TreeSHAP needs the discrete-hazard model"):
        batch.choose_explainer(ctx, "treeshap")


def test_auto_picks_treeshap_when_the_booster_is_there(ctx, monkeypatch):
    class WithBooster:
        booster = object()

        def predict_survival(self, X, times):        # pragma: no cover - not called
            raise AssertionError

    monkeypatch.setattr(ctx, "model", WithBooster())
    assert batch.choose_explainer(ctx, "auto") == "treeshap"
    assert batch.choose_explainer(ctx, "treeshap") == "treeshap"
    assert batch.choose_explainer(ctx, "survshap") == "survshap"
