"""Tests for the out-of-time holdout changes (fixes 1-6).

The end-to-end tests run the real stage scripts on a small *synthetic* project in
a temporary directory -- never on the real data. The synthetic loans are censored
at a fixed extraction date, so recent vintages have short follow-up exactly as in
Lending Club, which is what makes the censoring and vintage logic testable.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from creditsurv.explain.compare import coalition_noise_floor
from creditsurv.features.build import validation_split
from creditsurv.io.loaders import read_rejected_sample
from creditsurv.pipeline import (
    encode_data_source,
    parse_year_range,
    resolve_data_source,
)
from creditsurv.provenance import PROJECT_ROOT

SCRIPTS = PROJECT_ROOT / "scripts"


# --------------------------------------------------------------------------
# unit tests
# --------------------------------------------------------------------------

class _Cfg:
    """Minimal stand-in for Config.paths used by resolve_data_source."""

    def __init__(self, root: Path):
        class P:
            labeled_parquet = root / "accepted_labeled.parquet"
            dev_sample_parquet = root / "dev_sample.parquet"
        self.paths = P


class TestResolveDataSource:
    def test_recorded_relative_path_resolves_under_project(self, tmp_path):
        got = resolve_data_source({"data_source": "outputs/data/x.parquet"},
                                  _Cfg(tmp_path), "holdout")
        assert got == PROJECT_ROOT / "outputs/data/x.parquet"

    def test_recorded_absolute_path_is_used_as_is(self, tmp_path):
        target = tmp_path / "elsewhere.parquet"
        got = resolve_data_source({"data_source": target.as_posix()},
                                  _Cfg(tmp_path), "anything")
        assert got == target

    def test_recorded_source_wins_over_tag_spelling(self, tmp_path):
        got = resolve_data_source({"data_source": "outputs/data/x.parquet"},
                                  _Cfg(tmp_path), "full")
        assert got.name == "x.parquet"

    @pytest.mark.parametrize("tag,expected", [("full", "accepted_labeled.parquet"),
                                              ("full_strat", "accepted_labeled.parquet"),
                                              ("dev", "dev_sample.parquet")])
    def test_legacy_bundles_fall_back_loudly(self, tmp_path, capsys, tag, expected):
        assert resolve_data_source({}, _Cfg(tmp_path), tag).name == expected
        assert "predates data-source recording" in capsys.readouterr().err

    def test_legacy_bundle_with_unknown_tag_refuses(self, tmp_path):
        """The exact silent failure: tag 'holdout' must not fall through to dev."""
        with pytest.raises(ValueError, match="does not record its data source"):
            resolve_data_source({}, _Cfg(tmp_path), "holdout")

    def test_encode_is_relative_inside_the_project(self):
        assert encode_data_source(PROJECT_ROOT / "outputs" / "data" / "a.parquet") \
            == "outputs/data/a.parquet"


class TestParseYearRange:
    def test_range(self):
        assert parse_year_range("2016-2018") == [2016, 2017, 2018]

    def test_single_year(self):
        assert parse_year_range("2017") == [2017]

    def test_none(self):
        assert parse_year_range(None) is None

    @pytest.mark.parametrize("bad", ["2018-2016", "2016-2017-2018", "abc"])
    def test_rejects_malformed(self, bad):
        with pytest.raises(ValueError):
            parse_year_range(bad)


class TestValidationSplit:
    def test_disjoint_and_complete(self):
        idx = pd.Index(np.arange(1000))
        fit, val = validation_split(idx, frac=0.1, seed=1)
        assert not set(fit) & set(val)
        assert len(fit) + len(val) == 1000 and len(val) == 100

    def test_draws_only_from_the_given_training_index(self):
        idx = pd.Index(np.arange(500, 900))
        fit, val = validation_split(idx, frac=0.2, seed=1)
        assert set(val) <= set(idx) and set(fit) <= set(idx)

    def test_reproducible(self):
        idx = pd.Index(np.arange(300))
        assert list(validation_split(idx, seed=4)[1]) == list(validation_split(idx, seed=4)[1])

    def test_rejects_bad_fraction(self):
        with pytest.raises(ValueError):
            validation_split(pd.Index(np.arange(10)), frac=1.5)


def _rejected_frame(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    years = rng.integers(2012, 2019, n)
    months = rng.integers(1, 13, n)
    return pd.DataFrame({
        "loan_amnt": rng.uniform(1000, 35000, n).astype("float32"),
        "application_d": [f"{y}-{m:02d}-15" for y, m in zip(years, months)],
        "title": "Debt consolidation",
        "risk_score": np.where(rng.random(n) < 0.5, np.nan,
                               rng.normal(640, 40, n)).astype("float32"),
        "dti_raw": [f"{v:.1f}%" for v in rng.gamma(3, 8, n)],
        "zip_code": "900xx", "addr_state": "CA",
        "emp_length": rng.choice(["< 1 year", "2 years", "10+ years"], n,
                                 p=[0.6, 0.25, 0.15]),
        "policy_code": 0.0,
    })


class TestReadRejectedSample:
    def test_year_filter_is_exact(self, tmp_path):
        df = _rejected_frame()
        path = tmp_path / "rej.parquet"
        df.to_parquet(path, index=False)
        expected = int(df["application_d"].str[:4].isin(["2016", "2017", "2018"]).sum())
        out, n_avail = read_rejected_sample(path, ["loan_amnt"], years=[2016, 2017, 2018])
        assert n_avail == expected == len(out)
        assert set(out["application_d"].str[:4]) == {"2016", "2017", "2018"}

    def test_sampling_is_bounded_and_reproducible(self, tmp_path):
        path = tmp_path / "rej.parquet"
        _rejected_frame().to_parquet(path, index=False)
        a, n = read_rejected_sample(path, ["loan_amnt"], years=[2016, 2017], n=300, seed=5)
        b, _ = read_rejected_sample(path, ["loan_amnt"], years=[2016, 2017], n=300, seed=5)
        assert len(a) == 300 and n > 300
        pd.testing.assert_frame_equal(a, b)

    def test_year_first_formats_still_filter_correctly(self, tmp_path):
        df = _rejected_frame(200)
        df["application_d"] = ["20170315", "2017/03/15", "2016-01-01", "2019-02-01"] * 50
        path = tmp_path / "rej.parquet"
        df.to_parquet(path, index=False)
        out, n = read_rejected_sample(path, ["loan_amnt"], years=[2017])
        assert n == 100 and set(out["application_d"].str[:4]) == {"2017"}

    def test_month_first_dates_are_excluded_not_detected(self, tmp_path):
        """Documented limitation: a non-year-first date falls outside the pushed-down
        range and is silently excluded. The real file was verified to be ISO (the
        2016-2018 filter returns exactly the 21,339,229 rows the ingest counted)."""
        df = _rejected_frame(100)
        df["application_d"] = "03/15/2017"
        path = tmp_path / "rej.parquet"
        df.to_parquet(path, index=False)
        _, n = read_rejected_sample(path, ["loan_amnt"], years=[2017])
        assert n == 0


class _LinearModel:
    def __init__(self, w):
        self.w = w

    def predict_survival(self, X, times):
        lin = sum(v * X[k].to_numpy(float) for k, v in self.w.items())
        return np.exp(-np.outer(np.exp(lin), np.atleast_1d(times)) / 400.0)


class TestCoalitionNoiseFloor:
    def test_exact_enumeration_has_no_noise(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
        m = _LinearModel({"a": 0.5, "b": -0.3, "c": 0.1})
        # 3 features -> 2^3 = 8 coalitions; nsamples well above that is exact.
        out = coalition_noise_floor(m, X.iloc[:6], X, np.array([6.0, 24.0]),
                                    nsamples=64, n_background=10)
        assert out["spearman_between_draws"] == pytest.approx(1.0)
        assert out["top5_jaccard"] == 1.0
        assert out["mean_rel_disagreement"] < 1e-9

    def test_reports_the_setting_actually_used(self):
        X = pd.DataFrame(np.random.default_rng(1).normal(size=(20, 3)), columns=list("abc"))
        out = coalition_noise_floor(_LinearModel({"a": 1.0}), X.iloc[:4], X,
                                    np.array([12.0]), nsamples=2, n_background=5)
        assert out["nsamples"] == 6          # raised to the 2p floor


# --------------------------------------------------------------------------
# end-to-end on a synthetic project
# --------------------------------------------------------------------------

def _accepted_frame(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Loans issued 2012-2018 and censored at an early-2019 extraction date."""
    rng = np.random.default_rng(seed)
    year = rng.integers(2012, 2019, n).astype("int16")
    fico = rng.normal(700, 35, n)
    dti = rng.gamma(4, 4, n)
    lin = 0.04 * (dti - dti.mean()) - 0.02 * (fico - fico.mean())
    t_event = rng.exponential(1 / np.exp(lin - 3.3))
    followup = (2019.2 - (year + rng.uniform(0, 1, n))) * 12   # months observable
    dur = np.minimum(t_event, np.minimum(followup, 60))
    return pd.DataFrame({
        "issue_year": year,
        "duration_months": np.clip(np.ceil(dur), 1, 60).astype("int32"),
        "event": (t_event <= np.minimum(followup, 60)).astype("int8"),
        "term_months": rng.choice([36, 60], n).astype("int16"),
        "grade": rng.choice(list("ABCDEFG"), n),
        "loan_amnt": rng.uniform(1000, 35000, n).astype("float32"),
        "installment": rng.uniform(40, 1200, n).astype("float32"),
        "annual_inc": rng.lognormal(11, 0.5, n).astype("float32"),
        "dti": dti.astype("float32"),
        "fico_range_low": fico.astype("float32"),
        "fico_range_high": (fico + 4).astype("float32"),
        "revol_util": rng.uniform(0, 100, n).astype("float32"),
        "emp_length": rng.choice(["< 1 year", "2 years", "5 years", "10+ years"], n),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "other"], n),
        "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], n),
    })


def _run(script: str, cfg: Path, *args: str):
    return subprocess.run([sys.executable, str(SCRIPTS / script), "--config", str(cfg),
                           *args], capture_output=True, text=True, cwd=PROJECT_ROOT,
                          timeout=600)


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_project")
    for d in ("data", "tables", "figures", "models"):
        (root / d).mkdir()
    acc = _accepted_frame()
    acc.to_parquet(root / "data" / "dev_sample.parquet", index=False)
    # A decoy with different rows under the other name: if any stage picks its
    # data by the tag's spelling instead of the bundle, it reads this and fails.
    acc.sample(frac=0.5, random_state=9).assign(dti=-1.0).to_parquet(
        root / "data" / "accepted_labeled.parquet", index=False)
    _rejected_frame().to_parquet(root / "data" / "rejected_raw.parquet", index=False)
    cfg = root / "cfg.yaml"
    cfg.write_text(
        "paths:\n"
        + "".join(f"  {k}_dir: {(root / k).as_posix()}\n"
                  for k in ("data", "tables", "figures", "models"))
        + "model:\n  time_bin_months: 3\n  eval_horizons_months: [6, 12, 24]\n"
        + "explain:\n  n_background: 10\n  kernel_nsamples: 40\n"
        + "diagnostic:\n  rejected_sample_size: 2500\n",
        encoding="utf-8")
    r = _run("02_train_models.py", cfg, "--split", "out_of_time", "--oot-cutoff", "2016",
             "--tag", "oot_fixture", "--cox-max-rows", "5000")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return root, cfg, acc, r


# These fixtures write under 'oot_fixture', not 'holdout': that tag is reserved for
# the pre-registered full-data run and 02_train_models.py refuses anything else under
# it (tests/test_reserved_tags.py).
class TestStage2OutOfTime:
    def test_cutoff_is_respected(self, project):
        root, *_ = project
        m = json.loads((root / "tables" / "02_metrics_oot_fixture.json").read_text())
        assert max(m["train_years"]) == 2015 and min(m["test_years"]) == 2016
        assert m["oot_cutoff"] == 2016 and m["split_scheme"] == "out_of_time"

    def test_bundle_split_has_no_year_overlap(self, project):
        root, _, acc, _ = project
        b = pickle.load(open(root / "models" / "02_models_oot_fixture.pkl", "rb"))
        assert acc.loc[b["train_idx"], "issue_year"].max() <= 2015
        assert acc.loc[b["test_idx"], "issue_year"].min() >= 2016

    def test_early_stopping_uses_training_loans_only(self, project):
        root, *_ = project
        m = json.loads((root / "tables" / "02_metrics_oot_fixture.json").read_text())
        es = m["early_stopping"]
        assert es["n_fit"] + es["n_val"] == m["n_train"]
        assert es["n_val"] == round(0.1 * m["n_train"])
        assert "training loans" in es["validation"]

    def test_ipcw_fitted_on_the_evaluation_set(self, project):
        root, *_ = project
        m = json.loads((root / "tables" / "02_metrics_oot_fixture.json").read_text())
        assert m["ipcw_fitted_on"] == "test"

    def test_per_vintage_results_for_both_models(self, project):
        root, *_ = project
        m = json.loads((root / "tables" / "02_metrics_oot_fixture.json").read_text())
        for model in ("cox", "discrete_hazard"):
            assert set(m["results_by_vintage"][model]) == {"2016", "2017", "2018"}

    def test_vintage_horizons_stop_where_follow_up_stops(self, project):
        """2018 loans are observed ~14 months at most: no 24-month AUC for them."""
        root, *_ = project
        m = json.loads((root / "tables" / "02_metrics_oot_fixture.json").read_text())
        v2018 = m["results_by_vintage"]["discrete_hazard"]["2018"]
        assert "auc_24m" not in v2018 and "auc_6m" in v2018

    def test_data_source_recorded_in_bundle_and_metrics(self, project):
        root, *_ = project
        b = pickle.load(open(root / "models" / "02_models_oot_fixture.pkl", "rb"))
        m = json.loads((root / "tables" / "02_metrics_oot_fixture.json").read_text())
        assert b["data_source"].endswith("dev_sample.parquet")
        assert m["data_source"] == b["data_source"]
        assert b["split"]["oot_cutoff"] == 2016

    def test_high_variance_horizons_are_reported(self, project):
        root, *_ = project
        m = json.loads((root / "tables" / "02_metrics_oot_fixture.json").read_text())
        assert "high_variance_horizons" in m["results"][0]
        assert "n_controls_by_horizon" in m["results"][0]

    def test_out_of_time_without_cutoff_is_refused(self, project, tmp_path):
        root, cfg, *_ = project
        r = _run("02_train_models.py", cfg, "--split", "out_of_time", "--tag", "nocut")
        assert r.returncode == 2 and "--oot-cutoff" in r.stderr
        assert not list((root / "tables").glob("02_*_nocut.*"))

    def test_cutoff_without_out_of_time_is_refused(self, project):
        _, cfg, *_ = project
        r = _run("02_train_models.py", cfg, "--oot-cutoff", "2016", "--tag", "x")
        assert r.returncode == 2


class TestStage3ReadsTheBundlesData:
    def test_explain_uses_recorded_source_not_the_tag(self, project):
        root, cfg, *_ = project
        r = _run("03_explain.py", cfg, "--tag", "oot_fixture", "--n-explain", "8",
                 "--skip-naive")
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        out = json.loads((root / "tables" / "03_explain_oot_fixture.json").read_text())
        assert out["provenance"]["inputs"]["data"]["path"].endswith("dev_sample.parquet")

    def test_noise_floor_script_runs_and_does_not_block_stage3(self, project):
        root, cfg, *_ = project
        r = _run("03c_noise_floor.py", cfg, "--tag", "nf", "--model-tag", "oot_fixture",
                 "--n-explain", "6")
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        nf = json.loads((root / "tables" / "03c_noise_floor_nf.json").read_text())
        assert 0.0 <= nf["top5_jaccard"] <= 1.0 and "spearman_between_draws" in nf
        from creditsurv.provenance import find_existing_outputs
        assert not find_existing_outputs([root / "tables"], "03", "nf")

    def test_legacy_bundle_with_new_tag_fails_loudly(self, project):
        root, cfg, *_ = project
        b = pickle.load(open(root / "models" / "02_models_oot_fixture.pkl", "rb"))
        b.pop("data_source")
        pickle.dump(b, open(root / "models" / "02_models_legacy.pkl", "wb"))
        r = _run("03_explain.py", cfg, "--tag", "legacy", "--n-explain", "4",
                 "--skip-naive")
        assert r.returncode == 2 and "does not record its data source" in r.stderr


class TestStage4Holdout:
    def test_diagnostic_only_with_year_filters(self, project):
        root, cfg, acc, _ = project
        r = _run("04_reject_inference.py", cfg, "--tag", "oot_fixture", "--diagnostic-only",
                 "--accepted-years", "2016-2018", "--rejected-years", "2016-2018")
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        out = json.loads((root / "tables" / "04_reject_inference_oot_fixture.json").read_text())
        assert out["diagnostic_only"] is True and out["correction_applied"] is False
        assert out["explanation_shift"]["status"] == "not_run"
        comp = out["composition"]
        assert set(comp["accepted_by_year"]) == {"2016", "2017", "2018"}
        assert set(comp["rejected_by_year"]) <= {"2016", "2017", "2018"}
        assert out["n_accepted"] == int(acc["issue_year"].isin([2016, 2017, 2018]).sum())
        assert out["provenance"]["inputs"]["accepted"]["path"].endswith("dev_sample.parquet")

    def test_year_filter_without_diagnostic_only_is_refused(self, project):
        _, cfg, *_ = project
        r = _run("04_reject_inference.py", cfg, "--tag", "yf", "--model-tag", "oot_fixture",
                 "--accepted-years", "2016-2018")
        assert r.returncode == 2 and "--diagnostic-only" in r.stderr


def _load_report_module():
    spec = importlib.util.spec_from_file_location("report", SCRIPTS / "05_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestSection6:
    def _boot(self, lo, hi, inq_lo=0.9, inq_hi=1.4):
        def row(f, lo, hi):
            return {"feature": f, "share_ratio": (lo + hi) / 2, "share_ratio_lo": lo,
                    "share_ratio_hi": hi, "rank_shift": 1, "rank_shift_lo": 0,
                    "rank_shift_hi": 2, "shift_ci_excludes_zero": False,
                    "share_ci_excludes_one": False, "ci": 0.95}
        return {"n_boot": 5000, "reference": list("ABCDE"),
                "results": {"G": [row("annual_inc", lo, hi),
                                  row("mths_since_recent_inq", inq_lo, inq_hi)]}}

    @pytest.mark.parametrize("lo,hi,expected", [(0.4, 0.8, "CONFIRMED"),
                                                (0.6, 1.1, "NOT REPLICATED")])
    def test_preregistered_income_rule_is_applied_mechanically(self, tmp_path, lo, hi,
                                                               expected):
        mod = _load_report_module()
        (tmp_path / "03_segment_bootstrap_holdout_strat.json").write_text(
            json.dumps(self._boot(lo, hi)))
        text = mod.section_6(tmp_path)
        line = next(l for l in text.splitlines() if l.startswith("| `annual_inc`"))
        assert f"**{expected}**" in line

    def test_inquiry_retest_uses_its_own_rule(self, tmp_path):
        mod = _load_report_module()
        (tmp_path / "03_segment_bootstrap_holdout_strat.json").write_text(
            json.dumps(self._boot(0.4, 0.8, inq_lo=1.05, inq_hi=1.6)))
        line = next(l for l in mod.section_6(tmp_path).splitlines()
                    if l.startswith("| `mths_since_recent_inq`"))
        assert "**REPLICATED**" in line

    def test_comparison_without_holdout_noise_floor_is_not_judged(self, tmp_path):
        mod = _load_report_module()
        (tmp_path / "03_explain_holdout.json").write_text(json.dumps(
            {"naive_vs_survshap": {"spearman": 0.99, "top5_overlap": 1.0,
                                   "top5_sign_disagreements": []}}))
        text = mod.section_6(tmp_path)
        assert "cannot be evaluated" in text and "HOLDS" not in text

    def test_nothing_run_yet(self, tmp_path):
        assert "Not run yet" in _load_report_module().section_6(tmp_path)


class TestReportEndToEnd:
    def test_section6_added_without_touching_2_to_5_and_keep_block_survives(
            self, project, tmp_path):
        root, cfg, *_ = project
        findings = tmp_path / "F.md"
        findings.write_text(
            "# F\n\n## 1. Intro\n\nhand text\n\n## 2. Survival models\n\nx\n\n"
            "## 3. Explainability\n\nx\n\n## 4. Reject inference (diagnostic-gated)\n\n"
            "x\n\n## 5. Summary\n\nx\n", encoding="utf-8")
        first = _run("05_report.py", cfg, "--tag", "full", "--findings", str(findings))
        assert first.returncode == 0, first.stderr
        text = findings.read_text(encoding="utf-8")
        assert "## 6. Out-of-time holdout" in text
        before_2_5 = text[text.index("## 2."):text.index("## 6.")]

        # Pre-register inside section 6, then regenerate: the block must survive
        # and sections 2-5 must be unchanged.
        text = text.replace("## 6. Out-of-time holdout\n",
                            "## 6. Out-of-time holdout\n\n<!-- keep:prereg -->\n"
                            "RULE: fixed in advance\n<!-- /keep:prereg -->\n")
        findings.write_text(text, encoding="utf-8")
        second = _run("05_report.py", cfg, "--tag", "full", "--findings", str(findings),
                      "--holdout-tag", "oot_fixture",
                      "--holdout-strat-tag", "oot_fixture_strat", "--force")
        assert second.returncode == 0, second.stderr
        after = findings.read_text(encoding="utf-8")
        assert "RULE: fixed in advance" in after
        assert "hand text" in after
        assert after[after.index("## 2."):after.index("## 6.")] == before_2_5
        assert "6.2 Survival models on the holdout" in after

    def test_rendering_an_out_of_time_run_into_2_to_5_is_refused(self, project, tmp_path):
        _, cfg, *_ = project
        findings = tmp_path / "F.md"
        findings.write_text("## 2. Survival models\n\n## 3. Explainability\n\n"
                            "## 4. Reject inference (diagnostic-gated)\n\n## 5. Summary\n",
                            encoding="utf-8")
        before = findings.read_text(encoding="utf-8")
        r = _run("05_report.py", cfg, "--tag", "oot_fixture", "--findings", str(findings))
        assert r.returncode == 2 and "out-of-time run" in r.stderr
        assert findings.read_text(encoding="utf-8") == before


def test_section_6_is_bounded_by_the_next_heading(tmp_path):
    """A section written after the generated ones -- a hand-written analysis, or the
    explainer pre-registration -- must survive a report run. Section 6 used to end
    at end-of-file, which would have deleted it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "report_mod", Path(__file__).resolve().parents[1] / "scripts" / "05_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    text = ("## 5. Summary\n\nfive\n\n"
            "## 6. Out-of-time holdout\n\nold six\n\n"
            "## 7. Something written by hand\n\nkeep me\n")
    out = mod.replace_section(text, "## 6. Out-of-time holdout", mod.NEXT_HEADING,
                              "## 6. Out-of-time holdout\n\nnew six")
    assert "new six" in out and "old six" not in out
    assert "## 7. Something written by hand" in out and "keep me" in out
    assert "five" in out

    # With no later section there is nothing to protect, and it still works.
    plain = "## 6. Out-of-time holdout\n\nold six\n"
    out2 = mod.replace_section(plain, "## 6. Out-of-time holdout", mod.NEXT_HEADING,
                               "## 6. Out-of-time holdout\n\nnew six")
    assert "new six" in out2 and "old six" not in out2
