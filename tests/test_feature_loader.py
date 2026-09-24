"""One loader for a model's features, used by every stage that reads them.

A model trained with ``--with-derived`` has features the parquet never stored:
``fico_midpoint``, ``emp_length_years`` and the income ratios are computed at training
time and not written back. Two stages broke on that separately -- the cleaning-values
script and the ablation -- each asking pyarrow for a column that does not exist. The
fix is one function, so there is one place to get this right.
"""

from __future__ import annotations

import importlib.util
import pathlib
import pickle

import numpy as np
import pandas as pd
import pytest

from creditsurv import pipeline
from creditsurv.features.build import FeatureSpec
from creditsurv.pipeline import DERIVATION_INPUTS, load_feature_frame

class _Stub:
    """Stands in for a fitted model: the loader is what is under test, not scoring.
    Defined at module level so a bundle holding it can be pickled."""

    booster = None

    def predict_survival(self, X, times):          # pragma: no cover - not scored
        return np.ones((len(X), len(np.atleast_1d(times))))


RAW_SPEC = FeatureSpec(numeric=("loan_amnt", "annual_inc", "dti"),
                       categorical=("purpose",), structural_missing=())
DERIVED_SPEC = FeatureSpec(
    numeric=("loan_amnt", "annual_inc", "dti", "fico_midpoint", "emp_length_years",
             "loan_to_income", "log_annual_inc", "installment_to_income",
             "term_months"),
    categorical=("purpose",), structural_missing=())


@pytest.fixture
def raw_parquet(tmp_path) -> pathlib.Path:
    """A file holding only raw columns -- no derived feature is stored."""
    rng = np.random.default_rng(0)
    n = 200
    frame = pd.DataFrame({
        "loan_amnt": rng.uniform(1000, 35000, n),
        "installment": rng.uniform(50, 1200, n),
        "annual_inc": rng.lognormal(11, 0.4, n),
        "dti": rng.uniform(1, 38, n),
        "fico_range_low": rng.choice([660.0, 700.0, 740.0], n),
        "fico_range_high": rng.choice([664.0, 704.0, 744.0], n),
        "emp_length": rng.choice(["< 1 year", "3 years", "10+ years"], n),
        "term": rng.choice(["36 months", "60 months"], n),
        "purpose": rng.choice(["car", "credit_card"], n),
        "grade": rng.choice(list("ABC"), n),
        "duration_months": rng.integers(1, 36, n),
        "event": rng.integers(0, 2, n)})
    path = tmp_path / "train.parquet"
    frame.to_parquet(path)
    stored = set(pd.read_parquet(path).columns)
    assert not ({"fico_midpoint", "emp_length_years", "term_months"} & stored)
    return path


def test_derived_features_are_computed_not_demanded(raw_parquet):
    frame = load_feature_frame(raw_parquet, DERIVED_SPEC)
    for feature in DERIVED_SPEC.all_columns:
        assert feature in frame.columns, feature
    assert frame["fico_midpoint"].notna().all()
    assert set(frame["term_months"].dropna().unique()) <= {36.0, 60.0}


def test_the_survival_outcome_comes_too(raw_parquet):
    frame = load_feature_frame(raw_parquet, RAW_SPEC)
    assert {"duration_months", "event"} <= set(frame.columns)


def test_extra_columns_can_be_asked_for(raw_parquet):
    frame = load_feature_frame(raw_parquet, RAW_SPEC, columns=("grade",))
    assert "grade" in frame.columns


def test_a_feature_neither_stored_nor_derivable_raises_with_its_name(raw_parquet):
    spec = FeatureSpec(numeric=("loan_amnt", "invented_feature"), categorical=(),
                       structural_missing=())
    with pytest.raises(ValueError) as exc:
        load_feature_frame(raw_parquet, spec)
    assert "invented_feature" in str(exc.value)
    assert "train.parquet" in str(exc.value)


def test_a_missing_extra_is_reported_rather_than_fatal(raw_parquet, capsys):
    """A stage that does not need the outcome should still be able to load."""
    frame = load_feature_frame(raw_parquet, RAW_SPEC, extras=(),
                               columns=("issue_year",), verbose=True)
    assert "issue_year" not in frame.columns
    assert "issue_year" in capsys.readouterr().out


def test_what_was_computed_is_said_out_loud(raw_parquet, capsys):
    load_feature_frame(raw_parquet, DERIVED_SPEC, verbose=True)
    out = capsys.readouterr().out
    assert "computed from raw columns rather than read" in out
    assert "fico_midpoint" in out


def test_every_derivation_input_is_read():
    """A derivation whose inputs are not read cannot run, so the two lists must agree:
    DERIVATION_INPUTS has to cover what both derivation paths consume."""
    from creditsurv.derive import DERIVATIONS
    from creditsurv.features.build import DERIVED_NUMERIC

    needed = {c for rule in DERIVATIONS for c in rule.needs}
    computed = {rule.feature for rule in DERIVATIONS} | set(DERIVED_NUMERIC)
    raw_inputs = needed - computed          # inputs that must come from the file
    assert raw_inputs <= DERIVATION_INPUTS, raw_inputs - DERIVATION_INPUTS
    assert {"fico_range_low", "emp_length", "installment", "annual_inc",
            "loan_amnt"} <= DERIVATION_INPUTS


# --------------------------------- every stage goes through the one loader --------

ENTRY_POINTS = {
    "02r_recompute_metrics": "scripts/02r_recompute_metrics.py",
    "02s_save_cleaning_values": "scripts/02s_save_cleaning_values.py",
    "03d_explainer_validation": "scripts/03d_explainer_validation.py",
    "03e_feature_ablation": "scripts/03e_feature_ablation.py",
    "03f_settings_comparison": "scripts/03f_settings_comparison.py",
}


def _module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(f"entry_{name}",
                                                  pathlib.Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name,path", sorted(ENTRY_POINTS.items()))
def test_each_stage_can_load_a_model_that_uses_derived_features(name, path,
                                                                raw_parquet):
    """The load path is shared, so proving it here proves it for each stage: the
    function each module holds is this one, and it handles a derived spec."""
    module = _module(name, path)
    assert module.load_feature_frame is pipeline.load_feature_frame, name
    frame = module.load_feature_frame(raw_parquet, DERIVED_SPEC)
    assert "fico_midpoint" in frame.columns


@pytest.mark.parametrize("name,path", sorted(ENTRY_POINTS.items()))
def test_no_stage_reads_feature_columns_by_name_any_more(name, path):
    """The bug was `read_parquet(src, columns=list(spec.all_columns))`. It must not
    come back, in these files or a new one copied from them."""
    source = pathlib.Path(path).read_text(encoding="utf-8")
    assert "all_columns" not in source, f"{name} still selects columns by name"
    assert "load_feature_frame" in source, name


def test_the_application_loads_a_derived_model_too(tmp_path, raw_parquet):
    """06_score_upload.py and the upload page both go through load_context."""
    from creditsurv.batch import load_context
    from creditsurv.cleaning import fit_values
    from creditsurv.config import Config, DecisionConfig, Paths

    for sub in ("models", "tables", "figures"):
        (tmp_path / sub).mkdir()
    frame = load_feature_frame(raw_parquet, DERIVED_SPEC)
    values = fit_values(frame, DERIVED_SPEC, source=str(raw_parquet))
    bundle = {"artefacts": {"discrete_hazard": _Stub(),
                            "gbm_columns": list(DERIVED_SPEC.all_columns)},
              "spec": DERIVED_SPEC, "train_idx": frame.index[:150],
              "test_idx": frame.index[150:], "data_source": str(raw_parquet),
              "split": {"scheme": "random"}, "cleaning_values": values.to_dict()}
    with open(tmp_path / "models" / "02_models_d.pkl", "wb") as fh:
        pickle.dump(bundle, fh)

    cfg = Config(paths=Paths(data_dir=raw_parquet.parent,
                             models_dir=tmp_path / "models",
                             figures_dir=tmp_path / "figures",
                             tables_dir=tmp_path / "tables"),
                 decision=DecisionConfig(model_tag="d", background_rows=50))
    ctx = load_context(cfg, "d")
    for feature in ("fico_midpoint", "emp_length_years", "term_months"):
        assert feature in ctx.reference.columns, feature
