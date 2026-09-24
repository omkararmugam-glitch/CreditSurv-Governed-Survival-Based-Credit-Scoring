"""Where cleaning values come from: the model, once, or the run refuses.

A value learned from the file being scored would make every upload its own
yardstick; a value learned from a 20,000-row sample is not the one the model was
trained against. So scoring reads saved values and never fits.
"""

from __future__ import annotations

import json
import pathlib
import pickle

import numpy as np
import pandas as pd
import pytest

from creditsurv.batch import BatchError, load_context
from creditsurv.cleaning import fit_values
from creditsurv.config import Config, DecisionConfig, Paths
from creditsurv.features.build import FeatureSpec

SPEC = FeatureSpec(numeric=("loan_amnt", "annual_inc", "dti"),
                   categorical=("purpose",), structural_missing=())


class _Stub:
    booster = None

    def predict_survival(self, X, times):        # pragma: no cover - not scored here
        return np.ones((len(X), len(np.atleast_1d(times))))


@pytest.fixture
def workspace(tmp_path):
    for sub in ("models", "data", "tables", "figures"):
        (tmp_path / sub).mkdir()
    rng = np.random.default_rng(0)
    n = 400
    train = pd.DataFrame({
        "loan_amnt": rng.uniform(1000, 35000, n),
        "annual_inc": rng.lognormal(11, 0.4, n),
        "dti": rng.uniform(1, 38, n),
        "purpose": rng.choice(["car", "credit_card"], n),
        "duration_months": rng.integers(1, 36, n),
        "event": rng.integers(0, 2, n)})
    data = tmp_path / "data" / "train.parquet"
    train.to_parquet(data)
    cfg = Config(paths=Paths(data_dir=tmp_path / "data",
                             models_dir=tmp_path / "models",
                             figures_dir=tmp_path / "figures",
                             tables_dir=tmp_path / "tables"),
                 decision=DecisionConfig(model_tag="t", background_rows=100))
    return cfg, train, data


def _write_bundle(cfg, train, data, *, values=None):
    bundle = {
        "artefacts": {"discrete_hazard": _Stub(), "gbm_columns": list(SPEC.all_columns)},
        "spec": SPEC, "train_idx": train.index[:300], "test_idx": train.index[300:],
        "data_source": str(data), "split": {"scheme": "random"}}
    if values is not None:
        bundle["cleaning_values"] = values.to_dict()
    path = cfg.paths.models_dir / "02_models_t.pkl"
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh)
    return path


def test_a_bundle_without_values_refuses_and_names_the_command(workspace):
    cfg, train, data = workspace
    _write_bundle(cfg, train, data)
    with pytest.raises(BatchError) as exc:
        load_context(cfg, "t")
    assert "no saved cleaning values" in exc.value.message
    assert "02s_save_cleaning_values.py --model-tag t" in exc.value.fix


def test_values_in_the_bundle_are_used_as_they_are(workspace):
    cfg, train, data = workspace
    values = fit_values(train.loc[train.index[:300]], SPEC, source=str(data))
    _write_bundle(cfg, train, data, values=values)
    ctx = load_context(cfg, "t")
    assert ctx.values_from_bundle is True
    assert ctx.clean_values.to_dict() == values.to_dict()
    assert ctx.clean_values.fitted_rows == 300          # the whole training split


def test_a_sidecar_is_accepted_when_the_bundle_predates_the_field(workspace):
    cfg, train, data = workspace
    _write_bundle(cfg, train, data)
    values = fit_values(train.loc[train.index[:300]], SPEC, source=str(data))
    (cfg.paths.models_dir / "02_cleaning_values_t.json").write_text(
        json.dumps({"model_tag": "t", "values": values.to_dict()}), encoding="utf-8")
    ctx = load_context(cfg, "t")
    assert ctx.clean_values.to_dict() == values.to_dict()
    assert ctx.values_from_bundle is False              # reported, not hidden


def test_an_unreadable_sidecar_refuses_rather_than_falling_back(workspace):
    cfg, train, data = workspace
    _write_bundle(cfg, train, data)
    (cfg.paths.models_dir / "02_cleaning_values_t.json").write_text(
        "{ truncated", encoding="utf-8")
    with pytest.raises(BatchError) as exc:
        load_context(cfg, "t")
    assert "could not be read" in exc.value.message
    assert "--overwrite" in exc.value.fix


def test_scoring_never_fits_values(workspace, monkeypatch):
    """The guarantee, enforced: if load_context tried to fit, this test fails."""
    cfg, train, data = workspace
    values = fit_values(train.loc[train.index[:300]], SPEC, source=str(data))
    _write_bundle(cfg, train, data, values=values)

    import creditsurv.batch as batch_mod

    def boom(*args, **kwargs):
        raise AssertionError("scoring must not fit cleaning values")

    monkeypatch.setattr(batch_mod, "fit_values", boom)
    ctx = load_context(cfg, "t")
    assert ctx.clean_values.fitted_rows == 300

# ------------------------------- values for a model with derived features --------

def test_derived_features_are_computed_before_the_values_are_fitted(tmp_path,
                                                                    monkeypatch):
    """A model trained with --with-derived has features the parquet never stored.
    Reading them by name fails in pyarrow ("No match for fico_midpoint"), so the
    script must derive them from the raw columns first."""
    import importlib.util

    for sub in ("models", "data", "tables", "figures"):
        (tmp_path / sub).mkdir()
    rng = np.random.default_rng(1)
    n = 300
    raw = pd.DataFrame({
        "loan_amnt": rng.uniform(1000, 35000, n),
        "installment": rng.uniform(50, 1200, n),
        "annual_inc": rng.lognormal(11, 0.4, n),
        "dti": rng.uniform(1, 38, n),
        "fico_range_low": rng.choice([660.0, 700.0, 740.0], n),
        "fico_range_high": np.zeros(n),
        "emp_length": rng.choice(["< 1 year", "3 years", "10+ years"], n),
        "purpose": rng.choice(["car", "credit_card"], n),
        "duration_months": rng.integers(1, 36, n),
        "event": rng.integers(0, 2, n)})
    raw["fico_range_high"] = raw["fico_range_low"] + 4.0
    data = tmp_path / "data" / "train.parquet"
    raw.to_parquet(data)
    assert "fico_midpoint" not in pd.read_parquet(data).columns

    derived_spec = FeatureSpec(
        numeric=("loan_amnt", "annual_inc", "dti", "fico_midpoint",
                 "emp_length_years", "loan_to_income", "log_annual_inc",
                 "installment_to_income"),
        categorical=("purpose",), structural_missing=())
    bundle = {"artefacts": {"discrete_hazard": _Stub()}, "spec": derived_spec,
              "train_idx": raw.index[:250], "test_idx": raw.index[250:],
              "data_source": str(data), "split": {"scheme": "random"}}
    with open(tmp_path / "models" / "02_models_d.pkl", "wb") as fh:
        pickle.dump(bundle, fh)

    spec_file = importlib.util.spec_from_file_location(
        "save_values", pathlib.Path("scripts/02s_save_cleaning_values.py"))
    module = importlib.util.module_from_spec(spec_file)
    spec_file.loader.exec_module(module)

    config = tmp_path / "config.yaml"
    config.write_text(f"""paths:
  data_dir: {(tmp_path / 'data').as_posix()}
  models_dir: {(tmp_path / 'models').as_posix()}
  tables_dir: {(tmp_path / 'tables').as_posix()}
  figures_dir: {(tmp_path / 'figures').as_posix()}
""", encoding="utf-8")

    code = module.main(["--config", str(config), "--model-tag", "d"])
    assert code == 0, "the run should succeed, not fail on a derived feature"

    payload = json.loads(
        (tmp_path / "models" / "02_cleaning_values_d.json").read_text(encoding="utf-8"))
    values = payload["values"]
    for feature in ("fico_midpoint", "emp_length_years", "loan_to_income",
                    "log_annual_inc", "installment_to_income"):
        assert feature in values["ranges"], f"{feature} has no fitted range"
    assert payload["fitted_rows"] == 250              # the whole training split
    low, high = values["ranges"]["fico_midpoint"]
    assert 600 <= low <= high <= 850, (low, high)     # a real range, not a placeholder


def test_a_feature_that_is_neither_stored_nor_derivable_is_refused(tmp_path, capsys):
    """The honest failure: name the features and write nothing."""
    import importlib.util

    for sub in ("models", "data", "tables", "figures"):
        (tmp_path / sub).mkdir()
    raw = pd.DataFrame({"loan_amnt": [1000.0, 2000.0], "annual_inc": [50000.0, 60000.0],
                        "duration_months": [3, 4], "event": [0, 1]})
    data = tmp_path / "data" / "train.parquet"
    raw.to_parquet(data)
    bundle = {"artefacts": {}, "spec": FeatureSpec(
        numeric=("loan_amnt", "annual_inc", "invented_feature"), categorical=(),
        structural_missing=()), "train_idx": raw.index, "test_idx": raw.index[:0],
        "data_source": str(data), "split": {"scheme": "random"}}
    with open(tmp_path / "models" / "02_models_x.pkl", "wb") as fh:
        pickle.dump(bundle, fh)

    spec_file = importlib.util.spec_from_file_location(
        "save_values2", pathlib.Path("scripts/02s_save_cleaning_values.py"))
    module = importlib.util.module_from_spec(spec_file)
    spec_file.loader.exec_module(module)
    config = tmp_path / "config.yaml"
    config.write_text(f"""paths:
  data_dir: {(tmp_path / 'data').as_posix()}
  models_dir: {(tmp_path / 'models').as_posix()}
  tables_dir: {(tmp_path / 'tables').as_posix()}
  figures_dir: {(tmp_path / 'figures').as_posix()}
""", encoding="utf-8")

    assert module.main(["--config", str(config), "--model-tag", "x"]) == 2
    assert "invented_feature" in capsys.readouterr().err
    assert not (tmp_path / "models" / "02_cleaning_values_x.json").exists()
