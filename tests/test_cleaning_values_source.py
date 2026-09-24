"""Where cleaning values come from: the model, once, or the run refuses.

A value learned from the file being scored would make every upload its own
yardstick; a value learned from a 20,000-row sample is not the one the model was
trained against. So scoring reads saved values and never fits.
"""

from __future__ import annotations

import json
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
