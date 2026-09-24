"""A Cox fit that cannot be identified must not be published.

The failure this file is about: `holdout_applicant` was trained, evaluated and saved
with a test concordance of 0.502 -- random ranking -- because its partial likelihood
ran away along a direction three Idaho loans could not constrain. Nothing stopped it,
and its metrics file looked like a result. Two things now prevent a repeat: rare
one-hot levels are pooled instead of estimated, and a model that ranks no better than
chance fails the run instead of being saved (FINDINGS 7h).
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

from creditsurv.features.build import (MIN_EVENTS_FOR_POOLING, MIN_EVENTS_PER_LEVEL,
                                       FeatureSpec, build_design_matrix)

SPEC = FeatureSpec(numeric=("loan_amnt", "annual_inc"),
                   categorical=("addr_state",), structural_missing=())


def frame(n=30_000, seed=0, rare_events=1) -> pd.DataFrame:
    """Mostly two well-populated states, plus a handful of loans in a third.

    ``rare_events`` sets how many of the rare state's loans default -- the quantity
    that decides whether its coefficient can be estimated at all.
    """
    rng = np.random.default_rng(seed)
    state = np.array(["CA"] * (n // 2) + ["TX"] * (n - n // 2 - 5) + ["ID"] * 5)
    event = rng.integers(0, 2, n)
    event[state == "ID"] = 0
    idx = np.flatnonzero(state == "ID")[:rare_events]
    event[idx] = 1
    return pd.DataFrame({
        "loan_amnt": rng.uniform(1000, 35000, n),
        "annual_inc": rng.lognormal(11, 0.4, n),
        "addr_state": state,
        "duration_months": rng.integers(1, 36, n),
        "event": event})


def test_a_level_with_too_few_events_is_pooled_not_estimated():
    dm = build_design_matrix(frame(), SPEC, flavour="cox")
    assert "addr_state_ID" not in dm.X.columns
    reason = dm.dropped["addr_state_ID"]
    assert "1 events in this level" in reason
    assert "pooled into the reference level" in reason
    assert "addr_state_TX" in dm.X.columns          # the populated level survives


def test_a_test_split_uses_exactly_the_training_columns():
    """The pooling decision belongs to the training split. A test split must not
    re-derive it, or the two matrices stop matching."""
    train = build_design_matrix(frame(seed=1), SPEC, flavour="cox")
    test = build_design_matrix(
        frame(seed=2, rare_events=5), SPEC, flavour="cox",
        standardisation=train.standardisation, fill_values=train.fill_values,
        reference_columns=list(train.X.columns))
    assert list(test.X.columns) == list(train.X.columns)
    assert "addr_state_ID" not in test.X.columns


def test_pooling_is_off_when_there_are_too_few_events_to_judge():
    small = frame(n=400)
    assert int(small["event"].sum()) < MIN_EVENTS_FOR_POOLING
    dm = build_design_matrix(small, SPEC, flavour="cox")
    assert not any("pooled" in str(v) for v in dm.dropped.values())


def test_zero_disables_pooling():
    dm = build_design_matrix(frame(), SPEC, flavour="cox", min_events_per_level=0)
    assert "addr_state_ID" in dm.X.columns


def test_the_gbm_matrix_is_untouched_by_the_rule():
    """The fix is a Cox fix. Every published gradient-boosted number must stay
    exactly as it was, so the GBM flavour keeps all of its categories."""
    dm = build_design_matrix(frame(), SPEC, flavour="gbm")
    assert not dm.dropped
    assert set(dm.X["addr_state"].astype(str)) == {"CA", "TX", "ID"}


def test_the_threshold_is_the_recorded_one():
    assert MIN_EVENTS_PER_LEVEL == 100
    assert MIN_EVENTS_FOR_POOLING == 2_000


# ------------------------------------------------- the guard in Stage 2 ----------

@pytest.fixture(scope="module")
def stage2():
    spec = importlib.util.spec_from_file_location(
        "train_models_guard", pathlib.Path("scripts/02_train_models.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Res:
    def __init__(self, c, model="cox"):
        self.concordance, self.model = c, model


def test_a_concordance_at_chance_is_refused(stage2):
    for c in (0.5, 0.502, 0.485, 0.5199):   # 0.48 is the boundary itself
        problem = stage2.degenerate_fit(_Res(c))
        assert problem, c
        assert "random ranking" in problem
    assert stage2.refuse_degenerate(_Res(0.502), "t") == 5


def test_a_working_model_passes(stage2):
    for c in (0.6936, 0.7158, 0.4, 0.5201):
        assert stage2.degenerate_fit(_Res(c)) is None, c
    assert stage2.refuse_degenerate(_Res(0.6936), "t") == 0


def test_a_concordance_that_is_not_a_number_is_refused(stage2):
    assert "not a number" in stage2.degenerate_fit(_Res(float("nan")))


def test_the_band_and_the_coefficient_bound_are_the_recorded_ones(stage2):
    assert stage2.DEGENERATE_C_BAND == 0.02
    assert stage2.COX_MAX_ABS_COEF == 10.0
    assert stage2.COX_PENALIZER == 0.05


def test_the_refusal_says_nothing_was_saved(stage2, capsys):
    stage2.refuse_degenerate(_Res(0.5), "holdout_applicant",
                             written="the Cox tables")
    err = capsys.readouterr().err
    assert "no model bundle and no metrics file" in err
    assert "the Cox tables" in err            # what is already on disk and now stale
    assert "FINDINGS 7h" in err
