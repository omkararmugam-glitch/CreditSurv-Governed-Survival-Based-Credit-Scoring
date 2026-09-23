"""TreeSHAP attributions for the discrete-hazard model.

The point of this explainer is that it is exact, deterministic and fast enough for
bulk runs, so those three properties are what is tested here. Whether it *agrees*
with SurvSHAP(t) closely enough to write notices is a separate, pre-registered
question answered by scripts/03d_explainer_validation.py and FINDINGS section 7.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from creditsurv.explain.adverse_action import build_adverse_action_notice
from creditsurv.explain.tree_shap import TreeShapExplanation, explain_tree_shap
from creditsurv.models.discrete_hazard import DiscreteTimeHazardModel


@pytest.fixture
def fitted(synthetic_survival_frame):
    df = synthetic_survival_frame
    X = df[["fico_range_low", "dti", "loan_amnt"]].copy()
    model = DiscreteTimeHazardModel(time_bin_months=3, max_horizon_months=36,
                                    num_boost_round=40)
    model.fit(X, df["duration_months"].to_numpy(), df["event"].to_numpy())
    return model, X


def test_unfitted_model_is_refused():
    with pytest.raises(RuntimeError, match="not fitted"):
        explain_tree_shap(DiscreteTimeHazardModel(), pd.DataFrame({"x": [1.0]}))


def test_shape_and_interface(fitted):
    model, X = fitted
    expl = explain_tree_shap(model, X.iloc[:20], horizon_months=36.0)
    assert isinstance(expl, TreeShapExplanation)
    assert expl.phi.shape == (20, len(X.columns))       # period is not a feature
    assert model.period_col not in expl.feature_names
    assert set(expl.feature_names) == set(X.columns)
    assert expl.prediction.shape == (20, 1)
    assert expl.explainer == "treeshap"
    # The interface the notice code relies on.
    assert len(expl.importance(0)) == len(X.columns)
    assert len(expl.signed_importance(0)) == len(X.columns)
    assert (expl.importance(0) >= 0).all()
    assert expl.n_observations == 20


def test_deterministic_and_independent_of_batch_membership(fitted):
    """The property SurvSHAP(t) lacks: an applicant's attributions do not depend on
    who else is in the batch, so streaming cannot change a notice."""
    model, X = fitted
    alone = explain_tree_shap(model, X.iloc[[7]], horizon_months=36.0).phi[0]
    in_batch = explain_tree_shap(model, X.iloc[[3, 7, 11]], horizon_months=36.0).phi[1]
    again = explain_tree_shap(model, X.iloc[[7]], horizon_months=36.0).phi[0]
    assert np.allclose(alone, in_batch)
    assert np.array_equal(alone, again)                 # exact, not merely close


def test_attributions_point_the_right_way(fitted):
    """Sign convention matches SurvSHAP(t): positive raises survival. A borrower
    whose risk is above the batch's own average must carry net-negative
    attribution."""
    model, X = fitted
    expl = explain_tree_shap(model, X.iloc[:100], horizon_months=36.0)
    risk = 1.0 - expl.prediction[:, 0]
    net = expl.phi.sum(axis=1)
    riskiest = int(np.argmax(risk))
    safest = int(np.argmin(risk))
    assert net[riskiest] < net[safest]
    # Ranking by net attribution should track predicted risk closely.
    rho = pd.Series(net).corr(pd.Series(-risk), method="spearman")
    assert rho > 0.8


def test_horizon_changes_the_attributions(fitted):
    model, X = fitted
    short = explain_tree_shap(model, X.iloc[:30], horizon_months=6.0)
    long = explain_tree_shap(model, X.iloc[:30], horizon_months=36.0)
    assert not np.allclose(short.phi, long.phi)
    assert short.horizon_months == 6.0 and long.horizon_months == 36.0


def test_notice_builds_from_tree_shap_unchanged(fitted):
    """The whole reason for matching SurvShapExplanation's interface: Regulation B
    filtering and wording come from the same function either way."""
    model, X = fitted
    expl = explain_tree_shap(model, X.iloc[:5], horizon_months=36.0,
                             times=np.array([12.0, 36.0]))
    notice = build_adverse_action_notice(expl, obs=0, horizon_months=36,
                                         model_name="discrete_hazard, treeshap")
    text = notice.render()
    assert "STATEMENT OF ADVERSE ACTION" in text
    assert len(notice.reasons) <= 4
    assert notice.predicted_default_probability == pytest.approx(
        1.0 - expl.prediction[0, 1])


def test_batching_does_not_change_results(fitted):
    model, X = fitted
    one_pass = explain_tree_shap(model, X.iloc[:40], horizon_months=36.0)
    in_blocks = explain_tree_shap(model, X.iloc[:40], horizon_months=36.0,
                                  batch_rows=7)
    assert np.allclose(one_pass.phi, in_blocks.phi)


def test_fast_enough_for_bulk(fitted):
    """Not a benchmark, a floor: the whole argument for this explainer is that it
    is orders of magnitude cheaper than SurvSHAP(t)'s ~2.7 s per applicant."""
    model, X = fitted
    rows = pd.concat([X] * 4, ignore_index=True).head(400)
    t0 = time.perf_counter()
    explain_tree_shap(model, rows, horizon_months=36.0)
    per_row = (time.perf_counter() - t0) / len(rows)
    assert per_row < 0.1, f"{per_row * 1000:.1f} ms/row is too slow for bulk use"
