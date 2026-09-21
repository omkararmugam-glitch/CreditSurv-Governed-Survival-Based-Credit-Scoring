"""Tests for the person-period expansion and the discrete-time hazard model.

The expansion is where censoring enters the likelihood, so getting it wrong would
quietly bias every hazard estimate. Its arithmetic is therefore checked
row-by-row on hand-countable cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditsurv.models.discrete_hazard import (
    DiscreteTimeHazardModel,
    expand_person_period,
    expansion_row_estimate,
)


@pytest.fixture
def tiny():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    duration = np.array([3, 1, 5], dtype=float)
    event = np.array([1, 0, 1], dtype=int)
    return X, duration, event


class TestRowEstimate:
    def test_monthly_bins_sum_durations(self, tiny):
        _, duration, _ = tiny
        assert expansion_row_estimate(duration, 1) == 9  # 3 + 1 + 5

    def test_quarterly_bins_reduce_rows(self, tiny):
        _, duration, _ = tiny
        assert expansion_row_estimate(duration, 3) == 4  # 1 + 1 + 2

    def test_horizon_truncation_reduces_rows(self, tiny):
        _, duration, _ = tiny
        assert expansion_row_estimate(duration, 1, max_horizon_months=2) == 5


class TestExpansion:
    def test_one_row_per_month_under_observation(self, tiny):
        X, duration, event = tiny
        pp = expand_person_period(X, duration, event)
        assert len(pp) == 9
        assert pp.n_loans == 3

    def test_periods_restart_at_one_for_each_loan(self, tiny):
        X, duration, event = tiny
        pp = expand_person_period(X, duration, event)
        assert list(pp.period) == [1, 2, 3, 1, 1, 2, 3, 4, 5]

    def test_target_marks_only_the_final_period_of_an_event(self, tiny):
        X, duration, event = tiny
        pp = expand_person_period(X, duration, event)
        assert list(pp.y) == [0, 0, 1, 0, 0, 0, 0, 0, 1]

    def test_censored_loan_contributes_only_zeros(self, tiny):
        X, duration, event = tiny
        pp = expand_person_period(X, duration, event)
        assert pp.y[pp.loan_index == 1].sum() == 0

    def test_total_events_equal_number_of_defaulted_loans(self, tiny):
        X, duration, event = tiny
        pp = expand_person_period(X, duration, event)
        assert pp.y.sum() == event.sum()

    def test_features_are_repeated_correctly(self, tiny):
        X, duration, event = tiny
        pp = expand_person_period(X, duration, event)
        assert list(pp.X["x"][pp.loan_index == 2]) == [3.0] * 5

    def test_period_column_is_added(self, tiny):
        X, duration, event = tiny
        pp = expand_person_period(X, duration, event)
        assert "period" in pp.X.columns
        assert pp.X["period"].dtype == np.int16

    def test_quarterly_binning(self, tiny):
        X, duration, event = tiny
        pp = expand_person_period(X, duration, event, time_bin_months=3)
        assert len(pp) == 4
        assert pp.y.sum() == 2

    def test_horizon_truncation_censors_later_events(self, tiny):
        X, duration, event = tiny
        # Loan 2 defaults at month 5; truncating at 2 must censor it.
        pp = expand_person_period(X, duration, event, max_horizon_months=2)
        assert pp.y[pp.loan_index == 2].sum() == 0

    def test_zero_duration_still_gets_one_period(self):
        X = pd.DataFrame({"x": [1.0]})
        pp = expand_person_period(X, np.array([0.0]), np.array([1]))
        assert len(pp) == 1

    def test_length_mismatch_is_rejected(self, tiny):
        X, duration, _ = tiny
        with pytest.raises(ValueError, match="same length"):
            expand_person_period(X, duration, np.array([1, 0]))

    def test_invalid_bin_width_rejected(self, tiny):
        X, duration, event = tiny
        with pytest.raises(ValueError, match="time_bin_months"):
            expand_person_period(X, duration, event, time_bin_months=0)


class TestNegativeSubsampling:
    def test_all_events_are_retained(self, tiny):
        X, duration, event = tiny
        pp = expand_person_period(X, duration, event, negative_subsample=0.01, seed=1)
        assert pp.y.sum() == event.sum(), "case-control sampling must keep every event"

    def test_negatives_are_upweighted_to_compensate(self, tiny):
        X, duration, event = tiny
        pp = expand_person_period(X, duration, event, negative_subsample=0.5, seed=1)
        assert np.allclose(pp.weight[pp.y == 0], 2.0)
        assert np.allclose(pp.weight[pp.y == 1], 1.0)

    def test_invalid_fraction_rejected(self, tiny):
        X, duration, event = tiny
        with pytest.raises(ValueError, match="negative_subsample"):
            expand_person_period(X, duration, event, negative_subsample=1.5)


class TestLoanWeights:
    def test_weights_broadcast_to_every_period_of_a_loan(self, tiny):
        X, duration, event = tiny
        pp = expand_person_period(X, duration, event, loan_weight=np.array([2.0, 5.0, 1.0]))
        assert list(pp.weight[pp.loan_index == 0]) == [2.0, 2.0, 2.0]
        assert list(pp.weight[pp.loan_index == 1]) == [5.0]

    def test_wrong_length_rejected(self, tiny):
        X, duration, event = tiny
        with pytest.raises(ValueError, match="loan_weight"):
            expand_person_period(X, duration, event, loan_weight=np.array([1.0]))

    def test_weights_combine_with_subsampling(self, tiny):
        X, duration, event = tiny
        pp = expand_person_period(
            X, duration, event, loan_weight=np.array([2.0, 2.0, 2.0]),
            negative_subsample=0.5, seed=3,
        )
        assert np.allclose(pp.weight[pp.y == 0], 4.0)   # 2.0 / 0.5
        assert np.allclose(pp.weight[pp.y == 1], 2.0)


class TestModel:
    @pytest.fixture
    def fitted(self, synthetic_survival_frame):
        df = synthetic_survival_frame
        X = df[["fico_range_low", "dti", "loan_amnt"]].copy()
        m = DiscreteTimeHazardModel(
            time_bin_months=3, max_horizon_months=36, num_boost_round=40
        )
        return m.fit(X, df["duration_months"].to_numpy(), df["event"].to_numpy()), X, df

    def test_unfitted_model_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            DiscreteTimeHazardModel().predict_survival(pd.DataFrame({"x": [1.0]}),
                                                       np.array([12.0]))

    def test_hazards_are_valid_probabilities(self, fitted):
        m, X, _ = fitted
        h = m.predict_hazard(X.iloc[:20], np.arange(1, 5))
        assert h.shape == (20, 4)
        assert ((h > 0) & (h < 1)).all()

    def test_survival_is_monotone_non_increasing(self, fitted):
        m, X, _ = fitted
        times = np.array([3.0, 6.0, 12.0, 24.0, 36.0])
        s = m.predict_survival(X.iloc[:30], times)
        assert np.all(np.diff(s, axis=1) <= 1e-9)

    def test_survival_is_bounded(self, fitted):
        m, X, _ = fitted
        s = m.predict_survival(X.iloc[:30], np.array([6.0, 36.0]))
        assert ((s >= 0) & (s <= 1)).all()

    def test_survival_shape_matches_requested_times(self, fitted):
        m, X, _ = fitted
        times = np.array([6.0, 18.0, 30.0])
        assert m.predict_survival(X.iloc[:7], times).shape == (7, 3)

    def test_risk_is_one_minus_survival(self, fitted):
        m, X, _ = fitted
        risk = m.predict_risk(X.iloc[:10], at_month=24)
        surv = m.predict_survival(X.iloc[:10], np.array([24.0]))[:, 0]
        assert np.allclose(risk, 1.0 - surv)

    def test_recovers_the_true_signal_direction(self, fitted):
        """Hazard rises with dti in the fixture, so risk must too."""
        m, X, _ = fitted
        probe = X.iloc[[0]].copy()
        low, high = probe.copy(), probe.copy()
        low["dti"] = float(X["dti"].quantile(0.05))
        high["dti"] = float(X["dti"].quantile(0.95))
        r_low = m.predict_risk(low, at_month=36)[0]
        r_high = m.predict_risk(high, at_month=36)[0]
        assert r_high > r_low

    def test_feature_importance_includes_period(self, fitted):
        m, _, _ = fitted
        imp = m.feature_importance()
        assert "period" in set(imp["feature"])
        assert (imp["importance"] >= 0).all()

    def test_weighted_fit_changes_the_model(self, synthetic_survival_frame):
        df = synthetic_survival_frame
        X = df[["fico_range_low", "dti", "loan_amnt"]].copy()
        d, e = df["duration_months"].to_numpy(), df["event"].to_numpy()
        rng = np.random.default_rng(0)
        w = rng.uniform(0.2, 5.0, len(X))

        a = DiscreteTimeHazardModel(time_bin_months=3, max_horizon_months=36,
                                    num_boost_round=30).fit(X, d, e)
        b = DiscreteTimeHazardModel(time_bin_months=3, max_horizon_months=36,
                                    num_boost_round=30).fit(X, d, e, loan_weight=w)
        pa = a.predict_risk(X.iloc[:50], at_month=36)
        pb = b.predict_risk(X.iloc[:50], at_month=36)
        assert not np.allclose(pa, pb), "weights must actually influence the fit"
