"""Tests for the survival metrics.

These matter more than most: the metrics are hand-implemented because
`scikit-survival` is unavailable, so a silent error here would corrupt every
reported number with nothing to catch it. Each metric is therefore checked
against a case whose answer is known analytically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditsurv.models.evaluate import (
    CensoringModel,
    brier_score,
    calibration_table,
    concordance_index,
    cumulative_dynamic_auc,
    evaluate_survival,
    integrated_brier_score,
)


@pytest.fixture
def simple():
    """Ten loans, half events, no censoring -- so IPCW weights are all 1."""
    duration = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50], dtype=float)
    event = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=int)
    return duration, event


class TestConcordance:
    def test_perfect_ranking_is_one(self, simple):
        duration, event = simple
        # Risk decreasing in survival time == perfect ordering.
        assert concordance_index(duration, event, -duration) == pytest.approx(1.0)

    def test_reversed_ranking_is_zero(self, simple):
        duration, event = simple
        assert concordance_index(duration, event, duration) == pytest.approx(0.0)

    def test_constant_risk_is_a_half(self, simple):
        duration, event = simple
        assert concordance_index(duration, event, np.zeros(10)) == pytest.approx(0.5)


class TestCensoringModel:
    def test_survival_is_monotone_non_increasing(self, simple):
        duration, event = simple
        cm = CensoringModel(duration, event)
        g = cm.survival(np.arange(1, 51))
        assert np.all(np.diff(g) <= 1e-12)

    def test_floor_is_applied(self, simple):
        duration, event = simple
        cm = CensoringModel(duration, event, g_floor=0.25)
        assert cm.survival(np.array([50.0]))[0] >= 0.25

    def test_no_censoring_means_g_stays_one(self):
        duration = np.arange(1, 21, dtype=float)
        event = np.ones(20, dtype=int)
        cm = CensoringModel(duration, event)
        assert cm.survival(np.array([10.0]))[0] == pytest.approx(1.0)


class TestTimeDependentAUC:
    def test_perfect_separation_gives_auc_one(self, simple):
        duration, event = simple
        cm = CensoringModel(duration, event)
        times = np.array([22.0])
        # Cases (duration<=22, event=1) get survival 0; controls get 1.
        surv = np.where(duration <= 22, 0.0, 1.0).reshape(-1, 1)
        out = cumulative_dynamic_auc(surv, times, duration, event, cm)
        assert out["auc"].iloc[0] == pytest.approx(1.0)

    def test_reversed_gives_auc_zero(self, simple):
        duration, event = simple
        cm = CensoringModel(duration, event)
        times = np.array([22.0])
        surv = np.where(duration <= 22, 1.0, 0.0).reshape(-1, 1)
        out = cumulative_dynamic_auc(surv, times, duration, event, cm)
        assert out["auc"].iloc[0] == pytest.approx(0.0)

    def test_all_ties_give_auc_half(self, simple):
        duration, event = simple
        cm = CensoringModel(duration, event)
        times = np.array([22.0])
        surv = np.full((10, 1), 0.5)
        out = cumulative_dynamic_auc(surv, times, duration, event, cm)
        assert out["auc"].iloc[0] == pytest.approx(0.5)

    def test_case_and_control_counts_are_correct(self, simple):
        duration, event = simple
        cm = CensoringModel(duration, event)
        times = np.array([22.0])
        surv = np.full((10, 1), 0.5)
        out = cumulative_dynamic_auc(surv, times, duration, event, cm)
        assert out["n_cases"].iloc[0] == 4     # durations 5,10,15,20 with event
        assert out["n_controls"].iloc[0] == 6  # durations 25..50

    def test_horizon_with_no_cases_returns_nan_not_an_error(self, simple):
        duration, event = simple
        cm = CensoringModel(duration, event)
        out = cumulative_dynamic_auc(
            np.full((10, 1), 0.5), np.array([1.0]), duration, event, cm
        )
        assert np.isnan(out["auc"].iloc[0])
        assert out["reliable"].iloc[0] == False  # noqa: E712

    def test_shape_mismatch_is_rejected(self, simple):
        duration, event = simple
        cm = CensoringModel(duration, event)
        with pytest.raises(ValueError, match="n_samples"):
            cumulative_dynamic_auc(
                np.full((10, 3), 0.5), np.array([12.0]), duration, event, cm
            )


class TestBrierScore:
    def test_perfect_prediction_scores_zero(self, simple):
        duration, event = simple
        cm = CensoringModel(duration, event)
        times = np.array([22.0])
        surv = np.where(duration <= 22, 0.0, 1.0).reshape(-1, 1)
        out = brier_score(surv, times, duration, event, cm)
        assert out["brier"].iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_worst_prediction_scores_near_one(self, simple):
        duration, event = simple
        cm = CensoringModel(duration, event)
        times = np.array([22.0])
        surv = np.where(duration <= 22, 1.0, 0.0).reshape(-1, 1)
        out = brier_score(surv, times, duration, event, cm)
        assert out["brier"].iloc[0] > 0.9

    def test_constant_half_scores_a_quarter(self, simple):
        duration, event = simple
        cm = CensoringModel(duration, event)
        out = brier_score(
            np.full((10, 1), 0.5), np.array([22.0]), duration, event, cm
        )
        assert out["brier"].iloc[0] == pytest.approx(0.25, abs=0.02)

    def test_lower_is_better_ordering(self, simple):
        duration, event = simple
        cm = CensoringModel(duration, event)
        times = np.array([22.0])
        good = np.where(duration <= 22, 0.1, 0.9).reshape(-1, 1)
        bad = np.where(duration <= 22, 0.4, 0.6).reshape(-1, 1)
        b_good = brier_score(good, times, duration, event, cm)["brier"].iloc[0]
        b_bad = brier_score(bad, times, duration, event, cm)["brier"].iloc[0]
        assert b_good < b_bad

    def test_integrated_brier_averages_the_curve(self):
        tbl = pd.DataFrame({"time": [0.0, 10.0], "brier": [0.2, 0.2]})
        assert integrated_brier_score(tbl) == pytest.approx(0.2)

    def test_integrated_brier_needs_two_points(self):
        tbl = pd.DataFrame({"time": [5.0], "brier": [0.2]})
        assert np.isnan(integrated_brier_score(tbl))


class TestCalibration:
    def test_returns_one_row_per_populated_bin(self, synthetic_survival_frame):
        df = synthetic_survival_frame
        rng = np.random.default_rng(0)
        pred = rng.uniform(0.2, 0.9, len(df))
        out = calibration_table(
            pred, df["duration_months"].to_numpy(), df["event"].to_numpy(), t=24, n_bins=5
        )
        assert len(out) == 5
        assert {"predicted_survival", "observed_survival", "error"} <= set(out.columns)

    def test_predicted_survival_increases_across_bins(self, synthetic_survival_frame):
        df = synthetic_survival_frame
        rng = np.random.default_rng(0)
        pred = rng.uniform(0.1, 0.95, len(df))
        out = calibration_table(
            pred, df["duration_months"].to_numpy(), df["event"].to_numpy(), t=24, n_bins=5
        )
        assert out["predicted_survival"].is_monotonic_increasing


class TestEvaluateSurvival:
    def test_produces_a_complete_result(self, synthetic_survival_frame):
        df = synthetic_survival_frame
        d = df["duration_months"].to_numpy()
        e = df["event"].to_numpy()
        cm = CensoringModel(d, e)
        times = np.array([6.0, 12.0, 24.0])
        rng = np.random.default_rng(1)
        surv = np.sort(rng.uniform(0, 1, (len(df), 3)), axis=1)[:, ::-1]

        res = evaluate_survival(
            model_name="dummy", split_name="test", survival=surv, times=times,
            duration=d, event=e, censoring=cm, calibration_at=12.0,
        )
        assert res.model == "dummy" and res.n == len(df)
        assert 0.0 <= res.concordance <= 1.0
        assert len(res.auc_table) == 3 and len(res.brier_table) == 3
        assert not res.calibration.empty
        assert "concordance" in res.summary()
        assert "AUC" in str(res)

    def test_default_risk_is_derived_from_the_last_horizon(self, synthetic_survival_frame):
        df = synthetic_survival_frame
        d, e = df["duration_months"].to_numpy(), df["event"].to_numpy()
        cm = CensoringModel(d, e)
        times = np.array([12.0, 36.0])
        surv = np.column_stack([np.full(len(df), 0.9), np.linspace(0.1, 0.8, len(df))])
        res = evaluate_survival(
            model_name="m", split_name="s", survival=surv, times=times,
            duration=d, event=e, censoring=cm,
        )
        assert np.isfinite(res.concordance)
