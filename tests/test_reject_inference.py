"""Tests for the Stage 4 diagnostic gate and the reweighting correction.

The gate's job is to *block* a correction when one is not justified, so the tests
deliberately include cases where the right answer is "do nothing" and cases where
the right answer is "distortion exists but correction is still unjustified". A
diagnostic that always fired would be useless, and these tests would catch that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditsurv.reject_inference.correction import (
    effective_sample_size,
    fit_reweighting,
)
from creditsurv.reject_inference.diagnostics import (
    common_support,
    ks_statistic,
    parse_rejected_dti,
    rank_auc,
    run_selection_diagnostic,
    separability,
    standardized_mean_difference,
)


class TestStandardizedMeanDifference:
    def test_identical_samples_give_zero(self):
        a = np.arange(100, dtype=float)
        assert standardized_mean_difference(a, a) == pytest.approx(0.0)

    def test_one_pooled_sd_shift_gives_about_one(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 20_000)
        b = rng.normal(1, 1, 20_000)
        assert standardized_mean_difference(a, b) == pytest.approx(-1.0, abs=0.05)

    def test_sign_follows_direction(self):
        rng = np.random.default_rng(0)
        a = rng.normal(5, 1, 5_000)
        b = rng.normal(0, 1, 5_000)
        assert standardized_mean_difference(a, b) > 0

    def test_constant_input_is_nan_not_an_error(self):
        assert np.isnan(standardized_mean_difference(np.ones(50), np.ones(50)))

    def test_too_few_observations_is_nan(self):
        assert np.isnan(standardized_mean_difference(np.array([1.0]), np.arange(10.0)))


class TestRankAuc:
    def test_identical_distributions_give_half(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0, 1, 4_000)
        b = rng.normal(0, 1, 4_000)
        assert rank_auc(a, b) == pytest.approx(0.5, abs=0.03)

    def test_complete_separation_gives_one(self):
        assert rank_auc(np.arange(100, 200.0), np.arange(0, 100.0)) == pytest.approx(1.0)

    def test_is_invariant_to_monotone_rescaling(self):
        """The property that makes it valid across the FICO / Risk_Score mismatch."""
        rng = np.random.default_rng(2)
        a = rng.normal(700, 40, 3_000)
        b = rng.normal(640, 40, 3_000)
        plain = rank_auc(a, b)
        rescaled = rank_auc(a * 3.0 + 17.0, b * 3.0 + 17.0)
        assert plain == pytest.approx(rescaled, abs=1e-9)

    def test_smd_is_not_invariant_to_separate_rescaling(self):
        """Contrast: this is exactly why SMD cannot be trusted for that pair."""
        rng = np.random.default_rng(2)
        a = rng.normal(700, 40, 3_000)
        b = rng.normal(640, 40, 3_000)
        assert standardized_mean_difference(a, b) != pytest.approx(
            standardized_mean_difference(a, b * 1.3), abs=0.05
        )


class TestKs:
    def test_identical_samples_give_near_zero_statistic(self):
        a = np.linspace(0, 1, 2_000)
        stat, _ = ks_statistic(a, a)
        assert stat == pytest.approx(0.0, abs=1e-6)

    def test_disjoint_samples_give_one(self):
        stat, _ = ks_statistic(np.arange(0, 100.0), np.arange(200, 300.0))
        assert stat == pytest.approx(1.0)

    def test_huge_samples_give_vanishing_p_even_for_tiny_effects(self):
        """The reason the gate ignores p-values."""
        rng = np.random.default_rng(3)
        a = rng.normal(0, 1, 200_000)
        b = rng.normal(0.02, 1, 200_000)
        stat, p = ks_statistic(a, b, max_n=200_000)
        assert stat < 0.05, "effect is tiny"
        assert p < 0.01, "yet p is decisive -- which is why p is not used to gate"


class TestParseRejectedDti:
    def test_strips_percent_and_parses(self):
        got = parse_rejected_dti(pd.Series(["25.5%", "10%", " 8.1 %"]))
        # float32 output, so exact decimal equality does not hold.
        assert list(got) == pytest.approx([25.5, 10.0, 8.1], abs=1e-5)

    def test_clips_implausible_extremes(self):
        got = parse_rejected_dti(pd.Series(["50000031.5%", "-1%"]), clip_upper=100.0)
        assert got.iloc[0] == pytest.approx(100.0)
        assert got.iloc[1] == pytest.approx(0.0)

    def test_non_numeric_becomes_nan(self):
        assert parse_rejected_dti(pd.Series(["momentum"])).isna().all()


class TestSeparability:
    def test_identical_populations_give_auc_near_half(self):
        rng = np.random.default_rng(4)
        a = pd.DataFrame({"x": rng.normal(0, 1, 3_000), "y": rng.normal(0, 1, 3_000)})
        b = pd.DataFrame({"x": rng.normal(0, 1, 3_000), "y": rng.normal(0, 1, 3_000)})
        auc, _, _, _ = separability(a, b, seed=1)
        assert auc == pytest.approx(0.5, abs=0.05)

    def test_well_separated_populations_give_high_auc(self):
        rng = np.random.default_rng(4)
        a = pd.DataFrame({"x": rng.normal(5, 1, 3_000), "y": rng.normal(5, 1, 3_000)})
        b = pd.DataFrame({"x": rng.normal(0, 1, 3_000), "y": rng.normal(0, 1, 3_000)})
        auc, _, _, _ = separability(a, b, seed=1)
        assert auc > 0.95

    def test_returns_propensity_for_every_input_row(self):
        rng = np.random.default_rng(5)
        a = pd.DataFrame({"x": rng.normal(0, 1, 500)})
        b = pd.DataFrame({"x": rng.normal(1, 1, 900)})
        _, p_a, p_b, coefs = separability(a, b, seed=1)
        assert len(p_a) == 500 and len(p_b) == 900
        assert "x" in coefs.index

    def test_no_common_columns_rejected(self):
        with pytest.raises(ValueError, match="common"):
            separability(pd.DataFrame({"a": [1.0]}), pd.DataFrame({"b": [1.0]}))


class TestCommonSupport:
    def test_full_overlap_is_near_one(self):
        rng = np.random.default_rng(6)
        p = rng.uniform(0.3, 0.7, 5_000)
        out = common_support(p, rng.uniform(0.35, 0.65, 5_000))
        assert out["share_in_support"] > 0.97

    def test_disjoint_ranges_give_zero(self):
        out = common_support(np.linspace(0.8, 0.99, 1_000),
                             np.linspace(0.01, 0.2, 1_000))
        assert out["share_in_support"] == pytest.approx(0.0)

    def test_reports_counts_and_range(self):
        out = common_support(np.linspace(0.2, 0.8, 500), np.linspace(0.0, 1.0, 400))
        assert out["n_rejected"] == 400
        assert out["n_rejected_in_support"] <= 400
        assert len(out["accepted_propensity_range"]) == 2


class TestGate:
    @pytest.fixture
    def identical(self):
        rng = np.random.default_rng(7)
        n = 4_000
        cols = {"loan_amnt": rng.normal(15_000, 5_000, n),
                "dti": rng.normal(18, 8, n)}
        return pd.DataFrame(cols), pd.DataFrame(
            {k: rng.normal(v.mean(), v.std(), n) for k, v in cols.items()}
        )

    @pytest.fixture
    def separated(self):
        rng = np.random.default_rng(8)
        n = 4_000
        acc = pd.DataFrame({"loan_amnt": rng.normal(15_000, 4_000, n),
                            "dti": rng.normal(15, 5, n)})
        rej = pd.DataFrame({"loan_amnt": rng.normal(15_200, 4_000, n),
                            "dti": rng.normal(45, 5, n)})
        return acc, rej

    def test_no_difference_means_no_correction_needed(self, identical):
        diag = run_selection_diagnostic(*identical, seed=1)
        assert not diag.bias_detected
        assert diag.gate_decision == "no_correction_needed"
        assert "not warranted" in diag.rationale()

    def test_large_difference_is_detected(self, separated):
        diag = run_selection_diagnostic(*separated, seed=1)
        assert diag.bias_detected
        assert "dti" in diag.substantial_features

    def test_thin_overlap_blocks_correction_even_with_bias(self):
        """Distortion present but no common support: the honest answer is no."""
        rng = np.random.default_rng(9)
        n = 3_000
        acc = pd.DataFrame({"score": rng.normal(760, 15, n)})
        rej = pd.DataFrame({"score": rng.normal(560, 15, n)})
        diag = run_selection_diagnostic(acc, rej, seed=1, min_common_support=0.05)
        assert diag.bias_detected
        assert not diag.support_adequate
        assert diag.gate_decision == "correction_unjustified"
        assert "extrapolation" in diag.rationale()

    def test_partial_comparability_is_judged_on_rank_auc(self):
        rng = np.random.default_rng(10)
        n = 3_000
        acc = pd.DataFrame({"score": rng.normal(700, 40, n)})
        rej = pd.DataFrame({"score": rng.normal(700, 40, n) * 2.0})
        diag = run_selection_diagnostic(
            acc, rej, comparability={"score": "partial"}, seed=1
        )
        row = next(f for f in diag.features if f.feature == "score")
        assert row.primary_metric == "rank_auc"
        assert any("rank AUC" in n for n in diag.notes)

    def test_gate_decision_is_one_of_three_values(self, separated):
        diag = run_selection_diagnostic(*separated, seed=1)
        assert diag.gate_decision in {
            "apply_correction", "no_correction_needed", "correction_unjustified"
        }

    def test_thresholds_are_recorded_in_the_summary(self, identical):
        s = run_selection_diagnostic(*identical, seed=1).summary()
        assert "thresholds" in s and "min_common_support" in s["thresholds"]
        assert "gate_decision" in s and "rationale" in s

    def test_table_has_one_row_per_feature(self, separated):
        diag = run_selection_diagnostic(*separated, seed=1)
        assert len(diag.table()) == 2

    def test_no_common_columns_rejected(self):
        with pytest.raises(ValueError, match="share no columns"):
            run_selection_diagnostic(
                pd.DataFrame({"a": [1.0, 2.0]}), pd.DataFrame({"b": [1.0, 2.0]})
            )


class TestEffectiveSampleSize:
    def test_equal_weights_give_nominal_n(self):
        assert effective_sample_size(np.ones(100)) == pytest.approx(100.0)

    def test_one_dominant_weight_collapses_ess(self):
        w = np.concatenate([[1000.0], np.ones(99)])
        assert effective_sample_size(w) < 5.0

    def test_empty_is_zero(self):
        assert effective_sample_size(np.array([])) == 0.0


class TestReweighting:
    @pytest.fixture
    def populations(self):
        rng = np.random.default_rng(12)
        acc = pd.DataFrame({"score": rng.normal(710, 40, 3_000),
                            "dti": rng.normal(16, 6, 3_000)})
        rej = pd.DataFrame({"score": rng.normal(660, 45, 6_000),
                            "dti": rng.normal(26, 9, 6_000)})
        return acc, rej

    def test_returns_one_weight_per_accepted_row(self, populations):
        acc, rej = populations
        w = fit_reweighting(acc, rej, seed=1)
        assert len(w.weights) == len(acc)

    def test_weights_are_positive_and_mean_one(self, populations):
        acc, rej = populations
        w = fit_reweighting(acc, rej, seed=1)
        assert (w.weights > 0).all()
        assert float(np.mean(w.weights)) == pytest.approx(1.0, abs=1e-9)

    def test_reject_like_accepted_loans_get_larger_weights(self, populations):
        acc, rej = populations
        w = fit_reweighting(acc, rej, seed=1)
        # Low score looks like a typical reject, so it should stand in for more.
        low = acc["score"] < acc["score"].quantile(0.1)
        high = acc["score"] > acc["score"].quantile(0.9)
        assert w.weights[low.to_numpy()].mean() > w.weights[high.to_numpy()].mean()

    def test_clipping_is_reported(self, populations):
        acc, rej = populations
        w = fit_reweighting(acc, rej, clip_quantiles=(0.1, 0.9), seed=1)
        assert w.n_clipped > 0
        assert w.clip_bounds[0] < w.clip_bounds[1]

    def test_effective_sample_size_is_below_nominal(self, populations):
        acc, rej = populations
        w = fit_reweighting(acc, rej, seed=1)
        assert w.effective_n <= w.nominal_n
        assert 0.0 < w.ess_ratio <= 1.0

    def test_summary_states_the_blind_spot(self, populations):
        acc, rej = populations
        w = fit_reweighting(acc, rej, unavailable_features=("annual_inc",), seed=1)
        summary = w.summary()
        assert "annual_inc" in summary["unavailable_features"]
        assert "not addressed" in summary["caveat"]

    def test_no_common_features_rejected(self):
        with pytest.raises(ValueError, match="no common features"):
            fit_reweighting(pd.DataFrame({"a": [1.0]}), pd.DataFrame({"b": [1.0]}))
