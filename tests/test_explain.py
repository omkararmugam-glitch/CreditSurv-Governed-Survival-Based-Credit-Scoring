"""Tests for SurvSHAP(t), the naive comparison, adverse action and segments.

The central test is :class:`TestEfficiencyAxiom`. SurvSHAP(t) here is a
reimplementation rather than the reference package, so its correctness rests on
satisfying the Shapley efficiency axiom at every time point. An implementation
that mishandled coalition weighting or feature masking would produce
plausible-looking attributions and fail this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditsurv.explain.adverse_action import (
    MAX_PRINCIPAL_REASONS,
    NonDisclosableFeatureError,
    assert_disclosable,
    build_adverse_action_notice,
)
from creditsurv.explain.compare import (
    compare_explanations,
    time_variation_share,
    top_k_overlap,
)
from creditsurv.explain.naive_shap import explain_naive_shap
from creditsurv.explain.segments import analyse_segment_stability, segment_importance
from creditsurv.explain.survshap import (
    SurvShapExplanation,
    check_efficiency,
    explain_survshap,
    summarise_background,
)


class LinearSurvivalModel:
    """A deterministic stand-in with a known structure.

    ``S(t|x) = exp(-t * exp(w . x) / scale)``. Being analytic means the test does
    not depend on a trained model and runs in milliseconds.
    """

    def __init__(self, weights: dict[str, float], scale: float = 400.0):
        self.weights = weights
        self.scale = scale

    def predict_survival(self, X: pd.DataFrame, times) -> np.ndarray:
        times = np.atleast_1d(np.asarray(times, dtype=float))
        lin = np.zeros(len(X), dtype=float)
        for col, w in self.weights.items():
            if col in X.columns:
                v = X[col]
                if str(v.dtype) == "category":
                    v = v.cat.codes
                lin += w * pd.to_numeric(v, errors="coerce").fillna(0.0).to_numpy(float)
        hazard = np.exp(np.clip(lin, -20, 20))
        return np.exp(-np.outer(hazard, times) / self.scale)


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(11)
    n = 120
    return pd.DataFrame(
        {
            "dti": rng.uniform(0, 40, n),
            "fico_midpoint": rng.uniform(620, 820, n),
            "loan_amnt": rng.uniform(1000, 35000, n),
            "inq_last_6mths": rng.integers(0, 6, n).astype(float),
        }
    )


@pytest.fixture
def toy_model():
    return LinearSurvivalModel(
        {"dti": 0.05, "fico_midpoint": -0.01, "loan_amnt": 2e-5, "inq_last_6mths": 0.2}
    )


@pytest.fixture
def explanation(toy_model, toy_data):
    times = np.array([6.0, 12.0, 24.0, 36.0])
    return explain_survshap(
        toy_model, toy_data.iloc[:6], toy_data, times,
        nsamples=64, n_background=15, seed=3,
    )


class TestEfficiencyAxiom:
    """sum_j phi_j(t) + base(t) == S(t|x), at every t. The correctness proof."""

    def test_residual_is_at_machine_precision(self, explanation):
        eff = check_efficiency(explanation)
        assert float(eff["max_abs_residual"].max()) < 1e-8

    def test_holds_at_every_time_point(self, explanation):
        eff = check_efficiency(explanation)
        assert (eff["max_abs_residual"] < 1e-8).all()

    def test_reconstruction_matches_the_model_directly(self, explanation):
        rebuilt = explanation.base[None, :] + explanation.phi.sum(axis=1)
        np.testing.assert_allclose(rebuilt, explanation.prediction, atol=1e-8)

    def test_a_corrupted_explanation_fails_the_check(self, explanation):
        broken = SurvShapExplanation(
            phi=explanation.phi * 1.5,
            times=explanation.times,
            base=explanation.base,
            prediction=explanation.prediction,
            feature_names=explanation.feature_names,
            feature_values=explanation.feature_values,
        )
        assert float(check_efficiency(broken)["max_abs_residual"].max()) > 1e-6


class TestSurvShapShape:
    def test_phi_has_obs_feature_time_shape(self, explanation, toy_data):
        assert explanation.phi.shape == (6, toy_data.shape[1], 4)

    def test_curve_is_time_by_feature(self, explanation):
        curve = explanation.curve(0)
        assert curve.shape == (4, 4)
        assert list(curve.columns) == list(explanation.feature_names)

    def test_importance_is_sorted_and_non_negative(self, explanation):
        imp = explanation.importance()
        assert (imp >= 0).all()
        assert list(imp) == sorted(imp, reverse=True)

    def test_recovers_the_dominant_feature(self, toy_model, toy_data):
        """dti and inq drive the hazard most, so they should rank highly."""
        times = np.array([12.0, 36.0])
        expl = explain_survshap(
            toy_model, toy_data.iloc[:10], toy_data, times,
            nsamples=96, n_background=20, seed=5,
        )
        assert set(expl.importance().head(3).index) & {"dti", "inq_last_6mths"}

    def test_signed_importance_has_the_expected_direction(self, toy_model, toy_data):
        """Higher dti lowers survival, so its signed attribution is negative."""
        times = np.array([12.0, 36.0])
        high = toy_data.iloc[:8].copy()
        high["dti"] = 38.0
        expl = explain_survshap(
            toy_model, high, toy_data, times, nsamples=96, n_background=20, seed=5
        )
        assert expl.signed_importance()["dti"] < 0

    def test_rejects_non_3d_phi(self, explanation):
        with pytest.raises(ValueError, match="3-D"):
            SurvShapExplanation(
                phi=np.zeros((3, 4)), times=explanation.times,
                base=explanation.base, prediction=explanation.prediction,
                feature_names=explanation.feature_names,
                feature_values=explanation.feature_values,
            )


class TestBackgroundSummary:
    def test_reduces_to_requested_size(self, toy_data):
        assert len(summarise_background(toy_data, n=10, seed=1)) <= 10

    def test_returns_input_when_already_small(self, toy_data):
        small = toy_data.iloc[:5]
        assert len(summarise_background(small, n=50)) == 5

    def test_preserves_column_order(self, toy_data):
        out = summarise_background(toy_data, n=10, seed=1)
        assert list(out.columns) == list(toy_data.columns)

    def test_handles_categorical_columns(self, toy_data):
        df = toy_data.copy()
        df["purpose"] = pd.Categorical(["a", "b", "c"] * (len(df) // 3))
        out = summarise_background(df, n=8, seed=1)
        assert str(out["purpose"].dtype) == "category"
        assert len(out) <= 8


class TestNaiveComparison:
    @pytest.fixture
    def pair(self, toy_model, toy_data):
        times = np.array([6.0, 12.0, 24.0, 36.0])
        X = toy_data.iloc[:8]
        surv = explain_survshap(toy_model, X, toy_data, times,
                                nsamples=64, n_background=15, seed=3)
        naive = explain_naive_shap(toy_model, X, toy_data, at_month=36.0,
                                   nsamples=64, n_background=15, seed=3)
        return surv, naive

    def test_comparison_fields_are_populated(self, pair):
        result = compare_explanations(*pair)
        assert -1.0 <= result.spearman <= 1.0
        assert 0.0 <= result.top5_overlap <= 1.0
        assert 0.0 <= result.sign_agreement <= 1.0
        assert len(result.per_feature) == len(pair[0].feature_names)

    def test_verdict_is_one_of_the_expected_categories(self, pair):
        verdict = compare_explanations(*pair).verdict()
        assert verdict.split(":")[0] in {
            "BROAD AGREEMENT", "MEANINGFUL DISAGREEMENT", "MATERIAL DISAGREEMENT"
        }

    def test_sign_conventions_are_aligned_not_mirrored(self, toy_model, toy_data):
        """dti raises risk; after alignment both methods must agree on that.

        Explained on deliberately high-dti borrowers. Averaging signed
        attributions over a sample centred on the background would give ~0 with
        an arbitrary sign, which would test nothing.
        """
        times = np.array([12.0, 36.0])
        high = toy_data.iloc[:8].copy()
        high["dti"] = 38.0
        surv = explain_survshap(toy_model, high, toy_data, times,
                                nsamples=96, n_background=20, seed=3)
        naive = explain_naive_shap(toy_model, high, toy_data, at_month=36.0,
                                   nsamples=96, n_background=20, seed=3)
        row = compare_explanations(surv, naive).per_feature.loc["dti"]
        assert row["survshap_signed"] > 0, "aligned: positive means raises risk"
        assert row["naive_signed"] > 0

    def test_mismatched_features_rejected(self, pair):
        surv, naive = pair
        naive.feature_names = ("a", "b")
        with pytest.raises(ValueError, match="same feature ordering"):
            compare_explanations(surv, naive)

    def test_top_k_overlap_extremes(self):
        a = pd.Series([3.0, 2.0, 1.0], index=["x", "y", "z"])
        assert top_k_overlap(a, a, 2) == pytest.approx(1.0)
        b = pd.Series([1.0, 2.0, 3.0], index=["x", "y", "z"])
        assert top_k_overlap(a, b, 1) == pytest.approx(0.0)

    def test_flat_attribution_has_zero_time_variation(self, explanation):
        flat = SurvShapExplanation(
            phi=np.ones_like(explanation.phi) * 0.1,
            times=explanation.times, base=explanation.base,
            prediction=explanation.prediction,
            feature_names=explanation.feature_names,
            feature_values=explanation.feature_values,
        )
        assert np.allclose(time_variation_share(flat).to_numpy(), 0.0)

    def test_varying_attribution_has_positive_time_variation(self, explanation):
        phi = np.zeros_like(explanation.phi)
        phi[:, 0, :] = np.array([-1.0, -0.5, 0.5, 1.0])   # crosses zero
        varying = SurvShapExplanation(
            phi=phi, times=explanation.times, base=explanation.base,
            prediction=explanation.prediction,
            feature_names=explanation.feature_names,
            feature_values=explanation.feature_values,
        )
        assert time_variation_share(varying).iloc[0] > 0.5


class TestAdverseAction:
    def test_grade_is_refused_as_a_reason(self):
        with pytest.raises(NonDisclosableFeatureError, match="grade"):
            assert_disclosable(["dti", "grade"])

    def test_geography_is_allowed_as_a_model_input(self):
        """Geography is a legitimate feature; it is only barred as a stated reason.

        Refusing the whole notice would be wrong: `addr_state` is in the model, so
        that would make Stage 3(b) impossible to run at all.
        """
        assert_disclosable(["dti", "addr_state"])   # must not raise

    def test_geography_never_becomes_a_stated_reason(self, toy_model, toy_data):
        df = toy_data.copy()
        df["addr_state"] = pd.Categorical(
            np.resize(["CA", "NY", "TX"], len(df))
        )
        model = LinearSurvivalModel({"dti": 0.05, "addr_state": 0.9})
        expl = explain_survshap(
            model, df.iloc[:6], df, np.array([12.0, 36.0]),
            nsamples=96, n_background=15, seed=2,
        )
        notice = build_adverse_action_notice(expl, obs=0)
        assert "addr_state" not in {r.feature for r in notice.reasons}

    def test_dominant_geography_raises_a_fair_lending_flag(self, toy_data):
        """If geography IS the real driver, that is surfaced, not silently dropped."""
        df = toy_data.copy()
        df["addr_state"] = pd.Categorical(np.resize(["CA", "NY", "TX"], len(df)))
        model = LinearSurvivalModel({"addr_state": 3.0}, scale=100.0)
        expl = explain_survshap(
            model, df.iloc[:6], df, np.array([12.0, 36.0]),
            nsamples=96, n_background=15, seed=2,
        )
        notices = [build_adverse_action_notice(expl, obs=i) for i in range(6)]
        assert any(n.fair_lending_flags for n in notices)
        flagged = next(n for n in notices if n.fair_lending_flags)
        assert "addr_state" in flagged.fair_lending_flags
        # Surfaced in the internal record -- and never in the applicant's notice,
        # which is where test 1 found it in 166 of 255 notices.
        assert "FAIR-LENDING REVIEW FLAG" in flagged.render_internal()
        assert "addr_state" in flagged.render_internal()
        assert flagged.to_dict()["fair_lending_flags"] == ["addr_state"]
        applicant = flagged.render()
        assert "INTERNAL" not in applicant.upper()
        assert "addr_state" not in applicant
        assert "fair-lending" not in applicant.lower()

    def test_clean_feature_list_is_accepted(self):
        assert_disclosable(["dti", "fico_midpoint", "revol_util"])

    def test_notice_caps_reasons_at_four(self, explanation):
        notice = build_adverse_action_notice(explanation, obs=0)
        assert len(notice.reasons) <= MAX_PRINCIPAL_REASONS

    def test_reasons_are_ranked_from_one(self, explanation):
        notice = build_adverse_action_notice(explanation, obs=0)
        assert [r.rank for r in notice.reasons] == list(range(1, len(notice.reasons) + 1))

    def test_only_adverse_features_become_reasons(self, explanation):
        notice = build_adverse_action_notice(explanation, obs=0)
        assert all(r.attribution < 0 for r in notice.reasons), (
            "a feature that helped the applicant is not a reason for denial"
        )

    def test_helpful_features_are_recorded_separately(self, explanation):
        notice = build_adverse_action_notice(explanation, obs=0)
        overlap = set(notice.excluded_helpful_features) & {
            r.feature for r in notice.reasons
        }
        assert not overlap

    def test_rendered_notice_contains_every_required_element(self, explanation):
        text = build_adverse_action_notice(explanation, obs=0).render()
        assert "STATEMENT OF ADVERSE ACTION" in text
        assert "ACTION TAKEN" in text
        assert "PRINCIPAL REASONS" in text
        assert "Equal Credit Opportunity Act" in text
        assert "Consumer Financial Protection Bureau" in text

    def test_no_duplicate_statutory_reason(self, explanation):
        notice = build_adverse_action_notice(explanation, obs=0)
        texts = [r.reason for r in notice.reasons]
        assert len(texts) == len(set(texts))

    def test_score_block_appears_only_with_a_score(self, explanation):
        without = build_adverse_action_notice(explanation, obs=0).render()
        assert "CREDIT SCORE INFORMATION" not in without
        with_score = build_adverse_action_notice(
            explanation, obs=0, credit_score=640.0
        ).render()
        assert "CREDIT SCORE INFORMATION" in with_score
        assert "640" in with_score

    def test_to_dict_is_serialisable(self, explanation):
        import json

        payload = build_adverse_action_notice(explanation, obs=0).to_dict()
        json.dumps(payload)
        assert payload["n_reasons"] == len(payload["reasons"])


class TestSegments:
    @pytest.fixture
    def big_explanation(self, toy_model, toy_data):
        times = np.array([12.0, 36.0])
        return explain_survshap(
            toy_model, toy_data.iloc[:60], toy_data, times,
            nsamples=48, n_background=12, seed=9,
        )

    def test_segment_importance_is_features_by_levels(self, big_explanation):
        seg = pd.Series(["a", "b", "c"] * 20)
        imp = segment_importance(big_explanation, seg, min_size=10)
        assert imp.shape == (len(big_explanation.feature_names), 3)

    def test_small_levels_are_dropped(self, big_explanation):
        seg = pd.Series(["big"] * 57 + ["tiny"] * 3)
        imp = segment_importance(big_explanation, seg, min_size=10)
        assert list(imp.columns) == ["big"]

    def test_length_mismatch_rejected(self, big_explanation):
        with pytest.raises(ValueError, match="observations"):
            segment_importance(big_explanation, pd.Series(["a", "b"]))

    def test_stability_result_is_populated(self, big_explanation):
        seg = pd.Series(["a", "b", "c"] * 20)
        st = analyse_segment_stability(big_explanation, seg, segment_name="grp",
                                       min_size=10)
        assert st.segment_name == "grp"
        assert -1.0 <= st.min_rank_correlation <= 1.0
        assert 0.0 <= st.mean_topk_overlap <= 1.0
        assert st.verdict().split(":")[0] in {"STABLE", "DRIFTS", "INCONCLUSIVE"}

    def test_identical_segments_are_reported_stable(self, big_explanation):
        """Random assignment means no real between-segment structure."""
        rng = np.random.default_rng(4)
        seg = pd.Series(rng.choice(["a", "b"], 60))
        st = analyse_segment_stability(big_explanation, seg, min_size=10)
        assert st.min_rank_correlation > 0.5

    def test_inconclusive_when_no_level_is_large_enough(self, big_explanation):
        seg = pd.Series([f"lvl{i}" for i in range(60)])
        st = analyse_segment_stability(big_explanation, seg, min_size=30)
        assert st.verdict().startswith("INCONCLUSIVE")


class TestSegmentSummaryPerLevel:
    def test_summary_reports_every_level_individually(self, toy_model, toy_data):
        expl = explain_survshap(
            toy_model, toy_data.iloc[:60], toy_data, np.array([12.0, 36.0]),
            nsamples=48, n_background=12, seed=9,
        )
        seg = pd.Series(["a", "b", "c"] * 20)
        s = analyse_segment_stability(expl, seg, min_size=10).summary()
        assert set(s["rank_correlation_by_level"]) == {"a", "b", "c"}
        assert set(s["topk_overlap_by_level"]) == {"a", "b", "c"}
        k = min(5, len(expl.feature_names))   # toy model has only 4 features
        assert all(len(v) == k for v in s["topk_by_level"].values())
        assert s["min_rank_correlation"] == pytest.approx(
            min(s["rank_correlation_by_level"].values()), abs=1e-4
        )


class TestBootstrapSegmentRanks:
    """The bootstrap must be able to say 'no real difference' as readily as 'real'."""

    @staticmethod
    def _frame(shift_target: bool, n: int = 36, seed: int = 0):
        from creditsurv.explain.segments import bootstrap_segment_ranks  # noqa: F401

        rng = np.random.default_rng(seed)
        feats = [f"f{i}" for i in range(10)]
        base = np.linspace(1.0, 0.1, 10)          # f0 most important ... f9 least
        rows, seg = [], []
        for lvl in ["A", "B", "C", "G"]:
            mean = base.copy()
            if shift_target and lvl == "G":
                mean[1], mean[8] = base[8], base[1]   # f1 collapses, f8 rises
            rows.append(np.abs(rng.normal(mean, 0.05, size=(n, 10))))
            seg += [lvl] * n
        return pd.DataFrame(np.vstack(rows), columns=feats), pd.Series(seg)

    def test_real_shift_is_detected(self):
        from creditsurv.explain.segments import bootstrap_segment_ranks

        X, seg = self._frame(shift_target=True)
        out = bootstrap_segment_ranks(X, seg, target="G", reference_levels=["A", "B", "C"],
                                      features=["f1", "f8"], n_boot=400, seed=1)
        f1 = out.set_index("feature").loc["f1"]
        assert f1["rank_shift"] > 0 and f1["shift_ci_excludes_zero"]
        assert f1["p_outside_reference_range"] > 0.95
        assert f1["share_ci_excludes_one"] and f1["share_ratio_hi"] < 1

    def test_no_difference_is_reported_as_such(self):
        from creditsurv.explain.segments import bootstrap_segment_ranks

        X, seg = self._frame(shift_target=False)
        # adjacent near-tied features: the CI must straddle zero rather than
        # manufacture a difference
        X["f4"] = X["f5"] * np.random.default_rng(3).uniform(0.97, 1.03, len(X))
        out = bootstrap_segment_ranks(X, seg, target="G", reference_levels=["A", "B", "C"],
                                      features=["f4", "f5"], n_boot=400, seed=1)
        for _, r in out.iterrows():
            assert r["rank_shift_lo"] <= 0 <= r["rank_shift_hi"]
            assert not r["shift_ci_excludes_zero"]

    def test_point_estimate_matches_direct_computation(self):
        from creditsurv.explain.segments import bootstrap_segment_ranks

        X, seg = self._frame(shift_target=True)
        out = bootstrap_segment_ranks(X, seg, target="G", reference_levels=["A", "B", "C"],
                                      features=["f8"], n_boot=50, seed=1)
        direct = X[seg.to_numpy() == "G"].mean().rank(ascending=False)["f8"]
        assert out.iloc[0]["rank_target"] == direct

    def test_reproducible_under_seed(self):
        from creditsurv.explain.segments import bootstrap_segment_ranks

        X, seg = self._frame(shift_target=True)
        kw = dict(target="G", reference_levels=["A", "B"], features=["f1"], n_boot=100, seed=7)
        pd.testing.assert_frame_equal(bootstrap_segment_ranks(X, seg, **kw),
                                      bootstrap_segment_ranks(X, seg, **kw))

    def test_missing_level_raises(self):
        from creditsurv.explain.segments import bootstrap_segment_ranks

        X, seg = self._frame(shift_target=False)
        with pytest.raises(ValueError, match="no observations"):
            bootstrap_segment_ranks(X, seg, target="Z", reference_levels=["A"],
                                    features=["f1"], n_boot=10)
