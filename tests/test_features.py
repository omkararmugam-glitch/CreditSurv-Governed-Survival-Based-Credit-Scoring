"""Tests for encoders, design-matrix construction and splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditsurv.features.build import (
    FeatureSpec,
    STRUCTURAL_MISSING,
    add_derived_features,
    build_design_matrix,
    default_spec,
    train_test_split_loans,
)
from creditsurv.features.encoders import (
    fico_midpoint,
    income_band,
    parse_emp_length,
    winsorize,
)
from creditsurv.io import schema as sch


class TestEncoders:
    @pytest.mark.parametrize(
        "raw,expected",
        [("< 1 year", 0.5), ("1 year", 1.0), ("5 years", 5.0), ("10+ years", 10.0)],
    )
    def test_emp_length_is_ordinal_not_nominal(self, raw, expected):
        assert parse_emp_length(pd.Series([raw])).iloc[0] == pytest.approx(expected)

    def test_emp_length_missing_stays_nan_not_zero(self):
        got = parse_emp_length(pd.Series([None, "unknown", ""]))
        assert got.isna().all(), "missing employment must not become zero years"

    def test_fico_midpoint_averages_the_band(self):
        got = fico_midpoint(pd.Series([700.0]), pd.Series([704.0]))
        assert got.iloc[0] == pytest.approx(702.0)

    def test_income_band_is_ordered_and_covers_extremes(self):
        got = income_band(pd.Series([10_000, 60_000, 5_000_000]))
        assert list(got.astype(str)) == ["<30k", "50-75k", "150k+"]

    def test_winsorize_clips_to_quantiles(self):
        s = pd.Series([0.0] * 10 + [1e9])
        got = winsorize(s, lower=0.0, upper=0.9)
        assert got.max() < 1e9


class TestDerivedFeatures:
    @pytest.fixture
    def frame(self):
        return pd.DataFrame(
            {
                "fico_range_low": [700.0, 660.0],
                "fico_range_high": [704.0, 664.0],
                "emp_length": ["5 years", "< 1 year"],
                "installment": [300.0, 600.0],
                "annual_inc": [60_000.0, 24_000.0],
                "loan_amnt": [10_000.0, 20_000.0],
            }
        )

    def test_adds_expected_columns(self, frame):
        out = add_derived_features(frame)
        for col in ("fico_midpoint", "emp_length_years", "installment_to_income",
                    "loan_to_income", "log_annual_inc", "income_band"):
            assert col in out.columns

    def test_installment_to_income_uses_monthly_income(self, frame):
        out = add_derived_features(frame)
        assert out["installment_to_income"].iloc[0] == pytest.approx(300 / (60_000 / 12))

    def test_zero_income_does_not_produce_inf(self):
        out = add_derived_features(
            pd.DataFrame({"installment": [300.0], "annual_inc": [0.0],
                          "loan_amnt": [1000.0]})
        )
        assert out["installment_to_income"].isna().all()
        assert not np.isinf(out["loan_to_income"]).any()

    def test_original_frame_is_not_mutated(self, frame):
        before = list(frame.columns)
        add_derived_features(frame)
        assert list(frame.columns) == before


class TestDefaultSpec:
    def test_excludes_lc_grade_by_default(self):
        cols = ["loan_amnt", "dti", "grade", "sub_grade", "int_rate", "purpose"]
        spec = default_spec(cols)
        assert "grade" not in spec.categorical
        assert "int_rate" not in spec.numeric

    def test_includes_lc_grade_when_requested(self):
        cols = ["loan_amnt", "dti", "grade", "sub_grade", "int_rate", "purpose"]
        spec = default_spec(cols, with_lc_grade=True)
        assert "grade" in spec.categorical and "int_rate" in spec.numeric

    def test_superseded_columns_are_excluded(self):
        spec = default_spec(["fico_range_low", "fico_range_high", "emp_length", "dti"])
        assert "fico_range_low" not in spec.numeric
        assert "emp_length" not in spec.categorical

    def test_leaking_column_is_silently_excluded_fail_closed(self):
        """The allowlist means an unknown or leaking column can never get in.

        It is excluded rather than raising, because the allowlist is the defence:
        anything not named in the feature tuples simply never becomes a feature.
        The loud error is reserved for a hand-built spec, covered by
        ``TestDesignMatrix.test_leakage_column_in_spec_is_refused``.
        """
        spec = default_spec(["loan_amnt", "recoveries", "dti"])
        assert "recoveries" not in spec.numeric
        assert "recoveries" not in spec.categorical
        assert not (set(spec.all_columns) & sch.LEAKAGE_COLUMNS)

    def test_intersects_with_available_columns(self):
        spec = default_spec(["loan_amnt", "dti"])
        assert set(spec.numeric) <= {"loan_amnt", "dti"}


class TestDesignMatrix:
    @pytest.fixture
    def frame(self, synthetic_survival_frame):
        df = synthetic_survival_frame.copy()
        df["fico_range_high"] = df["fico_range_low"] + 4
        df["emp_length"] = "5 years"
        df["home_ownership"] = "RENT"
        df["mths_since_last_delinq"] = np.where(
            np.arange(len(df)) % 2 == 0, np.nan, 12.0
        )
        return df

    def test_gbm_flavour_keeps_categories_and_nan(self, frame):
        dm = build_design_matrix(frame, flavour="gbm")
        assert dm.flavour == "gbm"
        assert any(str(dm.X[c].dtype) == "category" for c in dm.X.columns)
        assert dm.X.isna().any().any(), "LightGBM should receive NaN directly"

    def test_cox_flavour_is_numeric_and_complete(self, frame):
        dm = build_design_matrix(frame, flavour="cox")
        assert not dm.X.isna().any().any(), "Cox cannot accept NaN"
        assert all(np.issubdtype(dm.X[c].dtype, np.number) for c in dm.X.columns)

    def test_cox_standardises_numeric_columns(self, frame):
        dm = build_design_matrix(frame, flavour="cox")
        col = "dti"
        assert abs(float(dm.X[col].mean())) < 1e-4
        assert abs(float(dm.X[col].std()) - 1.0) < 1e-2

    def test_missing_indicators_are_added_before_filling(self, frame):
        dm = build_design_matrix(frame, flavour="cox")
        assert "mths_since_last_delinq_missing" in dm.X.columns
        assert dm.X["mths_since_last_delinq_missing"].sum() > 0

    def test_indicators_are_not_standardised(self, frame):
        dm = build_design_matrix(frame, flavour="cox")
        vals = set(dm.X["mths_since_last_delinq_missing"].unique())
        assert vals <= {0, 1}

    def test_structural_missing_list_is_respected(self, frame):
        assert "mths_since_last_delinq" in STRUCTURAL_MISSING

    def test_test_split_reuses_training_transforms(self, frame):
        tr, te = train_test_split_loans(frame, test_size=0.3, seed=5)
        dm_tr = build_design_matrix(frame.loc[tr], flavour="cox")
        dm_te = build_design_matrix(
            frame.loc[te], flavour="cox",
            standardisation=dm_tr.standardisation,
            fill_values=dm_tr.fill_values,
            reference_columns=list(dm_tr.X.columns),
        )
        assert list(dm_te.X.columns) == list(dm_tr.X.columns)

    def test_unseen_category_does_not_change_column_set(self, frame):
        tr = frame.iloc[:400]
        te = frame.iloc[400:].copy()
        te["purpose"] = "a_brand_new_purpose"
        dm_tr = build_design_matrix(tr, flavour="cox")
        dm_te = build_design_matrix(
            te, flavour="cox",
            standardisation=dm_tr.standardisation,
            fill_values=dm_tr.fill_values,
            reference_columns=list(dm_tr.X.columns),
        )
        assert list(dm_te.X.columns) == list(dm_tr.X.columns)

    def test_high_cardinality_categorical_is_dropped(self, frame):
        df = frame.copy()
        df["purpose"] = [f"p{i}" for i in range(len(df))]
        dm = build_design_matrix(df, flavour="cox", max_categories=10)
        assert "purpose" in dm.dropped

    def test_duration_and_event_survive(self, frame):
        dm = build_design_matrix(frame, flavour="gbm")
        assert len(dm.duration) == len(frame)
        assert set(dm.event.unique()) <= {0, 1}

    def test_leakage_column_in_spec_is_refused(self, frame):
        from creditsurv.features.build import FeatureSpec

        bad = FeatureSpec(numeric=("loan_amnt", "recoveries"), categorical=())
        with pytest.raises(sch.LeakageError):
            build_design_matrix(frame, bad, flavour="gbm")

    def test_invalid_flavour_rejected(self, frame):
        with pytest.raises(ValueError, match="flavour"):
            build_design_matrix(frame, flavour="neural")


class TestSplit:
    def test_random_split_sizes(self, synthetic_survival_frame):
        tr, te = train_test_split_loans(synthetic_survival_frame, test_size=0.25, seed=1)
        n = len(synthetic_survival_frame)
        assert len(te) == round(n * 0.25)
        assert len(tr) + len(te) == n

    def test_split_is_disjoint(self, synthetic_survival_frame):
        tr, te = train_test_split_loans(synthetic_survival_frame, seed=1)
        assert not set(tr) & set(te)

    def test_random_split_is_reproducible(self, synthetic_survival_frame):
        a = train_test_split_loans(synthetic_survival_frame, seed=42)
        b = train_test_split_loans(synthetic_survival_frame, seed=42)
        assert list(a[0]) == list(b[0]) and list(a[1]) == list(b[1])

    def test_out_of_time_holds_out_latest_vintages(self, synthetic_survival_frame):
        df = synthetic_survival_frame.copy()
        df["issue_year"] = np.repeat([2015, 2016, 2017, 2018], len(df) // 4)[: len(df)]
        tr, te = train_test_split_loans(df, scheme="out_of_time", out_of_time_cutoff=2018)
        assert df.loc[te, "issue_year"].min() >= 2018
        assert df.loc[tr, "issue_year"].max() < 2018

    def test_out_of_time_requires_year_column(self, synthetic_survival_frame):
        with pytest.raises(ValueError, match="issue_year"):
            train_test_split_loans(synthetic_survival_frame, scheme="out_of_time")

    def test_unknown_scheme_rejected(self, synthetic_survival_frame):
        with pytest.raises(ValueError, match="unknown scheme"):
            train_test_split_loans(synthetic_survival_frame, scheme="kfold")


class TestSpecFlagsForVariantModels:
    """The Stage 2 flags that make an unpriced or score-aware variant possible.

    Both must leave the default spec untouched: the models in sections 2-7b were
    fitted on it, and a silent change here would make those results describe a
    model nobody trained.
    """

    def _frame(self):
        import numpy as np
        return pd.DataFrame({
            "loan_amnt": [10000.0, 20000.0], "installment": [300.0, 600.0],
            "annual_inc": [60000.0, 90000.0], "dti": [12.0, 20.0],
            "fico_range_low": [700.0, 660.0], "fico_range_high": [704.0, 664.0],
            "emp_length": ["5 years", "10+ years"], "open_acc": [5.0, 9.0],
            "revol_bal": [1000.0, 2000.0], "purpose": ["car", "car"],
            "home_ownership": ["RENT", "OWN"], "addr_state": ["CA", "TX"],
            "verification_status": ["Verified", "Verified"],
            "application_type": ["Individual", "Individual"],
            "initial_list_status": ["w", "w"],
        })

    def test_default_spec_excludes_the_derived_features(self):
        """Documents the defect recorded in FINDINGS 7c rather than hiding it."""
        spec = default_spec(self._frame().columns)
        for col in ("fico_midpoint", "emp_length_years", "installment_to_income",
                    "loan_to_income", "log_annual_inc"):
            assert col not in spec.all_columns
        assert "fico_range_low" not in spec.all_columns      # dropped as superseded

    def test_building_the_spec_after_deriving_recovers_them(self):
        df = self._frame()
        spec = default_spec(add_derived_features(df).columns)
        for col in ("fico_midpoint", "emp_length_years", "installment_to_income",
                    "loan_to_income", "log_annual_inc"):
            assert col in spec.numeric
        # And the design matrix built from it actually carries the score.
        df["duration_months"], df["event"] = [12, 24], [0, 1]
        dm = build_design_matrix(df, spec, flavour="gbm")
        assert "fico_midpoint" in dm.X.columns
        assert dm.X["fico_midpoint"].tolist() == [702.0, 662.0]

    def test_an_unpriced_spec_keeps_the_score_and_loses_the_rate(self):
        """What --with-derived --drop-features installment,installment_to_income
        produces: nothing computed from the lender's assigned rate."""
        spec = default_spec(add_derived_features(self._frame()).columns)
        wanted = ("installment", "installment_to_income")
        unpriced = FeatureSpec(
            numeric=tuple(c for c in spec.numeric if c not in wanted),
            categorical=spec.categorical,
            structural_missing=tuple(c for c in spec.structural_missing
                                     if c not in wanted))
        assert "fico_midpoint" in unpriced.numeric
        assert not set(wanted) & set(unpriced.all_columns)
        assert "loan_amnt" in unpriced.numeric          # the amount is not pricing
        for lender_field in ("grade", "sub_grade", "int_rate"):
            assert lender_field not in unpriced.all_columns
