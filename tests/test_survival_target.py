"""Tests for duration construction, date parsing and the exclusion audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditsurv.labeling.survival_target import (
    LabelConfig,
    build_survival_target,
    months_between,
    parse_month_series,
    parse_term_months,
)


class TestDateParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Dec-2018", "2018-12-01"),
            ("Jan-2015", "2015-01-01"),
            ("2018-12-01", "2018-12-01"),
            ("2018-12", "2018-12-01"),
            ("01-Dec-2018", "2018-12-01"),
            ("12/01/2018", "2018-12-01"),
            ("  Mar-2016  ", "2016-03-01"),
        ],
    )
    def test_parses_each_known_format(self, raw, expected):
        got = parse_month_series(pd.Series([raw]))
        assert got.iloc[0] == pd.Timestamp(expected)

    def test_mixed_formats_in_one_column(self):
        got = parse_month_series(pd.Series(["Dec-2018", "2015-06-01", "Jan-2010"]))
        assert list(got) == [
            pd.Timestamp("2018-12-01"),
            pd.Timestamp("2015-06-01"),
            pd.Timestamp("2010-01-01"),
        ]

    def test_unparsable_becomes_nat_not_an_exception(self):
        got = parse_month_series(pd.Series(["not-a-date", None, "", "Dec-2018"]))
        assert got.isna().tolist() == [True, True, True, False]

    def test_day_of_month_is_collapsed(self):
        got = parse_month_series(pd.Series(["2018-12-27"]))
        assert got.iloc[0].day == 1


class TestTermParsing:
    @pytest.mark.parametrize(
        "raw,expected", [(" 36 months", 36), ("60 months", 60), ("36", 36)]
    )
    def test_extracts_month_count(self, raw, expected):
        assert parse_term_months(pd.Series([raw])).iloc[0] == expected

    def test_missing_term_is_na(self):
        assert parse_term_months(pd.Series([None, "unknown"])).isna().all()


class TestMonthsBetween:
    def test_counts_whole_months(self):
        start = pd.Series([pd.Timestamp("2015-01-01")])
        end = pd.Series([pd.Timestamp("2015-11-01")])
        assert months_between(start, end).iloc[0] == 10

    def test_spans_year_boundaries(self):
        start = pd.Series([pd.Timestamp("2014-11-01")])
        end = pd.Series([pd.Timestamp("2016-02-01")])
        assert months_between(start, end).iloc[0] == 15

    def test_same_month_is_zero(self):
        s = pd.Series([pd.Timestamp("2016-07-01")])
        assert months_between(s, s).iloc[0] == 0

    def test_missing_propagates_as_na(self):
        start = pd.Series([pd.Timestamp("2015-01-01"), pd.NaT])
        end = pd.Series([pd.NaT, pd.Timestamp("2015-01-01")])
        assert months_between(start, end).isna().all()


class TestBuildSurvivalTarget:
    def test_event_duration_uses_last_payment(self, tiny_loans):
        out, _ = build_survival_target(tiny_loans)
        row = out.loc["event_month_10"]
        assert row["event"] == 1
        assert row["duration_months"] == 10

    def test_payoff_is_censored_at_payoff_date(self, tiny_loans):
        out, _ = build_survival_target(tiny_loans)
        row = out.loc["payoff_month_24"]
        assert row["event"] == 0
        assert row["duration_months"] == 24

    def test_current_loan_censored_at_inferred_cutoff(self, tiny_loans):
        out, audit = build_survival_target(tiny_loans)
        row = out.loc["current_admin_censor"]
        assert row["event"] == 0
        # Latest payment month anywhere in the fixture is Dec-2018, which is
        # the Current loan's own last payment.
        assert audit["data_cutoff"] == "2018-12-01"
        assert row["duration_months"] == 11  # Jan-2018 -> Dec-2018

    def test_explicit_cutoff_overrides_inference(self, tiny_loans):
        out, audit = build_survival_target(
            tiny_loans, LabelConfig(data_cutoff="2019-01")
        )
        assert audit["data_cutoff"] == "2019-01-01"
        assert audit["cutoff_source"] == "config"
        assert out.loc["current_admin_censor", "duration_months"] == 12

    def test_never_paid_charge_off_kept_at_floor_not_dropped(self, tiny_loans):
        out, audit = build_survival_target(tiny_loans)
        row = out.loc["event_never_paid"]
        assert row["event"] == 1
        assert row["duration_months"] == 1
        assert bool(row["never_paid"]) is True
        assert audit["n_never_paid"] == 1

    def test_same_month_payoff_is_floored_to_one(self, tiny_loans):
        out, audit = build_survival_target(tiny_loans)
        assert out.loc["same_month_payoff", "duration_months"] == 1
        assert audit["n_duration_floored"] >= 1

    def test_delinquent_statuses_are_censored_by_default(self, tiny_loans):
        out, _ = build_survival_target(tiny_loans)
        for scenario in ("late_31_120", "late_16_30", "grace_period"):
            assert out.loc[scenario, "event"] == 0

    def test_late_sensitivity_switch_flips_only_that_bucket(self, tiny_loans):
        out, _ = build_survival_target(
            tiny_loans, LabelConfig(late_31_120_is_event=True)
        )
        assert out.loc["late_31_120", "event"] == 1
        assert out.loc["late_16_30", "event"] == 0
        assert out.loc["grace_period", "event"] == 0

    def test_event_lag_shifts_events_only(self, tiny_loans):
        base, _ = build_survival_target(tiny_loans)
        lagged, _ = build_survival_target(tiny_loans, LabelConfig(event_lag_months=5))
        assert lagged.loc["event_month_10", "duration_months"] == 15
        assert (
            lagged.loc["payoff_month_24", "duration_months"]
            == base.loc["payoff_month_24", "duration_months"]
        )


class TestExclusions:
    def test_each_bad_row_is_dropped_with_the_right_reason(self, tiny_loans):
        out, audit = build_survival_target(tiny_loans)
        assert "policy_exception" not in out.index
        assert "bad_issue_date" not in out.index
        assert "missing_term" not in out.index
        assert "term_overrun" not in out.index
        assert audit["drop_counts"] == {
            "excluded_status": 1,
            "missing_issue_d": 1,
            "missing_term": 1,
            "term_overrun": 1,
        }

    def test_policy_exception_retained_when_enabled(self, tiny_loans):
        out, _ = build_survival_target(
            tiny_loans, LabelConfig(include_policy_exceptions=True)
        )
        assert out.loc["policy_exception", "event"] == 1

    def test_row_budget_reconciles(self, tiny_loans):
        _, audit = build_survival_target(tiny_loans)
        assert audit["n_retained"] + sum(audit["drop_counts"].values()) == audit["n_input"]

    def test_messy_whitespace_status_still_classified(self, tiny_loans):
        out, _ = build_survival_target(tiny_loans)
        assert out.loc["messy_whitespace", "event"] == 1
        assert out.loc["messy_whitespace", "duration_months"] == 6

    def test_term_overrun_grace_is_respected(self, tiny_loans):
        # Jan-2012 -> Jan-2018 is 72 months on a 36-month term: 36 over.
        loose, _ = build_survival_target(
            tiny_loans, LabelConfig(term_overrun_grace_months=48)
        )
        assert "term_overrun" in loose.index

    def test_no_duration_is_below_the_floor(self, tiny_loans):
        out, _ = build_survival_target(tiny_loans)
        assert (out["duration_months"] >= 1).all()

    def test_durations_are_plain_int32(self, tiny_loans):
        out, _ = build_survival_target(tiny_loans)
        assert out["duration_months"].dtype == np.int32


class TestAuditContent:
    def test_audit_reports_event_rate_and_config(self, tiny_loans):
        _, audit = build_survival_target(tiny_loans)
        assert 0.0 <= audit["event_rate"] <= 1.0
        assert audit["config"]["late_31_120_is_event"] is False
        assert audit["config"]["event_lag_months"] == 0

    def test_status_counts_cover_all_input_rows(self, tiny_loans):
        _, audit = build_survival_target(tiny_loans)
        assert sum(audit["status_counts"].values()) == len(tiny_loans)

    def test_unknown_status_surfaces_as_an_error(self, tiny_loans):
        broken = tiny_loans.copy()
        broken.loc["event_month_10", "loan_status"] = "Repossessed"
        with pytest.raises(ValueError, match="Unrecognised loan_status"):
            build_survival_target(broken)
