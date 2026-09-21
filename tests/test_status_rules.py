"""Tests for the event / censoring decision table.

This is the highest-stakes logic in the project: the mapping determines the
event rate, which determines every metric downstream. It is therefore tested
exhaustively rather than by sampling.
"""

from __future__ import annotations

import pytest

from creditsurv.labeling.status_rules import (
    PRIMARY_STATUS_MAP,
    Outcome,
    UnknownStatusError,
    classify_status,
    is_event,
    known_statuses,
    normalize_status,
    strip_policy_exception,
)


class TestNormalize:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Charged Off", "charged off"),
            ("  charged off  ", "charged off"),
            ("CHARGED OFF", "charged off"),
            ("Charged Off", "charged off"),
            ("Late (31–120 days)", "late (31-120 days)"),
            ("Late  (16-30   days)", "late (16-30 days)"),
            (None, ""),
            ("", ""),
        ],
    )
    def test_normalizes_real_world_variants(self, raw, expected):
        assert normalize_status(raw) == expected


class TestPolicyExceptionPrefix:
    def test_splits_prefix_and_flags(self):
        underlying, flagged = strip_policy_exception(
            "does not meet the credit policy. status:charged off"
        )
        assert (underlying, flagged) == ("charged off", True)

    def test_plain_status_unflagged(self):
        assert strip_policy_exception("charged off") == ("charged off", False)


class TestClassify:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Charged Off", Outcome.EVENT),
            ("Default", Outcome.EVENT),
            ("Fully Paid", Outcome.CENSORED_PAYOFF),
            ("Current", Outcome.CENSORED_ADMIN),
            ("Issued", Outcome.CENSORED_ADMIN),
            ("In Grace Period", Outcome.CENSORED_DELINQUENT),
            ("Late (16-30 days)", Outcome.CENSORED_DELINQUENT),
            ("Late (31-120 days)", Outcome.CENSORED_DELINQUENT),
        ],
    )
    def test_primary_specification(self, raw, expected):
        assert classify_status(raw) is expected

    def test_every_status_in_the_real_file_is_covered(self, status_examples):
        """No status in the real file may fall through to an error."""
        for raw in status_examples:
            assert isinstance(classify_status(raw), Outcome)

    def test_exactly_two_statuses_are_events_by_default(self, status_examples):
        events = {s for s in status_examples if is_event(classify_status(s))}
        assert events == {"Charged Off", "Default"}

    def test_unknown_status_raises_rather_than_defaulting_to_censored(self):
        with pytest.raises(UnknownStatusError, match="Unrecognised loan_status"):
            classify_status("Repossessed")

    def test_blank_status_is_excluded_not_censored(self):
        assert classify_status(None) is Outcome.EXCLUDED
        assert classify_status("") is Outcome.EXCLUDED


class TestPolicyExceptions:
    def test_excluded_by_default(self):
        assert (
            classify_status("Does not meet the credit policy. Status:Charged Off")
            is Outcome.EXCLUDED
        )
        assert (
            classify_status("Does not meet the credit policy. Status:Fully Paid")
            is Outcome.EXCLUDED
        )

    def test_mapped_by_underlying_status_when_included(self):
        assert (
            classify_status(
                "Does not meet the credit policy. Status:Charged Off",
                include_policy_exceptions=True,
            )
            is Outcome.EVENT
        )
        assert (
            classify_status(
                "Does not meet the credit policy. Status:Fully Paid",
                include_policy_exceptions=True,
            )
            is Outcome.CENSORED_PAYOFF
        )


class TestLateSensitivitySwitch:
    def test_late_31_120_becomes_event(self):
        assert (
            classify_status("Late (31-120 days)", late_31_120_is_event=True)
            is Outcome.EVENT
        )

    def test_switch_does_not_touch_other_delinquent_buckets(self):
        for raw in ("Late (16-30 days)", "In Grace Period"):
            assert (
                classify_status(raw, late_31_120_is_event=True)
                is Outcome.CENSORED_DELINQUENT
            )

    def test_switch_does_not_touch_performing_loans(self):
        assert classify_status("Current", late_31_120_is_event=True) is Outcome.CENSORED_ADMIN
        assert (
            classify_status("Fully Paid", late_31_120_is_event=True)
            is Outcome.CENSORED_PAYOFF
        )


class TestTableIntegrity:
    def test_map_keys_are_already_normalized(self):
        for key in PRIMARY_STATUS_MAP:
            assert normalize_status(key) == key

    def test_no_status_maps_to_excluded_in_the_primary_table(self):
        assert Outcome.EXCLUDED not in set(PRIMARY_STATUS_MAP.values())

    def test_known_statuses_grows_with_policy_exceptions(self):
        base = known_statuses()
        extended = known_statuses(include_policy_exceptions=True)
        assert base < extended
        assert len(extended) == 2 * len(base)
