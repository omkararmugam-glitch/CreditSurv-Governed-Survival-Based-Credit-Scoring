"""The shared cleaning module: one set of rules, fitted on training data only.

The parity tests are the important ones. They pin two properties the project
depends on: the same rows come out of training and scoring identically, and the
default policy changes no value -- so wiring cleaning into Stage 2 cannot have
moved the data the current models were fitted on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditsurv.cleaning import (NA_TOKENS, PARITY_VERSION, CleaningPolicy,
                                 CleaningValues, clean, coerce_numeric, fit_values)
from creditsurv.config import CleaningConfig, Config, load_config
from creditsurv.features.build import FeatureSpec

NUMERIC = ("loan_amnt", "annual_inc", "dti", "revol_util")
CATEGORICAL = ("purpose", "home_ownership")
SPEC = FeatureSpec(numeric=NUMERIC, categorical=CATEGORICAL, structural_missing=())


def training_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": np.arange(n),
        "loan_amnt": rng.uniform(1000, 35000, n).astype("float32"),
        "annual_inc": rng.lognormal(11, 0.4, n).astype("float32"),
        "dti": rng.uniform(1, 38, n).astype("float32"),
        "revol_util": rng.uniform(0, 100, n).astype("float32"),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "car"], n),
        "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n),
    })


@pytest.fixture
def values():
    return fit_values(training_frame(), SPEC, source="training.parquet")


# ------------------------------------------------------- parity / single source --

def test_training_and_scoring_clean_the_same_rows_identically(values):
    """The same 40 rows, cleaned on the training path and on the scoring path,
    come out byte-for-byte equal. This is the property that made the two paths
    disagree before cleaning.py existed."""
    train = training_frame()
    rows = train.iloc[:40].copy()

    cleaned_training, _, _ = clean(train, SPEC, values)
    cleaned_scoring, _, _ = clean(rows, SPEC, values)

    for col in NUMERIC + CATEGORICAL:
        pd.testing.assert_series_equal(
            cleaned_training[col].iloc[:40].reset_index(drop=True),
            cleaned_scoring[col].reset_index(drop=True),
            check_dtype=True, check_categorical=True)


def test_default_policy_changes_no_value(values):
    """v1-parity is what the trained models were built under: cleaning may read,
    count and flag, but not alter. If this fails, wiring cleaning into Stage 2
    silently changed the training data."""
    train = training_frame()
    cleaned, report, _ = clean(train, SPEC, values)
    assert not CleaningPolicy().changes_model_inputs()
    assert report.rows_in == report.rows_out == len(train)
    assert not report.clipped and not report.imputed and not report.pooled_to_other
    for col in NUMERIC:
        pd.testing.assert_series_equal(cleaned[col], train[col], check_dtype=False)
    for col in CATEGORICAL:
        assert list(cleaned[col].astype("string")) == list(train[col].astype("string"))


def test_config_defaults_match_the_parity_policy():
    cfg = load_config("config/config.yaml")
    assert cfg.cleaning == CleaningConfig()          # nothing enabled by accident
    assert cfg.cleaning.version == PARITY_VERSION
    assert not cfg.cleaning.clip_numeric
    assert not cfg.cleaning.unseen_category_to_other
    assert cfg.cleaning.rare_category_min_count == 0
    assert not cfg.cleaning.drop_duplicate_ids


# ------------------------------------------------- learned values come from train --

def test_values_are_fitted_on_training_data_only(values):
    """An upload whose values are wildly different must not move the reference
    ranges: they are the training ranges, whatever the file contains."""
    upload = pd.DataFrame({
        "loan_amnt": [999_999.0] * 20, "annual_inc": [50_000_000.0] * 20,
        "dti": [900.0] * 20, "revol_util": [-50.0] * 20,
        "purpose": ["moon_landing"] * 20, "home_ownership": ["CASTLE"] * 20})
    before = CleaningValues.from_dict(values.to_dict())
    clean(upload, SPEC, values)
    assert values.to_dict() == before.to_dict()       # untouched by scoring
    assert values.fitted_rows == 300
    assert values.ranges["annual_inc"][1] < 1e6
    # Every recorded value traces back to the training frame's own distribution.
    train = training_frame()
    lo, hi = values.ranges["dti"]
    assert train["dti"].min() <= lo <= hi <= train["dti"].max()


def test_clean_never_refits_categories(values):
    upload = pd.DataFrame({
        "loan_amnt": [5000.0], "annual_inc": [60000.0], "dti": [12.0],
        "revol_util": [30.0], "purpose": ["wedding"], "home_ownership": ["RENT"]})
    cleaned, _, _ = clean(upload, SPEC, values)
    assert list(cleaned["purpose"].cat.categories) == values.categories["purpose"]
    assert "wedding" not in values.categories["purpose"]


# ------------------------------------------------------------ value-level rules --

@pytest.mark.parametrize("raw,expected,rescued_by_stripping", [
    ("$85,000", 85000.0, 1),     # currency and separator: would be NaN without it
    ("45.6%", 45.6, 1),          # percentage: likewise
    ("1,234.5", 1234.5, 1),      # thousands separator: likewise
    (" 18.2 ", 18.2, 0),         # whitespace only: parses either way
])
def test_numeric_text_is_read_the_way_ingest_reads_it(raw, expected,
                                                      rescued_by_stripping):
    """The divergence that made $85,000 score as missing while training read it
    as 85000."""
    got, rescued = coerce_numeric(pd.Series([raw]))
    assert got.iloc[0] == pytest.approx(expected, rel=1e-5)
    assert rescued == rescued_by_stripping


@pytest.mark.parametrize("token", NA_TOKENS[1:])
def test_missing_tokens_become_missing(token):
    got, _ = coerce_numeric(pd.Series([token]))
    assert pd.isna(got.iloc[0])


def test_unreadable_numbers_are_counted_and_treated_as_missing(values):
    upload = pd.DataFrame({"loan_amnt": ["not a number", "5000"],
                           "annual_inc": [60000.0, 60000.0], "dti": [10.0, 10.0],
                           "revol_util": [30.0, 30.0],
                           "purpose": ["car", "car"],
                           "home_ownership": ["RENT", "RENT"]})
    cleaned, report, flags = clean(upload, SPEC, values)
    assert pd.isna(cleaned["loan_amnt"].iloc[0])
    assert report.unreadable_numbers["loan_amnt"] == 1
    assert flags["n_unreadable_numbers"].iloc[0] == 1
    assert flags["n_unreadable_numbers"].iloc[1] == 0


def test_unseen_category_is_flagged_and_left_missing_by_default(values):
    upload = pd.DataFrame({"loan_amnt": [5000.0, 6000.0], "annual_inc": [60000.0, 70000.0],
                           "dti": [10.0, 11.0], "revol_util": [30.0, 40.0],
                           "purpose": ["car", "moon_landing"],
                           "home_ownership": ["RENT", "RENT"]})
    cleaned, report, flags = clean(upload, SPEC, values)
    assert report.unseen_categories["purpose"] == {"moon_landing": 1}
    assert pd.isna(cleaned["purpose"].iloc[1])        # unknown level -> missing
    assert cleaned["purpose"].iloc[0] == "car"
    assert flags["n_unknown_categories"].to_list() == [0, 1]
    assert "purpose" in flags["unknown_categories_fields"].iloc[1]


def test_unseen_category_to_other_is_opt_in(values):
    upload = pd.DataFrame({"loan_amnt": [5000.0], "annual_inc": [60000.0], "dti": [10.0],
                           "revol_util": [30.0], "purpose": ["moon_landing"],
                           "home_ownership": ["RENT"]})
    policy = CleaningPolicy(unseen_category_to_other=True)
    assert policy.changes_model_inputs()
    cleaned, report, _ = clean(upload, SPEC, values, policy=policy)
    assert cleaned["purpose"].iloc[0] == "other"
    assert report.unseen_categories["purpose"] == {"moon_landing": 1}


def test_out_of_range_values_are_flagged_not_changed(values):
    upload = pd.DataFrame({"loan_amnt": [5000.0], "annual_inc": [50_000_000.0],
                           "dti": [12.0], "revol_util": [30.0],
                           "purpose": ["car"], "home_ownership": ["RENT"]})
    cleaned, report, flags = clean(upload, SPEC, values)
    assert cleaned["annual_inc"].iloc[0] == pytest.approx(50_000_000.0)   # unchanged
    assert report.out_of_range["annual_inc"] == 1
    assert flags["n_out_of_range"].iloc[0] == 1
    assert "annual_inc" in flags["out_of_range_fields"].iloc[0]
    assert not report.clipped


def test_clipping_is_opt_in_and_reported(values):
    upload = pd.DataFrame({"loan_amnt": [5000.0], "annual_inc": [50_000_000.0],
                           "dti": [12.0], "revol_util": [30.0],
                           "purpose": ["car"], "home_ownership": ["RENT"]})
    policy = CleaningPolicy(clip_numeric=True)
    cleaned, report, _ = clean(upload, SPEC, values, policy=policy)
    assert cleaned["annual_inc"].iloc[0] == pytest.approx(
        values.clip_bounds["annual_inc"][1], rel=1e-4)
    assert report.clipped["annual_inc"] == 1


def test_absent_columns_are_reported_not_invented(values):
    upload = pd.DataFrame({"loan_amnt": [5000.0], "purpose": ["car"]})
    cleaned, report, _ = clean(upload, SPEC, values)
    assert set(report.missing_columns) == {"annual_inc", "dti", "revol_util",
                                          "home_ownership"}
    assert "annual_inc" not in cleaned.columns


def test_duplicate_ids_are_counted_and_kept_by_default(values):
    upload = training_frame(6)
    upload.loc[5, "id"] = upload.loc[0, "id"]
    _, report, _ = clean(upload, SPEC, values)
    assert report.dropped_by_rule == {"duplicate_id_kept": 1}
    assert report.rows_out == 6

    dropped, report2, _ = clean(upload, SPEC, values,
                                policy=CleaningPolicy(drop_duplicate_ids=True))
    assert report2.dropped_by_rule == {"duplicate_id": 1}
    assert report2.rows_out == 5 == len(dropped)


# ------------------------------------------------------------------- reporting --

def test_report_round_trips_and_reads_as_sentences(values):
    upload = pd.DataFrame({"loan_amnt": ["$5,000"], "annual_inc": [50_000_000.0],
                           "dti": [12.0], "revol_util": [30.0],
                           "purpose": ["moon_landing"], "home_ownership": ["RENT"]})
    _, report, _ = clean(upload, SPEC, values)
    payload = report.to_dict()
    assert payload["policy_version"] == PARITY_VERSION
    assert payload["values_fitted_rows"] == 300
    frame = report.to_frame()
    assert {"scope", "rule", "column", "count"} <= set(frame.columns)
    assert (frame["rule"] == "rows_in").any()
    sentences = report.plain_english()
    assert any("outside the range seen in training" in s for s in sentences)
    assert any("categories the model never saw" in s for s in sentences)


def test_label_stage_drops_are_carried_into_the_report(values):
    audit = {"drop_counts": {"excluded_status": 1_234, "missing_term": 7}}
    _, report, _ = clean(training_frame(10), SPEC, values, label_audit=audit)
    assert report.label_stage_drops == {"excluded_status": 1234, "missing_term": 7}
    assert (report.to_frame()["rule"] == "label_stage_dropped").sum() == 2
