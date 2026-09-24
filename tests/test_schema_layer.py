"""The layer that works out what an uploaded file contains.

Recognition proposes, never applies silently; content vetoes a name that lies;
derivable columns are computed exactly rather than imputed; a missing optional
feature is scored with its measured cost; a missing required one is refused.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditsurv.batch import validate
from creditsurv.cleaning import fit_values
from creditsurv.derive import (DERIVATIONS, REQUIRED_DROP, FeatureCosts,
                               derive_features, load_costs)
from creditsurv.features.build import FeatureSpec
from creditsurv.schema_match import normalise, propose_mapping

SPEC = FeatureSpec(
    numeric=("loan_amnt", "installment", "annual_inc", "dti", "open_acc",
             "revol_bal", "inq_last_6mths", "fico_midpoint", "term_months"),
    categorical=("purpose", "home_ownership", "addr_state"),
    structural_missing=())


def training_frame(n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "loan_amnt": rng.uniform(1000, 35000, n),
        "installment": rng.uniform(50, 1200, n),
        "annual_inc": rng.lognormal(11, 0.4, n),
        "dti": rng.uniform(1, 38, n),
        "open_acc": rng.integers(1, 25, n).astype(float),
        "revol_bal": rng.uniform(0, 50000, n),
        "inq_last_6mths": rng.integers(0, 6, n).astype(float),
        "fico_midpoint": rng.uniform(620, 820, n),
        "term_months": rng.choice([36, 60], n).astype(float),
        "purpose": rng.choice(["car", "credit_card", "debt_consolidation"], n),
        "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n),
        "addr_state": rng.choice(["CA", "TX", "NY"], n)})


@pytest.fixture
def values():
    return fit_values(training_frame(), SPEC, source="train.parquet")


# ----------------------------------------------------------- E2 recognition --

def test_normalise_keeps_tokens_that_matter():
    assert normalise("Open Credit Lines (count)") == "opencreditlines"
    assert normalise("inquiries_last_6m") == "inquirieslast6m"   # the 'a' survives
    assert normalise("LoanAmount") == "amount"                   # camel case split
    assert normalise("state") == "state"


def test_renamed_columns_are_recognised_with_reasons(values):
    df = pd.DataFrame({
        "LoanAmount": [10000.0], "MonthlyPayment": [320.0],
        "annual_income": [65000.0], "debt_to_income": [14.0],
        "open_credit_lines": [7.0], "revolving_balance": [4200.0],
        "inquiries_last_6m": [1.0], "fico_midpoint": [702.0],
        "term": ["36 months"], "loan_purpose": ["car"],
        "home_ownership": ["RENT"], "state": ["CA"]})
    proposal = propose_mapping(df, SPEC, values=values)
    mapping = proposal.mapping()
    for uploaded, feature in [("LoanAmount", "loan_amnt"),
                              ("MonthlyPayment", "installment"),
                              ("annual_income", "annual_inc"),
                              ("debt_to_income", "dti"),
                              ("open_credit_lines", "open_acc"),
                              ("revolving_balance", "revol_bal"),
                              ("inquiries_last_6m", "inq_last_6mths"),
                              ("loan_purpose", "purpose"), ("state", "addr_state")]:
        assert mapping.get(uploaded) == feature, (uploaded, mapping.get(uploaded))
    frame = proposal.to_frame()
    assert (frame["confidence"] > 0).sum() >= 9
    assert frame["why"].str.len().gt(0).all()            # every row says why


def test_a_misleading_name_is_caught_by_content(values):
    """A column called annual_income holding values between 0 and 1 is not an income,
    whatever its name says, so the name match is vetoed."""
    df = pd.DataFrame({
        "loan_amnt": [10000.0] * 20, "dti": [12.0] * 20,
        "annual_income": np.linspace(0.01, 0.99, 20)})
    proposal = propose_mapping(df, SPEC, values=values)
    match = next(m for m in proposal.matches if m.column == "annual_income")
    assert match.best is None
    assert "annual_income" not in proposal.mapping()
    assert any("annual_inc" in v and "do not look like this feature" in v
               for v in match.vetoed), match.vetoed


def test_a_column_this_model_cannot_use_is_named_as_such(values):
    """credit_score is understood, and then reported as useless to a model whose
    spec has no score feature -- which is more informative than 'unknown column'."""
    df = pd.DataFrame({"loan_amnt": [10000.0], "annual_inc": [60000.0],
                       "dti": [12.0], "credit_score": [700]})
    proposal = propose_mapping(df, SPEC, values=values)
    assert "credit_score" in proposal.recognised_but_unused
    assert "does not use" in proposal.recognised_but_unused["credit_score"]
    assert "credit_score" not in proposal.mapping()


def test_two_columns_for_one_feature_are_never_mapped_silently(values):
    df = pd.DataFrame({"loan_amount": [10000.0], "amount_requested": [10000.0],
                       "annual_inc": [60000.0], "dti": [12.0]})
    proposal = propose_mapping(df, SPEC, values=values)
    assert "loan_amnt" in proposal.conflicts
    assert set(proposal.conflicts["loan_amnt"]) == {"loan_amount", "amount_requested"}
    assert "loan_amount" not in proposal.mapping()
    assert "amount_requested" not in proposal.mapping()


def test_category_levels_identify_a_column_whatever_it_is_called(values):
    df = pd.DataFrame({"housing_situation": ["RENT", "OWN", "MORTGAGE", "RENT"],
                       "loan_amnt": [1000.0] * 4, "annual_inc": [50000.0] * 4,
                       "dti": [10.0] * 4})
    proposal = propose_mapping(df, SPEC, values=values)
    assert proposal.mapping().get("housing_situation") == "home_ownership"


# ------------------------------------------------------------ E3 derivation --

def test_installment_is_derived_exactly():
    """Standard amortisation, against the closed form computed here."""
    df = pd.DataFrame({"loan_amnt": [10000.0], "int_rate": ["13.5%"],
                       "term": ["36 months"]})
    out, derived, blocked = derive_features(df, ["installment", "term_months"])
    assert derived["term_months"] and derived["installment"]
    rate = 0.135 / 12
    expected = 10000.0 * rate / (1 - (1 + rate) ** -36)
    assert out["installment"].iloc[0] == pytest.approx(expected, rel=1e-12)
    assert out["term_months"].iloc[0] == 36
    assert not blocked


def test_fico_midpoint_and_the_band_top_are_derived():
    df = pd.DataFrame({"fico_range_low": [700.0, 660.0]})
    out, derived, _ = derive_features(df, ["fico_midpoint", "fico_range_high"])
    assert out["fico_range_high"].tolist() == [704.0, 664.0]
    assert out["fico_midpoint"].tolist() == [702.0, 662.0]
    assert set(derived) == {"fico_range_high", "fico_midpoint"}


def test_a_derivation_says_what_it_would_have_needed():
    df = pd.DataFrame({"loan_amnt": [10000.0]})          # no rate, no term
    _, derived, blocked = derive_features(df, ["installment"])
    assert "installment" not in derived
    assert set(blocked["installment"]) == {"int_rate", "term_months"}


def test_derivations_never_invent_a_value():
    """Arithmetic on columns that are present, never a fill: all-missing inputs
    produce nothing rather than a number."""
    for rule in DERIVATIONS:
        assert rule.needs, rule.feature
        df = pd.DataFrame({c: [np.nan] for c in rule.needs})
        _, derived, _ = derive_features(df, [rule.feature])
        assert rule.feature not in derived


# ------------------------------------------------- E4/E5 tiers and messages --

def test_costs_split_features_into_tiers():
    costs = FeatureCosts(model_tag="full", baseline_concordance=0.69,
                         per_feature={"installment": 0.0187, "loan_amnt": 0.0169,
                                      "annual_inc": 0.0167, "bc_util": 0.0044,
                                      "tax_liens": 0.0001})
    assert costs.tier("installment") == "required"
    assert costs.tier("bc_util") == "optional_costed"
    assert costs.tier("tax_liens") == "optional_free"
    assert costs.tier("not_measured") == "unmeasured"
    assert costs.required() == ["annual_inc", "installment", "loan_amnt"]
    assert costs.cost_of(["bc_util", "tax_liens"]) == pytest.approx(0.0045)
    assert "at most 0.0045" in costs.describe(["bc_util", "tax_liens"])


def test_the_real_ablation_table_is_readable():
    costs = load_costs("full")
    assert costs.per_feature, "the full-model ablation should be on disk"
    assert costs.baseline_concordance > 0.6
    assert REQUIRED_DROP == 0.010
    assert set(costs.required()) >= {"loan_amnt", "annual_inc"}


def test_optional_missing_is_scored_with_its_cost(values):
    costs = FeatureCosts(per_feature={"revol_bal": 0.0033, "open_acc": 0.0009})
    df = pd.DataFrame({"loan_amnt": [10000.0], "installment": [320.0],
                       "annual_inc": [60000.0], "dti": [12.0],
                       "inq_last_6mths": [1.0], "fico_midpoint": [700.0],
                       "term_months": [36.0], "purpose": ["car"],
                       "home_ownership": ["RENT"], "addr_state": ["CA"]})
    _, report = validate(df, SPEC, values=values, costs=costs)
    assert report.required_missing == []
    assert set(report.optional_missing) == {"open_acc", "revol_bal"}
    assert "revol_bal" in report.message()
    assert "not in file" in report.message()
    assert "at most 0.0042" in costs.describe(report.optional_missing)


def test_a_required_feature_missing_is_reported_with_the_reason(values):
    costs = FeatureCosts(per_feature={"annual_inc": 0.0167, "loan_amnt": 0.0169})
    df = pd.DataFrame({"loan_amnt": [10000.0], "dti": [12.0], "purpose": ["car"]})
    _, report = validate(df, SPEC, values=values, costs=costs)
    assert "annual_inc" in report.required_missing
    assert "annual_inc (not in file)" in report.message()


def test_the_message_names_what_a_derivation_needed(values):
    costs = FeatureCosts(per_feature={"installment": 0.0187})
    df = pd.DataFrame({"loan_amnt": [10000.0], "annual_inc": [60000.0],
                       "dti": [12.0], "purpose": ["car"], "term": ["36 months"]})
    _, report = validate(df, SPEC, values=values, costs=costs)
    message = report.message()
    # 'term' is recognised as term_months rather than derived from it -- a rename is
    # better evidence than arithmetic -- so the instalment now needs only the rate.
    assert "Recognised: term -> term_months" in message
    assert "installment (needs int_rate)" in message


def test_the_simplified_name_file_is_understood(values):
    """outputs/data/synthetic_applicants.csv, the file with friendly column names."""
    df = pd.read_csv("outputs/data/synthetic_applicants.csv")
    cleaned, report = validate(df, SPEC, values=values, costs=FeatureCosts())
    assert report.recognised.get("open_credit_lines") == "open_acc"
    assert report.recognised.get("loan_purpose") == "purpose"
    assert report.recognised.get("state") == "addr_state"
    assert report.recognised.get("inquiries_last_6m") == "inq_last_6mths"
    for feature in ("loan_amnt", "annual_inc", "dti", "open_acc", "purpose",
                    "addr_state"):
        assert feature in cleaned.columns
    assert "Recognised:" in report.message()
