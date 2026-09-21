"""Column contracts for the Lending Club extracts.

The accepted-loans file has ~151 columns, and roughly 40 of them are recorded
*after* origination. Several are near-perfect proxies for the outcome
(``recoveries``, ``total_rec_prncp``, ``last_fico_range_low``), so letting one
slip into a feature matrix produces a model with a c-index near 1.0 that is
entirely worthless. This is the most common way Lending Club models go wrong.

The defence here is an **allowlist**, not a blocklist: a column reaches a model
only if it is named in one of the feature tuples below. New or renamed columns in
a future extract therefore fail closed (silently ignored as features) rather than
open. :data:`LEAKAGE_COLUMNS` is kept as well, but only so that
:func:`assert_no_leakage` can give a loud, specific error and so the exclusion is
documented rather than implicit.

A note on vintage coverage: the ``num_*``, ``mo_sin_*``, ``open_il_*`` and
``total_bal_il`` families were only added to Lending Club's reporting around
2012. They are almost entirely missing for 2007-2011 loans. They are therefore
kept in :data:`EXTENDED_NUMERIC` rather than :data:`CORE_NUMERIC`, so a model can
be fit on the full date range without ~40 columns that are structurally absent
for early vintages.
"""

from __future__ import annotations

__all__ = [
    "ID_COLUMNS",
    "LABEL_COLUMNS",
    "LC_ASSESSMENT_COLUMNS",
    "CORE_NUMERIC",
    "CORE_CATEGORICAL",
    "EXTENDED_NUMERIC",
    "LEAKAGE_COLUMNS",
    "REJECTED_COLUMNS",
    "REJECTED_RENAMES",
    "COMMON_FEATURE_MAP",
    "ingest_columns",
    "feature_columns",
    "assert_no_leakage",
    "LeakageError",
]


ID_COLUMNS: tuple[str, ...] = ("id",)

LABEL_COLUMNS: tuple[str, ...] = (
    "issue_d",
    "loan_status",
    "term",
    "last_pymnt_d",
)
"""Needed to build the survival target.

``last_pymnt_d`` is label-construction input only. It is genuine leakage as a
feature -- it encodes when the loan stopped performing -- and is dropped by
:func:`feature_columns`.
"""

LC_ASSESSMENT_COLUMNS: tuple[str, ...] = ("grade", "sub_grade", "int_rate")
"""Lending Club's *own* risk assessment.

Excluded from the primary model by decision: these are the output of another
model, so including them means partly predicting Lending Club's underwriter
rather than default itself. They also make adverse-action reasons circular --
"your grade was low" is not a permissible ECOA/Reg B reason. Retained here so
the ``with_lc_grade`` benchmark variant can quantify what the signal was worth.
"""

CORE_NUMERIC: tuple[str, ...] = (
    # Loan structure at origination
    "loan_amnt",
    "funded_amnt",
    "installment",
    # Borrower capacity
    "annual_inc",
    "dti",
    # Bureau score at origination
    "fico_range_low",
    "fico_range_high",
    # Credit history depth and utilisation
    "open_acc",
    "total_acc",
    "revol_bal",
    "revol_util",
    "mort_acc",
    # Adverse history
    "delinq_2yrs",
    "inq_last_6mths",
    "pub_rec",
    "pub_rec_bankruptcies",
    "tax_liens",
    "acc_now_delinq",
    "delinq_amnt",
    "chargeoff_within_12_mths",
    "collections_12_mths_ex_med",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "mths_since_last_major_derog",
    # Aggregate balances
    "tot_coll_amt",
    "tot_cur_bal",
    "total_rev_hi_lim",
    "acc_open_past_24mths",
)

CORE_CATEGORICAL: tuple[str, ...] = (
    "purpose",
    "home_ownership",
    "verification_status",
    "emp_length",
    "addr_state",
    "application_type",
    "initial_list_status",
)

EXTENDED_NUMERIC: tuple[str, ...] = (
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl",
    "mths_since_recent_bc",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_inq",
    "mths_since_recent_revol_delinq",
    "num_accts_ever_120_pd",
    "num_actv_bc_tl",
    "num_actv_rev_tl",
    "num_bc_sats",
    "num_bc_tl",
    "num_il_tl",
    "num_op_rev_tl",
    "num_rev_accts",
    "num_rev_tl_bal_gt_0",
    "num_sats",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m",
    "pct_tl_nvr_dlq",
    "percent_bc_gt_75",
    "tot_hi_cred_lim",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",
)

LEAKAGE_COLUMNS: frozenset[str] = frozenset(
    {
        # Payment history -- direct functions of the outcome
        "out_prncp",
        "out_prncp_inv",
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_amnt",
        "next_pymnt_d",
        "pymnt_plan",
        # Post-origination bureau refresh
        "last_credit_pull_d",
        "last_fico_range_high",
        "last_fico_range_low",
        # Loss mitigation -- only populated once a loan is in trouble
        "hardship_flag",
        "hardship_type",
        "hardship_reason",
        "hardship_status",
        "hardship_amount",
        "hardship_start_date",
        "hardship_end_date",
        "hardship_length",
        "hardship_dpd",
        "hardship_loan_status",
        "hardship_payoff_balance_amount",
        "hardship_last_payment_amount",
        "deferral_term",
        "payment_plan_start_date",
        "orig_projected_additional_accrued_interest",
        # Settlement -- implies the loan already defaulted
        "debt_settlement_flag",
        "debt_settlement_flag_date",
        "settlement_status",
        "settlement_date",
        "settlement_amount",
        "settlement_percentage",
        "settlement_term",
    }
)


class LeakageError(RuntimeError):
    """Raised when a post-origination column reaches a feature matrix."""


def ingest_columns(*, extended: bool = True, with_lc_grade: bool = True) -> list[str]:
    """Columns to read from the accepted-loans CSV.

    ``with_lc_grade`` defaults to ``True`` at *ingest* time even though the
    primary model excludes those columns: ingest is expensive and runs once, so
    the grade columns are materialised to Parquet and dropped later at
    feature-selection time by :func:`feature_columns`.
    """
    cols = list(ID_COLUMNS) + list(LABEL_COLUMNS) + list(CORE_NUMERIC) + list(
        CORE_CATEGORICAL
    )
    if extended:
        cols += list(EXTENDED_NUMERIC)
    if with_lc_grade:
        cols += list(LC_ASSESSMENT_COLUMNS)
    seen: dict[str, None] = {}
    for c in cols:
        seen.setdefault(c, None)
    return list(seen)


def feature_columns(*, extended: bool = True, with_lc_grade: bool = False) -> list[str]:
    """Columns a model may use. ``with_lc_grade=False`` is the primary spec."""
    cols = list(CORE_NUMERIC) + list(CORE_CATEGORICAL)
    if extended:
        cols += list(EXTENDED_NUMERIC)
    if with_lc_grade:
        cols += list(LC_ASSESSMENT_COLUMNS)
    assert_no_leakage(cols)
    return cols


def assert_no_leakage(columns: object) -> None:
    """Raise :class:`LeakageError` if any post-origination column is present.

    Called on every feature matrix before it reaches a model. ``last_pymnt_d`` is
    checked explicitly because it is a legitimate *label* input, which makes it
    the single most likely column to leak through by accident.
    """
    cols = set(map(str, columns))  # type: ignore[arg-type]
    offenders = sorted((cols & LEAKAGE_COLUMNS) | (cols & {"last_pymnt_d"}))
    if offenders:
        raise LeakageError(
            "post-origination columns present in feature matrix: "
            + ", ".join(offenders)
            + ". These are recorded after the loan was funded and encode the "
            "outcome; a model using them is invalid."
        )


# --------------------------------------------------------------------------
# Rejected-applications file
# --------------------------------------------------------------------------

REJECTED_COLUMNS: tuple[str, ...] = (
    "Amount Requested",
    "Application Date",
    "Loan Title",
    "Risk_Score",
    "Debt-To-Income Ratio",
    "Zip Code",
    "State",
    "Employment Length",
    "Policy Code",
)

REJECTED_RENAMES: dict[str, str] = {
    "Amount Requested": "loan_amnt",
    "Application Date": "application_d",
    "Loan Title": "title",
    "Risk_Score": "risk_score",
    "Debt-To-Income Ratio": "dti_raw",
    "Zip Code": "zip_code",
    "State": "addr_state",
    "Employment Length": "emp_length",
    "Policy Code": "policy_code",
}

COMMON_FEATURE_MAP: dict[str, dict[str, str]] = {
    "loan_amnt": {
        "accepted": "loan_amnt",
        "rejected": "loan_amnt",
        "comparability": "good",
        "caveat": "Accepted is funded amount, rejected is requested amount; "
        "these differ slightly where Lending Club partially funded a request.",
    },
    "emp_length": {
        "accepted": "emp_length",
        "rejected": "emp_length",
        "comparability": "good",
        "caveat": "Same bucketed encoding, but rejected file uses a bare "
        "'< 1 year' more often and has a higher missing rate.",
    },
    "addr_state": {
        "accepted": "addr_state",
        "rejected": "addr_state",
        "comparability": "good",
        "caveat": "Two-letter codes on both sides.",
    },
    "zip_code": {
        "accepted": "zip_code",
        "rejected": "zip_code",
        "comparability": "good",
        "caveat": "Both truncated to three digits with an 'xx' suffix.",
    },
    "fico_or_risk_score": {
        "accepted": "fico_range_low",
        "rejected": "risk_score",
        "comparability": "partial",
        "caveat": "Not the same instrument. Accepted reports a FICO band; "
        "rejected reports a single 'Risk_Score' whose source Lending Club "
        "changed over time, with a high missing rate in later vintages. "
        "Comparable in rank, not on an identical scale.",
    },
    "dti": {
        "accepted": "dti",
        "rejected": "dti_raw",
        "comparability": "partial",
        "caveat": "Different definitions. Accepted 'dti' excludes mortgage and "
        "is computed by Lending Club from verified income; rejected is a "
        "self-reported percentage string with implausible extremes (>1000%). "
        "Requires parsing and winsorising before any comparison.",
    },
    "purpose_or_title": {
        "accepted": "title",
        "rejected": "title",
        "comparability": "partial",
        "caveat": "Accepted has both a clean 'purpose' taxonomy and free-text "
        "'title'; rejected has free text only. Comparable only after mapping "
        "text to the accepted taxonomy, which is lossy.",
    },
}
"""Features present on *both* sides, with an honest comparability grade.

This dict is the hard ceiling on Stage 4. There is no income, no
``revol_util``, no ``home_ownership`` and no credit-history depth on the
rejected side, so the selection-bias diagnostic runs on roughly six features --
four solid and two only partially comparable. ``Application Date`` is
deliberately absent: an application date is not an issue date, so it is used for
vintage alignment only, never as a matched feature. ``Policy Code`` is also
absent: it is degenerate (almost entirely a single value) on both sides.
"""
