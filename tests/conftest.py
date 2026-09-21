"""Synthetic fixtures.

Everything here is generated in-process. No test in this suite requires the real
Lending Club download, so the labelling rules and the Stage 4 diagnostic logic
stay verifiable on a laptop in under a second.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def status_examples() -> list[str]:
    """Every ``loan_status`` value the real file is known to contain."""
    return [
        "Fully Paid",
        "Charged Off",
        "Current",
        "Late (31-120 days)",
        "In Grace Period",
        "Late (16-30 days)",
        "Default",
        "Issued",
        "Does not meet the credit policy. Status:Fully Paid",
        "Does not meet the credit policy. Status:Charged Off",
    ]


@pytest.fixture
def tiny_loans() -> pd.DataFrame:
    """Hand-built loans covering every branch of the labelling rule.

    Each row is a named scenario so a failure points at the rule that broke.
    """
    rows = [
        # scenario, loan_status, issue_d, last_pymnt_d, term
        ("event_month_10", "Charged Off", "Jan-2015", "Nov-2015", " 36 months"),
        ("event_never_paid", "Charged Off", "Jan-2015", None, " 36 months"),
        ("event_status_default", "Default", "Mar-2016", "Sep-2016", " 60 months"),
        ("payoff_month_24", "Fully Paid", "Jan-2014", "Jan-2016", " 36 months"),
        ("payoff_early_month_3", "Fully Paid", "Jun-2017", "Sep-2017", " 36 months"),
        ("current_admin_censor", "Current", "Jan-2018", "Dec-2018", " 60 months"),
        ("late_31_120", "Late (31-120 days)", "Feb-2017", "Aug-2018", " 60 months"),
        ("late_16_30", "Late (16-30 days)", "Feb-2017", "Oct-2018", " 60 months"),
        ("grace_period", "In Grace Period", "Apr-2017", "Nov-2018", " 36 months"),
        ("policy_exception", "Does not meet the credit policy. Status:Charged Off",
         "May-2008", "Jan-2009", " 36 months"),
        ("same_month_payoff", "Fully Paid", "Jul-2016", "Jul-2016", " 36 months"),
        ("bad_issue_date", "Fully Paid", "not-a-date", "Jan-2016", " 36 months"),
        ("missing_term", "Fully Paid", "Jan-2015", "Jan-2017", None),
        ("term_overrun", "Fully Paid", "Jan-2012", "Jan-2018", " 36 months"),
        ("messy_whitespace", "  charged off  ", "Feb-2015", "Aug-2015", "36 months"),
    ]
    df = pd.DataFrame(
        rows, columns=["scenario", "loan_status", "issue_d", "last_pymnt_d", "term"]
    )
    return df.set_index("scenario", drop=False)


@pytest.fixture
def synthetic_survival_frame() -> pd.DataFrame:
    """A larger synthetic survival dataset with a known signal.

    Hazard rises with ``dti`` and falls with ``fico_range_low``, so a correct
    model must recover a positive coefficient on the former and a negative one
    on the latter.
    """
    rng = np.random.default_rng(7)
    n = 800
    fico = rng.normal(700, 40, n)
    dti = rng.gamma(4, 4, n)
    amnt = rng.uniform(1_000, 35_000, n)
    log_hazard = 0.05 * (dti - dti.mean()) - 0.02 * (fico - fico.mean())
    true_time = rng.exponential(1 / np.exp(log_hazard - 3.0))
    censor_time = rng.uniform(6, 60, n)
    duration = np.minimum(true_time, censor_time)
    return pd.DataFrame(
        {
            "fico_range_low": fico.astype("float32"),
            "dti": dti.astype("float32"),
            "loan_amnt": amnt.astype("float32"),
            "purpose": rng.choice(["debt_consolidation", "credit_card", "other"], n),
            "grade": rng.choice(list("ABCDE"), n),
            "duration_months": np.clip(np.ceil(duration), 1, 60).astype("int32"),
            "event": (true_time <= censor_time).astype("int8"),
        }
    )


@pytest.fixture
def messy_accepted_csv(tmp_path):
    """A small accepted-loans CSV with the real file's structural quirks.

    Includes a prospectus preamble line, a footer totals row, thousands
    separators, a percent sign, and a blank line.
    """
    path = tmp_path / "accepted_messy.csv"
    lines = [
        "Notes offered by Prospectus (https://www.lendingclub.com/info/prospectus.action)",
        "id,loan_amnt,funded_amnt,installment,annual_inc,dti,fico_range_low,"
        "fico_range_high,int_rate,grade,sub_grade,purpose,home_ownership,"
        "verification_status,emp_length,addr_state,application_type,"
        "initial_list_status,term,issue_d,last_pymnt_d,loan_status",
        "1,10000,10000,325.50,\"60,000\",18.5,690,694,13.5%,C,C1,debt_consolidation,"
        "RENT,Verified,5 years,CA,Individual,w, 36 months,Jan-2015,Nov-2015,Charged Off",
        "2,5000,5000,160.00,45000,9.2,720,724,9.5%,B,B2,credit_card,MORTGAGE,"
        "Source Verified,10+ years,NY,Individual,f, 36 months,Feb-2014,Feb-2017,Fully Paid",
        "",
        "3,20000,20000,450.00,90000,22.1,660,664,17.2%,D,D3,home_improvement,OWN,"
        "Not Verified,2 years,TX,Joint App,w, 60 months,Mar-2018,Dec-2018,Current",
        "Total amount funded in policy code 1: 123456789",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def messy_rejected_csv(tmp_path):
    """A small rejected-applications CSV in the real file's format."""
    path = tmp_path / "rejected_messy.csv"
    lines = [
        "Amount Requested,Application Date,Loan Title,Risk_Score,"
        "Debt-To-Income Ratio,Zip Code,State,Employment Length,Policy Code",
        "5000,2014-03-15,Debt consolidation,640,25.5%,900xx,CA,< 1 year,0",
        "12000,2014-04-02,Business,,1200%,100xx,NY,3 years,0",
        "\"3,500\",2015-01-20,Car financing,momentum,15%,750xx,TX,10+ years,0",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
