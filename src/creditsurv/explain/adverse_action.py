"""Stage 3(b): convert SurvSHAP(t) output into an ECOA / Regulation B notice.

Regulatory basis
----------------
Regulation B, 12 CFR 1002.9, implements ECOA's adverse-action requirements. A
notice must contain:

1. a statement of the action taken;
2. the creditor's name and address;
3. the ECOA notice -- the substance of section 701(a);
4. the name and address of the federal agency enforcing compliance; and
5. either a statement of **specific reasons** for the action, or disclosure of
   the applicant's right to request one within 30 days.

Three constraints from 12 CFR 1002.9(b)(2) and its Official Staff Commentary
shape what this module does, and each is enforced in code rather than left to the
caller:

* **Reasons must be specific and principal.** The Commentary states that a
  notice reciting more than four reasons is "not likely to be helpful", so the
  notice is capped at four. :data:`MAX_PRINCIPAL_REASONS`.
* **"You failed to score high enough" is not a valid reason.** The Commentary is
  explicit that disclosing a failure to achieve a qualifying score, or a reference
  to internal policy, does not satisfy the requirement. The underlying factors
  must be named. This is the reason Lending Club's own `grade`, `sub_grade` and
  `int_rate` are excluded from the primary model: attributions pointing at a grade
  would be exactly the impermissible disclosure. :func:`assert_disclosable`
  refuses to render a notice built on them.
* **Only adverse factors are reasons.** A feature that *helped* the applicant is
  not a reason for denial. Attributions are filtered by direction, not just
  magnitude.
* **Geography is a model input but never a stated reason.** `addr_state` and
  `zip_code` are filtered out of reason selection rather than rejected outright,
  since they are legitimate features. But if one of them is among the strongest
  adverse drivers, the notice carries a fair-lending flag: a model whose real
  principal reason cannot lawfully be disclosed needs review, and hiding that
  would defeat the purpose of the exercise. :data:`NOT_DISCLOSABLE`.

Where a consumer report was used, FCRA 615(a) additionally requires the score,
its range, and the key factors that adversely affected it. That block is included
when a score is supplied.

This module produces a realistic, correctly-structured notice for a portfolio
project. It is not legal advice and has not been reviewed by counsel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from ..io import schema as sch
from .survshap import SurvShapExplanation

__all__ = [
    "MAX_PRINCIPAL_REASONS",
    "REASON_TEMPLATES",
    "PrincipalReason",
    "AdverseActionNotice",
    "assert_disclosable",
    "build_adverse_action_notice",
    "NonDisclosableFeatureError",
    "SCORE_FEATURES",
    "NOT_DISCLOSABLE",
]


MAX_PRINCIPAL_REASONS = 4
"""Per the Official Staff Commentary to 12 CFR 1002.9(b)(2)."""


# Reason language adapted from the sample forms in Regulation B Appendix C
# (Form C-1). Keys are model feature names; `higher_is_adverse` records which
# direction of the feature counts against the applicant, which is used as a
# consistency check against the sign of the attribution.
REASON_TEMPLATES: dict[str, dict] = {
    "dti": {
        "reason": "Excessive obligations in relation to income",
        "detail": "Your total monthly debt payments are high relative to your income.",
        "higher_is_adverse": True,
    },
    "installment_to_income": {
        "reason": "Payment on requested credit is high in relation to income",
        "detail": "The monthly payment for the amount requested is large relative to your monthly income.",
        "higher_is_adverse": True,
    },
    "loan_to_income": {
        "reason": "Amount of credit requested is high in relation to income",
        "detail": "The amount requested is large relative to your annual income.",
        "higher_is_adverse": True,
    },
    "annual_inc": {
        "reason": "Income insufficient for amount of credit requested",
        "detail": "Your stated income is low relative to the amount of credit requested.",
        "higher_is_adverse": False,
    },
    "log_annual_inc": {
        "reason": "Income insufficient for amount of credit requested",
        "detail": "Your stated income is low relative to the amount of credit requested.",
        "higher_is_adverse": False,
    },
    "loan_amnt": {
        "reason": "Amount of credit requested",
        "detail": "The amount of credit requested contributed to this decision.",
        "higher_is_adverse": True,
    },
    "fico_midpoint": {
        "reason": "Credit bureau score and the credit history behind it",
        "detail": "Information in your credit report indicates a higher likelihood of missed payments.",
        "higher_is_adverse": False,
    },
    "revol_util": {
        "reason": "Proportion of available revolving credit currently in use",
        "detail": "You are using a high share of the revolving credit lines available to you.",
        "higher_is_adverse": True,
    },
    "bc_util": {
        "reason": "Proportion of available bankcard credit currently in use",
        "detail": "You are using a high share of your available bankcard limits.",
        "higher_is_adverse": True,
    },
    "percent_bc_gt_75": {
        "reason": "Number of bankcard accounts near their credit limit",
        "detail": "Several of your bankcard accounts are close to their limits.",
        "higher_is_adverse": True,
    },
    "delinq_2yrs": {
        "reason": "Delinquent past or present credit obligations with others",
        "detail": "Your credit report shows payments past due within the last two years.",
        "higher_is_adverse": True,
    },
    "mths_since_last_delinq": {
        "reason": "Recency of delinquency on credit obligations",
        "detail": "A recent delinquency appears on your credit report.",
        "higher_is_adverse": False,
    },
    "mths_since_last_major_derog": {
        "reason": "Recency of a serious derogatory credit item",
        "detail": "A recent major derogatory item appears on your credit report.",
        "higher_is_adverse": False,
    },
    "pub_rec": {
        "reason": "Garnishment, attachment, foreclosure, repossession, collection action, or judgment",
        "detail": "Public records appear on your credit report.",
        "higher_is_adverse": True,
    },
    "pub_rec_bankruptcies": {
        "reason": "Bankruptcy",
        "detail": "A bankruptcy appears on your credit report.",
        "higher_is_adverse": True,
    },
    "tax_liens": {
        "reason": "Garnishment, attachment, foreclosure, repossession, collection action, or judgment",
        "detail": "One or more tax liens appear on your credit report.",
        "higher_is_adverse": True,
    },
    "inq_last_6mths": {
        "reason": "Number of recent inquiries on credit bureau report",
        "detail": "Your credit report shows several recent requests for new credit.",
        "higher_is_adverse": True,
    },
    "mths_since_recent_inq": {
        "reason": "Recency of inquiries on credit bureau report",
        "detail": "You have applied for credit recently.",
        "higher_is_adverse": False,
    },
    "acc_open_past_24mths": {
        "reason": "Number of accounts opened recently",
        "detail": "You have opened several new accounts in the last two years.",
        "higher_is_adverse": True,
    },
    "open_acc": {
        "reason": "Number of accounts currently open",
        "detail": "The number of open accounts on your credit report contributed to this decision.",
        "higher_is_adverse": True,
    },
    "total_acc": {
        "reason": "Insufficient number of credit references",
        "detail": "Your credit report shows limited credit experience.",
        "higher_is_adverse": False,
    },
    "num_accts_ever_120_pd": {
        "reason": "Delinquent past or present credit obligations with others",
        "detail": "Your credit report shows accounts that were seriously past due.",
        "higher_is_adverse": True,
    },
    "pct_tl_nvr_dlq": {
        "reason": "Payment history on credit obligations",
        "detail": "A share of your accounts have been past due at some point.",
        "higher_is_adverse": False,
    },
    "emp_length_years": {
        "reason": "Length of employment",
        "detail": "Your length of employment at your current job is short.",
        "higher_is_adverse": False,
    },
    "home_ownership": {
        "reason": "Type of residence",
        "detail": "Your housing status contributed to this decision.",
        "higher_is_adverse": None,
    },
    "verification_status": {
        "reason": "Unable to verify income",
        "detail": "We were unable to verify the income you reported.",
        "higher_is_adverse": None,
    },
    "purpose": {
        "reason": "Purpose of the credit requested",
        "detail": "The stated purpose of the requested credit contributed to this decision.",
        "higher_is_adverse": None,
    },
    "tot_cur_bal": {
        "reason": "Total balances on credit obligations",
        "detail": "The total balance across your accounts contributed to this decision.",
        "higher_is_adverse": True,
    },
    "total_rev_hi_lim": {
        "reason": "Amount of revolving credit available",
        "detail": "The total revolving credit available to you is limited.",
        "higher_is_adverse": False,
    },
    "mort_acc": {
        "reason": "Number of mortgage accounts",
        "detail": "Your mortgage account history contributed to this decision.",
        "higher_is_adverse": None,
    },
    "credit_history_months": {
        "reason": "Limited credit experience",
        "detail": "Your credit history is short.",
        "higher_is_adverse": False,
    },
}


class NonDisclosableFeatureError(RuntimeError):
    """Raised when a notice would rest on a score rather than on real factors.

    Reserved for Lending Club's grade / sub_grade / int_rate. The Official Staff
    Commentary to 12 CFR 1002.9(b)(2) is explicit that disclosing a failure to
    achieve a qualifying score does not satisfy the requirement, so a notice built
    on those columns cannot be made compliant by rewording -- the wrong model
    variant was used, and that is a hard error.
    """


SCORE_FEATURES: frozenset[str] = frozenset({"grade", "sub_grade", "int_rate"})
"""Lending Club's own score. Never a permissible stated reason."""

NOT_DISCLOSABLE: frozenset[str] = frozenset({"addr_state", "zip_code", "policy_code"})
"""Legitimate model inputs that must not appear as *stated reasons*.

"You live in the wrong ZIP code" is both unhelpful under Reg B and a redlining
exposure under ECOA and the Fair Housing Act. These are therefore filtered out of
reason selection rather than rejected outright -- but if one of them is among the
top adverse drivers, that is recorded as a fair-lending flag on the notice. A
model whose real principal reason cannot lawfully be disclosed is a finding, not
something to quietly drop.
"""


def assert_disclosable(features: object) -> None:
    """Reject a feature set built on Lending Club's own score.

    Geography is *not* rejected here; see :data:`NOT_DISCLOSABLE`.
    """
    names = {str(f) for f in features}  # type: ignore[arg-type]
    offenders = sorted(n for n in names if n in SCORE_FEATURES or n.startswith("grade"))
    if offenders:
        raise NonDisclosableFeatureError(
            "cannot build a notice on Lending Club's own score: "
            + ", ".join(offenders)
            + ". 12 CFR 1002.9(b)(2) commentary bars a failure-to-score reason, so "
            "the underlying factors must be named instead. Run the primary "
            "(no-LC-grade) model variant."
        )


@dataclass
class PrincipalReason:
    """One disclosed reason, traced back to the attribution that produced it."""

    rank: int
    feature: str
    reason: str
    detail: str
    attribution: float
    feature_value: object
    direction_consistent: bool = True

    def render(self) -> str:
        return f"{self.rank}. {self.reason}\n   {self.detail}"


@dataclass
class AdverseActionNotice:
    """A structured, renderable adverse-action notice."""

    applicant_id: str
    decision: str
    reasons: list[PrincipalReason]
    creditor_name: str
    creditor_address: str
    enforcement_agency: str
    notice_date: date
    credit_score: float | None = None
    score_range: tuple[float, float] | None = None
    score_source: str | None = None
    predicted_default_probability: float | None = None
    horizon_months: int | None = None
    model_name: str = "survival"
    excluded_helpful_features: tuple[str, ...] = field(default=())
    fair_lending_flags: tuple[str, ...] = field(default=())
    """Non-disclosable features that were nonetheless among the top adverse
    drivers. A non-empty value means the model's real principal reason cannot
    lawfully be stated, which warrants review before deployment."""

    ECOA_NOTICE = (
        "The federal Equal Credit Opportunity Act prohibits creditors from "
        "discriminating against credit applicants on the basis of race, color, "
        "religion, national origin, sex, marital status, age (provided the "
        "applicant has the capacity to enter into a binding contract); because "
        "all or part of the applicant's income derives from any public "
        "assistance program; or because the applicant has in good faith "
        "exercised any right under the Consumer Credit Protection Act."
    )

    def render(self) -> str:
        """The notice as plain text, in the order Regulation B requires."""
        w = 74
        out: list[str] = []
        out.append("=" * w)
        out.append("STATEMENT OF ADVERSE ACTION".center(w))
        out.append("=" * w)
        out.append("")
        out.append(f"Date:          {self.notice_date.isoformat()}")
        out.append(f"Applicant ID:  {self.applicant_id}")
        out.append("")
        out.append(f"{self.creditor_name}")
        for line in self.creditor_address.splitlines():
            out.append(f"{line}")
        out.append("")

        # (1) action taken
        out.append("-" * w)
        out.append("ACTION TAKEN")
        out.append("-" * w)
        out.append(f"We regret to inform you that we have {self.decision} your")
        out.append("application for credit.")
        out.append("")

        # (5) specific reasons
        out.append("-" * w)
        out.append("PRINCIPAL REASONS FOR OUR DECISION")
        out.append("-" * w)
        if not self.reasons:
            out.append("No individual factor was materially adverse. Please contact")
            out.append("us for a further explanation of this decision.")
        else:
            out.append("The principal reason(s) for this decision were:")
            out.append("")
            for r in self.reasons:
                out.append(r.render())
                out.append("")
        out.append("If you have questions about these reasons, you may contact us at")
        out.append("the address above.")
        out.append("")
        if self.fair_lending_flags:
            out.append("[INTERNAL REVIEW FLAG -- NOT PART OF THE APPLICANT NOTICE]")
            out.append("Non-disclosable features were among the strongest adverse")
            out.append(f"drivers for this applicant: {', '.join(self.fair_lending_flags)}.")
            out.append("These were excluded from the stated reasons. A model whose")
            out.append("principal driver cannot lawfully be disclosed requires")
            out.append("fair-lending review before deployment.")
            out.append("")

        # FCRA 615(a) score disclosure
        if self.credit_score is not None:
            out.append("-" * w)
            out.append("CREDIT SCORE INFORMATION")
            out.append("-" * w)
            out.append("Our decision was based in whole or in part on information")
            out.append("obtained in a report from a consumer reporting agency.")
            out.append("")
            out.append(f"Your credit score:  {self.credit_score:.0f}")
            if self.score_range:
                lo, hi = self.score_range
                out.append(f"Possible range:     {lo:.0f} to {hi:.0f}")
            if self.score_source:
                out.append(f"Score source:       {self.score_source}")
            out.append(f"Date of score:      {self.notice_date.isoformat()}")
            out.append("")
            out.append("You have a right to obtain a free copy of your consumer report")
            out.append("from the reporting agency and to dispute the accuracy or")
            out.append("completeness of any information in that report.")
            out.append("")

        # (3) ECOA notice
        out.append("-" * w)
        out.append("YOUR RIGHTS UNDER FEDERAL LAW")
        out.append("-" * w)
        for line in _wrap(self.ECOA_NOTICE, w):
            out.append(line)
        out.append("")
        # (4) enforcement agency
        out.append("The federal agency that administers compliance with this law")
        out.append("concerning this creditor is:")
        for line in self.enforcement_agency.splitlines():
            out.append(f"  {line}")
        out.append("")
        out.append("=" * w)
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {
            "applicant_id": self.applicant_id,
            "decision": self.decision,
            "notice_date": self.notice_date.isoformat(),
            "model": self.model_name,
            "predicted_default_probability": self.predicted_default_probability,
            "horizon_months": self.horizon_months,
            "n_reasons": len(self.reasons),
            "reasons": [
                {
                    "rank": r.rank,
                    "feature": r.feature,
                    "reason": r.reason,
                    "attribution": round(float(r.attribution), 6),
                    "feature_value": _plain(r.feature_value),
                    "direction_consistent": r.direction_consistent,
                }
                for r in self.reasons
            ],
            "excluded_helpful_features": list(self.excluded_helpful_features),
            "fair_lending_flags": list(self.fair_lending_flags),
        }


def _plain(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return str(value)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for word in words:
        if len(cur) + len(word) + 1 > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return lines


def build_adverse_action_notice(
    expl: SurvShapExplanation,
    obs: int = 0,
    *,
    applicant_id: str | None = None,
    decision: str = "declined",
    creditor_name: str = "Example Lending Co.",
    creditor_address: str = "123 Example Street\nSpringfield, IL 62701",
    enforcement_agency: str = (
        "Consumer Financial Protection Bureau\n"
        "1700 G Street NW\n"
        "Washington, DC 20552"
    ),
    notice_date: date | None = None,
    horizon_months: int | None = None,
    credit_score: float | None = None,
    score_range: tuple[float, float] | None = (300.0, 850.0),
    score_source: str | None = "Credit bureau risk score",
    max_reasons: int = MAX_PRINCIPAL_REASONS,
    model_name: str = "survival",
) -> AdverseActionNotice:
    """Build a notice from one observation's SurvSHAP(t) attributions.

    Attributions explain ``S(t|x)``, so a **negative** attribution lowers survival
    and therefore *raises* default risk -- those are the adverse factors. Features
    that helped the applicant are recorded separately and never disclosed as
    reasons, because they are not reasons for denial.
    """
    assert_disclosable(expl.feature_names)
    sch.assert_no_leakage(expl.feature_names)

    signed = expl.signed_importance(obs=obs)          # + raises survival
    magnitude = expl.importance(obs=obs)

    adverse = signed[signed < 0].sort_values()        # most negative first
    helpful = tuple(signed[signed > 0].sort_values(ascending=False).head(5).index)

    # A feature that cannot be disclosed but IS among the strongest adverse
    # drivers is a fair-lending finding: the model's real principal reason is one
    # the lender may not state. Recorded on the notice rather than dropped.
    top_adverse = list(adverse.head(max_reasons * 2).index)
    fair_lending_flags = tuple(f for f in top_adverse if f in NOT_DISCLOSABLE)

    row = expl.feature_values.iloc[obs]
    reasons: list[PrincipalReason] = []
    seen_reasons: set[str] = set()

    for feature in adverse.index:
        if len(reasons) >= max_reasons:
            break
        if feature in NOT_DISCLOSABLE:
            continue
        template = REASON_TEMPLATES.get(feature)
        if template is None:
            base = feature[:-8] if feature.endswith("_missing") else feature
            template = REASON_TEMPLATES.get(base)
        if template is None:
            continue
        # Two features can map to the same statutory reason; do not repeat it.
        if template["reason"] in seen_reasons:
            continue

        value = row.get(feature, None)
        expected = template.get("higher_is_adverse")
        consistent = True
        if expected is not None and pd.notna(value) and feature in magnitude.index:
            # An adverse attribution should line up with the feature being on its
            # adverse side relative to the population. Where it does not, the flag
            # is recorded rather than the reason being silently dropped.
            try:
                col = expl.feature_values[feature]
                if str(col.dtype) != "category":
                    median = float(pd.to_numeric(col, errors="coerce").median())
                    is_high = float(value) > median
                    consistent = bool(is_high == bool(expected))
            except (TypeError, ValueError):
                consistent = True

        seen_reasons.add(template["reason"])
        reasons.append(
            PrincipalReason(
                rank=len(reasons) + 1,
                feature=feature,
                reason=template["reason"],
                detail=template["detail"],
                attribution=float(adverse[feature]),
                feature_value=value,
                direction_consistent=consistent,
            )
        )

    k = -1 if horizon_months is None else int(
        np.argmin(np.abs(expl.times - horizon_months))
    )
    pd_hat = float(1.0 - expl.prediction[obs, k])

    return AdverseActionNotice(
        applicant_id=applicant_id or f"APP-{obs:06d}",
        decision=decision,
        reasons=reasons,
        creditor_name=creditor_name,
        creditor_address=creditor_address,
        enforcement_agency=enforcement_agency,
        notice_date=notice_date or date.today(),
        credit_score=credit_score,
        score_range=score_range if credit_score is not None else None,
        score_source=score_source if credit_score is not None else None,
        predicted_default_probability=pd_hat,
        horizon_months=horizon_months or int(expl.times[k]),
        model_name=model_name,
        excluded_helpful_features=helpful,
        fair_lending_flags=fair_lending_flags,
    )
