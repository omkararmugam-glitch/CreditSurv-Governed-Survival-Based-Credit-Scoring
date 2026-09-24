"""Work out which uploaded columns are which model features, and say so out loud.

Before this, a file whose columns were not named exactly as in training was refused.
That is the wrong failure: the model does not care what a column is called, only what
is in it. This layer sits in front of the model and changes nothing about it.

Three sources of evidence, in descending order of trust:

1. **A synonym table** (:data:`SYNONYMS`) -- fixed, written down, reviewable. A name
   in it is a strong match because a person decided so, not because a string looked
   similar.
2. **Normalised-name similarity** -- case, spaces, underscores and common words
   stripped, then compared. Good for ``LoanAmount`` or ``dti_ratio``; suggestive, not
   conclusive.
3. **Content** -- does the column hold what that feature holds? Type, plausible
   range, and the formats the real file uses (``"36 months"``, ``"13.5%"``,
   ``"$1,200"``). Content can *raise* confidence in a name match and, more
   importantly, **veto** one: a column called ``credit_score`` holding values between
   0 and 1 is not a credit score.

Nothing here applies a mapping. It proposes one, with a confidence and a reason per
column, for a person to confirm -- and it refuses to map two uploaded columns onto the
same feature, because then the choice between them is being made silently.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["SYNONYMS", "FEATURE_CONTENT", "Candidate", "ColumnMatch",
           "MappingProposal", "propose_mapping", "normalise"]

# Fixed, reviewable synonyms. Keys are normalised upload names, values are model
# features. Anything not here still has a chance through name similarity and content.
SYNONYMS: dict[str, str] = {
    "loanamount": "loan_amnt", "amountrequested": "loan_amnt",
    "amount": "loan_amnt", "principal": "loan_amnt",
    "monthlypayment": "installment", "payment": "installment",
    "monthlyinstalment": "installment", "instalment": "installment",
    "annualincome": "annual_inc", "income": "annual_inc",
    "yearlyincome": "annual_inc", "grossincome": "annual_inc",
    "debttoincome": "dti", "dtiratio": "dti", "debtratio": "dti",
    "creditscore": "fico_range_low", "ficoscore": "fico_range_low",
    "fico": "fico_range_low", "score": "fico_range_low",
    "opencreditlines": "open_acc", "openaccounts": "open_acc",
    "numopenaccounts": "open_acc", "openlines": "open_acc",
    "totalaccounts": "total_acc", "numaccounts": "total_acc",
    "revolvingbalance": "revol_bal", "revbalance": "revol_bal",
    "revolvingutilisation": "revol_util", "revolvingutilization": "revol_util",
    "utilisation": "revol_util", "utilization": "revol_util",
    "delinquencies2y": "delinq_2yrs", "delinquencies": "delinq_2yrs",
    "pastduecount": "delinq_2yrs",
    "inquirieslast6m": "inq_last_6mths", "inquiries6m": "inq_last_6mths",
    "recentinquiries": "inq_last_6mths", "inquiries": "inq_last_6mths",
    "loanpurpose": "purpose", "purposeofloan": "purpose", "reason": "purpose",
    "homeownership": "home_ownership", "housing": "home_ownership",
    "homeownershipstatus": "home_ownership", "residence": "home_ownership",
    "state": "addr_state", "borrowerstate": "addr_state",
    "statecode": "addr_state", "region": "addr_state",
    "publicrecords": "pub_rec", "publicrecord": "pub_rec",
    "bankruptcies": "pub_rec_bankruptcies",
    "mortgageaccounts": "mort_acc", "mortgages": "mort_acc",
    "employmentlength": "emp_length_years", "yearsemployed": "emp_length_years",
    "employmentyears": "emp_length_years",
    "term": "term_months", "loanterm": "term_months", "termmonths": "term_months",
    "interestrate": "int_rate", "rate": "int_rate", "apr": "int_rate",
    "verificationstatus": "verification_status",
    "applicationtype": "application_type",
}

_FILLER = frozenset(("the", "of", "a", "an", "num", "number", "count", "borrower",
                    "applicant", "customer", "cust", "loan"))


def normalise(name: str) -> str:
    """``"Open Credit Lines (count)"`` -> ``"opencreditlines"``.

    Case and punctuation removed so that names differing only in style compare
    equal, and filler words dropped -- but only as **whole tokens**. Substring
    removal would turn ``last`` into ``lst`` by deleting the ``a`` in it, and
    ``inquiries_last_6m`` would then match nothing.
    """
    tokens = [t for t in re.split(r"[^a-z0-9]+|(?<=[a-z])(?=[0-9])",
                                 _split_camel(str(name)).lower()) if t]
    kept = [t for t in tokens if t not in _FILLER]
    return "".join(kept or tokens)


def _split_camel(name: str) -> str:
    """``LoanAmount`` -> ``Loan Amount``, so camel case tokenises like snake case."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))


# What each feature should look like, used to confirm or veto a name match.
# (low, high) are generous plausibility bounds, not the training range.
FEATURE_CONTENT: dict[str, dict] = {
    "loan_amnt": {"kind": "numeric", "range": (100, 100_000)},
    "installment": {"kind": "numeric", "range": (5, 5_000)},
    "annual_inc": {"kind": "numeric", "range": (1_000, 10_000_000)},
    "dti": {"kind": "numeric", "range": (0, 200)},
    "fico_range_low": {"kind": "numeric", "range": (300, 850)},
    "fico_range_high": {"kind": "numeric", "range": (300, 850)},
    "int_rate": {"kind": "numeric", "range": (0, 45)},
    "term_months": {"kind": "numeric", "range": (6, 84),
                    "text_pattern": r"^\s*\d{1,2}\s*months?\s*$"},
    "revol_util": {"kind": "numeric", "range": (0, 200)},
    "bc_util": {"kind": "numeric", "range": (0, 200)},
    "revol_bal": {"kind": "numeric", "range": (0, 5_000_000)},
    "open_acc": {"kind": "numeric", "range": (0, 90)},
    "total_acc": {"kind": "numeric", "range": (0, 200)},
    "delinq_2yrs": {"kind": "numeric", "range": (0, 40)},
    "inq_last_6mths": {"kind": "numeric", "range": (0, 40)},
    "pub_rec": {"kind": "numeric", "range": (0, 90)},
    "mort_acc": {"kind": "numeric", "range": (0, 60)},
    "emp_length_years": {"kind": "numeric", "range": (0, 60)},
    "purpose": {"kind": "categorical"},
    "home_ownership": {"kind": "categorical"},
    "addr_state": {"kind": "categorical", "text_pattern": r"^\s*[A-Za-z]{2}\s*$"},
    "verification_status": {"kind": "categorical"},
    "application_type": {"kind": "categorical"},
    "initial_list_status": {"kind": "categorical"},
}


def _as_numeric(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    stripped = (text.str.replace(",", "", regex=False)
                    .str.replace("%", "", regex=False)
                    .str.replace("$", "", regex=False)
                    .str.replace(r"\s*months?\s*$", "", regex=True))
    return pd.to_numeric(stripped, errors="coerce")


@dataclass
class Candidate:
    feature: str
    confidence: float
    reason: str


@dataclass
class ColumnMatch:
    """One uploaded column and what it appears to be."""

    column: str
    best: Candidate | None
    others: list[Candidate] = field(default_factory=list)
    content_note: str = ""
    vetoed: list[str] = field(default_factory=list)

    @property
    def feature(self) -> str | None:
        return self.best.feature if self.best else None

    @property
    def confidence(self) -> float:
        return self.best.confidence if self.best else 0.0


@dataclass
class MappingProposal:
    """A proposed reading of the file: nothing is applied until it is confirmed."""

    matches: list[ColumnMatch] = field(default_factory=list)
    exact: dict[str, str] = field(default_factory=dict)
    ignored: list[str] = field(default_factory=list)
    conflicts: dict[str, list[str]] = field(default_factory=dict)
    recognised_but_unused: dict[str, str] = field(default_factory=dict)
    """Columns understood but useless to *this* model, with the reason -- a credit
    score in a file scored by a model that has no score feature is recognised and
    then not used, which is worth saying rather than filing under 'ignored'."""

    def mapping(self, *, min_confidence: float = 0.55) -> dict[str, str]:
        """Columns to rename, those at or above ``min_confidence`` pre-selected."""
        out = dict(self.exact)
        for m in self.matches:
            if m.best and m.best.confidence >= min_confidence:
                out[m.column] = m.best.feature
        return out

    def to_frame(self, *, min_confidence: float = 0.55) -> pd.DataFrame:
        rows = []
        for m in self.matches:
            rows.append({
                "uploaded column": m.column,
                "model feature": m.feature or "",
                "confidence": round(m.confidence, 2),
                "use it": bool(m.best and m.best.confidence >= min_confidence),
                "why": m.best.reason if m.best else "no plausible match",
                "content": m.content_note,
                "other candidates": ", ".join(
                    f"{c.feature} ({c.confidence:.2f})" for c in m.others[:3]),
            })
        return pd.DataFrame(rows)


def _content_check(values: pd.Series, feature: str) -> tuple[float, str]:
    """``(multiplier, note)``. A multiplier below 1 weakens a name match; 0 vetoes."""
    rules = FEATURE_CONTENT.get(feature)
    if rules is None:
        return 1.0, ""
    sample = values.dropna()
    if sample.empty:
        return 0.6, "column is empty, so its content could not be checked"
    if len(sample) > 5_000:
        sample = sample.sample(5_000, random_state=0)

    if rules["kind"] == "categorical":
        numeric = _as_numeric(sample)
        if numeric.notna().mean() > 0.95 and sample.astype("string").str.contains(
                r"[A-Za-z]").mean() < 0.05:
            return 0.25, ("holds numbers, but this feature is a category "
                          "(a code column is not a category name)")
        pattern = rules.get("text_pattern")
        if pattern:
            ok = sample.astype("string").str.match(pattern).mean()
            if ok < 0.6:
                return 0.4, f"only {ok:.0%} of values look like this feature's codes"
            return 1.15, f"{ok:.0%} of values match the expected code format"
        return 1.0, f"{sample.nunique()} distinct values"

    numeric = _as_numeric(sample)
    readable = float(numeric.notna().mean())
    if readable < 0.6:
        return 0.0, (f"only {readable:.0%} of values are numbers, but this feature "
                     f"is numeric")
    low, high = rules["range"]
    inside = float(((numeric >= low) & (numeric <= high)).mean())
    note = (f"{readable:.0%} numeric, {inside:.0%} within the plausible range "
            f"{low:g}-{high:g}")
    if inside < 0.5:
        return 0.0, note + " -- the values do not look like this feature"
    if inside < 0.9:
        return 0.7, note
    return 1.2, note


def propose_mapping(df: pd.DataFrame, spec, *, values=None,
                    min_confidence: float = 0.55) -> MappingProposal:
    """Propose which uploaded columns are which model features.

    ``spec`` is the model's feature spec; ``values`` is its saved cleaning values,
    used to recognise category levels when present.
    """
    features = [c for c in spec.all_columns]
    present = {c for c in df.columns if c in features}
    not_in_model: dict[str, str] = {}
    proposal = MappingProposal(exact={c: c for c in present})
    wanted = [f for f in features if f not in present]
    normalised_features = {normalise(f): f for f in wanted}
    known_levels = getattr(values, "categories", {}) or {}

    scored: dict[str, list[Candidate]] = {}
    for column in df.columns:
        if column in present:
            continue
        key = normalise(column)
        candidates: list[Candidate] = []

        target = SYNONYMS.get(key)
        if target and target in wanted:
            candidates.append(Candidate(target, 0.90, f"synonym table: {key}"))
        elif target and target not in features:
            not_in_model[column] = (
                f"recognised as {target}, which this model does not use")
        if key in normalised_features:
            feature = normalised_features[key]
            if all(c.feature != feature for c in candidates):
                candidates.append(Candidate(
                    feature, 0.85, "name matches after normalising case and spacing"))
        for name, feature in normalised_features.items():
            if any(c.feature == feature for c in candidates) or not name:
                continue
            ratio = difflib.SequenceMatcher(None, key, name).ratio()
            if ratio >= 0.82:
                candidates.append(Candidate(
                    feature, 0.55 + 0.25 * (ratio - 0.82) / 0.18,
                    f"name is {ratio:.0%} similar to {feature}"))

        # Category levels are strong evidence on their own: a column holding
        # RENT/OWN/MORTGAGE is home_ownership whatever it is called.
        text = df[column].astype("string").str.strip().dropna()
        if len(text):
            for feature, levels in known_levels.items():
                if feature not in wanted or not levels:
                    continue
                share = float(text.isin(list(levels)).mean())
                if share >= 0.9:
                    existing = next((c for c in candidates if c.feature == feature),
                                    None)
                    if existing:
                        existing.confidence = min(0.97, existing.confidence + 0.07)
                        existing.reason += f"; {share:.0%} of values are known levels"
                    else:
                        candidates.append(Candidate(
                            feature, 0.80,
                            f"{share:.0%} of values are levels seen in training"))

        match = ColumnMatch(column=column, best=None)
        checked: list[Candidate] = []
        for candidate in candidates:
            multiplier, note = _content_check(df[column], candidate.feature)
            if multiplier == 0.0:
                match.vetoed.append(f"{candidate.feature}: {note}")
                continue
            checked.append(Candidate(candidate.feature,
                                     min(0.99, candidate.confidence * multiplier),
                                     candidate.reason))
            if candidate is candidates[0]:
                match.content_note = note
        checked.sort(key=lambda c: -c.confidence)
        if checked:
            match.best, match.others = checked[0], checked[1:]
            if not match.content_note:
                match.content_note = _content_check(df[column], checked[0].feature)[1]
            scored.setdefault(checked[0].feature, []).append(match.column)
        elif column in not_in_model:
            proposal.recognised_but_unused[column] = not_in_model[column]
        else:
            proposal.ignored.append(column)
        proposal.matches.append(match)

    # Two columns for one feature is a choice, and a choice is not made silently.
    for feature, columns in scored.items():
        if len(columns) > 1:
            proposal.conflicts[feature] = columns
            for m in proposal.matches:
                if m.column in columns and m.best:
                    m.best.confidence = min(m.best.confidence, 0.49)
                    m.best.reason += (f" -- but {len(columns)} columns claim "
                                      f"{feature}, so none is pre-selected")
    return proposal
