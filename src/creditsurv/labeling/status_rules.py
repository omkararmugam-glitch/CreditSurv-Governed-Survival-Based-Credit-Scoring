"""Mapping from Lending Club ``loan_status`` to survival outcome classes.

This module is deliberately free of pandas and of any dependency on the real
dataset: the event/censoring decision is the single most consequential choice in
the whole project, so it lives in pure functions that can be exhaustively tested.

Decision record (see FINDINGS.md for rationale):

* ``Charged Off`` / ``Default``            -> EVENT
* ``Fully Paid``                           -> censored at payoff date
* ``Current`` / ``Issued``                 -> censored at the data cutoff
* ``In Grace Period`` / ``Late (*)``       -> censored at last payment date
* ``Does not meet the credit policy.*``    -> excluded from the primary sample

The two delinquent-but-not-charged-off buckets are censored rather than treated
as events, which is conservative in the sense that it *under*-counts defaults: a
loan 120 days late is largely destined to charge off. ``late_31_120_is_event``
exists so that this assumption can be attacked directly in a sensitivity run.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "Outcome",
    "PRIMARY_STATUS_MAP",
    "POLICY_EXCEPTION_PREFIX",
    "UnknownStatusError",
    "normalize_status",
    "strip_policy_exception",
    "classify_status",
    "is_event",
    "known_statuses",
]


class Outcome(str, Enum):
    """What a ``loan_status`` value means for the survival target."""

    EVENT = "event"
    """Default / charge-off. ``event = 1``."""

    CENSORED_PAYOFF = "censored_payoff"
    """Loan repaid in full. Censored at the payoff date.

    Formally this is a *competing* risk rather than clean censoring -- a loan
    paid off at month 14 can never default afterwards -- which makes the
    censoring mildly informative. We accept that (standard practice in the
    credit survival literature) and report it rather than hide it.
    """

    CENSORED_ADMIN = "censored_admin"
    """Still performing when the data was extracted. Censored at the cutoff."""

    CENSORED_DELINQUENT = "censored_delinquent"
    """Behind on payments but not yet charged off. Censored at last payment."""

    EXCLUDED = "excluded"
    """Not part of the primary modelling sample."""


POLICY_EXCEPTION_PREFIX = "does not meet the credit policy. status:"
"""Prefix marking 2007-2008 loans written under a different underwriting regime."""


PRIMARY_STATUS_MAP: dict[str, Outcome] = {
    "charged off": Outcome.EVENT,
    "default": Outcome.EVENT,
    "fully paid": Outcome.CENSORED_PAYOFF,
    "current": Outcome.CENSORED_ADMIN,
    "issued": Outcome.CENSORED_ADMIN,
    "in grace period": Outcome.CENSORED_DELINQUENT,
    "late (16-30 days)": Outcome.CENSORED_DELINQUENT,
    "late (31-120 days)": Outcome.CENSORED_DELINQUENT,
}


class UnknownStatusError(ValueError):
    """Raised when a ``loan_status`` value is not in the decision table.

    Failing loudly is intentional. Silently bucketing an unrecognised status
    into "censored" would quietly bias every downstream result.
    """


def normalize_status(raw: object) -> str:
    """Canonicalise a raw ``loan_status`` cell to lowercase, single-spaced text.

    Real Lending Club extracts contain trailing whitespace, non-breaking
    spaces and inconsistent capitalisation across vintages.
    """
    if raw is None:
        return ""
    text = str(raw).replace("\u00a0", " ").replace("\u2013", "-").replace("\u2014", "-")
    return " ".join(text.split()).lower()


def strip_policy_exception(status: str) -> tuple[str, bool]:
    """Split a normalized status into ``(underlying_status, was_policy_exception)``."""
    if status.startswith(POLICY_EXCEPTION_PREFIX):
        return " ".join(status[len(POLICY_EXCEPTION_PREFIX):].split()), True
    return status, False


def classify_status(
    raw: object,
    *,
    late_31_120_is_event: bool = False,
    include_policy_exceptions: bool = False,
) -> Outcome:
    """Classify one ``loan_status`` value into an :class:`Outcome`.

    Parameters
    ----------
    raw:
        Raw cell value, normalized internally.
    late_31_120_is_event:
        Sensitivity switch. When ``True``, ``Late (31-120 days)`` counts as an
        event instead of being censored.
    include_policy_exceptions:
        When ``True``, ``Does not meet the credit policy. Status:X`` is mapped
        by its underlying status ``X`` instead of being excluded.

    Raises
    ------
    UnknownStatusError
        If the status is non-empty but absent from :data:`PRIMARY_STATUS_MAP`.
    """
    status = normalize_status(raw)
    if not status:
        return Outcome.EXCLUDED

    underlying, is_exception = strip_policy_exception(status)
    if is_exception and not include_policy_exceptions:
        return Outcome.EXCLUDED

    try:
        outcome = PRIMARY_STATUS_MAP[underlying]
    except KeyError:
        raise UnknownStatusError(
            f"Unrecognised loan_status {raw!r} (normalized to {underlying!r}). "
            "Add it to PRIMARY_STATUS_MAP with an explicit decision rather than "
            "letting it default to censored."
        ) from None

    if late_31_120_is_event and underlying == "late (31-120 days)":
        return Outcome.EVENT
    return outcome


def is_event(outcome: Outcome) -> bool:
    """Whether an outcome contributes ``event = 1`` to the survival target."""
    return outcome is Outcome.EVENT


def known_statuses(*, include_policy_exceptions: bool = False) -> frozenset[str]:
    """Normalized statuses the decision table recognises."""
    base = set(PRIMARY_STATUS_MAP)
    if include_policy_exceptions:
        base |= {POLICY_EXCEPTION_PREFIX + s for s in PRIMARY_STATUS_MAP}
    return frozenset(base)
