"""Columns that can be computed exactly from others, and the costs of the rest.

Two separate jobs, both about a file that is missing something:

**Derivation.** Some features are not really missing -- they are implied. A monthly
instalment follows from the amount, the rate and the term by the standard
amortisation formula; a FICO midpoint follows from the range. Deriving those is
arithmetic, not imputation, and the distinction matters: an imputed value is a guess
that makes the output look more certain than it is, while a derived value is the same
number the lender's own system would produce. Every derived column is recorded and
reported as derived.

**Costs.** For a feature that genuinely is not in the file, the run says what its
absence costs, using the measured ablation table (FINDINGS 7d) rather than an
opinion. Features are required or optional by the thresholds recorded there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import PROJECT_ROOT

__all__ = ["DERIVATIONS", "Derivation", "derive_features", "FeatureCosts",
           "load_costs", "REQUIRED_DROP", "OPTIONAL_DROP", "ABLATION_COMMAND"]

# Thresholds recorded in FINDINGS 7d before use.
REQUIRED_DROP = 0.010
"""Concordance drop at or above which a feature is required (or must be derivable)."""
OPTIONAL_DROP = 0.002
"""Below this, an absence is reported without a cost claim."""

ABLATION_COMMAND = "python scripts/03e_feature_ablation.py --model-tag {tag}"
"""How to produce the table a model needs before the measured rule can apply."""

STRUCTURALLY_REQUIRED = ("loan_amnt",)
"""Required whatever the measurement says: there is no application without an
amount, and several derivations need it."""


@dataclass
class Derivation:
    """One feature that can be computed from others."""

    feature: str
    needs: tuple[str, ...]
    how: str
    compute: object                      # callable(frame) -> Series

    def missing_inputs(self, columns) -> list[str]:
        return [c for c in self.needs if c not in set(columns)]


def _installment(df: pd.DataFrame) -> pd.Series:
    """Standard amortisation: P = L * i / (1 - (1+i)^-n), i the monthly rate.

    ``int_rate`` is an annual percentage in this data, so it is converted to a
    monthly fraction. A zero rate degenerates to the straight division, which the
    formula cannot express.
    """
    amount = pd.to_numeric(df["loan_amnt"], errors="coerce")
    annual = pd.to_numeric(
        df["int_rate"].astype("string").str.replace("%", "", regex=False),
        errors="coerce")
    term = pd.to_numeric(
        df["term_months"].astype("string").str.extract(r"(\d+)", expand=False),
        errors="coerce")
    monthly = annual / 100.0 / 12.0
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = 1.0 - np.power(1.0 + monthly, -term)
        payment = np.where(monthly > 0, amount * monthly / factor, amount / term)
    return pd.Series(payment, index=df.index).astype("float64")


def _fico_midpoint(df: pd.DataFrame) -> pd.Series:
    low = pd.to_numeric(df["fico_range_low"], errors="coerce")
    high = pd.to_numeric(df["fico_range_high"], errors="coerce")
    return (low + high) / 2.0


def _fico_high_from_low(df: pd.DataFrame) -> pd.Series:
    """Lending Club reports FICO in 4-point bands, so the top of the band is the
    bottom plus four. Exact for this data, and stated as an assumption."""
    return pd.to_numeric(df["fico_range_low"], errors="coerce") + 4.0


def _term_months(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(
        df["term"].astype("string").str.extract(r"(\d+)", expand=False),
        errors="coerce")


def _loan_to_income(df: pd.DataFrame) -> pd.Series:
    amount = pd.to_numeric(df["loan_amnt"], errors="coerce")
    income = pd.to_numeric(df["annual_inc"], errors="coerce").replace(0.0, np.nan)
    return amount / income


def _installment_to_income(df: pd.DataFrame) -> pd.Series:
    payment = pd.to_numeric(df["installment"], errors="coerce")
    income = pd.to_numeric(df["annual_inc"], errors="coerce").replace(0.0, np.nan)
    return payment / (income / 12.0)


# Order matters: a feature derived here can feed a later one, and derive_features
# makes a second pass so a chain such as term -> term_months -> installment closes.
DERIVATIONS: tuple[Derivation, ...] = (
    Derivation("term_months", ("term",), "the number in a term like '36 months'",
               _term_months),
    Derivation("fico_range_high", ("fico_range_low",),
               "bottom of the 4-point FICO band plus four", _fico_high_from_low),
    Derivation("fico_midpoint", ("fico_range_low", "fico_range_high"),
               "midpoint of the reported FICO range", _fico_midpoint),
    Derivation("installment", ("loan_amnt", "int_rate", "term_months"),
               "standard amortisation from amount, rate and term", _installment),
    Derivation("loan_to_income", ("loan_amnt", "annual_inc"),
               "amount divided by annual income", _loan_to_income),
    Derivation("installment_to_income", ("installment", "annual_inc"),
               "instalment divided by monthly income", _installment_to_income),
)


def derive_features(df: pd.DataFrame, wanted) -> tuple[pd.DataFrame, dict, dict]:
    """Compute what can be computed exactly.

    Returns ``(frame, derived, blocked)``: ``derived`` maps feature to how it was
    obtained, ``blocked`` maps feature to the inputs it would have needed. Applied
    in order, so a feature derived early (``term_months``) can feed a later one
    (``installment``).
    """
    out = df.copy()
    derived: dict[str, str] = {}
    blocked: dict[str, list[str]] = {}
    wanted = set(wanted)
    for _pass in range(2):               # a second pass closes chains like
        blocked = {}                     # term -> term_months -> installment
        _apply(out, wanted, derived, blocked)
    return out, derived, blocked


def _apply(out: pd.DataFrame, wanted: set, derived: dict, blocked: dict) -> None:
    for rule in DERIVATIONS:
        if rule.feature not in wanted or rule.feature in out.columns:
            continue
        missing = rule.missing_inputs(out.columns)
        if missing:
            blocked[rule.feature] = missing
            continue
        try:
            computed = rule.compute(out)
        except Exception as exc:                        # pragma: no cover
            blocked[rule.feature] = [f"could not be computed: {exc!r}"]
            continue
        if computed.notna().sum() == 0:
            blocked[rule.feature] = [f"inputs present but unusable ({rule.how})"]
            continue
        out[rule.feature] = computed
        derived[rule.feature] = rule.how


@dataclass
class FeatureCosts:
    """Measured cost of a feature's absence, from the ablation table."""

    model_tag: str = ""
    baseline_concordance: float = float("nan")
    per_feature: dict[str, float] = field(default_factory=dict)
    per_group: dict[str, float] = field(default_factory=dict)
    source: str = ""

    def tier(self, feature: str) -> str:
        drop = self.per_feature.get(feature)
        if feature in STRUCTURALLY_REQUIRED:
            return "required"
        if drop is None:
            return "unmeasured"
        if drop >= REQUIRED_DROP:
            return "required"
        return "optional_costed" if drop >= OPTIONAL_DROP else "optional_free"

    @property
    def measured(self) -> bool:
        """Whether this model has an ablation table, and so a measured split."""
        return bool(self.per_feature)

    def required(self) -> list[str]:
        return sorted({f for f in list(self.per_feature) + list(STRUCTURALLY_REQUIRED)
                       if self.tier(f) == "required"})

    def rule_note(self) -> str:
        """One sentence naming the rule in force, so a reader knows which it was.

        Two rules exist and only one can apply to a given model. With a table, a
        feature is required because its absence was measured to cost at least
        REQUIRED_DROP concordance. Without one, a provisional list stands in, and
        the note says how to replace it with measurement.
        """
        if self.measured:
            return (f"Required features come from the measured ablation of "
                    f"{self.model_tag or 'this model'} ({self.source}): a feature is "
                    f"required when its absence costs at least {REQUIRED_DROP:.3f} "
                    f"concordance. {len(self.required())} of {len(self.per_feature)} "
                    f"features qualify.")
        return ("This model has no ablation table, so a provisional list of required "
                "columns is used instead of measurement. To replace it with measured "
                "costs: "
                + ABLATION_COMMAND.format(tag=self.model_tag or "TAG"))

    def cost_of(self, features) -> float:
        """Conservative upper bound on the joint cost: drops are not additive."""
        return round(sum(self.per_feature.get(f, 0.0) for f in features), 4)

    def describe(self, features) -> str:
        parts = [f"{f} ({self.per_feature[f]:.4f})" for f in features
                 if f in self.per_feature]
        if not parts:
            return ""
        return (", ".join(parts[:6]) + (f" and {len(parts) - 6} more" if len(parts) > 6
                                        else "")
                + f"; at most {self.cost_of(features):.4f} concordance in total")


def load_costs(model_tag: str = "full", tables_dir: Path | None = None) -> FeatureCosts:
    """Read the ablation result for a model, or an empty set of costs if none exists."""
    tables_dir = Path(tables_dir or PROJECT_ROOT / "outputs" / "tables")
    path = tables_dir / f"03e_ablation_{model_tag}.json"
    if not path.exists():
        return FeatureCosts(model_tag=model_tag, source="no ablation result")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return FeatureCosts(model_tag=model_tag, source=f"{path.name} unreadable")
    return FeatureCosts(
        model_tag=payload.get("model_tag", model_tag),
        baseline_concordance=(payload.get("baseline") or {}).get("concordance",
                                                                float("nan")),
        per_feature={r["name"]: float(r["concordance_drop"])
                     for r in payload.get("features", [])},
        per_group={r["name"]: float(r["concordance_drop"])
                   for r in payload.get("groups", [])},
        source=path.name)
