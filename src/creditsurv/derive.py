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
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .cleaning import coerce_numeric, parse_term_months
from .provenance import PROJECT_ROOT

__all__ = ["DERIVATIONS", "Derivation", "derive_features", "expand_wanted",
           "FeatureCosts",
           "load_costs", "REQUIRED_DROP", "OPTIONAL_DROP", "ABLATION_COMMAND",
           "UNLEARNED_MISSING_FLOOR"]

# Thresholds recorded in FINDINGS 7d before use.
REQUIRED_DROP = 0.010
"""Concordance drop at or above which a feature is required (or must be derivable)."""
OPTIONAL_DROP = 0.002
"""Below this, an absence is reported without a cost claim."""

UNLEARNED_MISSING_FLOOR = 0.01
"""Training missing rate below which an absent column has no *learned* NaN route.

The second rule of the required/optional gate, and the one the ablation table cannot
supply. A LightGBM split sends NaN whichever way the training data taught it; a
feature the training data never saw missing taught it nothing, so the booster falls
back to its default direction -- a fixed, arbitrary, unmeasured choice that applies
to every row at once. That is not degradation, it is a silent constant.

Ablation measures the *discrimination* lost when a feature goes missing, and
discrimination is a ranking. An unlearned default moves the whole PD distribution
instead, which a ranking metric is blind to by construction. Measured on
the file that found it (FINDINGS 7o): fico_midpoint costs 0.0057 concordance, below
REQUIRED_DROP, yet its absence alone moved mean PD by +0.056 and the rejection rate
by +13.9 points. The two rules catch different dangers and neither implies the other.
"""

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


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """A raw column as float64, read the way cleaning reads it, so ``$12,425`` and
    ``8.19%`` feed a derivation instead of silently becoming missing."""
    return coerce_numeric(df[col], dtype="float64")[0]


def _installment(df: pd.DataFrame) -> pd.Series:
    """Standard amortisation: P = L * i / (1 - (1+i)^-n), i the monthly rate.

    ``int_rate`` is an annual percentage in this data, so it is converted to a
    monthly fraction. A zero rate degenerates to the straight division, which the
    formula cannot express.
    """
    amount = _num(df, "loan_amnt")
    annual = _num(df, "int_rate")
    term = parse_term_months(df["term_months"])
    monthly = annual / 100.0 / 12.0
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = 1.0 - np.power(1.0 + monthly, -term)
        payment = np.where(monthly > 0, amount * monthly / factor, amount / term)
    return pd.Series(payment, index=df.index).astype("float64")


def _fico_midpoint(df: pd.DataFrame) -> pd.Series:
    low = _num(df, "fico_range_low")
    high = _num(df, "fico_range_high")
    return (low + high) / 2.0


def _fico_high_from_low(df: pd.DataFrame) -> pd.Series:
    """Lending Club reports FICO in 4-point bands, so the top of the band is the
    bottom plus four. Exact for this data, and stated as an assumption."""
    return _num(df, "fico_range_low") + 4.0


def _term_months(df: pd.DataFrame) -> pd.Series:
    """The cleaning module's parser, so a term reads the same whether the file
    called the column ``term`` or ``term_months``."""
    return parse_term_months(df["term"])


def _loan_to_income(df: pd.DataFrame) -> pd.Series:
    amount = _num(df, "loan_amnt")
    income = _num(df, "annual_inc").replace(0.0, np.nan)
    return amount / income


def _installment_to_income(df: pd.DataFrame) -> pd.Series:
    payment = _num(df, "installment")
    income = _num(df, "annual_inc").replace(0.0, np.nan)
    return payment / (income / 12.0)


def _emp_length_years(df: pd.DataFrame) -> pd.Series:
    """The parser training uses, so '5 years' is 5.0 here exactly as it was there."""
    from .features.encoders import parse_emp_length
    return parse_emp_length(df["emp_length"])


def _log_annual_inc(df: pd.DataFrame) -> pd.Series:
    """As features.build.add_derived_features computes it: log1p of the income
    clipped at zero, in float32."""
    income = _num(df, "annual_inc")
    return pd.Series(np.log1p(income.clip(lower=0).astype("float32")),
                     index=df.index).astype("float32")


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
    Derivation("emp_length_years", ("emp_length",),
               "years from an employment length like '5 years'", _emp_length_years),
    Derivation("log_annual_inc", ("annual_inc",),
               "log of annual income", _log_annual_inc),
)


def expand_wanted(wanted) -> list[str]:
    """``wanted`` plus any derivable intermediate that one of them needs.

    ``fico_midpoint`` needs ``fico_range_high``, which is itself derived from
    ``fico_range_low``. ``fico_range_high`` is superseded, so it is not a model
    feature and never appears in a spec -- and :func:`_apply` runs a rule only when
    its own feature is wanted. Without the intermediate in the set the chain breaks
    quietly: the raw column is recognised, the feature is not computed, and the file
    is scored without it while the report claims the column was understood.
    """
    out = list(wanted)
    derivable = {rule.feature for rule in DERIVATIONS}
    for _pass in range(len(DERIVATIONS)):
        for rule in DERIVATIONS:
            if rule.feature not in out:
                continue
            for col in rule.needs:
                if col in derivable and col not in out:
                    out.append(col)
    return out


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
    """What a feature's absence costs, and whether the model can absorb it at all.

    Two independent rules, because they answer different questions:

    * **Ablation cost** (``per_feature``, from the 03e table): how much ranking
      power the model loses when this column is missing. Required above
      :data:`REQUIRED_DROP`.
    * **Training missingness** (``train_missing``): whether the model has a *learned*
      route for this column being missing at all. Below
      :data:`UNLEARNED_MISSING_FLOOR` it does not, and its absence is an unlearned
      default rather than a degradation -- see that constant for the measurement.

    A feature can be cheap by the first rule and dangerous by the second. Nothing in
    the ablation table can reveal that, which is why both are applied here.
    """

    model_tag: str = ""
    baseline_concordance: float = float("nan")
    per_feature: dict[str, float] = field(default_factory=dict)
    per_group: dict[str, float] = field(default_factory=dict)
    source: str = ""
    train_missing: dict[str, float] = field(default_factory=dict)
    """Feature -> share of the model's training rows where it was missing. Empty
    when unknown, and an unknown rate is never treated as a low one."""
    unlearned_floor: float = UNLEARNED_MISSING_FLOOR
    train_missing_rows: int = 0
    train_missing_source: str = ""
    structural: tuple[str, ...] = ()
    """Features whose missingness is itself a trained input (FeatureSpec.
    structural_missing). They carry a <name>_missing indicator the model was fitted
    on, so a blank is an answer the model has a route for, and the unlearned-default
    rule does not apply to them however rare that blank was."""

    def tier(self, feature: str) -> str:
        drop = self.per_feature.get(feature)
        if feature in STRUCTURALLY_REQUIRED:
            return "required"
        # The ablation rule first, so a feature that is expensive *and* unlearned
        # keeps the stronger tier rather than being reclassified into the new one.
        if drop is not None and drop >= REQUIRED_DROP:
            return "required"
        if self.unlearned(feature):
            return "unlearned_missing"
        if drop is None:
            return "unmeasured"
        return "optional_costed" if drop >= OPTIONAL_DROP else "optional_free"

    def unlearned(self, feature: str) -> bool:
        """Whether this feature's absence would hit an unlearned default.

        False for a structural feature, whose missing indicator is trained, and
        false when the training rate is unknown -- a guess in this direction would
        quietly re-create the hole it exists to close.
        """
        if feature in self.structural:
            return False
        rate = self.train_missing.get(feature)
        return rate is not None and rate < self.unlearned_floor

    def unlearned_missing(self, features) -> list[str]:
        """Those of ``features`` whose absence has no learned route, in order."""
        return [f for f in features if self.unlearned(f)]

    @property
    def knows_train_missing(self) -> bool:
        return bool(self.train_missing)

    def with_train_missing(self, rates: dict[str, float], *, rows: int = 0,
                           source: str = "", structural=(),
                           floor: float | None = None) -> "FeatureCosts":
        """A copy carrying the training missing rates, which the ablation table does
        not hold. Separate from :func:`load_costs` because the rates come from the
        model's own training rows, which only a loaded bundle has."""
        return replace(self, train_missing={k: float(v) for k, v in rates.items()},
                       train_missing_rows=int(rows), train_missing_source=source,
                       structural=tuple(structural),
                       unlearned_floor=(self.unlearned_floor if floor is None
                                        else float(floor)))

    def unlearned_note(self, features=()) -> str:
        """One sentence naming the second rule and what it caught, for a reader who
        has to decide whether to trust the run."""
        if not self.knows_train_missing:
            return ("The training missing rates for this model are not available, so "
                    "the unlearned-default rule could not be applied. Absences were "
                    "judged on ablation cost alone, which cannot see a shift in the "
                    "level of predicted risk (FINDINGS 7o).")
        caught = self.unlearned_missing(features) if features else []
        head = (f"A feature missing in under {self.unlearned_floor:.0%} of the "
                f"{self.train_missing_rows:,} training rows has no learned route for "
                f"being absent, so its absence is an unlearned default rather than a "
                f"degradation, whatever its ablation cost says (FINDINGS 7o).")
        if not caught:
            return head
        return head + " Caught here: " + ", ".join(
            f"{f} (missing in {self.train_missing.get(f, 0.0):.2%} of training rows, "
            f"ablation cost {self.per_feature.get(f, float('nan')):.4f})"
            for f in caught) + "."

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
            note = (f"Required features come from the measured ablation of "
                    f"{self.model_tag or 'this model'} ({self.source}): a feature is "
                    f"required when its absence costs at least {REQUIRED_DROP:.3f} "
                    f"concordance. {len(self.required())} of {len(self.per_feature)} "
                    f"features qualify.")
            if self.knows_train_missing:
                n = sum(1 for f in self.per_feature if self.unlearned(f))
                note += (f" {n} of them are also governed by the second rule "
                         f"below, which ablation cost does not decide.")
            return note
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
