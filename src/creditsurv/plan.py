"""The single definition of the holdout stage sequence.

Both ``run_holdout.ps1`` and the Streamlit UI build their commands from this
module, so the command line and the UI cannot drift apart: if they ever disagree,
that is a bug here, and ``tests/test_plan.py`` pins the output to exactly what the
wrapper ran before it was switched over.

Every flag below is one the stage scripts actually accept (verified against their
argparse definitions). This module only *describes* commands; it runs nothing.

Usage (the wrapper calls this)::

    python -m creditsurv.plan --size Small --json
    python -m creditsurv.plan --size Full --overwrite --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass

__all__ = ["SIZES", "SizeSettings", "Stage", "Plan", "holdout_plan", "validate_tag",
           "default_tag", "STAGE_KEYS", "TagError"]


@dataclass(frozen=True)
class SizeSettings:
    tag: str
    full_data: bool
    n_explain: int
    n_background: int
    nsamples: int
    per_stratum: int
    noise_n: int
    n_boot: int


SIZES: dict[str, SizeSettings] = {
    # 200k dev sample, minimal SHAP work: a smoke test, not results to read.
    "Small": SizeSettings("holdout_small", False, 20, 25, 146, 8, 10, 500),
    # 200k dev sample, production SHAP settings, moderate n.
    "Medium": SizeSettings("holdout_medium", False, 100, 100, 600, 20, 20, 2000),
    # Full data, exactly the pre-registered design (commit afcd4a4).
    "Full": SizeSettings("holdout", True, 250, 100, 600, 36, 40, 5000),
}
DEFAULT_SIZE = "Small"

STAGE_KEYS = ("02", "03", "03c", "03s", "03b", "04", "05", "prov")
"""Stable keys, in run order, used by the UI's stage checkboxes."""


@dataclass(frozen=True)
class Stage:
    key: str
    name: str
    tag: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class Plan:
    size: str
    tag: str
    strat_tag: str
    overwrite: bool
    write_findings: bool
    stages: tuple[Stage, ...]

    def to_json(self) -> str:
        d = asdict(self)
        d["stages"] = [{**asdict(s), "args": list(s.args)} for s in self.stages]
        return json.dumps(d, indent=2)


class TagError(ValueError):
    """A tag that would collide with, or shadow, results it must not touch."""


_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,48}$")
_PRIMARY = {"full", "dev"}


def default_tag(size: str) -> str:
    return SIZES[size].tag


def validate_tag(tag: str, size: str) -> None:
    """Refuse tags that could overwrite or be mistaken for other results.

    * ``full`` / ``dev`` (and their ``_strat`` forms) hold the primary results.
    * Each size's default tag belongs to that size only. In particular ``holdout``
      is the pre-registered Full run: a Small run under it would fill section 6
      with smoke-test numbers and then block the real run.
    * The ``_strat`` suffix is added automatically, so a tag may not end in it.
    """
    if not _TAG_RE.match(tag or ""):
        raise TagError("tag must be 1-49 characters: lowercase letters, digits and "
                       "underscores, starting with a letter or digit")
    if tag.endswith("_strat"):
        raise TagError("do not end the tag in _strat; the stratified run adds it")
    if tag in _PRIMARY or tag.split("_")[0] in _PRIMARY:
        raise TagError(f"{tag!r} would collide with the primary results "
                       f"(tags starting 'full' or 'dev')")
    for other, s in SIZES.items():
        if tag == s.tag and other != size:
            raise TagError(f"{tag!r} is the default tag of -Size {other}; using it for "
                           f"{size} would mix results from two different run sizes")


def holdout_plan(size: str = DEFAULT_SIZE, *, overwrite: bool = False,
                 tag: str | None = None, only: tuple[str, ...] | None = None) -> Plan:
    """The out-of-time holdout sequence for one size.

    ``tag`` overrides the size's default tag (validated). ``only`` restricts the
    plan to a subset of :data:`STAGE_KEYS`, keeping run order.
    """
    if size not in SIZES:
        raise ValueError(f"size must be one of {list(SIZES)}, got {size!r}")
    s = SIZES[size]
    tag = tag or s.tag
    validate_tag(tag, size)
    strat = f"{tag}_strat"
    # FINDINGS.md is written only by the pre-registered Full run under its own tag.
    # Every other combination previews section 6 (--dry-run) and never writes.
    write = size == "Full" and tag == SIZES["Full"].tag
    ow = ["--overwrite"] if overwrite else []
    shap = ["--nsamples", str(s.nsamples), "--n-background", str(s.n_background)]

    stages = [
        Stage("02", "Stage 2  train 2007-2015, test 2016-2018", tag,
              ("scripts/02_train_models.py", *(["--full"] if s.full_data else []),
               "--split", "out_of_time", "--oot-cutoff", "2016", "--tag", tag,
               "--time-bin", "3", "--negative-subsample", "0.4", *ow)),
        Stage("03", "Stage 3  naive vs SurvSHAP(t) comparison", tag,
              ("scripts/03_explain.py", "--tag", tag, "--model-tag", tag,
               "--n-explain", str(s.n_explain), *shap, *ow)),
        Stage("03c", "Stage 3c noise floor on the holdout model", tag,
              ("scripts/03c_noise_floor.py", "--tag", tag, "--model-tag", tag,
               "--n-explain", str(s.noise_n), *shap, *ow)),
        Stage("03s", "Stage 3  grade-stratified explanations", strat,
              ("scripts/03_explain.py", "--tag", strat, "--model-tag", tag,
               "--stratify-by", "grade", "--per-stratum", str(s.per_stratum),
               "--skip-naive", *shap, *ow)),
        Stage("03b", "Stage 3b pre-registered grade G bootstrap", strat,
              ("scripts/03b_bootstrap_segments.py", "--tag", strat, "--target", "G",
               "--features", "annual_inc", "mths_since_recent_inq",
               "--n-boot", str(s.n_boot), *ow)),
        Stage("04", "Stage 4  selection-bias diagnostic only", tag,
              ("scripts/04_reject_inference.py", "--tag", tag, "--model-tag", tag,
               "--diagnostic-only", "--accepted-years", "2016-2018",
               "--rejected-years", "2016-2018", *ow)),
        Stage("05", "Stage 5  report (section 6" + (")" if write else " preview only)"),
              f"{tag} / {strat}",
              ("scripts/05_report.py", "--tag", "full", "--holdout-tag", tag,
               "--holdout-strat-tag", strat, *([] if write else ["--dry-run"]))),
        Stage("prov", "Provenance check", "(all)", ("scripts/check_provenance.py",)),
    ]
    if only is not None:
        unknown = set(only) - set(STAGE_KEYS)
        if unknown:
            raise ValueError(f"unknown stage keys: {sorted(unknown)}")
        stages = [st for st in stages if st.key in only]
    return Plan(size, tag, strat, overwrite, write, tuple(stages))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default=DEFAULT_SIZE, choices=list(SIZES))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--json", action="store_true", help="print the plan as JSON")
    args = ap.parse_args(argv)
    try:
        plan = holdout_plan(args.size, overwrite=args.overwrite, tag=args.tag)
    except (TagError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(plan.to_json())
    else:
        for st in plan.stages:
            print(f"[{st.key}] {st.name}  (--tag {st.tag})\n    python {' '.join(st.args)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
