"""The holdout stage plan: pinned to what run_holdout.ps1 ran before it was
switched to creditsurv.plan, and the tag guardrails."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from creditsurv.plan import STAGE_KEYS, TagError, holdout_plan, validate_tag
from creditsurv.provenance import PROJECT_ROOT

PINNED = json.loads((Path(__file__).parent / "fixtures" / "wrapper_commands_pinned.json")
                    .read_text(encoding="utf-8"))["plans"]
VARIANTS = [(k.split("|")[0], k.endswith("True")) for k in PINNED]


def _as_dicts(plan):
    return [{"name": s.name, "tag": s.tag, "args": list(s.args)} for s in plan.stages]


@pytest.mark.parametrize("size,overwrite", VARIANTS)
def test_plan_reproduces_pinned_wrapper_commands(size, overwrite):
    assert _as_dicts(holdout_plan(size, overwrite=overwrite)) == \
        PINNED[f"{size}|overwrite={overwrite}"]


@pytest.mark.skipif(shutil.which("powershell") is None, reason="needs Windows PowerShell")
@pytest.mark.parametrize("size,overwrite", VARIANTS)
def test_wrapper_plan_only_matches_pinned(size, overwrite):
    """The wrapper itself, via -PlanOnly (prints commands, runs nothing)."""
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
           str(PROJECT_ROOT / "run_holdout.ps1"), "-Size", size, "-PlanOnly"]
    if overwrite:
        cmd.append("-Overwrite")
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                         cwd=PROJECT_ROOT, timeout=120)
    assert out.returncode == 0, out.stderr
    got = []
    for line in out.stdout.splitlines():
        if line.startswith("STAGE|"):
            _, name, tag, args = line.split("|", 3)
            got.append({"name": name, "tag": tag, "args": args.split("\x1f")})
    assert got == PINNED[f"{size}|overwrite={overwrite}"]


def test_only_findings_writer_is_full_default_tag():
    assert holdout_plan("Full").write_findings
    for size, kw in [("Small", {}), ("Medium", {}), ("Full", {"tag": "holdout_rerun"})]:
        p = holdout_plan(size, **kw)
        assert not p.write_findings
        report = next(s for s in p.stages if s.key == "05")
        assert "--dry-run" in report.args


def test_stage5_always_reports_primary_tag_full():
    for size in ("Small", "Medium", "Full"):
        report = next(s for s in holdout_plan(size).stages if s.key == "05")
        i = report.args.index("--tag")
        assert report.args[i + 1] == "full"


def test_default_is_small_and_no_overwrite():
    p = holdout_plan()
    assert p.size == "Small" and not p.overwrite
    assert not any("--overwrite" in s.args for s in p.stages)
    assert not any("--full" in s.args for s in p.stages)


@pytest.mark.parametrize("tag", ["full", "dev", "full_v2", "dev_x", "x_strat", "Bad",
                                 "", "a b", "holdout"])
def test_rejected_tags(tag):
    with pytest.raises(TagError):
        validate_tag(tag, "Small")


def test_size_default_tags_belong_to_their_size():
    validate_tag("holdout", "Full")
    validate_tag("holdout_small", "Small")
    with pytest.raises(TagError):
        validate_tag("holdout_medium", "Small")
    validate_tag("fullness_check", "Small")     # only the whole word 'full' is refused


def test_only_subset_keeps_order():
    p = holdout_plan("Small", only=("04", "02"))
    assert [s.key for s in p.stages] == ["02", "04"]
    assert [s.key for s in holdout_plan("Small").stages] == list(STAGE_KEYS)
    with pytest.raises(ValueError):
        holdout_plan("Small", only=("99",))
