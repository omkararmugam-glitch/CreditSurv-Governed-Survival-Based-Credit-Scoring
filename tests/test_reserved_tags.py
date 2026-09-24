"""Reserved tags may only be written by the run they were registered for.

This exists because 02_models_holdout.pkl spent a day holding a 200k dev-sample
monthly-bin model while FINDINGS section 6 was reserved for the pre-registered
full-data run (FINDINGS 7d).
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from creditsurv.provenance import PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "02_train_models.py"


@pytest.fixture(scope="module")
def stage2():
    spec = importlib.util.spec_from_file_location("stage2_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(**kw):
    defaults = dict(full=False, time_bin=None, negative_subsample=1.0,
                    split="random", oot_cutoff=2016)
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_registered_holdout_settings_are_accepted(stage2):
    ok = _args(full=True, time_bin=3, negative_subsample=0.4,
               split="out_of_time", oot_cutoff=2016)
    assert stage2.check_reserved_tag("holdout", ok) == []


def test_every_mismatch_is_named(stage2):
    problems = stage2.check_reserved_tag("holdout", _args())
    joined = " | ".join(problems)
    assert "--full" in joined and "--time-bin" in joined
    assert "--negative-subsample" in joined and "--split" in joined
    for problem in problems:                     # each says what was expected and why
        assert "expected" in problem and ":" in problem


@pytest.mark.parametrize("wrong,setting", [
    (dict(full=False), "--full"),
    (dict(time_bin=1), "--time-bin"),
    (dict(negative_subsample=1.0), "--negative-subsample"),
    (dict(split="random"), "--split"),
    (dict(oot_cutoff=2015), "--oot-cutoff"),
])
def test_one_wrong_setting_is_enough_to_refuse(stage2, wrong, setting):
    base = dict(full=True, time_bin=3, negative_subsample=0.4,
                split="out_of_time", oot_cutoff=2016)
    base.update(wrong)
    problems = stage2.check_reserved_tag("holdout", _args(**base))
    assert len(problems) == 1 and problems[0].startswith(setting)


def test_full_tag_is_reserved_too(stage2):
    assert stage2.check_reserved_tag("full", _args(full=True, time_bin=3,
                                                   negative_subsample=0.4)) == []
    assert stage2.check_reserved_tag("full", _args(full=False, time_bin=3,
                                                   negative_subsample=0.4))


def test_derived_tags_are_not_reserved(stage2):
    """holdout_lcgrade, holdout_unpriced and the rest are ordinary tags: only the
    exact names other sections cite are protected."""
    for tag in ("holdout_lcgrade", "holdout_nograde", "holdout_small",
                "holdout_devsample", "full_v2", "dev"):
        assert stage2.check_reserved_tag(tag, _args()) == []


def test_the_command_that_caused_the_mixup_is_refused_end_to_end(stage2, monkeypatch,
                                                                capsys):
    """The real entry point, with the real arguments, refusing before it reads
    anything: exit code 2 and every mismatch named."""
    monkeypatch.setattr(sys, "argv", ["02_train_models.py", "--split", "out_of_time",
                                      "--oot-cutoff", "2016", "--tag", "holdout"])
    code = stage2.main()
    err = capsys.readouterr().err
    assert code == 2
    assert "'holdout' is a reserved tag" in err
    assert "--full is False, expected True" in err
    assert "Nothing has been run" in err
