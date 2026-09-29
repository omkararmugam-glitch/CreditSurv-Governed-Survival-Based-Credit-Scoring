"""UI status helpers. The key check: the pre-flight prediction of which stages would
refuse to overwrite matches what the real scripts do (they refuse before loading
any data, so this runs them safely against an empty temporary output tree)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from creditsurv.config import load_config
from creditsurv.plan import holdout_plan
from creditsurv.provenance import OVERWRITE_REFUSED, PROJECT_ROOT, build_stamp
from creditsurv.status import (discover_tags, figures_for_tag, fingerprint_cache,
                               preflight, stage_status)

TAG = "uitest"


@pytest.fixture
def tmp_cfg(tmp_path):
    text = (PROJECT_ROOT / "config" / "config.yaml").read_text(encoding="utf-8")
    for d in ("data", "models", "figures", "tables"):
        (tmp_path / d).mkdir()
        text = text.replace(f"{d}_dir: outputs/{d}", f"{d}_dir: {(tmp_path / d).as_posix()}")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(text, encoding="utf-8")
    paths = load_config(cfg).paths
    assert paths.tables_dir == tmp_path / "tables"      # really redirected
    return cfg, paths


def test_empty_tree_nothing_blocked(tmp_cfg):
    _, paths = tmp_cfg
    assert preflight(holdout_plan("Small", tag=TAG), paths) == {}


def test_preflight_matches_script_guards(tmp_cfg):
    cfg, paths = tmp_cfg
    strat = f"{TAG}_strat"
    for name in (f"02_metrics_{TAG}.json", f"03_explain_{TAG}.json",
                 f"03c_noise_floor_{TAG}.json", f"03_explain_{strat}.json",
                 f"03_segment_bootstrap_{strat}.json", f"04_reject_inference_{TAG}.json"):
        (paths.tables_dir / name).write_text("{}", encoding="utf-8")
    plan = holdout_plan("Small", tag=TAG, only=("02", "03", "03c", "03s", "03b", "04"))
    predicted = preflight(plan, paths)
    assert set(predicted) == {"02", "03", "03c", "03s", "03b", "04"}
    for st in plan.stages:
        r = subprocess.run([sys.executable, *st.args, "--config", str(cfg)],
                           cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300)
        assert r.returncode == OVERWRITE_REFUSED, (st.key, r.stdout, r.stderr)
        for p in predicted[st.key]:                      # the script names every file
            assert p.name in r.stderr, (st.key, p.name)


def test_stage2_models_dir_is_guarded(tmp_cfg):
    _, paths = tmp_cfg
    (paths.models_dir / f"02_models_{TAG}.pkl").write_bytes(b"x")
    hits = preflight(holdout_plan("Small", tag=TAG), paths)
    assert list(hits) == ["02"]


def test_strat_variant_does_not_block_base(tmp_cfg):
    _, paths = tmp_cfg
    (paths.tables_dir / f"03_explain_{TAG}_strat.json").write_text("{}", encoding="utf-8")
    hits = preflight(holdout_plan("Small", tag=TAG), paths)
    assert "03" not in hits and "03s" in hits


def test_stage_status_states(tmp_path, monkeypatch):
    tables = tmp_path / "tables"
    tables.mkdir()
    inp = tmp_path / "input.bin"
    inp.write_bytes(b"one")
    assert stage_status(tables, TAG, "02").state == "not run"

    (tables / f"02_metrics_{TAG}.json").write_text(json.dumps({"n": 1}), encoding="utf-8")
    assert stage_status(tables, TAG, "02").state == "unverified"

    stamp = build_stamp(stage="02", inputs={"data": inp})
    (tables / f"02_metrics_{TAG}.json").write_text(
        json.dumps({"provenance": stamp}), encoding="utf-8")
    fp = fingerprint_cache()
    assert stage_status(tables, TAG, "02", fp).state == "verified"

    inp.write_bytes(b"two")                              # input replaced after the run
    s = stage_status(tables, TAG, "02", fingerprint_cache())
    assert s.state == "changed" and "data changed" in s.detail

    inp.unlink()
    assert "data missing" in stage_status(tables, TAG, "02").detail


def test_discover_tags_folds_strat(tmp_path):
    for n in ("02_metrics_a.json", "03_explain_a_strat.json", "04_reject_inference_b.json",
              "02_model_comparison_c.csv"):
        (tmp_path / n).write_text("{}", encoding="utf-8")
    assert discover_tags(tmp_path) == ["a", "b"]


def test_figures_for_tag_exact_suffix(tmp_path):
    for n in ("02_km_x.png", "03_imp_x_strat.png", "02_km_xy.png"):
        (tmp_path / n).write_bytes(b"")
    assert [p.name for p in figures_for_tag(tmp_path, "x")] == ["02_km_x.png"]


def test_findings_diff_ignores_line_endings(tmp_path):
    import subprocess

    from creditsurv.status import findings_diff

    body = "# F\n\n## 1. a\n\nx\n\n## 6. holdout\n\ny\n"
    (tmp_path / "FINDINGS.md").write_text(body, encoding="utf-8", newline="\n")
    git = lambda *a: subprocess.run(["git", "-C", str(tmp_path), *a], check=True,  # noqa: E731
                                    capture_output=True)
    git("init", "-q")
    git("-c", "user.email=t@t", "-c", "user.name=t", "add", "FINDINGS.md")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "c")
    (tmp_path / "FINDINGS.md").write_bytes(body.replace("\n", "\r\n").encode())
    d = findings_diff(tmp_path)
    assert d["available"] and not d["diff"] and not d["above_s6"]

    (tmp_path / "FINDINGS.md").write_text(body.replace("x", "changed"), encoding="utf-8")
    assert findings_diff(tmp_path)["above_s6"]
    (tmp_path / "FINDINGS.md").write_text(body + "z\n", encoding="utf-8")
    d = findings_diff(tmp_path)
    assert d["diff"] and not d["above_s6"]
