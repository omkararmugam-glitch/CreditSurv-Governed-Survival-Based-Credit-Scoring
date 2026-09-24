"""Tests for overwrite guards and provenance stamps.

The script-level tests point each stage at a temporary config whose directories
are empty apart from one pre-existing output. A correct guard refuses before
loading anything; if a guard were missing, the script would still fail at once on
the absent data -- so no pipeline stage can actually run inside this suite.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from creditsurv.provenance import (
    OVERWRITE_REFUSED,
    PROJECT_ROOT,
    build_stamp,
    file_fingerprint,
    find_existing_outputs,
    git_state,
    guard_outputs,
    verify_stamp,
)


class TestFingerprint:
    def test_sha256_matches_hashlib(self, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"lending club" * 1000)
        assert file_fingerprint(f)["sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()

    def test_streaming_matches_for_multi_chunk_files(self, tmp_path):
        f = tmp_path / "big.bin"
        f.write_bytes(bytes(range(256)) * 20_000)
        fp = file_fingerprint(f, chunk=4096)
        assert fp["sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()
        assert fp["bytes"] == f.stat().st_size

    def test_missing_file_is_recorded_not_raised(self, tmp_path):
        assert file_fingerprint(tmp_path / "nope")["exists"] is False


class TestFindExisting:
    def test_tag_does_not_match_a_longer_variant_tag(self, tmp_path):
        for name in ("03_explain_full.json", "03_explain_full_strat.json",
                     "03_segment_importance_grade_full.csv", "02_metrics_full.json"):
            (tmp_path / name).write_text("{}")
        hits = {p.name for p in find_existing_outputs([tmp_path], "03", "full")}
        assert hits == {"03_explain_full.json", "03_segment_importance_grade_full.csv"}

    def test_variant_tag_matches_only_itself(self, tmp_path):
        for name in ("03_explain_full.json", "03_explain_full_strat.json"):
            (tmp_path / name).write_text("{}")
        hits = {p.name for p in find_existing_outputs([tmp_path], "03", "full_strat")}
        assert hits == {"03_explain_full_strat.json"}

    def test_missing_directory_is_fine(self, tmp_path):
        assert find_existing_outputs([tmp_path / "absent"], "02", "dev") == []


class TestGuard:
    def test_refuses_when_output_exists(self, tmp_path, capsys):
        f = tmp_path / "02_metrics_dev.json"
        f.write_text("{}")
        assert guard_outputs([f], overwrite=False, script="x") == OVERWRITE_REFUSED
        err = capsys.readouterr().err
        assert "--overwrite" in err and "Nothing has been run" in err

    def test_allows_and_lists_with_overwrite(self, tmp_path, capsys):
        f = tmp_path / "02_metrics_dev.json"
        f.write_text("{}")
        assert guard_outputs([f], overwrite=True, script="x") is None
        assert "replacing 1 existing output" in capsys.readouterr().out

    def test_passes_silently_when_nothing_exists(self, tmp_path, capsys):
        assert guard_outputs([tmp_path / "new.json"], overwrite=False, script="x") is None
        assert capsys.readouterr().err == ""


class TestStampAndVerify:
    def test_stamp_records_inputs_outputs_and_environment(self, tmp_path):
        inp, out = tmp_path / "in.parquet", tmp_path / "model.pkl"
        inp.write_bytes(b"data")
        out.write_bytes(b"model")
        s = build_stamp(stage="t", inputs={"data": inp}, outputs={"model": out},
                        args={"path": tmp_path, "n": 3})
        assert s["inputs"]["data"]["sha256"] == hashlib.sha256(b"data").hexdigest()
        assert s["outputs"]["model"]["sha256"] == hashlib.sha256(b"model").hexdigest()
        assert s["args"]["path"] == str(tmp_path) and s["args"]["n"] == 3
        assert s["environment"]["python"] and "pandas" in s["environment"]["packages"]

    def test_verify_detects_overwrite_and_deletion(self, tmp_path):
        f = tmp_path / "model.pkl"
        f.write_bytes(b"version 1")
        s = build_stamp(stage="t", inputs={"model": f})
        assert verify_stamp(s)[0]["status"] == "ok"
        f.write_bytes(b"version 2")
        assert verify_stamp(s)[0]["status"] == "changed"
        f.unlink()
        assert verify_stamp(s)[0]["status"] == "missing"

    def test_verify_distinguishes_already_missing(self, tmp_path):
        s = build_stamp(stage="t", inputs={"model": tmp_path / "never.pkl"})
        assert verify_stamp(s)[0]["status"] == "was_missing"

    def test_git_state_is_none_outside_this_repo(self, tmp_path):
        assert git_state(tmp_path) is None

    def test_git_state_reads_this_repo(self):
        state = git_state(PROJECT_ROOT)
        if state is None:
            pytest.skip("project is not a git repository in this environment")
        assert len(state["commit"]) == 40


def _config(tmp_path: Path) -> Path:
    for d in ("data", "tables", "figures", "models"):
        (tmp_path / d).mkdir(exist_ok=True)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "paths:\n"
        f"  data_dir: {(tmp_path / 'data').as_posix()}\n"
        f"  tables_dir: {(tmp_path / 'tables').as_posix()}\n"
        f"  figures_dir: {(tmp_path / 'figures').as_posix()}\n"
        f"  models_dir: {(tmp_path / 'models').as_posix()}\n",
        encoding="utf-8",
    )
    return cfg


SCRIPT_CASES = [
    ("01_build_labels.py", "data/accepted_labeled.parquet", []),
    ("02_train_models.py", "tables/02_metrics_dev.json", []),
    # A non-reserved tag: "full" may only be written by its registered
    # settings (tests/test_reserved_tags.py).
    ("02_train_models.py", "models/02_models_probe.pkl",
     ["--full", "--tag", "probe"]),
    ("03_explain.py", "tables/03_explain_dev.json", []),
    ("03_explain.py", "tables/03_explain_full_strat.json", ["--tag", "full_strat"]),
    ("04_reject_inference.py", "tables/04_reject_inference_dev.json", []),
    ("03b_bootstrap_segments.py", "tables/03_segment_bootstrap_full_strat.json", []),
]


def _run(script: str, cfg: Path, extra: list[str]):
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script), "--config", str(cfg),
         *extra],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=300,
    )


@pytest.mark.parametrize("script,existing,extra", SCRIPT_CASES)
def test_script_refuses_to_overwrite(tmp_path, script, existing, extra):
    cfg = _config(tmp_path)
    (tmp_path / existing).write_text("previous result")
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file())

    r = _run(script, cfg, extra)

    assert r.returncode == OVERWRITE_REFUSED, r.stderr
    assert "--overwrite" in r.stderr
    assert (tmp_path / existing).read_text() == "previous result"
    after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file())
    assert after == before, "a refused run must not write anything"


@pytest.mark.parametrize("script,existing,extra", SCRIPT_CASES)
def test_overwrite_flag_gets_past_the_guard(tmp_path, script, existing, extra):
    """With --overwrite the guard steps aside; the script then stops on the missing
    data in the empty temp directories, so nothing is actually run."""
    cfg = _config(tmp_path)
    (tmp_path / existing).write_text("previous result")
    r = _run(script, cfg, [*extra, "--overwrite"])
    assert r.returncode not in (0, OVERWRITE_REFUSED), r.stdout + r.stderr
    assert "replacing" in r.stdout


def test_checker_detects_an_overwritten_input_end_to_end(tmp_path):
    """Stamp a result, overwrite its input, and the checker must fail loudly."""
    import json

    from creditsurv.provenance import build_stamp

    cfg = _config(tmp_path)
    model = tmp_path / "models" / "02_models_x.pkl"
    model.write_bytes(b"model v1")
    result = {"metric": 0.7,
              "provenance": build_stamp(stage="t", inputs={"model": model})}
    (tmp_path / "tables" / "03_explain_x.json").write_text(json.dumps(result))
    (tmp_path / "tables" / "05_report_state.json").write_text("{}")

    ok = _run("check_provenance.py", cfg, [])
    assert ok.returncode == 0, ok.stdout
    assert "[OK     ] 03_explain_x.json" in ok.stdout
    assert "05_report_state" not in ok.stdout

    model.write_bytes(b"model v2 -- silently retrained")
    bad = _run("check_provenance.py", cfg, [])
    assert bad.returncode == 1
    assert "CHANGED" in bad.stdout and "02_models_x.pkl" in bad.stdout


def test_checker_baseline_catches_untracked_artefact_change(tmp_path):
    cfg = _config(tmp_path)
    pkl = tmp_path / "models" / "02_models_full.pkl"
    pkl.write_bytes(b"original")
    assert _run("check_provenance.py", cfg, ["--write-baseline"]).returncode == 0
    assert (tmp_path / "provenance_baseline.json").exists()
    assert _run("check_provenance.py", cfg, []).returncode == 0
    pkl.write_bytes(b"replaced")
    r = _run("check_provenance.py", cfg, [])
    assert r.returncode == 1 and "CHANGED" in r.stdout
