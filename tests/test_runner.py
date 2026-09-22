"""Background runner, on tiny fake stage scripts (never the real pipeline)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from creditsurv.runner import (LockHeld, active_lock, effective_state, launch,
                               list_runs, read_status, tail_log)


def _script(tmp_path: Path, name: str, body: str) -> str:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return str(p)


def _stages(*specs):
    return [{"key": k, "name": f"Stage {k}", "tag": "t", "args": [script]}
            for k, script in specs]


def test_all_stages_complete(tmp_path):
    ok = _script(tmp_path, "ok.py", "print('hello from stage')\n")
    runs = tmp_path / "runs"
    run_dir = launch(_stages(("a", ok), ("b", ok)), lock_tag="t", runs_dir=runs,
                     detach=False)
    st = read_status(run_dir)
    assert st["state"] == "completed"
    assert [s["state"] for s in st["stages"]] == ["completed", "completed"]
    log = tail_log(run_dir)
    assert log.count("hello from stage") == 2
    assert "[1/2] Stage a  completed in" in log and "Sequence complete." in log
    assert active_lock("t", runs) is None                 # lock released
    assert effective_state(run_dir) == "completed"


def test_stops_at_first_failure_without_retry(tmp_path):
    ok = _script(tmp_path, "ok.py", "print('ran')\n")
    refuse = _script(tmp_path, "refuse.py", "import sys; print('refusing'); sys.exit(4)\n")
    marker = tmp_path / "later_ran"
    later = _script(tmp_path, "later.py", f"open(r'{marker}', 'w').close()\n")
    runs = tmp_path / "runs"
    run_dir = launch(_stages(("a", ok), ("b", refuse), ("c", later)), lock_tag="t",
                     runs_dir=runs, detach=False)
    st = read_status(run_dir)
    assert st["state"] == "failed" and st["failed_stage"] == "Stage b"
    assert [s["state"] for s in st["stages"]] == ["completed", "failed", "not run"]
    assert st["stages"][1]["exit_code"] == 4
    assert not marker.exists()
    log = tail_log(run_dir)
    assert log.count("refusing") == 1                     # run once, not retried
    assert "FAILED (exit code 4)" in log and "Exit 4 = the stage refused" in log
    assert "Stopped. No later stage was run" in log
    assert active_lock("t", runs) is None


def test_live_lock_refuses_second_run(tmp_path):
    runs = tmp_path / "runs"
    (runs / "locks").mkdir(parents=True)
    (runs / "locks" / "t.lock").write_text(json.dumps(
        {"run_id": "other", "pid": os.getpid(), "started": "now"}), encoding="utf-8")
    with pytest.raises(LockHeld):
        launch(_stages(("a", "x.py")), lock_tag="t", runs_dir=runs, detach=False)
    assert list_runs(runs) == []                          # nothing was created


def test_stale_lock_is_ignored(tmp_path):
    ok = _script(tmp_path, "ok.py", "pass\n")
    runs = tmp_path / "runs"
    (runs / "locks").mkdir(parents=True)
    (runs / "locks" / "t.lock").write_text(json.dumps(
        {"run_id": "dead", "pid": 2 ** 22 + 12345, "started": "then"}), encoding="utf-8")
    assert active_lock("t", runs) is None
    run_dir = launch(_stages(("a", ok)), lock_tag="t", runs_dir=runs, detach=False)
    assert read_status(run_dir)["state"] == "completed"


def test_running_status_without_live_lock_is_interrupted(tmp_path):
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "plan.json").write_text(json.dumps({"run_id": "r1", "lock_tag": "t",
                                                   "stages": []}), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"state": "running", "stages": []}),
                                         encoding="utf-8")
    assert effective_state(run_dir) == "interrupted"


def test_detached_run_survives_and_finishes(tmp_path):
    ok = _script(tmp_path, "ok.py", "import time; time.sleep(0.5); print('detached ok')\n")
    runs = tmp_path / "runs"
    run_dir = launch(_stages(("a", ok)), lock_tag="t", runs_dir=runs,
                     python=sys.executable, detach=True)
    deadline = time.time() + 60
    while time.time() < deadline and read_status(run_dir)["state"] not in ("completed", "failed"):
        time.sleep(0.25)
    assert read_status(run_dir)["state"] == "completed", \
        (run_dir / "runner.out").read_text(encoding="utf-8", errors="replace")
    assert "detached ok" in tail_log(run_dir)
    assert active_lock("t", runs) is None


def test_unreadable_lock_is_not_mistaken_for_a_dead_run(tmp_path):
    """A lock file caught mid-replacement must not make a live run look dead:
    that false reading is what ended a background scoring run early."""
    from creditsurv.runner import read_lock

    runs = tmp_path / "runs"
    run_dir = runs / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "plan.json").write_text(json.dumps({"run_id": "r1", "lock_tag": "t",
                                                   "stages": []}), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"state": "running", "stages": []}),
                                         encoding="utf-8")
    lock = runs / "locks" / "t.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("{ truncated", encoding="utf-8")        # half-written

    info, readable = read_lock("t", runs)
    assert info is None and readable is False
    assert effective_state(run_dir) == "running"            # not "interrupted"
    assert active_lock("t", runs)["unreadable"] is True     # and still held

    lock.unlink()                                           # genuinely gone
    assert effective_state(run_dir) == "interrupted"
