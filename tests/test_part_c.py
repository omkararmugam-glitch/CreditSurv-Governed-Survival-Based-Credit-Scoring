"""Part C: evidence jobs report real progress (C2); the WSL copy keeps itself in
step with Windows and sends results back (C3).

The Linux-only tests drive the real launcher script and a real rsync; they run in
the WSL suite and are skipped on Windows, where none of this applies.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from creditsurv import evidence_jobs as jobs
from creditsurv import wsl_sync
from creditsurv.provenance import PROJECT_ROOT
from creditsurv import runner
from creditsurv.runner import PROGRESS_ENV, launch, progress_env, tail_log

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                reason="the WSL copy exists only under Linux")


# ================================================================ C2 ==

class _Linear:
    def predict_survival(self, X, times):
        r = 0.002 * X["dti"].to_numpy() + 0.001 * X["revol_util"].to_numpy()
        return np.exp(-np.outer(r, np.atleast_1d(times)))


def _toy():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"dti": rng.uniform(1, 40, 60),
                         "revol_util": rng.uniform(0, 100, 60)})


class TestProgress:
    def test_progress_is_display_only(self, monkeypatch, capsys):
        """The same attributions with the progress bar on as off."""
        from creditsurv.explain.survshap import SHOW_PROGRESS_ENV, explain_survshap
        df = _toy()
        args = dict(nsamples=40, n_background=10, seed=3)
        monkeypatch.delenv(SHOW_PROGRESS_ENV, raising=False)
        quiet = explain_survshap(_Linear(), df.iloc[:4], df, np.array([12.0, 36.0]), **args)
        assert "4/4" not in capsys.readouterr().err
        monkeypatch.setenv(SHOW_PROGRESS_ENV, "1")
        loud = explain_survshap(_Linear(), df.iloc[:4], df, np.array([12.0, 36.0]), **args)
        assert "4/4" in capsys.readouterr().err          # shap's bar, on stderr
        np.testing.assert_array_equal(quiet.phi, loud.phi)

    def test_the_runner_passes_progress_settings_through_wsl(self, monkeypatch):
        monkeypatch.setenv("WSLENV", "USERPROFILE/p")
        env = progress_env(True)
        for k, v in PROGRESS_ENV.items():
            assert env[k] == v
        passed = env["WSLENV"].split(":")
        assert "USERPROFILE/p" in passed                 # existing entries kept
        assert all(f"{k}/u" in passed for k in PROGRESS_ENV)
        quiet = progress_env(False)
        assert "CREDITSURV_SHOW_PROGRESS" not in quiet or \
            os.environ.get("CREDITSURV_SHOW_PROGRESS") == quiet["CREDITSURV_SHOW_PROGRESS"]

    def test_evidence_jobs_ask_for_progress(self, monkeypatch, tmp_path):
        seen = {}
        monkeypatch.setattr(jobs, "launch", lambda st, **kw: seen.update(kw) or tmp_path)
        jobs.launch_job("explainer_validation", "m", runs_dir=tmp_path, via_wsl=False)
        assert seen["meta"]["show_progress"] is True

    @pytest.mark.parametrize("log, want", [
        (" 34%|###4      | 340/1000 [15:02<29:10,  2.65s/it]\r"
         " 35%|###5      | 350/1000 [15:28<28:44,  2.65s/it]\n",
         {"done": 350, "total": 1000, "eta": "28:44", "what": "applicants explained"}),
        ("64 ablation passes\n  features 20/64\n  features 30/64\n",
         {"done": 30, "total": 64, "eta": None, "what": "ablation features"}),
        (" 0%|          | 0/300 [00:00<?, ?it/s]",
         {"done": 0, "total": 300, "eta": None, "what": "applicants explained"}),
        ("model m; data x\n", None),
    ])
    def test_progress_is_read_from_the_log(self, log, want):
        assert jobs.parse_progress(log) == want

    def test_a_real_stage_prints_progress_the_page_can_read(self, tmp_path):
        """A background stage running the SurvSHAP engine, as 03d does, with the
        runner's progress settings: its log carries a bar parse_progress reads."""
        script = tmp_path / "stage.py"
        script.write_text(
            "import sys, numpy as np\n"
            f"sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r})\n"
            f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
            "from test_part_c import _Linear, _toy\n"
            "from creditsurv.explain.survshap import explain_survshap\n"
            "df = _toy()\n"
            "explain_survshap(_Linear(), df.iloc[:6], df, np.array([12.0, 36.0]),\n"
            "                 nsamples=40, n_background=10)\n", encoding="utf-8")
        run = launch([{"key": "explainer_validation", "name": "03d-like", "tag": "t",
                       "args": [], "argv": [sys.executable, str(script)]}],
                     lock_tag="t", runs_dir=tmp_path / "runs", detach=False,
                     meta={"show_progress": True})
        prog = jobs.parse_progress(tail_log(run))
        assert prog and prog["total"] == 6 and prog["done"] == 6, tail_log(run)


# the registry page, drawing a running job's progress bar
def test_the_registry_page_shows_a_running_jobs_progress(tmp_path):
    import test_registry_page as trp

    cfg_path = trp._candidate(tmp_path)
    runs = tmp_path / "runs"
    run = runs / "20260927_000000_registry_evidence"
    run.mkdir(parents=True)
    (run / "plan.json").write_text(json.dumps({
        "run_id": run.name, "lock_tag": jobs.LOCK_TAG,
        "meta": {"source": jobs.SOURCE, "job": "explainer_validation"},
        "stages": [{"key": "explainer_validation", "name": "Run explainer validation "
                    "for m (in WSL)", "tag": "m", "args": []}]}))
    (run / "status.json").write_text(json.dumps({
        "run_id": run.name, "state": "running", "created": "x", "finished": None,
        "failed_stage": None, "stages": [{
            "key": "explainer_validation", "name": "Run explainer validation for m",
            "tag": "m", "state": "running", "started": "2026-09-27T00:00:00",
            "ended": None, "minutes": None, "exit_code": None}]}))
    (run / "log.txt").write_text("TreeSHAP: 6.7s\n 35%|###5      | 350/1000 "
                                 "[15:28<28:44,  2.65s/it]\n")
    (runs / "locks").mkdir()
    (runs / "locks" / f"{jobs.LOCK_TAG}.lock").write_text(json.dumps(
        {"run_id": run.name, "pid": os.getpid(), "started": "x"}))
    at = trp._page(cfg_path, runs, tmp_path / "marker")
    bars = [str(p.proto) for p in at.get("progress")]
    assert any("350 of 1,000 applicants explained" in b and "28:44" in b for b in bars), bars


# ================================================================ C3 ==

class TestSafety:
    def test_a_reused_pid_does_not_count_as_our_job(self, tmp_path):
        stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        ours = subprocess.Popen([sys.executable, "-c",
                                 "import time; time.sleep(30)  # creditsurv.runner"])
        try:
            time.sleep(0.5)
            assert wsl_sync._pid_alive(stranger.pid) is False
            assert wsl_sync._pid_alive(ours.pid) is True
            locks = tmp_path / "outputs" / "logs" / "runs" / "locks"
            locks.mkdir(parents=True)
            (locks / "old.lock").write_text(json.dumps({"pid": stranger.pid}))
            assert wsl_sync.sync_blockers(tmp_path) == []   # stale lock, reused pid
            (locks / "live.lock").write_text(json.dumps({"pid": ours.pid}))
            assert wsl_sync.sync_blockers(tmp_path) == ["background job live"]
        finally:
            stranger.kill()
            ours.kill()

    def test_a_background_job_is_detached_on_both_platforms(self, tmp_path,
                                                             monkeypatch):
        """"Outlive whoever started me" has a different spelling per platform, and
        only the Windows one was ever said. POSIX spells it setsid(), which Popen
        calls start_new_session; without it every job in WSL -- the only place this
        app runs -- stayed in the server's process group and terminal."""
        seen: dict = {}

        class _Fake:
            pid = 4321

            def __init__(self, cmd, **kw):
                seen.update(kw)

        monkeypatch.setattr(runner.subprocess, "Popen", _Fake)
        launch([{"key": "s", "name": "n", "tag": "t", "args": ["-c", "pass"]}],
               lock_tag="detach", runs_dir=tmp_path)
        if os.name == "nt":
            flags = seen["creationflags"]
            assert flags & subprocess.DETACHED_PROCESS
            assert flags & subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            assert seen["start_new_session"] is True
        assert seen["close_fds"] is True

    @linux_only
    def test_a_background_job_gets_its_own_session(self, tmp_path):
        """The live version of the test above. A job shares nothing that can hang it
        up: a Ctrl+C in the window that started the app, or closing that window,
        signals every process in the server's foreground group -- which used to
        include an hour-long 03d run."""
        launch([{"key": "s", "name": "probe", "tag": "probe",
                 "args": ["-c", "import time; time.sleep(30)"]}],
               lock_tag="detachprobe", runs_dir=tmp_path / "runs")
        lock = tmp_path / "runs" / "locks" / "detachprobe.lock"
        pid = json.loads(lock.read_text(encoding="utf-8"))["pid"]
        try:
            for _ in range(50):
                if os.getsid(pid) == pid:
                    break
                time.sleep(0.1)
            assert os.getsid(pid) == pid                 # its own session leader
            assert os.getpgid(pid) == pid                # its own process group
            # /proc/<pid>/stat after the comm: state, ppid, pgrp, session, tty_nr.
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            assert stat[4] == "0"                        # no controlling terminal
        finally:
            os.kill(pid, 9)

    def test_session_heartbeats_age_out(self, tmp_path):
        wsl_sync.note_session("a", True, tmp_path)
        wsl_sync.note_session("b", False, tmp_path)
        assert wsl_sync.open_sessions(tmp_path) == 1
        f = tmp_path / "outputs" / "logs" / "sessions" / "a.json"
        f.write_text(json.dumps({"open_work": True, "at": time.time() - 700}))
        assert wsl_sync.open_sessions(tmp_path) == 0      # a closed tab stops blocking


def _pair(tmp_path):
    src, dst = tmp_path / "win", tmp_path / "wsl"
    for base in (src, dst):
        (base / "src").mkdir(parents=True)
        (base / "src" / "a.py").write_text("x = 1")
    now = time.time()
    os.utime(dst / "src" / "a.py", (now - 100, now - 100))
    return src, dst


class TestAutoSync:
    def test_every_outcome(self, tmp_path, monkeypatch):
        src, dst = _pair(tmp_path)
        restarted = []
        calls = []

        def fake_sync(root, source, timeout=600):
            calls.append((root, source))
            shutil.copy2(source / "src" / "a.py", root / "src" / "a.py")
            return True, "synced"
        monkeypatch.setattr(wsl_sync, "sync_now", fake_sync)
        go = lambda **kw: wsl_sync.auto_sync(dst, source=src,          # noqa: E731
                                             restart=lambda: restarted.append(1), **kw)
        # someone has work open: deferred, nothing touched
        assert go(open_work=True).startswith("deferred") and not calls
        # a job running: blocked
        locks = dst / "outputs" / "logs" / "runs" / "locks"
        locks.mkdir(parents=True)
        (locks / "j.lock").write_text(json.dumps({"pid": os.getpid()}))
        monkeypatch.setattr(wsl_sync, "_pid_alive", lambda pid: True)
        assert go(open_work=False).startswith("blocked") and not calls
        (locks / "j.lock").unlink()
        # safe: synced, noted, restarted
        assert go(open_work=False) == "synced 1"
        assert restarted == [1]
        note = json.loads((dst / "outputs" / "logs" / "last_code_sync.json").read_text())
        assert note["files"] == 1 and note["examples"] == ["src/a.py"]
        # and now current
        assert go(open_work=False) == "current"

    def test_a_failed_sync_does_not_restart(self, tmp_path, monkeypatch):
        src, dst = _pair(tmp_path)
        monkeypatch.setattr(wsl_sync, "sync_now", lambda *a, **k: (False, "rsync: boom"))
        restarted = []
        what = wsl_sync.auto_sync(dst, source=src, open_work=False,
                                  restart=lambda: restarted.append(1))
        assert what.startswith("failed") and "boom" in what and not restarted

    def test_the_watcher_starts_only_in_the_wsl_copy(self, tmp_path, monkeypatch):
        # In the WSL copy an earlier test renders app.py, which starts the real
        # watcher; start this check from a process that has none.
        monkeypatch.setattr(wsl_sync, "_WATCHER", {})
        assert wsl_sync.start_watcher(tmp_path) is False
        app = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
        assert "start_watcher(ROOT)" in app

    def test_both_phases_send_results_back(self):
        import inspect

        from creditsurv import batch, phase2
        assert "copy_back_soon()" in inspect.getsource(batch.score_file)
        assert "copy_back_soon()" in inspect.getsource(phase2.materialize)


@linux_only
def test_the_real_launcher_syncs_a_copy_and_marks_it(tmp_path):
    """sync_now runs scripts/wsl_launch.sh check -- rsync and the import check --
    into a copy, exactly as run_linux.ps1 -CheckOnly does, and the copy is then
    recognised as the WSL copy with nothing stale."""
    if shutil.which("rsync") is None:
        pytest.skip("rsync is not installed")
    win, dest = tmp_path / "win", tmp_path / "dest"
    for d in ("src", "app", "scripts", "config", ".streamlit"):
        if (PROJECT_ROOT / d).is_dir():
            shutil.copytree(PROJECT_ROOT / d, win / d,
                            ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("FINDINGS.md", "README.md", "pyproject.toml", "requirements-linux.txt"):
        shutil.copy2(PROJECT_ROOT / f, win / f)
    dest.mkdir()
    (dest / ".venv").symlink_to(Path(sys.prefix))        # the environment we run in
    ok, out = wsl_sync.sync_now(dest, win)
    assert ok, out
    assert (dest / "src" / "creditsurv" / "wsl_sync.py").exists()
    assert wsl_sync.windows_source(dest) == win
    assert wsl_sync.stale_files(dest, win) == []
    time.sleep(2.2)
    (win / "src" / "creditsurv" / "wsl_sync.py").touch()
    assert wsl_sync.stale_files(dest, win) == ["src/creditsurv/wsl_sync.py"]


@linux_only
def test_a_restart_re_executes_the_original_command(tmp_path):
    """Streamlit rewrites sys.argv; the restart must not use it. A process started
    as `python -m <module> run app.py --flag x`, with sys.argv overwritten the way
    Streamlit does, re-executes with exactly its original command line."""
    mod = tmp_path / "fakeserver.py"
    mod.write_text(
        "import os, socket, sys\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r})\n"
        "from creditsurv import wsl_sync\n"
        "port = int(sys.argv[-1])\n"
        "def serve():\n"
        "    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        "    s.bind(('127.0.0.1', port)); s.listen(); s.set_inheritable(True)  # as uvicorn\n"
        "    return s\n"
        "if os.environ.get('RESTARTED'):\n"
        "    serve()                               # the port must be free again\n"
        "    print('ARGS', wsl_sync.launch_command()[1:]); sys.exit(0)\n"
        "held = serve()\n"
        "sys.argv = ['app/app.py']              # what Streamlit does\n"
        "os.environ['RESTARTED'] = '1'\n"
        "wsl_sync.restart_server()\n", encoding="utf-8")
    import socket
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    out = subprocess.run([sys.executable, "-m", "fakeserver", "run", "app.py",
                          "--flag", str(port)], cwd=tmp_path, capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert f"ARGS ['-m', 'fakeserver', 'run', 'app.py', '--flag', '{port}']" in out.stdout


@linux_only
def test_copy_back_runs_the_launcher_in_the_background(tmp_path):
    root, win = tmp_path / "wsl", tmp_path / "win"
    (win / "scripts").mkdir(parents=True)
    (win / "scripts" / "wsl_launch.sh").write_text(
        'echo "copy-back called: $1 $2 into $CREDITSURV_WSL_DIR" > "$2/copied.txt"\n')
    root.mkdir()
    (root / ".synced_from_windows").write_text(str(win))
    assert wsl_sync.copy_back_soon(root) is True
    for _ in range(50):
        if (win / "copied.txt").exists():
            break
        time.sleep(0.1)
    assert (win / "copied.txt").read_text().startswith(f"copy-back called: copy-back {win}")


@linux_only
def test_copy_backs_run_one_at_a_time(tmp_path):
    """Two requests seconds apart (end of Phase 1, end of Phase 2): the second
    waits for the first, so the newest files are the ones that land last."""
    if shutil.which("flock") is None:
        pytest.skip("flock is not installed")
    root, win = tmp_path / "wsl", tmp_path / "win"
    (win / "scripts").mkdir(parents=True)
    (win / "scripts" / "wsl_launch.sh").write_text(
        'echo "start $(date +%s.%N)" >> "$2/order.txt"; sleep 1.5; '
        'echo "end $(date +%s.%N)" >> "$2/order.txt"\n')
    root.mkdir()
    (root / ".synced_from_windows").write_text(str(win))
    assert wsl_sync.copy_back_soon(root)
    # Ask again only once the first copy is running, which is what "seconds apart"
    # means here. The queue slot is held by the running request until its child has
    # the one-at-a-time lock, so a request made in that first instant is dropped --
    # harmlessly, because the copy it asked for has not begun either and the one
    # that does begin copies whatever is on disk by then.
    for _ in range(80):
        if (win / "order.txt").exists():
            break
        time.sleep(0.1)
    assert wsl_sync.copy_back_soon(root)
    for _ in range(80):
        if (win / "order.txt").exists() and \
                (win / "order.txt").read_text().count("end") == 2:
            break
        time.sleep(0.1)
    kinds = [line.split()[0] for line in (win / "order.txt").read_text().splitlines()]
    assert kinds == ["start", "end", "start", "end"]           # never overlapping


pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@linux_only
@pytest.mark.parametrize("open_work, busy, expect", [
    (False, False, "restarted"), (True, False, "button"), (False, True, "blocked")])
def test_the_page_banner(tmp_path, monkeypatch, open_work, busy, expect):
    src, dst = _pair(tmp_path)
    (dst / ".synced_from_windows").write_text(str(src))
    monkeypatch.setattr(wsl_sync, "_pid_alive", lambda pid: pid == os.getpid())
    # The page script below runs in this same process and replaces sync_now by
    # assignment, which it cannot undo; registering the real one here is what puts
    # it back at teardown, instead of leaving the fake for every later test.
    monkeypatch.setattr(wsl_sync, "sync_now", wsl_sync.sync_now)
    if busy:
        locks = dst / "outputs" / "logs" / "runs" / "locks"
        locks.mkdir(parents=True)
        (locks / "j.lock").write_text(json.dumps({"pid": os.getpid()}))

    def script(views, src_dir, root, open_work):
        import sys
        sys.path[:0] = [views, src_dir]
        from pathlib import Path
        import shutil
        import streamlit as st
        from creditsurv import wsl_sync
        import _common

        def fake_sync(r, s, timeout=600):
            shutil.copy2(Path(s) / "src" / "a.py", Path(r) / "src" / "a.py")
            return True, "ok"
        wsl_sync.sync_now = fake_sync
        if open_work:
            st.session_state["run_id"] = "x"        # a run open on the page
        _common.code_freshness(Path(root), restart=lambda: st.session_state.update(
            restarted=True))

    at = AppTest.from_function(script, default_timeout=60, kwargs={
        "views": str(PROJECT_ROOT / "app" / "views"), "src_dir": str(PROJECT_ROOT / "src"),
        "root": str(dst), "open_work": open_work})
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    warnings = " ".join(w.value for w in at.warning)
    if expect == "restarted":
        assert at.session_state["restarted"] is True
    elif expect == "button":
        assert "restarted" not in at.session_state
        assert "Syncing restarts the dashboard" in warnings
        next(b for b in at.button if b.label == "Sync from Windows and restart").click().run()
        assert at.session_state["restarted"] is True
    else:
        assert "Not syncing while background job j runs" in warnings
    heartbeat = list((dst / "outputs" / "logs" / "sessions").glob("*.json"))
    assert heartbeat and json.loads(heartbeat[0].read_text())["open_work"] is open_work


@linux_only
def test_copy_back_queues_one_and_drops_the_rest(tmp_path):
    """Phase 2 asks again at every checkpoint (60 s) while a copy over /mnt/c takes
    longer than that, so queueing every request piled them up in the hundreds, each
    one another full rsync still to run. One waiting is enough: it copies whatever is
    on disk when it runs, which includes what the dropped requests asked about."""
    if shutil.which("flock") is None:
        pytest.skip("flock is not installed")
    root, win = tmp_path / "wsl", tmp_path / "win"
    (win / "scripts").mkdir(parents=True)
    (win / "scripts" / "wsl_launch.sh").write_text(
        'echo run >> "$2/runs.txt"; sleep 3\n')
    root.mkdir()
    (root / ".synced_from_windows").write_text(str(win))

    assert wsl_sync.copy_back_soon(root) is True
    time.sleep(0.8)                       # the first one holds the lock by now
    for _ in range(5):
        assert wsl_sync.copy_back_soon(root) is True
    for _ in range(100):
        time.sleep(0.1)
        if (win / "runs.txt").exists() and (win / "runs.txt").read_text().count("run") == 2:
            break
    time.sleep(2)                         # a third would have started by now
    assert (win / "runs.txt").read_text().count("run") == 2
