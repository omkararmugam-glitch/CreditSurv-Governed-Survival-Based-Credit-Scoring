"""The WSL workflow: what it installs, what it checks, and how it labels itself.

Smart App Control blocks lightgbm and shap on the Windows machine this project is
developed on, so the app is served from WSL through run_linux.ps1. These tests pin
the pieces that can drift silently: the Linux requirements against pyproject.toml,
the fix each failed import is given, and the label that tells a Windows instance
of the app from a Linux one.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
import tomllib

import pytest

from creditsurv import environment as env
from creditsurv.environment import ImportCheck, fix_for, runtime_label, runtime_report
from creditsurv.provenance import PROJECT_ROOT


def _names(specs) -> set[str]:
    return {re.split(r"[<>=!~\[; ]", s.strip(), maxsplit=1)[0].lower().replace("_", "-")
            for s in specs if s.strip()}


def test_linux_requirements_cover_everything_pyproject_declares():
    """A package added to pyproject.toml but not to requirements-linux.txt is exactly
    how streamlit and lifelines went missing from the first WSL install."""
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = _names(project["dependencies"])
    for extra in ("ui", "dev"):
        declared |= _names(project["optional-dependencies"][extra])
    lines = (PROJECT_ROOT / "requirements-linux.txt").read_text(encoding="utf-8").splitlines()
    listed = _names(l for l in lines if l.strip() and not l.lstrip().startswith(("#", "-")))
    assert declared - listed == set(), "missing from requirements-linux.txt"
    assert listed - declared == set(), "in requirements-linux.txt but not pyproject.toml"
    assert "-e ." in [l.strip() for l in lines]
    assert "libgomp1" in "\n".join(lines)


def test_the_runtime_check_covers_what_the_launcher_promises():
    names = [n for n, _ in env.RUNTIME_REQUIREMENTS]
    for pkg in ("lifelines", "lightgbm", "shap", "streamlit"):
        assert pkg in names


@pytest.mark.parametrize("platform, wsl_var, expected", [
    ("win32", None, "Windows"),
    ("linux", "Ubuntu", "Linux (WSL)"),
    ("darwin", None, "macOS"),
])
def test_runtime_label(monkeypatch, platform, wsl_var, expected):
    monkeypatch.setattr(env.sys, "platform", platform)
    if wsl_var:
        monkeypatch.setenv("WSL_DISTRO_NAME", wsl_var)
    else:
        monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    assert runtime_label() == expected


def test_plain_linux_is_not_called_wsl(monkeypatch, tmp_path):
    monkeypatch.setattr(env.sys, "platform", "linux")
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    real_open = open

    def fake_open(path, *a, **k):
        if path == "/proc/sys/kernel/osrelease":
            p = tmp_path / "osrelease"
            p.write_text("6.8.0-45-generic\n")
            return real_open(p, *a, **k)
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", fake_open)
    assert runtime_label() == "Linux"


def test_each_failure_gets_the_fix_that_matches_its_cause(monkeypatch):
    monkeypatch.setattr(env.sys, "platform", "linux")
    gomp = ImportCheck("lightgbm", "x", False,
                       "OSError: libgomp.so.1: cannot open shared object file")
    missing = ImportCheck("streamlit", "x", False,
                          "ModuleNotFoundError: No module named 'streamlit'")
    blocked = ImportCheck("shap", "x", False,
                          "OSError: [WinError 4551] An Application Control policy "
                          "has blocked this file")
    assert "sudo apt install libgomp1" in fix_for(gomp)
    assert "requirements-linux.txt" in fix_for(missing)
    assert "cannot fix" in fix_for(blocked) and "run_linux.ps1" in fix_for(blocked)


def test_the_report_names_what_is_missing_and_refuses_to_start():
    ok, report = runtime_report((("json", "a"), ("no_such_pkg_xyz", "the thing")))
    assert not ok
    assert "MISSING  no_such_pkg_xyz -- needed for the thing" in report
    assert "fix:" in report
    assert "the app was not started" in report
    ok, report = runtime_report((("json", "a"),))
    assert ok and "ok       json" in report


def test_the_windows_message_points_at_the_launcher():
    blocked = ImportCheck("lightgbm", "the model", False,
                          "OSError: [WinError 4551] An Application Control policy")
    message = env.policy_block_message([blocked])
    assert "run_linux.ps1" in message
    assert "\r" not in message


def test_the_linux_script_parses_and_has_lf_endings():
    script = PROJECT_ROOT / "scripts" / "wsl_launch.sh"
    assert b"\r\n" not in script.read_bytes(), "CRLF would make bash fail inside WSL"
    bash = shutil.which("bash")
    if bash is None or sys.platform == "win32":
        # On Windows `bash` may be WSL's launcher, which cannot take a C:\ path.
        pytest.skip("syntax check runs where bash is native")
    subprocess.run([bash, "-n", str(script)], check=True)


def test_the_launcher_serves_headless_on_all_interfaces_and_never_ships_the_venv():
    sh = (PROJECT_ROOT / "scripts" / "wsl_launch.sh").read_text(encoding="utf-8")
    assert "--server.headless true" in sh and "--server.address 0.0.0.0" in sh
    assert ".venv" not in re.findall(r"for d in ([^;]+); do", sh)[0]


# --------------------------------------------------- starting things detached ---

def _detach_helper() -> str:
    """The launcher's detach() function on its own, to run in a test shell."""
    sh = (PROJECT_ROOT / "scripts" / "wsl_launch.sh").read_text(encoding="utf-8")
    found = re.search(r"^detach\(\) \{\n.*?^\}", sh, re.S | re.M)
    assert found, "scripts/wsl_launch.sh has no detach() helper"
    return found.group(0)


def test_the_launcher_starts_nothing_in_the_background_except_through_detach():
    """A bare ``cmd &`` leaves the job in this script's own process group -- the
    foreground group of the terminal wsl.exe gave it -- and the kernel hangs that
    group up the moment the script exits. ``setsid cmd &`` is no better by itself:
    it moves the job out only once it has run, and the shell exits in the same
    instant it forks, so the hangup can arrive first. That is how starting the API
    left no process and an empty log. Only detach() waits for the job to report
    from its new session, so every background start goes through it.
    """
    sh = (PROJECT_ROOT / "scripts" / "wsl_launch.sh").read_text(encoding="utf-8")
    outside = sh.replace(_detach_helper(), "")
    backgrounded = [line for line in outside.splitlines()
                    if re.search(r"(?<!&)&\s*$", line)
                    and not line.lstrip().startswith("#")]
    assert backgrounded == [], backgrounded
    # nohup only makes the job ignore SIGHUP, and not before it has started: it
    # leaves the job in the dying session and has the same race. (The comments in
    # the script say so, hence only the code is looked at here.)
    code = [line for line in sh.splitlines() if not line.lstrip().startswith("#")]
    assert [line for line in code if "nohup" in line] == []


@pytest.mark.skipif(sys.platform == "win32", reason="a pty and sessions are POSIX")
def test_a_detached_job_outlives_the_launcher_and_its_terminal(tmp_path):
    """The reported failure, reproduced and pinned.

    ``wsl.exe -e bash scripts/wsl_launch.sh`` makes the launcher the session leader
    of a fresh pty. When it returns, the kernel hangs that pty's foreground group
    up -- and a job started with ``setsid cmd &`` is still in that group while its
    setsid() has not run yet, which is the case when the shell exits in the same
    instant it forks. The job then died before its first line of output: nothing in
    ps, an empty log, immediately.

    Driven through a real pty here, because with no controlling terminal there is
    no hangup to survive and the test would pass either way. ``setsid --ctty`` does
    what wsl.exe does -- new session, that pty as its controlling terminal -- and,
    unlike ``pty.spawn``, forks no copy of this test process.
    """
    marker, log, done, out = (tmp_path / name for name in
                              ("job.pid", "job.log", "job.done", "launcher.out"))
    launcher = tmp_path / "launcher.sh"
    launcher.write_text(
        "set -euo pipefail\n"
        + _detach_helper() + "\n"
        + """job='echo running; sleep 3; echo finished > "$1"'\n"""
        + 'detach "%s" "%s" bash -c "$job" job "%s" > "%s"\n'
          % (marker, log, done, out),
        encoding="utf-8")

    # The launcher starts the job and returns at once: the failing case exactly.
    master, slave = os.openpty()
    try:
        started = subprocess.Popen(["setsid", "--ctty", "bash", str(launcher)],
                                   stdin=slave, stdout=slave, stderr=slave)
        os.close(slave)
        assert started.wait(timeout=60) == 0, "the launcher itself failed"
    finally:
        os.close(master)

    pid = int(out.read_text(encoding="utf-8").strip())
    try:
        session = os.getsid(pid)
    except ProcessLookupError:                 # the symptom that was reported
        wrote = (log.read_text(encoding="utf-8") if log.exists()
                 else "<the log was never even created>")
        raise AssertionError(
            f"pid {pid} is already gone: the job was hung up along with the "
            f"launcher that started it, and wrote {wrote!r}") from None
    assert session == pid, f"pid {pid} is still in the launcher's session"

    for _ in range(100):
        if done.exists():
            break
        time.sleep(0.1)
    assert done.read_text(encoding="utf-8").strip() == "finished", \
        "the job did not outlive the launcher that started it"
    assert "running" in log.read_text(encoding="utf-8"), \
        "the job wrote nothing to its log"


def test_the_detached_modes_are_offered_and_documented():
    """The two starts that were being hand-rolled with setsid, and lost: the API in
    the background, and one long script (03d) in the background."""
    sh = (PROJECT_ROOT / "scripts" / "wsl_launch.sh").read_text(encoding="utf-8")
    for mode in ("api-bg", "run-bg"):
        assert f"    {mode})" in sh, f"no {mode} mode"
        assert f"#   {mode}" in sh, f"{mode} is missing from the usage comment"
    assert "api-bg" in sh.split("unknown mode")[1]


def test_the_windows_half_frees_the_port_but_not_the_wsl_forwarder():
    path = PROJECT_ROOT / "run_linux.ps1"
    if not path.exists():
        pytest.skip("run_linux.ps1 is Windows-only and is not synced into WSL")
    ps1 = path.read_text(encoding="utf-8")
    # Both ports: the dashboard's and the API's.
    assert "Get-NetTCPConnection -LocalPort @($Port, $ApiPort)" in ps1
    assert "wslrelay" in ps1          # WSL's forwarder must never be stopped
