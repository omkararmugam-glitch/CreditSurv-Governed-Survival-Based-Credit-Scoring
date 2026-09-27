"""Keep the WSL copy in step with Windows, and results in step with WSL.

The app is served from ``~/creditsurv`` in WSL (Smart App Control blocks lightgbm on
Windows), which ``scripts/wsl_launch.sh`` syncs from the Windows folder -- but only
when someone runs ``run_linux.ps1 -CheckOnly``, and results travel back only with
``-CopyBack``. Between those, the app could serve stale code without saying so.
This module does the two by itself where that is safe:

* :func:`stale_files` -- which code files on Windows are newer than the copy this
  process is running. The page says so whenever the answer is not empty.
* :func:`sync_blockers` -- anything that makes a sync unsafe *now*: a background
  job or a Phase 2 running (it would finish on old code while new code arrives
  under it). With none, :func:`sync_now` runs ``wsl_launch.sh check`` (the same
  sync and import check as ``-CheckOnly``); the page then restarts the server so
  the new code is what runs.
* :func:`copy_back_soon` -- after a run finishes, ``wsl_launch.sh copy-back`` in
  the background (newer files only; it never deletes).

Everything here is a no-op except in the WSL copy, which the launcher marks with
``.synced_from_windows``: on Windows, or in any other checkout, nothing happens.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .provenance import PROJECT_ROOT
from .registry import SYNC_MARKER

__all__ = ["windows_source", "stale_files", "sync_blockers", "sync_now",
           "copy_back_soon", "restart_server", "CODE_DIRS", "auto_sync",
           "start_watcher", "note_session", "open_sessions", "launch_command"]

CODE_DIRS = ("src", "app", "scripts", "config", ".streamlit")
_SKIP = ("__pycache__", ".pytest_cache", ".egg-info")


def windows_source(root: Path = PROJECT_ROOT) -> Path | None:
    """The Windows folder (as a WSL path) this copy was synced from, or None."""
    marker = Path(root) / SYNC_MARKER
    if sys.platform == "win32" or not marker.is_file():
        return None
    try:
        source = Path(marker.read_text(encoding="utf-8").strip())
    except OSError:
        return None
    return source if source.is_dir() else None


def stale_files(root: Path, source: Path, dirs=CODE_DIRS, slack: float = 2.0) -> list[str]:
    """Code files under ``source`` that are missing from ``root`` or newer there.

    ``slack`` absorbs the timestamp rounding between NTFS and ext4 (the launcher's
    rsync uses the same two-second window).
    """
    out = []
    for d in dirs:
        base = Path(source) / d
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [n for n in dirnames if not n.endswith(_SKIP)]
            for name in filenames:
                if name.endswith((".pyc", ".tmp")):
                    continue
                src = Path(dirpath) / name
                rel = src.relative_to(source)
                dst = Path(root) / rel
                try:
                    if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime + slack:
                        out.append(rel.as_posix())
                except OSError:
                    continue
    return sorted(out)


OUR_PROCESSES = ("creditsurv.runner", "06_score_upload", "streamlit")
"""What a job of ours runs as: the runner, the scoring script (Phase 1 and a
background Phase 2), or the app (an on-demand explanation). Not "creditsurv" alone:
in WSL every Python process's path contains it, because the venv lives in
~/creditsurv/.venv."""


def _pid_alive(pid) -> bool:
    """Whether ``pid`` is a live process of ours.

    A lock or status file records a pid; after WSL restarts, the same number can
    belong to an unrelated process. Counting that as "a job is running" would block
    every sync for good, so the process must also look like ours.
    """
    try:
        import psutil
        if not pid or not psutil.pid_exists(int(pid)):
            return False
        cmd = " ".join(psutil.Process(int(pid)).cmdline())
        return any(k in cmd for k in OUR_PROCESSES)
    except ImportError:                                 # pragma: no cover
        return True
    except Exception:                                   # vanished, or not readable
        return False


def sync_blockers(root: Path = PROJECT_ROOT) -> list[str]:
    """What is running that a code sync could pull the floor from under."""
    busy = []
    locks = Path(root) / "outputs" / "logs" / "runs" / "locks"
    if locks.is_dir():
        for lock in locks.glob("*.lock"):
            try:
                info = json.loads(lock.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                busy.append(lock.stem)
                continue
            if _pid_alive(info.get("pid")):
                busy.append(f"background job {lock.stem}")
    runs = Path(root) / "outputs" / "runs"
    if runs.is_dir():
        for status in sorted(runs.glob("*/phase2/status.json"))[-30:]:
            try:
                st = json.loads(status.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if st.get("state") == "running" and _pid_alive(st.get("pid")):
                busy.append(f"Phase 2 of {status.parent.parent.name}")
    return busy


SESSION_MAX_AGE = 600
"""A browser session counts as having work open for this long after its last page
load with an upload or a run on screen; a closed tab stops blocking after it."""


def _sessions_dir(root: Path) -> Path:
    return Path(root) / "outputs" / "logs" / "sessions"


def note_session(session_id: str, open_work: bool, root: Path = PROJECT_ROOT) -> None:
    """A page load's heartbeat: whether this browser session has work on screen
    that a restart would interrupt. Read by the server-side watcher, which cannot
    see sessions otherwise."""
    d = _sessions_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{session_id}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"open_work": bool(open_work), "at": time.time()}),
                   encoding="utf-8")
    os.replace(tmp, path)


def open_sessions(root: Path = PROJECT_ROOT, max_age: float = SESSION_MAX_AGE) -> int:
    """How many browser sessions reported open work in the last ``max_age`` s."""
    n, now = 0, time.time()
    for f in _sessions_dir(root).glob("*.json") if _sessions_dir(root).is_dir() else []:
        try:
            info = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if now - info.get("at", 0) > 4 * max_age:
            f.unlink(missing_ok=True)                  # long gone: tidy up
        elif info.get("open_work") and now - info.get("at", 0) <= max_age:
            n += 1
    return n


def _launcher(source: Path) -> list[str]:
    return ["bash", str(Path(source) / "scripts" / "wsl_launch.sh")]


def sync_now(root: Path, source: Path, timeout: float = 600) -> tuple[bool, str]:
    """Run the launcher's ``check`` mode into ``root``: sync, then check imports."""
    env = {**os.environ, "CREDITSURV_WSL_DIR": str(root)}
    out = subprocess.run([*_launcher(source), "check", str(source)], env=env,
                         capture_output=True, text=True, timeout=timeout)
    return out.returncode == 0, (out.stdout + out.stderr).strip()


def copy_back_soon(root: Path = PROJECT_ROOT) -> bool:
    """Copy finished results back to Windows in the background, if this is the WSL
    copy. Returns whether a copy was started. Never blocks, never deletes."""
    source = windows_source(root)
    if source is None:
        return False
    log = Path(root) / "outputs" / "logs" / "copy_back.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CREDITSURV_WSL_DIR": str(root)}
    cmd = [*_launcher(source), "copy-back", str(source)]
    # One copy-back at a time, in the order asked for. Phase 1 of a large run can
    # still be copying a gigabyte when Phase 2 finishes and asks again; run side by
    # side, the older copy could land last and put older files over newer ones.
    import shutil
    if shutil.which("flock"):
        cmd = ["flock", str(log.with_name("copy_back.lock")), *cmd]
    with open(log, "a", encoding="utf-8") as fh:
        subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    return True


NOTE_NAME = "last_code_sync.json"


def auto_sync(root: Path = PROJECT_ROOT, *, open_work: bool | None = None,
              restart=None, source: Path | None = None) -> str:
    """Sync newer Windows code and restart, if -- and only if -- that is safe now.

    Returns what happened: ``"current"`` (nothing newer), ``"blocked: ..."`` (a job
    or Phase 2 is running), ``"deferred: ..."`` (someone has work open),
    ``"failed: ..."`` or ``"synced N"`` (after which ``restart`` is called). Both
    the server-side watcher and the pages call this, so the rules are one set.
    ``open_work`` None means: ask the session heartbeats.
    """
    source = source or windows_source(root)
    if source is None:
        return "not the WSL copy"
    stale = stale_files(root, source)
    if not stale:
        return "current"
    busy = sync_blockers(root)
    if busy:
        return "blocked: " + ", ".join(busy)
    if open_work is None:
        open_work = open_sessions(root) > 0
    if open_work:
        return f"deferred: {len(stale)} newer file(s), and a session has work open"
    ok, out = sync_now(root, source)
    if not ok:
        return "failed: " + out[-2000:]
    note = Path(root) / "outputs" / "logs" / NOTE_NAME
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(json.dumps({"at": time.time(), "files": len(stale),
                                "examples": stale[:5],
                                "when": time.strftime("%H:%M:%S")}), encoding="utf-8")
    (restart or restart_server)()
    return f"synced {len(stale)}"


_WATCHER: dict = {}


def start_watcher(root: Path = PROJECT_ROOT, interval: float = 20.0,
                  restart=None) -> bool:
    """Once per server process, in the WSL copy only: check every ``interval``
    seconds and :func:`auto_sync` when safe, so the app picks up Windows edits even
    when no page is open. Returns whether a watcher runs (started now or before)."""
    if _WATCHER.get("thread") is not None:
        return True
    if windows_source(root) is None:
        return False
    import threading

    log = Path(root) / "outputs" / "logs" / "code_sync.log"

    def write(what: str) -> None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {what}" + chr(10))

    def restart_logged() -> None:
        # Written first: after the restart this process no longer exists.
        write("synced newer Windows code; restarting the server on it")
        (restart or restart_server)()

    def loop():
        last = ""
        while not _WATCHER.get("stop"):
            try:
                what = auto_sync(root, restart=restart_logged)
            except Exception as exc:                   # never take the server down
                what = f"failed: {exc!r}"
            if what != last and what not in ("current",):
                write(what)
            last = what
            time.sleep(interval)

    t = threading.Thread(target=loop, name="creditsurv-code-sync", daemon=True)
    _WATCHER["thread"] = t
    t.start()
    return True


def launch_command() -> list[str]:
    """The command line this process was started with, as the kernel recorded it.

    Not ``sys.argv``: Streamlit rewrites that for the script it runs (it becomes
    ``["app/app.py"]``), so re-executing from it started ``python -m streamlit``
    with no command at all -- the server printed its usage and exited, found by
    the live test of this function."""
    raw = Path("/proc/self/cmdline").read_bytes()
    return [a.decode() for a in raw.split(bytes(1)) if a]


def restart_server() -> None:
    """Re-execute this Streamlit server so the synced code is the code that runs.

    ``src/creditsurv`` is imported once per server process and is outside the
    folder Streamlit watches, so without a restart the pages would be new and the
    library old. Same pid, same port, same working directory; the browser
    reconnects by itself. Background jobs are separate processes and are not
    affected (and a sync never happens while one runs).
    """
    cmd = launch_command()
    # Everything but stdin/out/err is closed first. Uvicorn marks its listening
    # socket inheritable, so it survived the exec, and the new server found its
    # own port taken and exited ("Port 8599 is not available", live test).
    os.closerange(3, 1 << 16)
    os.execv(sys.executable, [sys.executable, *cmd[1:]])
