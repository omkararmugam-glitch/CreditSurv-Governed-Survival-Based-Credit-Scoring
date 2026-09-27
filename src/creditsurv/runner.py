"""Background execution of a stage plan, for the Streamlit UI.

Why a detached process and a log file rather than a subprocess tied to the page:
Streamlit re-executes the page script on every click and browser refresh. A
subprocess owned by the page would lose its output the moment the user navigated
away during a 25-minute Stage 3, and a second click could launch a duplicate run
on the same tag. Here the run is its own process, writing to
``outputs/logs/runs/<run_id>/``; the page only *reads* that directory, so progress
survives navigation, and a per-tag lock refuses a second concurrent run.

Semantics match ``run_holdout.ps1`` exactly: the same commands (from
:mod:`creditsurv.plan`), in order, each run to completion, stopping at the first
non-zero exit with no retry and no cleanup. Status lines use the wrapper's wording.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .provenance import OVERWRITE_REFUSED, PROJECT_ROOT

__all__ = ["LockHeld", "launch", "execute", "read_status", "effective_state",
           "tail_log", "list_runs", "active_lock", "read_lock", "RUNS_DIR"]

RUNS_DIR = PROJECT_ROOT / "outputs" / "logs" / "runs"


class LockHeld(RuntimeError):
    """Another run is already using this tag."""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)          # atomic: the UI never reads a half-written file


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except ImportError:            # pragma: no cover - psutil is installed here
        return True                # unknown -> treat as alive, i.e. stay locked


def _lock_path(runs_dir: Path, tag: str) -> Path:
    return runs_dir / "locks" / f"{tag}.lock"


def read_lock(tag: str, runs_dir: Path = RUNS_DIR, *,
              attempts: int = 4) -> tuple[dict | None, bool]:
    """``(lock info, readable)``.

    ``(None, True)`` means there is no lock file. ``(None, False)`` means one is
    there but could not be read -- which happens transiently on Windows while the
    file is being replaced, and must not be mistaken for a run that has ended.
    """
    p = _lock_path(runs_dir, tag)
    for i in range(attempts):
        if not p.exists():
            return None, True
        try:
            return json.loads(p.read_text(encoding="utf-8")), True
        except (OSError, json.JSONDecodeError):
            if i < attempts - 1:
                time.sleep(0.05)
    # Still there but unparseable -> not readable. Vanished meanwhile -> no lock.
    return None, not p.exists()


def active_lock(tag: str, runs_dir: Path = RUNS_DIR) -> dict | None:
    """The live lock for ``tag``, or None. A lock whose process has died is stale
    and reported as absent (with its details kept on disk for inspection).

    An unreadable lock counts as held: refusing a second run for a moment is
    harmless, whereas launching a duplicate over a live one is not.
    """
    info, readable = read_lock(tag, runs_dir)
    if info is None:
        return {"run_id": None, "pid": None, "unreadable": True} if not readable else None
    return info if _pid_alive(info.get("pid")) else None


def launch(plan_stages: list[dict], *, lock_tag: str, meta: dict | None = None,
           runs_dir: Path = RUNS_DIR, python: str | None = None,
           detach: bool = True) -> Path:
    """Start a run in the background and return its run directory.

    ``plan_stages`` is a list of ``{"key", "name", "tag", "args"}``: each stage runs
    ``python -u <args>`` from the project root. A stage may instead carry
    ``"argv"``, a complete command run as given -- how a page served from Windows
    runs a stage inside WSL (``wsl.exe ...``) while this runner, its lock and its
    log stay on the Windows side where the page can watch them. ``meta`` with
    ``"keep_awake": True`` asks Windows not to sleep while the run lasts.

    Raises :class:`LockHeld` if a live run already holds ``lock_tag``.
    """
    held = active_lock(lock_tag, runs_dir)
    if held:
        raise LockHeld(f"tag {lock_tag!r} is in use by run {held.get('run_id')} "
                       f"(pid {held.get('pid')}, started {held.get('started')})")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{lock_tag}"
    run_dir = runs_dir / run_id
    (runs_dir / "locks").mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "plan.json", {"run_id": run_id, "lock_tag": lock_tag,
                                        "meta": meta or {}, "stages": plan_stages})
    _write_json(run_dir / "status.json", {
        "run_id": run_id, "state": "starting", "created": _now(), "finished": None,
        "failed_stage": None,
        "stages": [{"key": s["key"], "name": s["name"], "tag": s["tag"],
                    "state": "pending", "started": None, "ended": None,
                    "minutes": None, "exit_code": None} for s in plan_stages]})
    (run_dir / "log.txt").write_text("", encoding="utf-8")

    python = python or sys.executable
    env = {**os.environ, "PYTHONUNBUFFERED": "1",
           "PYTHONPATH": str(PROJECT_ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    cmd = [python, "-m", "creditsurv.runner", str(run_dir)]
    lock = _lock_path(runs_dir, lock_tag)
    # Held under this process's pid until the child exists, so there is no window
    # in which a second launch could pass the check above.
    _write_json(lock, {"run_id": run_id, "pid": os.getpid(), "started": _now(),
                       "run_dir": str(run_dir)})
    if detach:
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        with open(run_dir / "runner.out", "w", encoding="utf-8") as out:
            proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, stdout=out,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    creationflags=flags, close_fds=True)
        # Hand the lock to the child, unless it has already finished and released it.
        if lock.exists() and read_status(run_dir).get("finished") is None:
            _write_json(lock, {"run_id": run_id, "pid": proc.pid, "started": _now(),
                               "run_dir": str(run_dir)})
    else:
        execute(run_dir, python=python)
    return run_dir


PROGRESS_ENV = {"CREDITSURV_SHOW_PROGRESS": "1",   # the SurvSHAP engine's bar on
                "TQDM_MININTERVAL": "15",           # one update per 15 s in a log
                "TQDM_NCOLS": "100"}
# Not TQDM_ASCII: tqdm reads "1" as the bar's character set -- one symbol -- and
# divides by zero drawing it, which would crash the run it reports on. It falls
# back to ASCII by itself where the log cannot take Unicode.


def progress_env(show: bool) -> dict:
    """The environment a stage runs in. With ``show``, long computations print a
    progress bar the pages can read (see :func:`creditsurv.evidence_jobs.parse_progress`),
    and ``WSLENV`` carries those settings through ``wsl.exe`` into WSL, which does
    not inherit Windows environment variables otherwise."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if show:
        env.update(PROGRESS_ENV)
        passed = [v for v in env.get("WSLENV", "").split(":") if v]
        passed += [f"{k}/u" for k in PROGRESS_ENV if f"{k}/u" not in passed]
        env["WSLENV"] = ":".join(passed)
    return env


def execute(run_dir: Path, *, python: str | None = None) -> int:
    """Run every stage in order, stopping at the first failure. Returns the exit code."""
    run_dir = Path(run_dir)
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    runs_dir = run_dir.parent
    python = python or sys.executable
    stages = plan["stages"]
    code = 0
    status["state"] = "running"
    _write_json(run_dir / "status.json", status)
    awake = bool((plan.get("meta") or {}).get("keep_awake"))
    heartbeat = float((plan.get("meta") or {}).get("heartbeat_seconds", 300))
    stage_env = progress_env(bool((plan.get("meta") or {}).get("show_progress")))
    if awake:
        from .explain.parallel import keep_awake
        keep_awake(True)
    try:
        with open(run_dir / "log.txt", "a", encoding="utf-8", errors="replace") as log:
            def say(line: str) -> None:
                log.write(line + "\n")
                log.flush()

            for i, st in enumerate(stages, start=1):
                label = f"[{i}/{len(stages)}] {st['name']}"
                rec = status["stages"][i - 1]
                rec.update(state="running", started=_now())
                _write_json(run_dir / "status.json", status)
                say("")
                say("-" * 78)
                say(f"{label}   (--tag {st['tag']})   started {datetime.now():%H:%M:%S}")
                cmd = list(st["argv"]) if st.get("argv") else [python, "-u", *st["args"]]
                say(f"  > {' '.join(cmd) if st.get('argv') else 'python ' + ' '.join(st['args'])}")
                say("-" * 78)
                t0 = time.perf_counter()
                proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT,
                                        stdout=log, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL,
                                        env=stage_env)
                # A heartbeat in the log, whatever the stage prints: a stage that
                # is silent for most of an hour (03d's SurvSHAP step) still shows
                # it is alive and how long it has been going.
                while True:
                    try:
                        code = proc.wait(timeout=heartbeat)
                        break
                    except subprocess.TimeoutExpired:
                        say(f"  ... still running: {(time.perf_counter() - t0) / 60:.0f} "
                            f"min elapsed ({datetime.now():%H:%M:%S})")
                mins = (time.perf_counter() - t0) / 60
                rec.update(ended=_now(), minutes=round(mins, 1), exit_code=code)
                if code != 0:
                    rec["state"] = "failed"
                    say("")
                    say(f"{label}  FAILED (exit code {code}) after {mins:.1f} min, "
                        f"see output above.")
                    if code == OVERWRITE_REFUSED:
                        say("  Exit 4 = the stage refused to overwrite existing outputs "
                            "for this tag.")
                    say("  Stopped. No later stage was run and nothing was cleaned up "
                        "or retried.")
                    for later in status["stages"][i:]:
                        later["state"] = "not run"
                    status.update(state="failed", failed_stage=st["name"])
                    break
                rec["state"] = "completed"
                say(f"{label}  completed in {mins:.1f} min")
                _write_json(run_dir / "status.json", status)
            else:
                status["state"] = "completed"
                say("")
                say("Sequence complete.")
    except Exception as exc:                     # the runner itself failed
        status.update(state="failed", failed_stage=f"runner error: {exc!r}")
        code = code or 1
    finally:
        if awake:
            from .explain.parallel import keep_awake
            keep_awake(False)
        status["finished"] = _now()
        _write_json(run_dir / "status.json", status)
        lock = _lock_path(runs_dir, plan["lock_tag"])
        try:
            info = json.loads(lock.read_text(encoding="utf-8"))
            if info.get("run_id") == plan["run_id"]:
                lock.unlink()
        except (OSError, json.JSONDecodeError):
            pass
    return code


def read_status(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "status.json").read_text(encoding="utf-8"))


def effective_state(run_dir: Path) -> str:
    """The run's state, correcting for a runner that died without finishing.

    A status file still saying ``running`` while no live process holds the run's
    lock means the runner was killed (machine slept, process ended); that is shown
    as ``interrupted`` rather than left looking alive.
    """
    run_dir = Path(run_dir)
    st = read_status(run_dir)
    if st["state"] not in ("starting", "running"):
        return st["state"]
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    info, readable = read_lock(plan["lock_tag"], run_dir.parent)
    if info is None and not readable:
        return st["state"]          # cannot tell; never claim a live run has died
    if info is None or not _pid_alive(info.get("pid")):
        return "interrupted"
    return st["state"] if info.get("run_id") == plan["run_id"] else "interrupted"


def tail_log(run_dir: Path, max_lines: int = 400) -> str:
    p = Path(run_dir) / "log.txt"
    if not p.exists():
        return ""
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def list_runs(runs_dir: Path = RUNS_DIR) -> list[Path]:
    """Run directories, newest first."""
    if not runs_dir.exists():
        return []
    return sorted((p for p in runs_dir.iterdir() if (p / "status.json").exists()),
                  reverse=True)


if __name__ == "__main__":
    raise SystemExit(execute(Path(sys.argv[1])))
