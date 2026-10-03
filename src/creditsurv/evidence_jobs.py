"""The approval evidence a candidate model still needs, run in the background.

Two of the seven approval rules need a long run to produce their evidence:

=========================  ======  =====================================  ==========
job                        rule    script                                 takes
=========================  ======  =====================================  ==========
``ablation``               A5      ``03e_feature_ablation.py``            5-10 min
``explainer_validation``   A6      ``03d_explainer_validation.py``        about 1 h
=========================  ======  =====================================  ==========

This module only moves the commands already given in FINDINGS 7l into the Model
registry page. The scripts, their arguments and their overwrite guards are
unchanged, and nothing here approves anything: a job produces a file, and the rule
is then evaluated on that file like any other (:func:`creditsurv.registry.evaluate_rules`).

A job is a :mod:`creditsurv.runner` run, the same detached-process machinery a large
upload uses, so it outlives the browser tab and its log can be followed from the
page at any time. Where it executes depends on where the page is served:

* **Windows** (where Smart App Control blocks lightgbm): the runner stays on
  Windows and runs three stages through ``wsl.exe`` -- sync the code into the WSL
  copy and check its environment (``wsl_launch.sh check``), run the script there
  (``wsl_launch.sh run``), and copy its results back (``wsl_launch.sh copy-back``)
  so the rules on this side can read them. The lock and the log are Windows files,
  so the page can tell a live job from a dead one.
* **Linux / WSL**: the script runs directly, as a large upload does.

One job at a time, across all models: both scripts load the full training data,
and two of them at once do not fit in this machine's memory.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .provenance import PROJECT_ROOT
from .runner import RUNS_DIR, active_lock, launch

__all__ = ["EvidenceJob", "JOBS", "LOCK_TAG", "SOURCE", "runs_in_wsl",
           "wsl_available", "windows_to_wsl", "job_stages", "launch_job",
           "existing_outputs", "evidence_runs", "job_running", "RUNS_DIR",
           "parse_progress", "refusal"]

LOCK_TAG = "registry_evidence"
SOURCE = "ui-registry"


@dataclass(frozen=True)
class EvidenceJob:
    key: str
    label: str
    rule: str
    script: str
    minutes: str

    def args(self, tag: str) -> list[str]:
        if self.key == "ablation":
            # The command FINDINGS 7l gives; --tag defaults to the model tag.
            return ["--model-tag", tag, "--sample", "50000"]
        # Its own output tag, so it never collides with the `explainer` run that
        # belongs to `full` -- which would refuse to overwrite, correctly.
        return ["--model-tag", tag, "--tag", f"explainer_{tag}", "--n-explain", "1000"]

    def outputs(self, tag: str, tables_dir: Path) -> list[Path]:
        """What the script writes, and refuses to overwrite without --overwrite."""
        if self.key == "ablation":
            names = [f"03e_ablation_{tag}.json", f"03e_ablation_features_{tag}.csv",
                     f"03e_ablation_groups_{tag}.csv"]
        else:
            names = [f"03d_explainer_validation_explainer_{tag}.json",
                     f"03d_explainer_per_applicant_explainer_{tag}.csv"]
        return [Path(tables_dir) / n for n in names]


JOBS: dict[str, EvidenceJob] = {
    "ablation": EvidenceJob("ablation", "Run ablation", "A5_ablation",
                            "scripts/03e_feature_ablation.py", "5-10 minutes"),
    "explainer_validation": EvidenceJob(
        "explainer_validation", "Run explainer validation", "A6_explainer_validation",
        "scripts/03d_explainer_validation.py", "about an hour"),
}


def runs_in_wsl() -> bool:
    """Whether a job must be sent to WSL: yes when this process is on Windows."""
    return sys.platform == "win32"


def wsl_available() -> bool:
    return shutil.which("wsl.exe") is not None


def windows_to_wsl(path: Path) -> str:
    """The WSL path of a Windows folder, as WSL itself converts it (handles spaces
    and OneDrive, as run_linux.ps1 does)."""
    out = subprocess.run(["wsl.exe", "-e", "wslpath", "-a", str(path)],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"WSL could not convert {path}: {out.stderr.strip()}")
    return out.stdout.strip()


def job_stages(kind: str, tag: str, *, wsl_root: str | None = None) -> list[dict]:
    """The runner stages for one job. ``wsl_root`` set means: run through WSL."""
    job = JOBS[kind]
    run_args = [job.script, *job.args(tag)]
    if wsl_root is None:
        return [{"key": kind, "name": f"{job.label} for {tag}", "tag": tag,
                 "args": run_args}]
    launcher = ["wsl.exe", "-e", "bash", f"{wsl_root}/scripts/wsl_launch.sh"]
    return [
        {"key": "sync", "name": "Sync code to WSL and check its environment",
         "tag": tag, "args": [], "argv": [*launcher, "check", wsl_root]},
        {"key": kind, "name": f"{job.label} for {tag} (in WSL)", "tag": tag,
         "args": run_args, "argv": [*launcher, "run", wsl_root, "--", *run_args]},
        {"key": "copy-back", "name": "Copy the results back to Windows", "tag": tag,
         "args": [], "argv": [*launcher, "copy-back", wsl_root]},
    ]


_EVIDENCE_GLOB = {"ablation": "03e_ablation_*.json",
                  "explainer_validation": "03d_explainer_validation_*.json"}


def existing_outputs(kind: str, tag: str, tables_dir: Path) -> list[Path]:
    """Files that mean this job has already been run for ``tag``.

    Its own outputs (the script would refuse to overwrite them), and any other
    evidence file the rule reads for this model, whatever its output tag. The page
    offers no second run in either case: evidence is replaced only deliberately,
    with --overwrite from the command line, never by running again until a result
    passes.
    """
    import json

    found = [p for p in JOBS[kind].outputs(tag, tables_dir) if p.exists()]
    for p in sorted(Path(tables_dir).glob(_EVIDENCE_GLOB[kind])):
        try:
            if json.loads(p.read_text(encoding="utf-8")).get("model_tag") == tag:
                found.append(p)
        except (OSError, ValueError):
            continue
    return list(dict.fromkeys(found))


def refusal(kind: str, tag: str, *, rule_passed: bool, tables_dir: Path,
            runs_dir: Path = RUNS_DIR, via_wsl: bool | None = None) -> str | None:
    """Why this job may not be started now, or None when it may.

    The one set of conditions for every caller -- the registry page and the API --
    so no route to a job skips one. In order: WSL is needed and absent; another
    evidence job is running (one at a time, since each loads the full training
    data); the rule already passes; or evidence for this model already exists.
    The last is the important one: running again until a result passes would
    defeat the bar, so replacing evidence is left to ``--overwrite`` from the
    command line, deliberately.
    """
    if kind not in JOBS:
        return f"unknown job {kind!r}; one of {', '.join(JOBS)}"
    job = JOBS[kind]
    via_wsl = runs_in_wsl() if via_wsl is None else via_wsl
    if via_wsl and not wsl_available():
        return "WSL is not installed on this machine."
    running = job_running(runs_dir)
    if running:
        return (f"Another evidence job is running ({running.get('run_id')}); one at "
                f"a time, since each loads the full training data.")
    short = job.rule.split("_", 1)[0]
    if rule_passed:
        return f"{short} already passes."
    existing = existing_outputs(kind, tag, tables_dir)
    if existing:
        return (f"Evidence for this model already exists ({existing[0].name}) and the "
                f"rule reads it. No second run is offered: running again until a "
                f"result passes would defeat the bar. Replacing evidence needs "
                f"--overwrite, from the command line.")
    return None


def launch_job(kind: str, tag: str, *, runs_dir: Path = RUNS_DIR,
               root: Path = PROJECT_ROOT, via_wsl: bool | None = None) -> Path:
    """Start one evidence job in the background; returns its run directory.

    Raises :class:`creditsurv.runner.LockHeld` while another evidence job runs.
    """
    via_wsl = runs_in_wsl() if via_wsl is None else via_wsl
    wsl_root = windows_to_wsl(root) if via_wsl else None
    job = JOBS[kind]
    return launch(job_stages(kind, tag, wsl_root=wsl_root), lock_tag=LOCK_TAG,
                  runs_dir=runs_dir,
                  meta={"source": SOURCE, "job": kind, "model_tag": tag,
                        "rule": job.rule, "via_wsl": bool(via_wsl),
                        "keep_awake": True, "show_progress": True})


_TQDM = re.compile(r"(\d+)/(\d+) \[(?:(\d+:)?\d+:\d+)<((?:\d+:)?\d+:\d+|\?)")
_FEATURES = re.compile(r"(features|groups?) (\d+)/(\d+)")


def parse_progress(log: str) -> dict | None:
    """How far the running step of an evidence job has got, read from its log.

    Understands the SurvSHAP engine's progress bar (printed when the runner sets
    ``CREDITSURV_SHOW_PROGRESS``, one update per 15 s) and the ablation's
    ``features i/N`` lines. Returns ``{"done", "total", "eta", "what"}`` for the
    latest one, or None when the log has neither yet.
    """
    tail = log[-20_000:].replace("\r", "\n")
    best = None
    for m in _TQDM.finditer(tail):
        best = (m.end(), int(m.group(1)), int(m.group(2)),
                None if m.group(4) == "?" else m.group(4), "applicants explained")
    for m in _FEATURES.finditer(tail):
        if best is None or m.end() > best[0]:
            best = (m.end(), int(m.group(2)), int(m.group(3)), None,
                    f"ablation {m.group(1)}")
    if best is None:
        return None
    return {"done": best[1], "total": best[2], "eta": best[3], "what": best[4]}


def job_running(runs_dir: Path = RUNS_DIR) -> dict | None:
    return active_lock(LOCK_TAG, runs_dir)


def evidence_runs(runs_dir: Path = RUNS_DIR) -> list[Path]:
    """Evidence-job run directories, newest first."""
    import json

    from .runner import list_runs
    out = []
    for run in list_runs(runs_dir):
        try:
            plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (plan.get("meta") or {}).get("source") == SOURCE:
            out.append(run)
    return out
