"""Run pipeline: launch the holdout sequence in the background and follow it from
the first stage to the moment the process stops.

The commands come from creditsurv.plan, the same definition run_holdout.ps1 uses.
The run is a separate process (creditsurv.runner), so leaving or refreshing this
page does not stop it, and a per-tag lock refuses a second run on the same tag.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from _common import STATE_ICON, page_header, paths, rel
from creditsurv.environment import blocked_imports
from creditsurv.evidence_jobs import parse_progress
from creditsurv.plan import SIZES, STAGE_KEYS, TagError, holdout_plan
from creditsurv.runner import (LockHeld, active_lock, effective_state, launch,
                               list_runs, read_status, tail_log)
from creditsurv.status import preflight

page_header("Run pipeline",
            "Out-of-time holdout: train 2007-2015, test 2016-2018. Same commands as "
            "run_holdout.ps1. Stops at the first failure; nothing is retried or "
            "cleaned up.")

# A line in the log every 30 s while a stage prints nothing, so a quiet stage
# still visibly lives (the runner's default is every 5 min).
HEARTBEAT_SECONDS = 30
LIVE_LINES = 60


def free_tag(size: str) -> str:
    """The size's own tag if nothing has been written under it, else the first
    ``<tag>_<n>`` that is free -- so Start is not greyed out by default because a
    Small run was done once before. Full keeps ``holdout``: that tag is the
    pre-registered run, and its preflight refusal is the point."""
    base = SIZES[size].tag
    if size == "Full":
        return base
    for tag in [base] + [f"{base}_{n}" for n in range(2, 100)]:
        if not preflight(holdout_plan(size, tag=tag), paths()):
            return tag
    return base


# ------------------------------------------------------------------ settings --
c1, c2 = st.columns([1, 2])
with c1:
    size = st.radio("Size", list(SIZES), index=0, horizontal=True,
                    help="Small ~3-6 min, Medium ~12-20 min, Full ~35-55 min.")
with c2:
    tag = st.text_input("Tag", value=free_tag(size), key=f"tag_{size}",
                        help="Results are written under this tag. 'full' and 'dev' "
                             "(the primary results) are refused.").strip()

labels = {"02": "Stage 2 train", "03": "Stage 3 comparison", "03c": "Stage 3c noise floor",
          "03s": "Stage 3 stratified", "03b": "Stage 3b bootstrap",
          "04": "Stage 4 diagnostic", "05": "Stage 5 report", "prov": "Provenance check"}
chosen = st.multiselect("Stages (run in this order)", STAGE_KEYS, default=list(STAGE_KEYS),
                        format_func=lambda k: labels[k])
overwrite = st.checkbox("Overwrite existing outputs for this tag (--overwrite)", value=False)

try:
    plan = holdout_plan(size, overwrite=overwrite, tag=tag or None, only=tuple(chosen))
except TagError as exc:
    st.error(f"Tag refused: {exc}")
    st.stop()

if not plan.stages:
    st.warning("No stages selected.")
    st.stop()

if plan.write_findings:
    st.warning("**This is the pre-registered Full run.** Stage 5 will WRITE section 6 "
               "of FINDINGS.md. Every other size or tag only previews it (--dry-run).")
else:
    st.info("FINDINGS.md will not be written: Stage 5 runs with --dry-run and prints "
            "a preview of section 6 into the log.")

st.subheader("Commands")
st.dataframe(pd.DataFrame([{"#": i, "stage": s.name, "tag": s.tag,
                            "command": "python " + " ".join(s.args)}
                           for i, s in enumerate(plan.stages, 1)]),
             hide_index=True, width="stretch")

# ---------------------------------------------------------------- preflight --
hits = preflight(plan, paths())
blocked = False
confirmed = True
if hits:
    n = sum(len(v) for v in hits.values())
    untracked = [p for v in hits.values() for p in v
                 if rel(p).startswith(("outputs/models/", "outputs/data/"))]
    listing = "\n".join(f"- `{rel(p)}`" for v in hits.values() for p in v[:40])
    if not overwrite:
        blocked = True
        st.error(f"**{len(hits)} stage(s) would refuse to run:** {n} output file(s) "
                 f"already exist for this tag (stages {', '.join(hits)}). The scripts "
                 "would stop with exit code 4 before loading any data. Use a new tag, "
                 "or tick Overwrite.")
        with st.expander("Existing files"):
            st.markdown(listing)
    else:
        st.warning(f"--overwrite will REPLACE {n} existing file(s)."
                   + (f" {len(untracked)} are in outputs/models or outputs/data, which "
                      "are NOT in git and cannot be recovered." if untracked else ""))
        with st.expander("Files that will be replaced"):
            st.markdown(listing)
        confirmed = st.checkbox(f"I understand these {n} files will be replaced.")

if size == "Full":
    confirmed = confirmed and st.checkbox(
        "Run on the FULL data (about 35-55 minutes, heavy on memory).")

held = active_lock(plan.tag)
if held:
    st.error(f"Tag {plan.tag!r} is already in use by run {held['run_id']} "
             f"(pid {held['pid']}). Wait for it to finish.")

# Stage 2 trains a LightGBM model; where Smart App Control blocks it, every run
# would fail in its first minute. Say so here rather than in the log.
native_blocked = blocked_imports()
if native_blocked:
    st.error("This server cannot run the pipeline: "
             f"{', '.join(native_blocked)} cannot be loaded here. Serve the app from WSL "
             "(`run_linux.ps1`) to run it.")

if st.button("Start run", type="primary",
             disabled=blocked or not confirmed or bool(held) or bool(native_blocked)):
    try:
        run_dir = launch([{"key": s.key, "name": s.name, "tag": s.tag, "args": list(s.args)}
                          for s in plan.stages],
                         lock_tag=plan.tag,
                         meta={"size": size, "tag": plan.tag, "overwrite": overwrite,
                               "writes_findings": plan.write_findings, "source": "ui",
                               "heartbeat_seconds": HEARTBEAT_SECONDS,
                               "show_progress": True})
        st.session_state["run_dir"] = str(run_dir)
        st.success(f"Started run {run_dir.name}.")
    except LockHeld as exc:
        st.error(str(exc))

# ----------------------------------------------------------------- progress --
# Only pipeline runs: uploads and Phase 2 jobs share the runner, and have their
# own views on the Score applicants page.


def is_pipeline(run_dir: Path) -> bool:
    try:
        plan_ = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (plan_.get("meta") or {}).get("source") == "ui"


def minutes_since(stamp: str | None, until: str | None = None) -> float:
    if not stamp:
        return 0.0
    end = datetime.fromisoformat(until) if until else datetime.now()
    return max((end - datetime.fromisoformat(stamp)).total_seconds() / 60, 0.0)


def stage_log(log: str, number: int, total: int) -> str:
    """The part of the log written by stage ``number`` (1-based)."""
    start = log.rfind(f"[{number}/{total}] ")
    return log[start:] if start >= 0 else ""


def draw(run_dir: Path, live: bool) -> None:
    state = effective_state(run_dir)
    status = read_status(run_dir)
    stages = status["stages"]
    n = len(stages)
    done = sum(s["state"] == "completed" for s in stages)
    current = next((i for i, s in enumerate(stages) if s["state"] == "running"), None)
    total_min = minutes_since(status.get("created"),
                              None if live else status.get("finished"))
    log = tail_log(run_dir, 100_000)

    # ---- where the run is
    if state in ("starting", "running"):
        if current is None:
            st.info(f"🔵 **Starting** — {total_min:.1f} min since launch.")
        else:
            s = stages[current]
            here = stage_log(log, current + 1, n)
            last = [ln for ln in here.replace("\r", "\n").splitlines() if ln.strip()]
            st.info(f"🔵 **Stage {current + 1} of {n}: {s['name']}** — running for "
                    f"{minutes_since(s['started']):.1f} min"
                    + (f"\n\nLast output: `{last[-1].strip()[:140]}`" if last else ""))
            prog = parse_progress(here)
            if prog:
                st.progress(min(prog["done"] / max(prog["total"], 1), 1.0),
                            text=f"This stage: {prog['done']:,} of {prog['total']:,} "
                                 f"{prog['what']}"
                                 + (f" · about {prog['eta']} left" if prog.get("eta")
                                    else ""))
        st.progress(done / n, text=f"Whole run: {done} of {n} stages completed · "
                                   f"{total_min:.1f} min elapsed")
    elif state == "completed":
        st.success(f"🟢 **Completed** — all {n} stages in {total_min:.1f} min "
                   f"(finished {status.get('finished', '')}).")
    elif state == "failed":
        bad = next((s for s in stages if s["state"] == "failed"), None)
        st.error(f"🔴 **Failed at {status.get('failed_stage') or '?'}**"
                 + (f" (exit code {bad['exit_code']})" if bad else "")
                 + f" after {total_min:.1f} min. {done} of {n} stages completed; later "
                   "stages were not run. The end of the log below says why.")
    elif state == "interrupted":
        st.warning("🟠 **Interrupted** — the runner process is gone but the run never "
                   f"finished ({done} of {n} stages completed). The machine slept or the "
                   "process was ended. Nothing was cleaned up; check the log and re-run "
                   "with the same settings if needed.")

    # ---- every stage
    st.dataframe(pd.DataFrame([{
        "": STATE_ICON.get(s["state"], ""), "#": i, "stage": s["name"], "tag": s["tag"],
        "state": s["state"],
        "minutes": (round(minutes_since(s["started"]), 1) if s["state"] == "running"
                    else s["minutes"]),
        "exit": s["exit_code"]}
        for i, s in enumerate(stages, 1)]), hide_index=True, width="stretch")

    # ---- the output, newest line at the bottom
    lines = log.splitlines()
    if live:
        st.caption(f"Live output: the last {LIVE_LINES} lines, refreshed every 2 s. "
                   f"A quiet stage logs a heartbeat every {HEARTBEAT_SECONDS} s.")
        st.code("\n".join(lines[-LIVE_LINES:]) or "(no output yet)", language="text")
        with st.expander(f"Whole log so far ({len(lines):,} lines)"):
            st.code(log or "(no output yet)", language="text", height=500)
    else:
        st.caption(f"The end of the run's output ({len(lines):,} lines in all):")
        st.code("\n".join(lines[-LIVE_LINES:]) or "(no output)", language="text")
        with st.expander("Whole log, from launch to the end", expanded=False):
            st.code(log or "(no output)", language="text", height=500)
        st.download_button("Download the log", log.encode("utf-8"),
                           file_name=f"{run_dir.name}.log", mime="text/plain",
                           on_click="ignore")
        if state == "completed":
            st.page_link("views/results.py", label="Open the Results viewer")


st.divider()
st.subheader("Progress")
runs = [r for r in list_runs() if is_pipeline(r)]
if not runs:
    st.caption("No pipeline run started from this app yet.")
    st.stop()
names = [r.name for r in runs]
last = Path(st.session_state.get("run_dir", "")).name
default = names.index(last) if last in names else 0
pick = st.selectbox("Run", names, index=default,
                    format_func=lambda r: f"{r}  ({effective_state(runs[names.index(r)])})")
run_dir = runs[names.index(pick)]


@st.fragment(run_every=2)
def follow():
    """Redrawn every 2 s while the process runs; once it stops, the whole page is
    redrawn once more into the finished view, which no longer polls."""
    draw(run_dir, live=True)
    if effective_state(run_dir) not in ("starting", "running"):
        st.rerun(scope="app")


if effective_state(run_dir) in ("starting", "running"):
    follow()
else:
    draw(run_dir, live=False)
