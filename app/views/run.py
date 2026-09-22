"""Run pipeline: launch the holdout sequence in the background and follow its log.

The commands come from creditsurv.plan, the same definition run_holdout.ps1 uses.
The run is a separate process (creditsurv.runner), so leaving or refreshing this
page does not stop it, and a per-tag lock refuses a second run on the same tag.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from _common import STATE_ICON, page_header, paths, rel
from creditsurv.plan import SIZES, STAGE_KEYS, TagError, holdout_plan
from creditsurv.runner import (LockHeld, active_lock, effective_state, launch,
                               list_runs, read_status, tail_log)
from creditsurv.status import preflight

page_header("Run pipeline",
            "Out-of-time holdout: train 2007-2015, test 2016-2018. Same commands as "
            "run_holdout.ps1. Stops at the first failure; nothing is retried or "
            "cleaned up.")

# ------------------------------------------------------------------ settings --
c1, c2 = st.columns([1, 2])
with c1:
    size = st.radio("Size", list(SIZES), index=0, horizontal=True,
                    help="Small ~3-6 min, Medium ~12-20 min, Full ~35-55 min.")
with c2:
    tag = st.text_input("Tag", value=SIZES[size].tag, key=f"tag_{size}",
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

if st.button("Start run", type="primary", disabled=blocked or not confirmed or bool(held)):
    try:
        run_dir = launch([{"key": s.key, "name": s.name, "tag": s.tag, "args": list(s.args)}
                          for s in plan.stages],
                         lock_tag=plan.tag,
                         meta={"size": size, "tag": plan.tag, "overwrite": overwrite,
                               "writes_findings": plan.write_findings, "source": "ui"})
        st.session_state["run_dir"] = str(run_dir)
        st.success(f"Started run {run_dir.name}.")
    except LockHeld as exc:
        st.error(str(exc))

# ----------------------------------------------------------------- progress --
st.divider()
st.subheader("Progress")
runs = list_runs()
if not runs:
    st.caption("No runs yet.")
    st.stop()
names = [r.name for r in runs]
last = Path(st.session_state.get("run_dir", "")).name
default = names.index(last) if last in names else 0
pick = st.selectbox("Run", names, index=default)
run_dir = runs[names.index(pick)]


@st.fragment(run_every=2)
def progress():
    state = effective_state(run_dir)
    status = read_status(run_dir)
    st.markdown(f"**{STATE_ICON.get(state, '')} {state}**"
                + (f" at: {status['failed_stage']}" if status.get("failed_stage") else ""))
    if state == "interrupted":
        st.warning("The runner process is gone but the run never finished (machine "
                   "slept, or the process was ended). Nothing was cleaned up; check the "
                   "log and re-run with the same settings if needed.")
    st.dataframe(pd.DataFrame([{
        "": STATE_ICON.get(s["state"], ""), "stage": s["name"], "tag": s["tag"],
        "state": s["state"], "minutes": s["minutes"], "exit": s["exit_code"]}
        for s in status["stages"]]), hide_index=True, width="stretch")
    st.code(tail_log(run_dir) or "(no output yet)", language="text")
    if state == "completed":
        st.page_link("views/results.py", label="Open the Results viewer")


progress()
