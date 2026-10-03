"""Retrain models (research): launch the out-of-time holdout sequence and follow it.

The research stages, not scoring (scoring is Run pipeline). The commands come from
creditsurv.plan, the same definition run_holdout.ps1 uses, and the API starts them
in the background runner (POST /retrain) -- after enforcing every refusal this page
shows: a primary tag, outputs that already exist, an unconfirmed overwrite or Full
run, a tag already in use, a machine that cannot load lightgbm.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from _client import ApiError, api
from _common import dot, nav_link, page_header

page_header("Retrain models (research)",
            "Out-of-time holdout: train 2007-2015, test 2016-2018. Same commands as "
            "run_holdout.ps1. Stops at the first failure; nothing is retried or cleaned "
            "up. Not needed for scoring.")

HEARTBEAT_SECONDS = 30
LIVE_LINES = 60
LABELS = {"02": "Stage 2 train", "03": "Stage 3 comparison", "03c": "Stage 3c noise floor",
          "03s": "Stage 3 stratified", "03b": "Stage 3b bootstrap",
          "04": "Stage 4 diagnostic", "05": "Stage 5 report", "prov": "Provenance check"}

try:
    opts = api().get("/retrain/options")
except ApiError as exc:
    exc.show()
    st.stop()

c1, c2 = st.columns([1, 2])
with c1:
    size = st.radio("Size", list(opts["sizes"]), index=0, horizontal=True,
                    help="Small ~3-6 min, Medium ~12-20 min, Full ~35-55 min.")
with c2:
    tag = st.text_input("Tag", value=opts["sizes"][size]["free_tag"], key=f"tag_{size}",
                        help="Results are written under this tag. 'full' and 'dev' "
                             "(the primary results) are refused.").strip()
chosen = st.multiselect("Stages (run in this order)", opts["stage_keys"],
                        default=opts["stage_keys"], format_func=lambda k: LABELS[k])
overwrite = st.checkbox("Overwrite existing outputs for this tag (--overwrite)", value=False)

try:
    plan = api().get("/retrain/plan", size=size, tag=tag or None, overwrite=overwrite,
                     stages=",".join(chosen))
except ApiError as exc:
    st.error(exc.message)
    st.stop()
if not plan["stages"]:
    st.warning("No stages selected.")
    st.stop()

if plan["write_findings"]:
    st.warning("**This is the pre-registered Full run.** Stage 5 will WRITE section 6 "
               "of FINDINGS.md. Every other size or tag only previews it (--dry-run).")
else:
    st.info("FINDINGS.md will not be written: Stage 5 runs with --dry-run and prints a "
            "preview of section 6 into the log.")

st.subheader("Commands")
st.dataframe(pd.DataFrame([{"#": i, "stage": s["name"], "tag": s["tag"],
                            "command": s["command"]}
                           for i, s in enumerate(plan["stages"], 1)]),
             hide_index=True, width="stretch")

blocked, confirm_replace, confirm_full = False, False, False
n = plan["n_existing"]
if n:
    listing = "\n".join(f"- `{p}`" for v in plan["preflight"].values() for p in v[:40])
    if not overwrite:
        blocked = True
        st.error(f"**{len(plan['preflight'])} stage(s) would refuse to run:** {n} output "
                 f"file(s) already exist for this tag (stages "
                 f"{', '.join(plan['preflight'])}). The scripts would stop with exit "
                 f"code 4 before loading any data. Use a new tag, or tick Overwrite.")
        with st.expander("Existing files"):
            st.markdown(listing)
    else:
        st.warning(f"--overwrite will REPLACE {n} existing file(s)."
                   + (f" {plan['n_untracked']} are in outputs/models or outputs/data, "
                      "which are NOT in git and cannot be recovered."
                      if plan["n_untracked"] else ""))
        with st.expander("Files that will be replaced"):
            st.markdown(listing)
        confirm_replace = st.checkbox(f"I understand these {n} files will be replaced.")
if size == "Full":
    confirm_full = st.checkbox("Run on the FULL data (about 35-55 minutes, heavy on "
                               "memory).")
if plan["lock"]:
    st.error(f"Tag {plan['tag']!r} is already in use by run {plan['lock']['run_id']}. "
             f"Wait for it to finish.")
if plan["blocked_imports"]:
    st.error("The API cannot run the pipeline: "
             f"{', '.join(plan['blocked_imports'])} cannot be loaded there. Start it in "
             "WSL (`run_linux.ps1`).")

ready = (not blocked and (not n or confirm_replace) and (size != "Full" or confirm_full)
         and not plan["lock"] and not plan["blocked_imports"])
if st.button("Start run", type="primary", disabled=not ready):
    try:
        out = api().post("/retrain", {"size": size, "tag": tag or None,
                                      "overwrite": overwrite, "stages": chosen,
                                      "confirm_replace": confirm_replace,
                                      "confirm_full": confirm_full})
    except ApiError as exc:
        exc.show()
    else:
        st.session_state["retrain_job"] = out["job_id"]
        st.success(f"Started run {out['job_id']}.")


# ----------------------------------------------------------------- progress --
def minutes_since(stamp, until=None) -> float:
    if not stamp:
        return 0.0
    end = datetime.fromisoformat(until) if until else datetime.now()
    return max((end - datetime.fromisoformat(stamp)).total_seconds() / 60, 0.0)


def draw(job: dict, live: bool) -> None:
    state = job["state"]
    stages = job["stages"]
    total = len(stages)
    done = sum(s["state"] == "completed" for s in stages)
    current = next((i for i, s in enumerate(stages) if s["state"] == "running"), None)
    total_min = minutes_since(job.get("created"), None if live else job.get("finished"))
    log = job.get("log") or ""
    if state in ("starting", "running"):
        if current is None:
            st.info(f"🔵 **Starting** — {total_min:.1f} min since launch.")
        else:
            s = stages[current]
            last = [ln for ln in log.replace("\r", "\n").splitlines() if ln.strip()]
            st.info(f"🔵 **Stage {current + 1} of {total}: {s['name']}** — running for "
                    f"{minutes_since(s['started']):.1f} min"
                    + (f"\n\nLast output: `{last[-1].strip()[:140]}`" if last else ""))
            prog = job.get("progress")
            if prog:
                st.progress(min(prog["done"] / max(prog["total"], 1), 1.0),
                            text=f"This stage: {prog['done']:,} of {prog['total']:,} "
                                 f"{prog['what']}"
                                 + (f" · about {prog['eta']} left" if prog.get("eta")
                                    else ""))
        st.progress(done / total, text=f"Whole run: {done} of {total} stages completed · "
                                       f"{total_min:.1f} min elapsed")
    elif state == "completed":
        st.success(f"🟢 **Completed** — all {total} stages in {total_min:.1f} min "
                   f"(finished {job.get('finished', '')}).")
    elif state == "failed":
        bad = next((s for s in stages if s["state"] == "failed"), None)
        st.error(f"🔴 **Failed at {job.get('failed_stage') or '?'}**"
                 + (f" (exit code {bad['exit_code']})" if bad else "")
                 + f" after {total_min:.1f} min. {done} of {total} stages completed; "
                   "later stages were not run. The end of the log below says why.")
    elif state == "interrupted":
        st.warning("🟠 **Interrupted** — the runner process is gone but the run never "
                   f"finished ({done} of {total} stages completed). Nothing was cleaned "
                   "up; check the log and re-run with the same settings if needed.")
    st.dataframe(pd.DataFrame([{
        "": dot(s["state"]), "#": i, "stage": s["name"], "tag": s["tag"],
        "state": s["state"],
        "minutes": (round(minutes_since(s["started"]), 1) if s["state"] == "running"
                    else s["minutes"]),
        "exit": s["exit_code"]} for i, s in enumerate(stages, 1)]),
        hide_index=True, width="stretch")
    lines = log.splitlines()
    st.caption(f"Live output: the last {LIVE_LINES} lines, refreshed every 2 s. A quiet "
               f"stage logs a heartbeat every {HEARTBEAT_SECONDS} s." if live else
               f"The end of the run's output:")
    st.code("\n".join(lines[-LIVE_LINES:]) or "(no output yet)", language="text")
    if not live and state == "completed":
        nav_link("views/results.py", label="Open the Results viewer")


st.divider()
st.subheader("Progress")
try:
    jobs = api().get("/jobs", source="ui", limit=40)["jobs"]
except ApiError as exc:
    exc.show()
    st.stop()
if not jobs:
    st.caption("No retraining run started from this app yet.")
    st.stop()
names = [j["job_id"] for j in jobs]
last = st.session_state.get("retrain_job", "")
pick = st.selectbox("Run", names, index=names.index(last) if last in names else 0,
                    format_func=lambda r: f"{dot(jobs[names.index(r)]['state'])} {r}  "
                                          f"({jobs[names.index(r)]['state']})")
live = jobs[names.index(pick)]["state"] in ("starting", "running")


@st.fragment(run_every=2 if live else None)
def follow():
    try:
        job = api().get(f"/jobs/{pick}", lines=100_000 if not live else 400)
    except ApiError as exc:
        exc.show()
        return
    draw(job, live=job["state"] in ("starting", "running"))
    if live and job["state"] not in ("starting", "running"):
        st.rerun(scope="app")


follow()
