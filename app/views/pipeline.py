"""Run pipeline: a scoring run as the automated system it is.

Every stage of a run -- check, clean, score, decide, drift, run checks, explain,
notices -- as a box that changes state as the run moves, with Phase 2's progress
and the live log, all polled from the API. Opens on the run this session started
last, or the newest one; any run can be picked. New runs are started from Score
applicants, where the file's columns are confirmed first.
"""

from __future__ import annotations

import streamlit as st

from _client import ApiError, api
from _common import badge, dot, kpi_row, nav_link, page_header, when
from _phase2_view import render_phase2
from _pipeline_view import follow

meta = page_header("Run pipeline",
                   "Each stage of a scoring run, live: what has finished, what is "
                   "running, what failed and why. Follows runs started from any page "
                   "or from the API.")

try:
    runs = api().get("/runs", limit=40)["runs"]
except ApiError as exc:
    exc.show()
    st.stop()
if not runs:
    st.info("No scoring run yet. Start one from **Score applicants**.")
    nav_link("views/home.py", label="Score applicants", icon=":material/upload_file:")
    st.stop()

ids = [r["run_id"] for r in runs]
by_id = {r["run_id"]: r for r in runs}
mine = st.session_state.get("run_id")
pick = st.selectbox(
    "Run", ids, index=ids.index(mine) if mine in ids else 0,
    format_func=lambda i: f"{dot(by_id[i]['state'])} {when(by_id[i]['created_at'])} · "
                          f"{by_id[i].get('source_file') or i} · {by_id[i]['state']}")
st.session_state["run_id"] = pick
r = by_id[pick]

c = st.columns([3, 2, 2, 2])
c[0].markdown(f"**{r.get('source_file') or pick}**  \n`{pick}`")
c[1].markdown(f"State  \n{badge(r['state'])}", unsafe_allow_html=True)
c[2].markdown(f"Model  \n`{r.get('model_tag') or '—'}`")
if r.get("n_rows"):
    c[3].markdown(f"Applicants  \n**{int(r['n_rows']):,}**"
                  + (f" ({int(r['n_duplicates_removed'])} duplicates removed)"
                     if r.get("n_duplicates_removed") else ""))

st.subheader("Stages")
status = follow(pick, log_lines=60)
if status is None:
    st.stop()

err = status.get("error")
if err:
    st.error(f"**Stopped at {err.get('stage', '?')}: {err['message']}**"
             + (f"\n\n{err['fix']}" if err.get("fix") else ""), icon=":material/error:")
    if err.get("detail") and err["detail"] != err["message"]:
        with st.expander("Details"):
            st.code(err["detail"], language="text")

if status["finished"] and r.get("n_rejected"):
    try:
        detail = api().get(f"/runs/{pick}", preview_rows=0)
    except ApiError as exc:
        exc.show()
        st.stop()
    render_phase2(pick, meta, summary=detail["summary"])

if status["finished"]:
    s = r
    kpi_row([("Approved", f"{int(s.get('n_approved') or 0):,}"),
             ("Rejected", f"{int(s.get('n_rejected') or 0):,}"),
             ("Approval rate", f"{float(s.get('approval_rate') or 0):.1%}"),
             ("Phase 1", f"{float(s.get('phase1_seconds') or 0):.1f} s")])
    nav_link("views/results.py", label="Open this run in the Results viewer",
             icon=":material/table:")
