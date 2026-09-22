"""Overview: what has been run, under which tag, and whether it still verifies."""

import pandas as pd
import streamlit as st

from _common import STATE_ICON, page_header, paths
from creditsurv.runner import effective_state, list_runs, read_status
from creditsurv.status import overview

page_header("Overview",
            "Status of every result, read from the files the stage scripts wrote. "
            "Nothing on this page runs or changes anything.")

with st.expander("What the colours mean", expanded=False):
    st.markdown(
        "- 🟢 **verified**: the result's provenance stamp re-hashes cleanly; every "
        "input and model it recorded is unchanged on disk.\n"
        "- 🟠 **unverified**: produced before provenance stamping existed, so its "
        "inputs cannot be checked (file date shown instead).\n"
        "- 🔴 **changed**: an input or model the result recorded has since been "
        "replaced or is missing. The result no longer describes what is on disk.\n"
        "- ⚪ **not run**: no result file for this tag.")

with st.spinner("Verifying provenance stamps..."):
    ov = overview(paths().tables_dir)

if not ov:
    st.info("No stage results found in outputs/tables.")
for tag, rows in ov.items():
    counts = pd.Series([r.state for r in rows]).value_counts().to_dict()
    head = "  ".join(f"{STATE_ICON[k]} {v}" for k, v in counts.items())
    with st.expander(f"**{tag}**   {head}", expanded=tag.startswith("holdout")):
        st.dataframe(pd.DataFrame([{
            "": STATE_ICON[r.state], "stage": r.label, "tag": r.tag,
            "status": r.state, "run at": r.run_time or "", "commit": r.commit or "",
            "file": r.file, "detail": r.detail} for r in rows]),
            hide_index=True, width="stretch")

st.subheader("Runs started from this app")
runs = list_runs()
if not runs:
    st.caption("None yet.")
else:
    st.dataframe(pd.DataFrame([{
        "": STATE_ICON.get(s := effective_state(r), ""), "run": r.name, "state": s,
        "failed stage": read_status(r).get("failed_stage") or "",
        "finished": read_status(r).get("finished") or ""} for r in runs[:15]]),
        hide_index=True, width="stretch")
