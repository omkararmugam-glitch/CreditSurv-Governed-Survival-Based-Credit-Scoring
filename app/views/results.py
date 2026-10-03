"""Results viewer: every past run, without needing to know a folder name.

Scoring runs first -- browse, filter, open one to see its summary, checks, charts
and downloads -- then the research results (tables, figures and notices the stage
scripts wrote, with their provenance status). Read-only, all through the API.
"""

from __future__ import annotations

import io
import json

import pandas as pd
import streamlit as st

from _client import ApiError, api
from _common import dot, page_header, when
from _dashboard import render, render_failed, safe_load

meta = page_header("Results viewer",
                   "Every scoring run to date and every research result, with what "
                   "each one produced.")

tab_runs, tab_research = st.tabs(["Scoring runs", "Research results"])

# ------------------------------------------------------------- scoring runs --
with tab_runs:
    try:
        runs = api().get("/runs", limit=5000)["runs"]
    except ApiError as exc:
        exc.show()
        runs = []
    if not runs:
        st.info("No scoring run yet. Start one from Score applicants.")
    else:
        table = pd.DataFrame([{
            "": dot(r["state"]), "when": when(r["created_at"]),
            "file": r.get("source_file") or "", "state": r["state"],
            "model": r.get("model_tag") or "",
            "applicants": r.get("n_rows"),
            "approval rate": (None if r.get("approval_rate") is None
                              else 100 * float(r["approval_rate"])),
            "flag share": (None if r.get("fair_lending_flag_share") is None
                           else 100 * float(r["fair_lending_flag_share"])),
            "for lending": r.get("for_lending_decisions"),
            "run": r["run_id"]} for r in runs])
        f1, f2, f3 = st.columns([2, 2, 1])
        states = f1.multiselect("State", sorted(table["state"].unique()))
        text = f2.text_input("File name contains")
        lending = f3.checkbox("For lending only")
        shown = table
        if states:
            shown = shown[shown["state"].isin(states)]
        if text:
            shown = shown[shown["file"].str.contains(text, case=False, regex=False)]
        if lending:
            shown = shown[shown["for lending"] == True]  # noqa: E712
        st.caption(f"{len(shown):,} of {len(table):,} runs. Pick one below to open it.")
        st.dataframe(shown, hide_index=True, width="stretch", height=260,
                     column_config={
                         "approval rate": st.column_config.NumberColumn(format="%.1f%%"),
                         "flag share": st.column_config.NumberColumn(format="%.1f%%")},
                     )
        ids = shown["run"].tolist()
        if ids:
            wanted = st.session_state.get("run_id")
            pick = st.selectbox(
                "Open run", ids, index=ids.index(wanted) if wanted in ids else 0,
                format_func=lambda i: f"{table.set_index('run').loc[i, '']} "
                                      f"{table.set_index('run').loc[i, 'when']} · "
                                      f"{table.set_index('run').loc[i, 'file']} · {i}",
                key="results_pick")
            st.session_state["run_id"] = pick
            view = safe_load(pick)
            if view is not None:
                st.divider()
                if view.state == "running":
                    st.info("This run is still being scored; follow it on Run pipeline.")
                elif view.state in ("failed checks", "refused", "interrupted",
                                    "incomplete"):
                    render_failed(view)
                else:
                    render(view, meta)

# --------------------------------------------------------- research results --
with tab_research:
    try:
        tags = api().get("/research/tags")["tags"]
    except ApiError as exc:
        exc.show()
        tags = []
    if not tags:
        st.info("No research results in outputs/tables yet.")
    else:
        with st.expander("Provenance of every research result"):
            st.caption("🟢 verified: the stamp re-hashes cleanly · 🟠 unverified: "
                       "made before stamping · 🔴 changed: an input it recorded has "
                       "changed or gone · ⚪ not run.")
            if st.button("Check provenance now (hashes every input)",
                         key="check_prov"):
                try:
                    ov = api().get("/research/overview")["tags"]
                except ApiError as exc:
                    exc.show()
                    ov = {}
                rows = [{"": dot(x["state"]), "tag": t, "stage": x["label"],
                         "status": x["state"], "run at": x["run_time"] or "",
                         "file": x["file"], "detail": x["detail"]}
                        for t, xs in ov.items() for x in xs]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        base = st.selectbox("Tag", tags, index=tags.index("full") if "full" in tags
                            else 0)
        variant = st.radio("Variant", [base, f"{base}_strat"], horizontal=True,
                           format_func=lambda t: "main" if t == base
                           else "grade-stratified (_strat)")
        try:
            info = api().get(f"/research/tags/{variant}")
        except ApiError as exc:
            exc.show()
            info = {"stages": [], "files": [], "figures": []}
        st.markdown("  ".join(f"{dot(x['state'])} {x['label'].split('  ')[0]} "
                              f"{x['state']}" for x in info["stages"]))
        if any(x["state"] == "changed" for x in info["stages"]):
            st.error("At least one result for this tag no longer matches its inputs on "
                     "disk.")
        files = info["files"]
        t_tables, t_figs, t_notice, t_json = st.tabs(
            ["Tables", "Figures", "Adverse-action notice", "Raw JSON"])
        with t_tables:
            csvs = [f["name"] for f in files if f["kind"] == "csv"]
            if csvs:
                cmp_ = next((n for n in csvs if n.startswith("02_model_comparison_")), None)
                if cmp_:
                    st.markdown("**Model comparison**")
                    st.dataframe(pd.read_csv(io.BytesIO(api().raw(
                        f"/research/files/{cmp_}"))), hide_index=True, width="stretch")
                pick = st.selectbox("Table", csvs)
                st.dataframe(pd.read_csv(io.BytesIO(api().raw(f"/research/files/{pick}"))),
                             hide_index=True, width="stretch")
            else:
                st.caption("No CSV tables for this tag.")
        with t_figs:
            if not info["figures"]:
                st.caption("No figures for this tag.")
            cols = st.columns(2)
            for i, name in enumerate(info["figures"]):
                with cols[i % 2]:
                    st.image(api().raw(f"/research/files/{name}"), caption=name,
                             width="stretch")
        with t_notice:
            notices = [f["name"] for f in files if f["kind"] == "txt"]
            if not notices:
                st.caption("No adverse-action notice for this tag.")
            for name in notices:
                st.markdown(f"`{name}`")
                st.code(api().raw(f"/research/files/{name}").decode("utf-8", "replace"),
                        language="text")
        with t_json:
            jsons = [f["name"] for f in files if f["kind"] == "json"]
            if jsons:
                pick = st.selectbox("Result file", jsons)
                st.json(json.loads(api().raw(f"/research/files/{pick}")), expanded=1)
            else:
                st.caption("No JSON results for this tag.")
