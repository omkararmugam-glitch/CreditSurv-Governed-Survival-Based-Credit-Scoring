"""Results viewer: tables, figures and notices a tag's stages wrote. Read-only."""

import json

import pandas as pd
import streamlit as st

from _common import STATE_ICON, page_header, paths
from creditsurv.status import discover_tags, figures_for_tag, stage_status, RESULT_STAGES

page_header("Results viewer", "Tables, figures and notices written by a completed run.")
P = paths()
base_tags = discover_tags(P.tables_dir)
if not base_tags:
    st.info("No results in outputs/tables yet.")
    st.stop()

default = base_tags.index("full") if "full" in base_tags else 0
base = st.selectbox("Tag", base_tags, index=default)
variant = st.radio("Variant", [base, f"{base}_strat"], horizontal=True,
                   format_func=lambda t: "main" if t == base else "grade-stratified (_strat)")

# Provenance for the chosen tag, so a result is never read without its status.
rows = [stage_status(P.tables_dir, base, k) for k in RESULT_STAGES]
rows = [r for r in rows if r.tag == variant]
st.markdown("  ".join(f"{STATE_ICON[r.state]} {r.label.split('  ')[0]} {r.state}"
                      for r in rows))
if any(r.state == "changed" for r in rows):
    st.error("At least one result for this tag no longer matches its inputs on disk. "
             "See Overview for which files changed.")

files = sorted(p for p in P.tables_dir.iterdir()
               if p.is_file() and p.stem.endswith(f"_{variant}"))
tab_t, tab_f, tab_n, tab_j = st.tabs(["Tables", "Figures", "Adverse-action notice", "Raw JSON"])

with tab_t:
    csvs = [p for p in files if p.suffix == ".csv"]
    cmp_ = next((p for p in csvs if p.name.startswith("02_model_comparison_")), None)
    if cmp_:
        st.markdown("**Model comparison**")
        st.dataframe(pd.read_csv(cmp_), hide_index=True, width="stretch")
    if csvs:
        pick = st.selectbox("Table", [p.name for p in csvs])
        st.dataframe(pd.read_csv(P.tables_dir / pick), hide_index=True, width="stretch")
    else:
        st.caption("No CSV tables for this tag.")

with tab_f:
    figs = figures_for_tag(P.figures_dir, variant)
    if not figs:
        st.caption("No figures for this tag.")
    cols = st.columns(2)
    for i, f in enumerate(figs):
        with cols[i % 2]:
            st.image(str(f), caption=f.name, width="stretch")

with tab_n:
    notices = [p for p in files if p.suffix == ".txt"]
    if not notices:
        st.caption("No adverse-action notice for this tag.")
    for p in notices:
        st.markdown(f"`{p.name}`")
        st.code(p.read_text(encoding="utf-8", errors="replace"), language="text")

with tab_j:
    jsons = [p for p in files if p.suffix == ".json"]
    if jsons:
        pick = st.selectbox("Result file", [p.name for p in jsons])
        st.json(json.loads((P.tables_dir / pick).read_text(encoding="utf-8")),
                expanded=1)
    else:
        st.caption("No JSON results for this tag.")
