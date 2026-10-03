"""Overview: how much has been processed, what one run decided, and what stands out.

A filter panel on the left and three sections stacked on the right:

1. **Processing history** -- volume and outcomes across every run the filter kept.
2. **This run** -- the selected run's decisions, in the Results viewer's chart style.
3. **Insights** -- four cards, each a template with one of this page's numbers in it.

Then the model making the decisions, and the most recent runs.

Every number is read through the API (GET /overview, /runs/{id}, /models) from files
runs wrote and from config/models.yaml -- nothing is estimated, sampled or
hard-coded. Rates use finished runs only (decisions final, every check passed).
Read-only: no scoring happens on this page.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from _client import ApiError, api
from _common import badge, dot, keep, kpi_row, nav_link, page_header, when
from _dashboard import safe_load
from _overview_view import (filter_panel, insight_cards, render_history,
                            render_insights, render_run)

meta = page_header("Overview",
                   "How much has been processed, what the selected run decided, and "
                   "what stands out.")

# The run list and an unfiltered history first: the panel needs them to offer a run,
# a date range and the model tags that exist.
try:
    runs = api().get("/runs", limit=5000)["runs"]
    first = api().get("/overview", trend_n=20)
except ApiError as exc:
    exc.show()
    st.stop()

# Nothing has ever been scored: every card here would be a zero and every chart
# empty, which reads as a broken page rather than a new one. Say so instead, and
# point at the page that fills it.
if not runs:
    st.info("**No runs yet.** Upload a file on **Score applicants** to get started; "
            "this page fills in as soon as the first run finishes.",
            icon=":material/monitoring:")
    nav_link("views/home.py", label="Go to Score applicants",
             icon=":material/upload_file:")
    st.caption("Volume, decisions and the model in use are all read from finished "
               "runs in outputs/runs. There are none.")
    st.stop()

choice = filter_panel(runs, first.get("history") or {},
                      finished_states=meta.get("finished_states") or ())
keep("ov_run", "ov_dates", "ov_model", "ov_lending")

try:
    ov = api().get("/overview", trend_n=20,
                   for_lending_only=choice["for_lending_only"],
                   date_from=choice["date_from"], date_to=choice["date_to"],
                   model_tag=choice["model_tag"] or "")
    reg = api().get("/models")
except ApiError as exc:
    exc.show()
    st.stop()

history = ov.get("history") or {}
approved = ov["models"]["approved"]

by_state = ov["runs_by_state"]
if by_state:
    st.markdown(" ".join(badge(k, f"{v} {k}") for k, v in
                         sorted(by_state.items(), key=lambda kv: -kv[1])),
                unsafe_allow_html=True)
if not approved:
    st.error("No model is approved: nothing can score for lending decisions.",
             icon=":material/gpp_bad:")

# ------------------------------------------------- 1. processing history --
render_history(history, meta)
st.divider()

# ------------------------------------------------------- 2. this run --
view = safe_load(choice["run_id"]) if choice["run_id"] else None
if view is None:
    st.subheader("This run")
    st.info("No run selected yet. Score a file on **Score applicants** and it "
            "appears here.")
elif view.state in ("failed checks", "refused", "interrupted", "incomplete"):
    st.subheader("This run")
    st.warning(f"`{view.run_id}` is **{view.state}**, so it has no decisions to show. "
               f"Open it in the Results viewer for what it did write.")
    view = None
elif view.state == "running":
    st.subheader("This run")
    st.info(f"`{view.run_id}` is still being scored; follow it on **Run pipeline**.")
    view = None
else:
    render_run(view, meta)

st.divider()

# ---------------------------------------------------------- 3. insights --
if view is not None:
    render_insights(insight_cards(view, history, meta, reg))
    st.divider()

# ----------------------------------------------------------- the model --
st.subheader("The model making decisions")
kpi_row([
    ("Runs to date", f"{ov['total_runs']:,}",
     "Every scoring run with a folder in outputs/runs, whatever its state. Not "
     "affected by the filter panel."),
    # A model tag is long and the metric value is set in 2.25rem, so this card asks
    # for whatever width its own tag needs; the row wraps around it.
    ("Approved model", approved[0]["tag"] if approved else "none"),
    ("Awaiting approval", f"{ov['models']['awaiting_approval']}",
     "Models with registry status candidate."),
])
if approved:
    a = approved[0]
    rec = next((m for m in reg["models"] if m["tag"] == a["tag"]), {})
    left, right = st.columns([2, 3])
    with left:
        st.markdown(f"### `{a['tag']}` {badge('approved')}", unsafe_allow_html=True)
        st.markdown(f"{a.get('summary') or ''}  \n"
                    f"Approved by **{a.get('approved_by') or '?'}** on "
                    f"{when(a.get('approved_on'))} (FINDINGS {a.get('findings_section')})  \n"
                    f"Trained on {int(a.get('n_train') or 0):,} loans "
                    f"({a.get('split_scheme') or '?'} split)")
        rules = rec.get("rules") or []
        st.caption(f"Approval rules: {sum(r['passed'] for r in rules)} of {len(rules)} "
                   f"pass now." if rules else "")
    with right:
        vm = a.get("validation_metrics") or {}
        rows = [{"model": k, "concordance": v.get("concordance"),
                 "AUC 12m": v.get("auc_12m"), "AUC 36m": v.get("auc_36m"),
                 "integrated Brier": v.get("ibs"), "test loans": v.get("n"),
                 "default rate": v.get("event_rate")}
                for k, v in vm.items() if isinstance(v, dict)]
        if rows:
            st.markdown("**Held-out test metrics** (from the registry)")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.4f")
                                        for c in ("concordance", "AUC 12m", "AUC 36m",
                                                  "integrated Brier", "default rate")})
    if len(approved) > 1:
        st.caption("Also approved: " + ", ".join(x["tag"] for x in approved[1:]))
cands = [m for m in reg["models"] if m["status"] == "candidate"]
if cands:
    st.warning(f"**{len(cands)} model(s) awaiting approval:** "
               + "; ".join(f"`{m['tag']}` ({sum(r['passed'] for r in m['rules'])}/"
                           f"{len(m['rules'])} rules pass)" for m in cands)
               + ". See Model registry.", icon=":material/pending_actions:")

# ------------------------------------------------------------- recent runs --
st.subheader("Most recent runs")
recent = ov["recent"]
if not recent:
    st.caption("None yet.")
else:
    st.dataframe(pd.DataFrame([{
        "": dot(r["state"]), "when": when(r["created_at"]),
        "file": r.get("source_file") or "", "state": r["state"],
        "applicants": r.get("n_rows"),
        "approval rate": (None if r.get("approval_rate") is None
                          else 100 * float(r["approval_rate"])),
        "model": r.get("model_tag") or "", "run": r["run_id"]} for r in recent]),
        hide_index=True, width="stretch",
        column_config={"approval rate": st.column_config.NumberColumn(format="%.1f%%")})
    nav_link("views/results.py", label="All runs in the Results viewer",
             icon=":material/table:")
st.caption(f"Colours: {badge('done', 'done / passed')} {badge('running', 'working')} "
           f"{badge('overridden', 'needs a look')} {badge('failed', 'failed')} "
           f"{badge('pending', 'not yet')}", unsafe_allow_html=True)
