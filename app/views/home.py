"""Score Applicants: upload a CSV, get finished decision files.

The page is presentation only. Every step is one call into creditsurv.batch,
which in turn calls the pipeline's own validation, encoding, scoring, SurvSHAP(t)
and notice code. Nothing here trains anything.
"""

import hashlib
import re
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from _common import APPROVE, MUTED, REJECT, page_header, paths, rel
from creditsurv.batch import (CAP_NOTE, RUNS_DIR, BatchError, available_models,
                              bundle_zip, load_context, load_result, run_batch)
from creditsurv.config import load_config
from creditsurv.runner import LockHeld, effective_state, launch, read_status, tail_log

CONFIG = "config/config.yaml"
STEPS = [("check", "Checking file"), ("clean", "Cleaning data"),
         ("profile", "Profiling data"), ("score", "Scoring applicants"),
         ("explain", "Explaining decisions"), ("files", "Preparing files")]
ICON = {"pending": "⚪", "running": "🔵", "done": "🟢", "failed": "🔴"}

page_header("Score Applicants",
            "Upload a file of applicants. Every one is scored by the trained "
            "survival model, decided against the published threshold, and written "
            "out as finished CSVs with Regulation B reasons.")

cfg = load_config(CONFIG)
d = cfg.decision
models = available_models(paths().models_dir)
if not models:
    st.error("No trained model found, so nothing can be scored yet.")
    st.page_link("views/run.py", label="Go to Advanced > Run Pipeline to train one")
    st.stop()

# Settings stay in the sidebar: the main area shows only the drop zone until a
# file arrives.
with st.sidebar:
    with st.expander("Options", expanded=False):
        tag = st.selectbox("Model", models,
                           index=models.index(d.model_tag) if d.model_tag in models else 0)
        model_name = st.selectbox("Type", ["discrete_hazard", "cox"], index=0)
        threshold = st.slider(f"Reject at {d.horizon_months}-month default probability",
                              0.05, 0.60, float(d.reject_at_or_above), 0.01)
        max_explained = st.number_input("Rejected applicants explained (~2.7s each)",
                                        0, 2000, int(d.max_explained), 10)
        force_bg = st.checkbox(
            "Always run in the background", value=False,
            help=f"Uploads of {d.background_above_mb:.0f} MB or more always do. A "
                 f"background run survives leaving or refreshing this page, but "
                 f"reloads the model from scratch (20-40s).")
        st.caption(f"Defaults come from `decision:` in {CONFIG}.")

up = st.file_uploader("Upload applicant dataset (CSV)", type=["csv", "txt"],
                      accept_multiple_files=False)
if up is None:
    st.caption(f"Decision rule in force: reject when the predicted "
               f"{d.horizon_months}-month default probability is "
               f"{threshold:.0%} or higher. Model: {tag} ({model_name}).")
    st.stop()

# ----------------------------------------------------------------- process --
raw = up.getvalue()
size_mb = len(raw) / 1e6
key = hashlib.sha256(raw + f"{tag}{model_name}{threshold}{max_explained}"
                     .encode()).hexdigest()
background = force_bg or size_mb >= d.background_above_mb

# A background run is a detached process (creditsurv.runner) writing into a run
# folder this page created, so closing the tab does not stop it and the page only
# reads what the run wrote.
job = st.session_state.get("upload_job")
if job and job["key"] == key and st.session_state.get("batch_key") != key:
    state = effective_state(Path(job["log_dir"]))
    if state == "completed":
        try:
            st.session_state["batch_result"] = load_result(Path(job["run_dir"]))
            st.session_state["batch_key"] = key
        except BatchError as exc:
            st.error(f"**{exc.message}**")
            with st.expander("Details"):
                st.code(exc.detail, language="text")
            st.stop()
        st.session_state.pop("upload_job", None)
    elif state in ("failed", "interrupted"):
        status = read_status(Path(job["log_dir"]))
        st.error("**Scoring stopped before it finished.** Nothing was cleaned up or "
                 "retried; the output folder holds whatever had been written."
                 + (f" ({status['failed_stage']})" if status.get("failed_stage") else ""))
        with st.expander("Details", expanded=True):
            st.code(tail_log(Path(job["log_dir"])) or "(no output)", language="text")
        if st.button("Clear and try again"):
            st.session_state.pop("upload_job", None)
            st.rerun()
        st.stop()
    else:
        st.info(f"Scoring {size_mb:,.0f} MB in the background. You can leave this "
                f"page; the run keeps going and this view picks it up again.")

        @st.fragment(run_every=2)
        def watch():
            now = effective_state(Path(job["log_dir"]))
            st.markdown(f"**{ICON.get('running' if now in ('running', 'starting') else 'failed' if now in ('failed', 'interrupted') else 'done', '')} {now}**")
            st.code(tail_log(Path(job["log_dir"]), 60) or "(starting...)", language="text")
            if now not in ("running", "starting"):
                st.rerun(scope="app")

        watch()
        st.stop()

if st.session_state.get("batch_key") != key and background:
    run_dir = RUNS_DIR / (f"{datetime.now():%Y%m%d_%H%M%S}_"
                          + (re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(up.name).stem)[:40]
                             or "upload"))
    run_dir.mkdir(parents=True, exist_ok=True)
    saved = run_dir / f"input_{Path(up.name).name}"
    saved.write_bytes(raw)
    args = ["scripts/06_score_upload.py", "--file", str(saved),
            "--run-dir", str(run_dir), "--model-tag", tag, "--model", model_name,
            "--threshold", f"{threshold}", "--max-explained", f"{int(max_explained)}"]
    try:
        log_dir = launch([{"key": "score", "name": f"Score {up.name}",
                           "tag": run_dir.name, "args": args}],
                        lock_tag=run_dir.name,
                        meta={"source": "ui-upload", "file": up.name,
                              "size_mb": round(size_mb, 1), "model_tag": tag,
                              "threshold": threshold})
    except LockHeld as exc:
        st.error(str(exc))
        st.stop()
    st.session_state["upload_job"] = {"key": key, "run_dir": str(run_dir),
                                      "log_dir": str(log_dir)}
    st.rerun()

if st.session_state.get("batch_key") != key:          # a download click reruns the
    state = {k: "pending" for k, _ in STEPS}          # page; do not re-score then
    notes: dict[str, str] = {}
    box = st.container()

    def draw(placeholder):
        with placeholder.container():
            for k, label in STEPS:
                note = f" — {notes[k]}" if notes.get(k) else ""
                st.markdown(f"{ICON[state[k]]} **{label}**{note}")

    slot = box.empty()
    draw(slot)

    def progress(step, st_state, message=""):
        state[step] = st_state
        if message:
            notes[step] = message
        draw(slot)

    try:
        with st.spinner("Processing..."):
            ctx = load_context(cfg, tag, model_name)
            result = run_batch(raw, up.name, cfg, ctx=ctx, threshold=threshold,
                               max_explained=int(max_explained), progress=progress)
    except BatchError as exc:
        st.error(f"**{exc.message}**" + (f"\n\n{exc.fix}" if exc.fix else ""))
        with st.expander("Details"):
            st.code(exc.detail, language="text")
        st.stop()
    st.session_state["batch_key"] = key
    st.session_state["batch_result"] = result

result = st.session_state["batch_result"]
s, rep = result.summary, result.report
scored = result.scored

st.success(f"Done in {result.seconds:.1f}s "
           f"({result.seconds / max(len(scored), 1) * 1000:.0f}s per 1,000 applicants). "
           f"Files saved to `{rel(result.run_dir)}`.")

for w in rep.warnings:
    st.warning(w)
if s["degraded_coverage"]:
    st.error(f"**Degraded run:** only {s['features_present']} of "
             f"{s['features_expected']} model features were in this file "
             f"({s['feature_coverage']:.0%}). The scores are weaker than the "
             f"model's published performance.")
DRIFT_BOX = {"stable": st.success, "moderate": st.warning, "large": st.error,
             "unknown": st.warning, "insufficient": st.info}
if result.drift is not None:
    label = ("NOT ASSESSED" if result.drift.status == "insufficient"
             else result.drift.colour.upper())
    DRIFT_BOX[result.drift.status](f"**Data drift: {label}** — "
                                   + result.drift.headline())

# A declined applicant with no stated reasons is a compliance gap, so it is said
# here rather than left to a column in the CSV.
missing_reasons = int(s.get("n_rejected_without_reasons", 0) or 0)
if missing_reasons:
    st.error(f"**{missing_reasons:,} of {s['n_rejected']:,} rejected applicants have "
             f"no adverse-action reasons**: the explanation cap "
             f"(`decision.max_explained` = {s.get('max_explained')}) was reached. "
             f"Those rows are marked \"{CAP_NOTE}\" in every output file. Raise the "
             f"cap in the sidebar and re-run to generate the rest "
             f"(about 2.7s each, so roughly "
             f"{missing_reasons * 2.7 / 60:,.0f} more minutes).")

st.info(f"Decision rule: reject at a {s['horizon_months']}-month default "
        f"probability of **{s['threshold']:.0%}** or higher. This is a policy "
        f"choice, not a model output. Model `{s['model_tag']}` ({s['model']}) "
        f"does not use credit score or employment length — see FINDINGS.")

# --------------------------------------------------------------- dashboard --
c = st.columns(5)
c[0].metric("Applicants", f"{s['n_rows']:,}")
c[1].metric("Approved", f"{s['n_approved']:,}")
c[2].metric("Rejected", f"{s['n_rejected']:,}")
c[3].metric("Approval rate", f"{s['approval_rate']:.1%}")
mean_pd = s[f"mean_pd_{s['horizon_months']}m"]
c[4].metric(f"Average {s['horizon_months']}m risk", f"{mean_pd:.1%}")
if s["n_rejected"]:
    st.caption(f"Reasons generated for {s['n_explained']:,} of {s['n_rejected']:,} "
               f"rejected applicants by **{s.get('explainer', 'survshap')}**"
               + (f" — {missing_reasons:,} still without reasons."
                  if missing_reasons else " (all of them).")
               + (f"  Profiled and drift-checked on the first "
                  f"{s['profiled_rows']:,} of {s['n_rows']:,} rows."
                  if s.get("profiled_rows", 0) < s["n_rows"] else ""))

# Every chart below is drawn from counts accumulated while the file streamed
# through, so a 500 MB run costs no more to display than a small one.
agg = result.aggregates
left, right = st.columns(2)
with left:
    st.markdown("**Predicted risk**")
    if agg is not None:
        risk = agg.risk_frame(s["threshold"])
        hist = alt.Chart(risk).mark_bar().encode(
            x=alt.X("bin_start:Q", title="default probability",
                    scale=alt.Scale(domain=[0, 1])),
            x2="bin_end:Q",
            y=alt.Y("applicants:Q", title="applicants"),
            color=alt.Color("decision:N", scale=alt.Scale(
                domain=["approve", "reject"], range=[APPROVE, REJECT]), title=None),
            tooltip=["bin_start", "bin_end", "applicants"])
        rule = alt.Chart(pd.DataFrame({"t": [s["threshold"]]})).mark_rule(
            color=MUTED, strokeDash=[6, 4]).encode(x="t:Q")
        st.altair_chart(hist + rule, use_container_width=True)
with right:
    by = agg.group_frame() if agg is not None else pd.DataFrame()
    if not by.empty:
        st.markdown(f"**Decisions by loan {agg.group_column}**")
        st.altair_chart(alt.Chart(by).mark_bar().encode(
            y=alt.Y("group:N", sort="-x", title=None),
            x=alt.X("applicants:Q", stack="normalize", title="share"),
            color=alt.Color("decision:N", scale=alt.Scale(
                domain=["approve", "reject"], range=[APPROVE, REJECT]), title=None),
            tooltip=["group", "decision", "applicants"]),
            use_container_width=True)

top = agg.reason_frame().head(8) if agg is not None else pd.DataFrame()
if not top.empty:
    st.markdown("**Most common rejection reasons** (Regulation B wording)")
    st.altair_chart(alt.Chart(top).mark_bar(color=REJECT).encode(
        y=alt.Y("reason:N", sort="-x", title=None),
        x=alt.X("times cited:Q")), use_container_width=True)

st.markdown(f"**Preview** (first 25 of {s['n_rows']:,} rows; the full report is in "
            f"the download)")
st.dataframe(scored.head(25), hide_index=True, width="stretch")

tab_clean, tab_profile, tab_drift = st.tabs(
    ["Cleaning report", "Data profile", "Drift check"])

with tab_clean:
    cr = result.clean_report
    st.caption(f"Cleaning policy `{s['cleaning_policy_version']}`, using values "
               f"fitted on {s['cleaning_values_fitted_rows']:,} training rows"
               + ("" if s["cleaning_values_from_model_bundle"] else
                  " (this model bundle predates saved cleaning values, so they were "
                  "re-fitted from its training split — still training data, never "
                  "this upload)") + ".")
    for line in cr.plain_english():
        st.markdown(f"- {line}")
    st.dataframe(cr.to_frame(), hide_index=True, width="stretch")

with tab_profile:
    prof = result.profile
    o = prof["overview"]
    if s.get("profiled_rows", 0) < s["n_rows"]:
        st.caption(f"Computed on the first {s['profiled_rows']:,} of "
                   f"{s['n_rows']:,} rows.")
    cols = st.columns(4)
    cols[0].metric("Rows", f"{o['rows']:,}")
    cols[1].metric("Columns", f"{o['columns']:,}")
    cols[2].metric("Numeric", f"{o['numeric_columns']:,}")
    cols[3].metric("Categorical", f"{o['categorical_columns']:,}")
    miss = prof["missing"]
    if not miss.empty:
        st.markdown("**Missing values by column**")
        affected = miss[miss["missing_share"] > 0]
        if affected.empty:
            st.caption("No missing values in any model feature.")
        else:
            st.altair_chart(alt.Chart(affected.head(25)).mark_bar(color=ACCENT).encode(
                y=alt.Y("column:N", sort="-x", title=None),
                x=alt.X("missing_share:Q", axis=alt.Axis(format="%"),
                        title="share of rows missing")), use_container_width=True)
        st.dataframe(miss, hide_index=True, width="stretch", height=240)
    if not prof["numeric"].empty:
        st.markdown("**Numeric summary**")
        st.dataframe(prof["numeric"], hide_index=True, width="stretch", height=260)
    if not prof["categorical"].empty:
        st.markdown("**Category counts**")
        st.dataframe(prof["categorical"], hide_index=True, width="stretch", height=260)
    if not prof["outliers"].empty:
        st.markdown("**Outliers** (counts only; nothing was altered)")
        st.dataframe(prof["outliers"], hide_index=True, width="stretch", height=240)

with tab_drift:
    dr = result.drift
    st.markdown(f"**{dr.headline()}**")
    if dr.status == "insufficient":
        st.caption("Nothing is wrong with the file; there is simply not enough of it "
                   "to tell whether it resembles the training population.")
    st.caption("Numeric features: population stability index against the training "
               "distribution (below 0.10 stable, 0.10–0.25 moderate, above 0.25 "
               "large). Categorical: total variation distance on the same bands.")
    table = dr.table
    shifted = table[table["status"].isin(["large", "moderate", "unknown"])]
    if not shifted.empty:
        st.altair_chart(alt.Chart(shifted.head(20)).mark_bar().encode(
            y=alt.Y("feature:N", sort="-x", title=None),
            x=alt.X("score:Q", title="PSI / TVD"),
            color=alt.Color("status:N", scale=alt.Scale(
                domain=["large", "moderate", "unknown", "stable"],
                range=[REJECT, "#d9a02b", MUTED, APPROVE]), title=None)),
            use_container_width=True)
    st.dataframe(table, hide_index=True, width="stretch", height=320)

# --------------------------------------------------------------- downloads --
st.subheader("Downloads")
labels = {
    "scored_applicants.csv": "Every applicant, with risk, decision and top reasons",
    "approved_applicants.csv": f"Approved only ({s['n_approved']:,} rows)",
    "rejected_applicants.csv": f"Rejected only ({s['n_rejected']:,} rows), with "
                               "Regulation B reasons and fair-lending flags",
    "adverse_action_notices.zip": f"{s['n_notices']:,} formatted adverse-action notices",
    "cleaning_report.csv": "What cleaning did: per rule and per column",
    "data_drift.csv": "Per-feature drift of this file against the training data",
    "run_summary.csv": "One row describing this run, for traceability",
}
cols = st.columns(2)
for i, (name, label) in enumerate(labels.items()):
    path = result.files.get(name)
    with cols[i % 2]:
        if path is None:
            st.caption(f"{name} — not produced ({'no rejected applicants' if 'notice' in name else 'n/a'})")
            continue
        st.download_button(f"⬇ {name}", path.read_bytes(), file_name=name,
                           mime="application/zip" if name.endswith(".zip") else "text/csv",
                           width="stretch")
        st.caption(label)
st.download_button("⬇ Download all (ZIP)", bundle_zip(result),
                   file_name=f"{result.run_dir.name}.zip", mime="application/zip",
                   type="primary")

with st.expander("Run details"):
    st.json(s, expanded=False)
    st.caption("Also written to the run folder as provenance.json, with SHA-256 "
               "hashes of the upload, the model and every output file.")
