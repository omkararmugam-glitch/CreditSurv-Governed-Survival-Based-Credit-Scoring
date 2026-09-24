"""Score Applicants: upload a CSV, get finished decision files.

The page is presentation only. Every step is one call into creditsurv.batch,
which in turn calls the pipeline's own validation, encoding, scoring, SurvSHAP(t)
and notice code. Nothing here trains anything.
"""

import hashlib
import re
from datetime import datetime
from pathlib import Path

import streamlit as st

from _common import page_header, paths
from _dashboard import render
from creditsurv.batch import (RUNS_DIR, BatchError, available_models, load_context,
                              load_result, read_upload, run_batch)
from creditsurv.derive import derive_features, load_costs
from creditsurv.schema_match import propose_mapping
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

# --------------------------------------------------- confirm what is what --
# The mapping is proposed, shown and confirmed before any scoring happens. A
# column is never read as a feature because the software guessed quietly.
raw = up.getvalue()
try:
    ctx_preview = load_context(cfg, tag, model_name)
    head = read_upload(raw[: 2_000_000], up.name).head(500)
except BatchError as exc:
    st.error(f"**{exc.message}**" + (f"\n\n{exc.fix}" if exc.fix else ""))
    with st.expander("Details"):
        st.code(exc.detail, language="text")
    st.stop()

proposal = propose_mapping(head, ctx_preview.spec, values=ctx_preview.clean_values)
table = proposal.to_frame()
st.subheader("Columns")
if table.empty:
    st.caption("Every column in this file already carries its model feature name.")
else:
    st.caption("Proposed reading of the file. Tick or untick, then confirm — nothing "
               "is scored until you do.")
    edited = st.data_editor(
        table, hide_index=True, width="stretch", key="mapping_editor",
        column_config={
            "use it": st.column_config.CheckboxColumn(required=True),
            "model feature": st.column_config.SelectboxColumn(
                options=[""] + list(ctx_preview.spec.all_columns)),
            "confidence": st.column_config.NumberColumn(format="%.2f",
                                                        disabled=True),
            "why": st.column_config.TextColumn(disabled=True),
            "content": st.column_config.TextColumn(disabled=True),
            "other candidates": st.column_config.TextColumn(disabled=True)},
        disabled=["uploaded column", "confidence", "why", "content",
                  "other candidates"])
    chosen = {r["uploaded column"]: r["model feature"] for _, r in edited.iterrows()
              if r["use it"] and r["model feature"]}
    duplicates = [f for f in set(chosen.values())
                  if list(chosen.values()).count(f) > 1]
    if duplicates:
        st.error(f"Two columns are mapped to the same feature: "
                 f"{', '.join(duplicates)}. Untick one — the choice between them is "
                 f"yours, not the software's.")
        st.stop()
    if proposal.conflicts:
        st.warning("Several columns could be the same feature, so none was "
                   "pre-selected: "
                   + "; ".join(f"{f} <- {', '.join(cs)}"
                               for f, cs in proposal.conflicts.items()))
    if proposal.recognised_but_unused:
        st.info("Recognised, but this model has no such feature: "
                + "; ".join(f"{k} ({v})" for k, v
                            in proposal.recognised_but_unused.items()))

    renamed = dict(chosen)
    # What the file will still be missing after the confirmed renames, and what of
    # that can be computed exactly instead of guessed.
    after = head.rename(columns=renamed)
    wanted = [c for c in ctx_preview.spec.all_columns if c not in after.columns]
    _, derived, blocked = derive_features(after, wanted)
    costs = load_costs(ctx_preview.model_tag, paths().tables_dir)
    required = set(costs.required())
    still_missing = [c for c in wanted if c not in derived]
    missing_required = [c for c in still_missing if c in required]
    if derived:
        st.success("Will be computed exactly from other columns: "
                   + "; ".join(f"{k} ({v})" for k, v in derived.items()))
    if missing_required:
        st.error("**Required and not in the file:** "
                 + "; ".join(f"{c} (needs {' and '.join(blocked[c])})" if c in blocked
                             else f"{c} (not in file)" for c in missing_required)
                 + ". Required means the measured cost of its absence is at least "
                   "0.010 concordance (FINDINGS 7d).")
    optional_missing = [c for c in still_missing if c not in required]
    if optional_missing:
        detail = costs.describe(optional_missing)
        st.warning(f"{len(optional_missing)} optional feature(s) missing; the file "
                   f"will be scored without them."
                   + (f" Measured cost: {detail}" if detail else ""))
    st.session_state["confirmed_mapping"] = renamed
    if not st.checkbox("These columns are read correctly", value=not missing_required,
                       disabled=bool(missing_required)):
        st.stop()

mapping = st.session_state.get("confirmed_mapping") or None

# ----------------------------------------------------------------- process --
size_mb = len(raw) / 1e6
key = hashlib.sha256(
    raw + f"{tag}{model_name}{threshold}{max_explained}{sorted((mapping or {}).items())}"
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
    for uploaded, feature in (mapping or {}).items():
        args += ["--map", f"{uploaded}={feature}"]
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
                               max_explained=int(max_explained), mapping=mapping,
                               progress=progress)
    except BatchError as exc:
        st.error(f"**{exc.message}**" + (f"\n\n{exc.fix}" if exc.fix else ""))
        with st.expander("Details"):
            st.code(exc.detail, language="text")
        st.stop()
    st.session_state["batch_key"] = key
    st.session_state["batch_result"] = result

# The view lives in _dashboard.render so a test can draw every panel and tab of a
# finished run without uploading anything.
render(st.session_state["batch_result"], cfg)
