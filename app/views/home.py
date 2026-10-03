"""Score Applicants: upload a CSV, see decisions in minutes, reasons as they come.

Presentation only, over the API. The upload is held by the API (POST /uploads) and
this page keeps only its id, so the file never disappears on a rerun or a change of
page. The columns are checked by the API (POST /uploads/{id}/check -- the same
check scoring makes), the run is started by it (POST /runs), and the result is
read from it (GET /runs/{id}).

Once the run finishes, "What happened" is drawn first: that is
``_overview_view.render_run``, the very panel Overview draws as its section 2,
called here with the run that just finished. Nothing about it is a second copy --
the numbers and the charts cannot differ between the two pages, because there is
only one function. Everything after it (the stamps and checks, the Phase 2 panel,
the tabs and the downloads) is where it was.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from _client import ApiError, api, frame
from _common import keep, page_header, remember, status_box
from _dashboard import render, render_failed, safe_load
from _overview_view import render_run
from _phase2_view import render_phase2
from _pipeline_view import follow

meta = page_header("Score Applicants",
                   "Upload a file of applicants. Decisions come first, in minutes for "
                   "any file size; Regulation B reasons and notices follow, and you can "
                   "watch them arrive.")
d = meta["decision"]

try:
    reg = api().get("/models")
except ApiError as exc:
    exc.show()
    st.stop()
models = reg["available"]
if not models:
    st.error("No trained model found, so nothing can be scored yet.")
    st.stop()
labels = {m["tag"]: m for m in reg["models"]}

# Settings stay in the sidebar; each is kept across page changes, since a threshold
# that fell back to its default on return would change the run.
OPTION_KEYS = ("opt_model", "opt_type", "opt_threshold", "opt_background", "opt_override")
if remember("opt_model", d["model_tag"]) not in models:
    st.session_state["opt_model"] = d["model_tag"] if d["model_tag"] in models else models[0]
remember("opt_type", "discrete_hazard")
remember("opt_threshold", float(d["reject_at_or_above"]))
remember("opt_background", False)
remember("opt_override", False)
with st.sidebar:
    with st.expander("Options", expanded=False):
        tag = st.selectbox("Model", models, key="opt_model",
                           format_func=lambda m: f"{m} — "
                           f"{labels.get(m, {}).get('label', 'unregistered')}")
        model_name = st.selectbox("Type", ["discrete_hazard", "cox"], key="opt_type")
        threshold = st.slider(f"Reject at {d['horizon_months']}-month default "
                              f"probability", 0.05, 0.60, step=0.01, key="opt_threshold")
        force_bg = st.checkbox(
            "Always run in the background", key="opt_background",
            help=f"Uploads of {d['background_above_mb']:.0f} MB or more always do. A "
                 f"background run survives the API being restarted.")
        st.caption(f"Defaults come from `decision:` in config/config.yaml. Phase 2 "
                   f"asks first above {d['explain_confirm_above']:,} rejected applicants.")
keep(*OPTION_KEYS)

# Only an approved model decides. Anything else needs an explicit override, and the
# run is then stamped "not for lending decisions" in every output. The API refuses
# the run without it; the page says so first.
m = labels.get(tag, {"approved": False, "label": "unregistered", "problems": [],
                     "status": "unregistered"})
allow_unapproved = False
if not m["approved"]:
    st.error(f"**Model `{tag}` is not approved for lending decisions** (registry "
             f"status: {m['status']}). {' '.join(m['problems'])}",
             icon=":material/gpp_bad:")
    allow_unapproved = st.checkbox(
        "Override: score with this unapproved model anyway. Every output and notice "
        "will be stamped NOT FOR LENDING DECISIONS.", key="opt_override")
    keep("opt_override")
if abs(threshold - float(d["reject_at_or_above"])) > 1e-12:
    st.warning(f"The threshold {threshold:.0%} is not the published "
               f"{d['reject_at_or_above']:.0%}; this run will be stamped not for "
               f"lending decisions.")

# ------------------------------------------------------------------ upload --
new = st.file_uploader("Upload applicant dataset (CSV)", type=["csv", "txt"],
                       accept_multiple_files=False)
if new is not None and st.session_state.get("upload_file_id") != new.file_id:
    try:
        with st.spinner(f"Sending {new.name} to the API..."):
            up = api().upload(new.name, new.getvalue())
    except ApiError as exc:
        exc.show()
        st.stop()
    old = st.session_state.get("upload_id")
    if old:
        try:
            api().delete(f"/uploads/{old}")
        except ApiError:
            pass
    st.session_state.update(upload_id=up["upload_id"], upload_name=up["name"],
                            upload_size=up["size_bytes"], upload_file_id=new.file_id)
    st.session_state.pop("run_id", None)

upload_id = st.session_state.get("upload_id")
if upload_id and new is None:
    c1, c2 = st.columns([4, 1], vertical_alignment="center")
    c1.info(f"Working on **{st.session_state['upload_name']}** "
            f"({st.session_state['upload_size'] / 1e6:,.1f} MB), held by the API. "
            f"Upload another file above to replace it.", icon=":material/description:")
    if c2.button("Clear this file", width="stretch"):
        try:
            api().delete(f"/uploads/{upload_id}")
        except ApiError:
            pass
        for k in ("upload_id", "upload_name", "upload_size", "upload_file_id", "run_id"):
            st.session_state.pop(k, None)
        st.rerun()


def show_result(run_id: str) -> None:
    view = safe_load(run_id)
    if view is None:
        return
    if view.state in ("failed checks", "refused", "interrupted", "incomplete"):
        render_failed(view)
        return
    # What the file came to, before the detail of how: the same per-run panel
    # Overview draws (its section 2), given the run that just finished. No click and
    # no change of page -- the moment Phase 1 is done, this is what replaces the
    # column-matching step, and uploading another file replaces it again with that
    # run's. While Phase 2 is still writing, its reasons chart says so; the Phase 2
    # panel and the partial-snapshot banner below already redraw the whole page when
    # it finishes (st.rerun(scope="app")), which brings this section with them, so
    # nothing here polls on its own.
    render_run(view, meta, heading="What happened")
    st.divider()
    render(view, meta, middle=lambda: render_phase2(run_id, meta, summary=view.summary))


if not upload_id:
    st.caption(f"Decision rule in force: reject when the predicted "
               f"{d['horizon_months']}-month default probability is {threshold:.0%} or "
               f"higher. Model: {tag} ({model_name}), {m['label']}.")
    st.caption("Past runs are in the Results viewer.")
    st.stop()
if not m["approved"] and not allow_unapproved:
    st.error(f"Nothing was scored: `{tag}` is not approved and the override is not "
             f"ticked.")
    st.stop()

# --------------------------------------------------- confirm what is what --
# The mapping is proposed, shown and confirmed before any scoring happens. A column
# is never read as a feature because the software guessed quietly.
key = (upload_id, tag, model_name)
confirmed = st.session_state.setdefault("confirmed_mappings", {})
try:
    with st.spinner("Checking the file's columns (the first time, this loads the "
                    "model)..."):
        check = api().post(f"/uploads/{upload_id}/check",
                           {"model_tag": tag, "model": model_name,
                            "mapping": confirmed.get(key)})
except ApiError as exc:
    exc.show()
    st.stop()

table = frame(check["proposal"])
st.subheader("Columns")
if table.empty:
    st.caption("Every column in this file already carries its model feature name.")
    mapping = {}
else:
    if key in confirmed:           # coming back: the confirmed ticks, not the proposal
        kept = confirmed[key]
        table = table.assign(**{
            "use it": table["uploaded column"].isin(kept),
            "model feature": [kept.get(c, f) for c, f
                              in zip(table["uploaded column"], table["model feature"])]})
    st.caption("Proposed reading of the file. Tick or untick, then confirm — nothing "
               "is scored until you do.")
    edited = st.data_editor(
        table, hide_index=True, width="stretch", key=f"mapping_editor_{upload_id}",
        column_config={
            "use it": st.column_config.CheckboxColumn(required=True),
            "model feature": st.column_config.SelectboxColumn(
                options=[""] + list(check["features"])),
            "confidence": st.column_config.NumberColumn(format="%.2f", disabled=True),
            "why": st.column_config.TextColumn(disabled=True),
            "content": st.column_config.TextColumn(disabled=True),
            "other candidates": st.column_config.TextColumn(disabled=True)},
        disabled=["uploaded column", "confidence", "why", "content", "other candidates"])
    mapping = {r["uploaded column"]: r["model feature"] for _, r in edited.iterrows()
               if r["use it"] and r["model feature"]}
    if mapping != check["mapping"]:
        try:
            check = api().post(f"/uploads/{upload_id}/check",
                               {"model_tag": tag, "model": model_name,
                                "mapping": mapping})
        except ApiError as exc:
            exc.show()
            st.stop()
    if check["duplicate_targets"]:
        st.error(f"Two columns are mapped to the same feature: "
                 f"{', '.join(check['duplicate_targets'])}. Untick one — the choice "
                 f"between them is yours, not the software's.")
        st.stop()
    if check["conflicts"]:
        st.warning("Several columns could be the same feature, so none was "
                   "pre-selected: " + "; ".join(f"{f} <- {', '.join(cs)}"
                                                for f, cs in check["conflicts"].items()))
    if check["recognised_but_unused"]:
        st.info("Recognised, but this model has no such feature: "
                + "; ".join(f"{k} ({v})" for k, v in check["recognised_but_unused"].items()))

if check["derived"]:
    st.success("Will be computed exactly from other columns: "
               + "; ".join(f"{k} ({v})" for k, v in check["derived"].items()))
if check["required_missing"]:
    st.error("**Required and not in the file:** "
             + "; ".join(f"{c} (needs {' and '.join(check['blocked'][c])})"
                         if c in check["blocked"] else f"{c} (not in file)"
                         for c in check["required_missing"])
             + ". " + check["rule_note"])
if check["optional_missing"]:
    st.warning(f"{len(check['optional_missing'])} optional feature(s) missing; the file "
               f"will be scored without them."
               + (f" Measured cost: {check['optional_cost']}"
                  if check["optional_cost"] else ""))
st.caption(check["rule_note"])
if not table.empty or check["required_missing"]:
    if not st.checkbox("These columns are read correctly",
                       value=not check["required_missing"],
                       disabled=bool(check["required_missing"]),
                       key=f"confirm_{upload_id}_{tag}_{model_name}"):
        st.stop()
confirmed[key] = mapping

# ----------------------------------------------------------------- Phase 1 --
run_key = f"{upload_id}|{tag}|{model_name}|{threshold}|{allow_unapproved}|" \
          f"{sorted(mapping.items())}"
runs = st.session_state.setdefault("upload_runs", {})
if run_key not in runs:
    try:
        started = api().post("/runs", {
            "upload_id": upload_id, "model_tag": tag, "model": model_name,
            "threshold": threshold, "mapping": mapping or None,
            "allow_unapproved_model": allow_unapproved,
            "background": True if force_bg else None})
    except ApiError as exc:
        exc.show()
        st.stop()
    runs[run_key] = started["run_id"]
run_id = runs[run_key]
st.session_state["run_id"] = run_id

status = api().get(f"/runs/{run_id}/status")
if not status["finished"] and status["state"] in ("running", "starting"):
    st.info("Scoring: decisions first, reasons after. You can leave this page; the "
            "API keeps the run, and the Run pipeline page follows it too.")
    follow(run_id, log_lines=20)
    st.stop()
if status["state"] in ("refused", "failed checks", "interrupted"):
    status_box("failed", f"Run `{run_id}` did not finish.")
    if st.button("Clear and try again"):
        runs.pop(run_key, None)
        st.rerun()
show_result(run_id)
