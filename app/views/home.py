"""Score Applicants: upload a CSV, see decisions in minutes, reasons as they come.

The page is presentation only. Decisions are creditsurv.batch.score_file (Phase 1);
reasons and notices are creditsurv.phase2.explain_run (Phase 2), run in the
background or, for one applicant, on demand. The same two functions serve
06_score_upload.py and every other caller. Nothing here trains anything.
"""

import re
import shutil
from datetime import datetime
from pathlib import Path

import streamlit as st

from _common import (CONFIG_PATH, cached_context, cached_result, drop_upload,
                     hold_upload, keep, model_statuses, page_header, paths, remember)
from _dashboard import render
from _phase2_view import render_phase2
from creditsurv.batch import (PROVISIONAL_REQUIRED, RUNS_DIR, BatchError,
                              available_models, read_upload, score_file)
from creditsurv.config import load_config
from creditsurv.derive import derive_features, load_costs
from creditsurv.phase2 import explain_run
from creditsurv.registry import assess, registry_path
from creditsurv.runner import LockHeld, effective_state, launch, read_status, tail_log
from creditsurv.schema_match import propose_mapping

CONFIG = str(CONFIG_PATH)
STEPS = [("check", "Checking file"), ("clean", "Cleaning data"),
         ("profile", "Profiling data"), ("score", "Scoring applicants"),
         ("files", "Writing and checking decision files")]
ICON = {"pending": "⚪", "running": "🔵", "done": "🟢", "failed": "🔴"}


def show_failed_checks(run_dir) -> None:
    """A run that failed its own checks wrote validation_checks.csv and nothing that
    marks it finished; show which checks failed, whichever path ran it."""
    import pandas as pd

    path = Path(run_dir) / "validation_checks.csv" if run_dir else None
    if path is None or not path.exists():
        return
    checks = pd.read_csv(path)
    st.error(f"**Run checks: {int((checks['status'] == 'FAIL').sum())} of "
             f"{len(checks)} FAILED.** The run is not marked finished and no notices "
             f"will be produced for it.")
    st.dataframe(checks, hide_index=True, width="stretch")


def show_run(run_dir, cfg, tag, model_name) -> None:
    """The result view for a run whose decisions are on disk, with Phase 2 in it."""
    result = cached_result(run_dir)

    def explain_one(row_id: int) -> None:
        ctx = cached_context(result.summary["model_tag"], result.summary["model"])
        explain_run(run_dir, cfg, only=[row_id], model=ctx.model,
                    model_path=ctx.model_path, workers=1)

    render(result, cfg, middle=lambda: render_phase2(
        run_dir, cfg, summary=result.summary, explain_one=explain_one))


page_header("Score Applicants",
            "Upload a file of applicants. Decisions come first, in minutes for any file "
            "size; Regulation B reasons and notices follow, and you can watch them "
            "arrive.")

cfg = load_config(CONFIG)
d = cfg.decision
models = available_models(paths().models_dir)
if not models:
    st.error("No trained model found, so nothing can be scored yet.")
    st.page_link("views/run.py", label="Go to Advanced > Run Pipeline to train one")
    st.stop()

reg_file = registry_path(cfg)
statuses = model_statuses(CONFIG, tuple(models),
                          reg_file.stat().st_mtime_ns if reg_file.exists() else 0)

# Settings stay in the sidebar: the main area shows only the drop zone until a
# file arrives.
# Each option is kept across page changes (remember/keep): a threshold that fell
# back to its default on return would change the run key and score the file again.
OPTION_KEYS = ("opt_model", "opt_type", "opt_threshold", "opt_background", "opt_override")
if remember("opt_model", d.model_tag) not in models:
    st.session_state["opt_model"] = d.model_tag if d.model_tag in models else models[0]
remember("opt_type", "discrete_hazard")
remember("opt_threshold", float(d.reject_at_or_above))
remember("opt_background", False)
remember("opt_override", False)
with st.sidebar:
    with st.expander("Options", expanded=False):
        tag = st.selectbox("Model", models, key="opt_model",
                           format_func=lambda m: f"{m} — {statuses[m]}")
        model_name = st.selectbox("Type", ["discrete_hazard", "cox"], key="opt_type")
        threshold = st.slider(f"Reject at {d.horizon_months}-month default probability",
                              0.05, 0.60, step=0.01, key="opt_threshold")
        force_bg = st.checkbox(
            "Always run in the background", key="opt_background",
            help=f"Uploads of {d.background_above_mb:.0f} MB or more always do. A "
                 f"background run survives leaving or refreshing this page.")
        st.caption(f"Defaults come from `decision:` in config/config.yaml. Phase 2 "
                   f"asks first above {d.explain_confirm_above:,} rejected applicants.")
keep(*OPTION_KEYS)

# Only an approved model decides. Anything else needs an explicit override, and
# the run is then stamped "not for lending decisions" in every output.
approval = assess(tag, cfg)
allow_unapproved = False
if not approval.approved:
    st.error(f"**Model `{tag}` is not approved for lending decisions** (registry "
             f"status: {approval.status}). {' '.join(approval.problems)}",
             icon=":material/gpp_bad:")
    allow_unapproved = st.checkbox(
        "Override: score with this unapproved model anyway. Every output and notice "
        "will be stamped NOT FOR LENDING DECISIONS.", key="opt_override")
    keep("opt_override")
    if not allow_unapproved:
        st.caption("Pick an approved model under Options, or approve this one: "
                   f"`python scripts/07_model_registry.py rules --model-tag {tag}`.")
if abs(threshold - float(d.reject_at_or_above)) > 1e-12:
    st.warning(f"The threshold {threshold:.0%} is not the published "
               f"{d.reject_at_or_above:.0%}; this run will be stamped not for lending "
               f"decisions.")

new = st.file_uploader("Upload applicant dataset (CSV)", type=["csv", "txt"],
                       accept_multiple_files=False)
# The uploader comes back empty after another page has been shown; the file it
# held does not. It stays until another file is uploaded or it is cleared here.
up = hold_upload(new)
if up is not None and new is None:
    c1, c2 = st.columns([4, 1], vertical_alignment="center")
    c1.info(f"Working on **{up.name}** ({up.size / 1e6:,.1f} MB), uploaded earlier "
            f"in this session. Upload another file above to replace it.",
            icon=":material/description:")
    if c2.button("Clear this file", width="stretch"):
        drop_upload()
        st.rerun()
if up is None:
    st.caption(f"Decision rule in force: reject when the predicted "
               f"{d.horizon_months}-month default probability is "
               f"{threshold:.0%} or higher. Model: {tag} ({model_name}), "
               f"{approval.label}.")
    # A run is on disk whether or not this browser session saw it finish: open it.
    finished = sorted((p for p in RUNS_DIR.glob("*") if (p / "provenance.json").exists()),
                      reverse=True)[:15] if RUNS_DIR.exists() else []
    if finished:
        with st.expander("Open a previous run"):
            pick = st.selectbox("Run", ["—"] + [p.name for p in finished],
                                key="open_run_pick")
            if pick != "—":
                st.session_state["open_run"] = str(RUNS_DIR / pick)
    if st.session_state.get("open_run"):
        show_run(Path(st.session_state["open_run"]), cfg, tag, model_name)
    st.stop()
st.session_state.pop("open_run", None)
if not approval.approved and not allow_unapproved:
    st.error(f"Nothing was scored: `{tag}` is not approved and the override is not "
             f"ticked.")
    st.stop()

# --------------------------------------------------- confirm what is what --
# The mapping is proposed, shown and confirmed before any scoring happens. A
# column is never read as a feature because the software guessed quietly.
try:
    ctx = cached_context(tag, model_name)
    head = read_upload(up.head(2_000_000), up.name).head(500)
except BatchError as exc:
    st.error(f"**{exc.message}**" + (f"\n\n{exc.fix}" if exc.fix else ""))
    with st.expander("Details"):
        st.code(exc.detail, language="text")
    st.stop()

proposal = propose_mapping(head, ctx.spec, values=ctx.clean_values)
table = proposal.to_frame()
# The editor loses its ticks when the page is left. Coming back, it starts from the
# mapping confirmed last time for this file and model, not from the proposal, so
# the run key -- and so the result on screen -- is the same one.
mapping_key = (up.file_id, tag, model_name)
confirmed_mappings = st.session_state.setdefault("confirmed_mappings", {})
base = st.session_state.get("mapping_base")
if base is None or base[0] != mapping_key:
    base = st.session_state["mapping_base"] = (mapping_key, table)
elif "mapping_editor" not in st.session_state and mapping_key in confirmed_mappings \
        and not table.empty:
    kept = confirmed_mappings[mapping_key]
    table = table.assign(**{
        "use it": table["uploaded column"].isin(kept),
        "model feature": [kept.get(c, f) for c, f
                          in zip(table["uploaded column"], table["model feature"])]})
    base = st.session_state["mapping_base"] = (mapping_key, table)
table = base[1]
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
                options=[""] + list(ctx.spec.all_columns)),
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
    after = head.rename(columns=renamed)
    wanted = [c for c in ctx.spec.all_columns if c not in after.columns]
    _, derived, blocked = derive_features(after, wanted)
    costs = load_costs(ctx.model_tag, paths().tables_dir)
    # The same one-rule-or-the-other choice scoring makes, so the page cannot
    # promise a file will be accepted and then have it refused.
    required = set(costs.required() if costs.measured else PROVISIONAL_REQUIRED)
    still_missing = [c for c in wanted if c not in derived]
    missing_required = [c for c in still_missing if c in required]
    if derived:
        st.success("Will be computed exactly from other columns: "
                   + "; ".join(f"{k} ({v})" for k, v in derived.items()))
    if missing_required:
        st.error("**Required and not in the file:** "
                 + "; ".join(f"{c} (needs {' and '.join(blocked[c])})" if c in blocked
                             else f"{c} (not in file)" for c in missing_required)
                 + ". " + costs.rule_note())
    optional_missing = [c for c in still_missing if c not in required]
    if optional_missing:
        detail = costs.describe(optional_missing)
        st.warning(f"{len(optional_missing)} optional feature(s) missing; the file "
                   f"will be scored without them."
                   + (f" Measured cost: {detail}" if detail else ""))
    st.caption(costs.rule_note())
    if not st.checkbox("These columns are read correctly", value=not missing_required,
                       disabled=bool(missing_required)):
        st.stop()
    confirmed_mappings[mapping_key] = renamed

mapping = confirmed_mappings.get(mapping_key) or None

# ----------------------------------------------------------------- Phase 1 --
# Keyed on Streamlit's own id for the upload, not a hash of its bytes: hashing a
# 450 MB file on every click was a full pass over it per rerun.
size_mb = up.size / 1e6
key = (f"{up.file_id}|{tag}|{model_name}|{threshold}|{allow_unapproved}|"
       f"{sorted((mapping or {}).items())}")
runs = st.session_state.setdefault("upload_runs", {})
background = force_bg or size_mb >= d.background_above_mb

if key not in runs:
    run_dir = RUNS_DIR / (f"{datetime.now():%Y%m%d_%H%M%S}_"
                          + (re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(up.name).stem)[:40]
                             or "upload"))
    if background:
        # A detached process (creditsurv.runner) runs Phase 1 in a folder this page
        # created, so closing the tab does not stop it; the page only reads it.
        run_dir.mkdir(parents=True, exist_ok=True)
        saved = run_dir / f"input_{Path(up.name).name}"
        shutil.copyfile(up.path, saved)
        args = ["scripts/06_score_upload.py", "--file", str(saved),
                "--run-dir", str(run_dir), "--model-tag", tag, "--model", model_name,
                "--threshold", f"{threshold}", "--phase2", "defer"]
        for uploaded, feature in (mapping or {}).items():
            args += ["--map", f"{uploaded}={feature}"]
        if allow_unapproved:
            args.append("--allow-unapproved-model")
        try:
            log_dir = launch([{"key": "score", "name": f"Decisions for {up.name}",
                               "tag": run_dir.name, "args": args}],
                             lock_tag=run_dir.name,
                             meta={"source": "ui-upload", "file": up.name,
                                   "size_mb": round(size_mb, 1), "model_tag": tag,
                                   "threshold": threshold})
        except LockHeld as exc:
            st.error(str(exc))
            st.stop()
        runs[key] = {"run_dir": str(run_dir), "log_dir": str(log_dir)}
    else:
        state = {k: "pending" for k, _ in STEPS}
        notes: dict[str, str] = {}
        slot = st.empty()

        def draw():
            with slot.container():
                for k, label in STEPS:
                    note = f" — {notes[k]}" if notes.get(k) else ""
                    st.markdown(f"{ICON[state[k]]} **{label}**{note}")

        def progress(step, st_state, message=""):
            if step in state:
                state[step] = st_state
                if message:
                    notes[step] = message
                draw()

        draw()
        try:
            result = score_file(up.getvalue(), up.name, cfg, ctx=ctx,
                                threshold=threshold, mapping=mapping,
                                progress=progress, run_dir=run_dir,
                                allow_unapproved_model=allow_unapproved)
        except BatchError as exc:
            st.error(f"**{exc.message}**" + (f"\n\n{exc.fix}" if exc.fix else ""))
            show_failed_checks(exc.run_dir)
            with st.expander("Details"):
                st.code(exc.detail, language="text")
            st.stop()
        slot.empty()
        runs[key] = {"run_dir": str(result.run_dir), "log_dir": None}

run = runs[key]
run_dir = Path(run["run_dir"])

if run["log_dir"] and not (run_dir / "provenance.json").exists():
    log_dir = Path(run["log_dir"])
    state = effective_state(log_dir)
    if state in ("failed", "interrupted", "completed"):
        status = read_status(log_dir)
        st.error("**Scoring stopped before the decisions were written.** Nothing was "
                 "cleaned up or retried."
                 + (f" ({status['failed_stage']})" if status.get("failed_stage") else ""))
        show_failed_checks(run_dir)
        with st.expander("Details", expanded=True):
            st.code(tail_log(log_dir) or "(no output)", language="text")
        if st.button("Clear and try again"):
            runs.pop(key, None)
            st.rerun()
        st.stop()
    st.info(f"Scoring {size_mb:,.0f} MB in the background: decisions first, reasons "
            f"after. You can leave this page; the run keeps going and this view picks "
            f"it up again (or open it later under *Open a previous run*).")

    @st.fragment(run_every=2)
    def watch():
        now = effective_state(log_dir)
        st.markdown(f"**{ICON['running'] if now in ('running', 'starting') else ICON['done']}"
                    f" {now}**")
        st.code(tail_log(log_dir, 40) or "(starting...)", language="text")
        if (run_dir / "provenance.json").exists() or now not in ("running", "starting"):
            st.rerun(scope="app")

    watch()
    st.stop()

# The view lives in _dashboard / _phase2_view so a test can draw every panel of a
# run without uploading anything.
show_run(run_dir, cfg, tag, model_name)
