"""The Model registry page, as a function a test can draw with its own config.

Presentation only. Every rule is evaluated by creditsurv.registry.evaluate_rules and
every approval is made by creditsurv.registry.approve_model -- the two functions
07_model_registry.py's ``rules`` and ``approve`` commands run -- and every job is a
creditsurv.evidence_jobs run. Nothing on this page can mark a rule as passed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from _common import STATE_ICON, page_header
from creditsurv import evidence_jobs as jobs
from creditsurv.environment import runtime_label
from creditsurv.registry import (APPROVAL_RULES, RegistryError, approve_model,
                                 evaluate_rules, load_registry, synced_copy_source)
from creditsurv.runner import LockHeld, effective_state, read_status, tail_log

RULES = list(APPROVAL_RULES)
SHORT = {r: r.split("_", 1)[0] for r in RULES}          # A1_model_file -> A1


def _fingerprint(cfg, registry_file: Path) -> tuple:
    """Changes whenever a file a rule reads changes, so cached results never
    outlive the evidence they were computed from."""
    items = []
    for folder in (cfg.paths.models_dir, cfg.paths.tables_dir):
        folder = Path(folder)
        if folder.is_dir():
            items += [(p.name, p.stat().st_mtime_ns, p.stat().st_size)
                      for p in folder.iterdir() if p.is_file()]
    if registry_file.exists():
        items.append(("registry", registry_file.stat().st_mtime_ns))
    return tuple(sorted(items))


@st.cache_data(show_spinner="Checking every model against the seven rules...")
def _all_rules(_cfg, registry_file: str, fingerprint: tuple) -> dict:
    reg = load_registry(_cfg, path=Path(registry_file))
    return {tag: [(r.rule, r.passed, r.detail) for r in evaluate_rules(_cfg, rec)]
            for tag, rec in reg.models.items()}


def render(cfg, *, launcher=jobs.launch_job, runs_dir=None,
           via_wsl: bool | None = None) -> None:
    runs_dir = Path(runs_dir or jobs.RUNS_DIR)
    page_header("Model registry",
                "Which models may make lending decisions, the seven rules each is "
                "held to, and the runs that produce the missing evidence. Approval "
                "here is the same check as 07_model_registry.py approve.")
    try:
        reg = load_registry(cfg)
    except RegistryError as exc:
        st.error(f"**The model registry could not be read.** {exc}")
        st.stop()

    if st.session_state.get("registry_flash"):
        st.success(st.session_state.pop("registry_flash"))
    via_wsl = jobs.runs_in_wsl() if via_wsl is None else via_wsl
    synced_from = synced_copy_source(reg.path.parent.parent)
    if via_wsl:
        st.caption(f"Page served from {runtime_label()}. Background jobs run in WSL: "
                   "the code is synced there first, and the results are copied back "
                   "so the rules below can read them.")
    else:
        st.caption(f"Page served from {runtime_label()}. Background jobs run here.")

    results = _all_rules(cfg, str(reg.path), _fingerprint(cfg, reg.path))

    # ------------------------------------------------------------- the table --
    st.subheader("Every model, every rule")
    rows = []
    for tag, rec in sorted(reg.models.items()):
        row = {"model": tag, "status": rec.status}
        for rule, passed, detail in results[tag]:
            row[SHORT[rule]] = f"{'✅ PASS' if passed else '❌ FAIL'} — {detail}"
        row["all seven"] = "✅" if all(p for _, p, _ in results[tag]) else "❌"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={SHORT[r]: st.column_config.TextColumn(
                     SHORT[r], help=APPROVAL_RULES[r], width="medium") for r in RULES})
    with st.expander("What each rule requires"):
        st.markdown("\n".join(f"- **{SHORT[r]}** `{r}` — {t}"
                              for r, t in APPROVAL_RULES.items()))

    running = jobs.job_running(runs_dir)

    # ------------------------------------------------------------ candidates --
    candidates = [t for t, r in sorted(reg.models.items()) if r.status == "candidate"]
    st.subheader("Candidates")
    if not candidates:
        st.caption("No model has status candidate.")
    for tag in candidates:
        rule_state = {rule: (passed, detail) for rule, passed, detail in results[tag]}
        with st.container(border=True):
            st.markdown(f"**{tag}**")
            cols = st.columns(len(jobs.JOBS))
            for col, (kind, job) in zip(cols, jobs.JOBS.items()):
                passed, _ = rule_state[job.rule]
                existing = jobs.existing_outputs(kind, tag, cfg.paths.tables_dir)
                why = None
                if via_wsl and not jobs.wsl_available():
                    why = "WSL is not installed on this machine."
                elif running:
                    why = (f"Another evidence job is running ({running.get('run_id')}); "
                           f"one at a time, since each loads the full training data.")
                elif passed:
                    why = f"{SHORT[job.rule]} already passes."
                elif existing:
                    why = (f"Evidence for this model already exists "
                           f"({existing[0].name}) and the rule above reads it. No "
                           f"second run is offered: running again until a result "
                           f"passes would defeat the bar. Replacing evidence needs "
                           f"--overwrite, from the command line.")
                with col:
                    clicked = st.button(job.label, key=f"{kind}_{tag}",
                                        disabled=why is not None, width="stretch")
                    st.caption(f"{SHORT[job.rule]} · {job.minutes} · "
                               f"`python {job.script} {' '.join(job.args(tag))}`"
                               + (f"  \n{why}" if why else ""))
                if clicked:
                    try:
                        run_dir = launcher(kind, tag)
                    except LockHeld as exc:
                        st.error(str(exc))
                    except Exception as exc:        # WSL did not answer, etc.
                        st.error(f"**The job could not be started.** {exc}")
                    else:
                        st.session_state["registry_job"] = str(run_dir)
                        st.rerun()

    # --------------------------------------------------------------- approve --
    st.subheader("Approve")
    ready = [t for t, r in sorted(reg.models.items())
             if r.status not in ("approved", "deprecated")
             and all(p for _, p, _ in results[t])]
    if not ready:
        st.caption("The Approve button appears for a model once all seven rules pass. "
                   "None does yet.")
    if ready and synced_from:
        st.warning(f"This app is serving the WSL copy synced from {synced_from}. Its "
                   "config/ is replaced from Windows on every sync, so an approval "
                   "written here would be lost. Approve from an app started on "
                   "Windows, or with `python scripts/07_model_registry.py approve`.")
    for tag in ready:
        with st.form(f"approve_{tag}", border=True):
            st.markdown(f"**{tag}** passes all seven rules. Approving re-checks every "
                        "rule now, then records who approved it and when in "
                        "config/models.yaml.")
            by = st.text_input("Approved by", key=f"by_{tag}")
            findings = st.text_input("FINDINGS section recording the decision",
                                     value="7l", key=f"findings_{tag}")
            note = st.text_input("Note (optional)", key=f"note_{tag}")
            go = st.form_submit_button(f"Approve {tag}", type="primary",
                                       disabled=synced_from is not None)
        if go:
            outcome = approve_model(cfg, tag, by=by, findings=findings, note=note)
            if outcome.approved:
                st.session_state["registry_flash"] = (
                    f"{tag} is approved by {by.strip()}. Recorded in {reg.path.name}.")
                st.rerun()
            st.error(f"**Not approved.** {outcome.refusal}")
            if outcome.results:
                st.dataframe(pd.DataFrame(
                    [{"rule": r.rule, "result": "PASS" if r.passed else "FAIL",
                      "reason": r.detail} for r in outcome.results]),
                    hide_index=True, width="stretch")

    approved = [(t, r) for t, r in sorted(reg.models.items()) if r.status == "approved"]
    if approved:
        st.dataframe(pd.DataFrame([{
            "approved model": t, "by": r.approval.get("approved_by"),
            "when": r.approval.get("approved_at") or r.approval.get("approved_on"),
            "FINDINGS": r.approval.get("findings_section"),
            "explainers": ", ".join(r.approval.get("explainers") or []),
            "settings": str(r.approval.get("explain_settings"))}
            for t, r in approved]), hide_index=True, width="stretch")

    # ------------------------------------------------------------------ jobs --
    st.subheader("Evidence jobs")
    runs = jobs.evidence_runs(runs_dir)
    if not runs:
        st.caption("No evidence job has been run from this page yet.")
        return
    names = [r.name for r in runs]
    last = Path(st.session_state.get("registry_job", "")).name
    pick = st.selectbox("Job", names, index=names.index(last) if last in names else 0)
    run_dir = runs[names.index(pick)]
    st.caption("The job is its own process: close this tab and come back, and this "
               "log picks up where it is.")
    live = effective_state(run_dir) in ("starting", "running")

    @st.fragment(run_every=2 if live else None)
    def watch():
        state = effective_state(run_dir)
        status = read_status(run_dir)
        st.markdown(f"**{STATE_ICON.get(state, '')} {state}**"
                    + (f" at: {status['failed_stage']}" if status.get("failed_stage")
                       else ""))
        stage = next((s for s in status["stages"] if s["state"] == "running"), None)
        if state == "running" and stage and stage.get("started"):
            from datetime import datetime as _dt
            import json as _json
            minutes = (_dt.now() - _dt.fromisoformat(stage["started"])).total_seconds() / 60
            meta = _json.loads((run_dir / "plan.json").read_text(encoding="utf-8")).get(
                "meta", {})
            job = jobs.JOBS.get(meta.get("job", ""))
            log = tail_log(run_dir, 400)
            last = [ln for ln in log.replace("\r", "\n").splitlines() if ln.strip()]
            prog = jobs.parse_progress(log) if job and stage["key"] == job.key else None
            if prog:
                st.progress(min(prog["done"] / max(prog["total"], 1), 1.0),
                            text=f"{prog['done']:,} of {prog['total']:,} "
                                 f"{prog['what']}"
                                 + (f" · about {prog['eta']} left in this step"
                                    if prog.get("eta") else ""))
            st.info(f"**{stage['name']}** — running for {minutes:.0f} min"
                    + (f" (expected {job.minutes})" if job and stage["key"] == job.key
                       else "")
                    + (f". Last output: `{last[-1].strip()[:120]}`" if last else ""))
        if state == "interrupted":
            st.warning("The runner process is gone but the job never finished (the "
                       "machine slept, or the process was ended). Nothing was cleaned "
                       "up; read the log, then start it again.")
        st.dataframe(pd.DataFrame([{
            "": STATE_ICON.get(s["state"], ""), "stage": s["name"],
            "state": s["state"], "minutes": s["minutes"], "exit": s["exit_code"]}
            for s in status["stages"]]), hide_index=True, width="stretch")
        st.code(tail_log(run_dir, 200) or "(no output yet)", language="text")
        if live and state not in ("starting", "running"):
            st.rerun(scope="app")                 # re-read the rules on finish

    watch()
