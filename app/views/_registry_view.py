"""The Model registry page, over the API.

Presentation only. Every rule is evaluated, every job refused or started, and every
approval made by the API (GET /models, POST /models/{tag}/evidence/{kind},
POST /models/{tag}/approve), which calls the functions 07_model_registry.py runs.
Nothing on this page can mark a rule as passed, start a job the rules refuse, or
write the registry.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from _client import ApiError, api
from _common import badge, dot, page_header


def render() -> None:
    page_header("Model registry",
                "Which models may make lending decisions, the seven rules each is held "
                "to, and the runs that produce the missing evidence. Approval here is "
                "the same check as 07_model_registry.py approve.")
    try:
        reg = api().get("/models")
        rules = api().get("/models/rules")["rules"]
        ctx = api().get("/registry/context")
    except ApiError as exc:
        exc.show()
        st.stop()

    if st.session_state.get("registry_flash"):
        st.success(st.session_state.pop("registry_flash"))
    if ctx["jobs_via_wsl"]:
        st.caption(f"API on {ctx['runtime']}. Evidence jobs run in WSL: the code is "
                   "synced there first, and the results are copied back so the rules "
                   "below can read them.")
    else:
        st.caption(f"API on {ctx['runtime']}. Evidence jobs run there.")
    short = {r["rule"]: r["short"] for r in rules}
    models = reg["models"]

    # ------------------------------------------------------------- the table --
    st.subheader("Every model, every rule")
    rows = []
    for m in models:
        row = {"": dot(m["status"]), "model": m["tag"], "status": m["status"]}
        for r in m["rules"]:
            row[short[r["rule"]]] = f"{'✅ PASS' if r['passed'] else '❌ FAIL'} — {r['detail']}"
        row["all seven"] = "✅" if m["all_rules_pass"] else "❌"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).drop(columns=[""]), hide_index=True,
                 width="stretch",
                 column_config={r["short"]: st.column_config.TextColumn(
                     r["short"], help=r["requires"], width="medium") for r in rules})
    st.markdown(" ".join(badge(m["status"], f"{m['tag']}: {m['status']}")
                         for m in models), unsafe_allow_html=True)
    with st.expander("What each rule requires"):
        st.markdown("\n".join(f"- **{r['short']}** `{r['rule']}` — {r['requires']}"
                              for r in rules))

    # ------------------------------------------------------------ candidates --
    candidates = [m["tag"] for m in models if m["status"] == "candidate"]
    st.subheader("Candidates")
    if not candidates:
        st.caption("No model has status candidate.")
    for tag in candidates:
        try:
            detail = api().get(f"/models/{tag}")
        except ApiError as exc:
            exc.show()
            continue
        with st.container(border=True):
            st.markdown(f"**{tag}** {badge('candidate')}", unsafe_allow_html=True)
            cols = st.columns(len(detail["evidence_jobs"]))
            for col, job in zip(cols, detail["evidence_jobs"]):
                why = job["refusal"]
                with col:
                    clicked = st.button(job["label"], key=f"{job['kind']}_{tag}",
                                        disabled=why is not None, width="stretch")
                    st.caption(f"{short.get(job['rule'], job['rule'])} · {job['minutes']} "
                               f"· `{job['command']}`" + (f"  \n{why}" if why else ""))
                if clicked:
                    try:
                        out = api().post(f"/models/{tag}/evidence/{job['kind']}")
                    except ApiError as exc:
                        st.error(f"**The job could not be started.** {exc.message}")
                    else:
                        st.session_state["registry_job"] = out["job_id"]
                        st.rerun()

    # --------------------------------------------------------------- approve --
    st.subheader("Approve")
    ready = [m["tag"] for m in models
             if m["status"] not in ("approved", "deprecated") and m["all_rules_pass"]]
    synced_from = ctx.get("synced_from")
    if not ready:
        st.caption("The Approve button appears for a model once all seven rules pass. "
                   "None does yet.")
    if ready and synced_from:
        st.warning(f"The API is serving the WSL copy synced from {synced_from}. Its "
                   "config/ is replaced from Windows on every sync, so an approval "
                   "written here would be lost. Approve from an API started on Windows, "
                   "or with `python scripts/07_model_registry.py approve`.")
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
            try:
                api().post(f"/models/{tag}/approve",
                           {"by": by, "findings": findings, "note": note})
            except ApiError as exc:
                st.error(f"**{exc.message}**")
                if exc.body.get("results"):
                    st.dataframe(pd.DataFrame(
                        [{"rule": r["rule"], "result": "PASS" if r["passed"] else "FAIL",
                          "reason": r["detail"]} for r in exc.body["results"]]),
                        hide_index=True, width="stretch")
            else:
                st.session_state["registry_flash"] = (
                    f"{tag} is approved by {by.strip()}. Recorded in the registry.")
                st.rerun()

    approved = [m for m in models if m["status"] == "approved"]
    if approved:
        st.dataframe(pd.DataFrame([{
            "approved model": m["tag"], "by": m["approval"].get("approved_by"),
            "when": m["approval"].get("approved_at") or m["approval"].get("approved_on"),
            "FINDINGS": m["approval"].get("findings_section"),
            "explainers": ", ".join(m["approval"].get("explainers") or []),
            "settings": str(m["approval"].get("explain_settings"))}
            for m in approved]), hide_index=True, width="stretch")

    # ------------------------------------------------------------------ jobs --
    st.subheader("Evidence jobs")
    try:
        runs = api().get("/jobs", source="ui-registry")["jobs"]
    except ApiError as exc:
        exc.show()
        return
    if not runs:
        st.caption("No evidence job has been run from this page yet.")
        return
    names = [j["job_id"] for j in runs]
    last = st.session_state.get("registry_job", "")
    pick = st.selectbox("Job", names, index=names.index(last) if last in names else 0,
                        format_func=lambda n: f"{dot(runs[names.index(n)]['state'])} {n}")
    st.caption("The job is its own process: close this tab and come back, and this "
               "log picks up where it is.")
    live = runs[names.index(pick)]["state"] in ("starting", "running")

    @st.fragment(run_every=2 if live else None)
    def watch():
        try:
            job = api().get(f"/jobs/{pick}", lines=200)
        except ApiError as exc:
            exc.show()
            return
        state = job["state"]
        st.markdown(f"**{dot(state)} {state}**"
                    + (f" at: {job['failed_stage']}" if job.get("failed_stage") else ""))
        stage = next((s for s in job["stages"] if s["state"] == "running"), None)
        if state == "running" and stage and stage.get("started"):
            minutes = (datetime.now() - datetime.fromisoformat(stage["started"])
                       ).total_seconds() / 60
            prog = job.get("progress")
            if prog:
                st.progress(min(prog["done"] / max(prog["total"], 1), 1.0),
                            text=f"{prog['done']:,} of {prog['total']:,} {prog['what']}"
                                 + (f" · about {prog['eta']} left in this step"
                                    if prog.get("eta") else ""))
            lines = [ln for ln in (job.get("log") or "").replace("\r", "\n").splitlines()
                     if ln.strip()]
            st.info(f"**{stage['name']}** — running for {minutes:.0f} min"
                    + (f". Last output: `{lines[-1].strip()[:120]}`" if lines else ""))
        if state == "interrupted":
            st.warning("The runner process is gone but the job never finished (the "
                       "machine slept, or the process was ended). Nothing was cleaned "
                       "up; read the log, then start it again.")
        st.dataframe(pd.DataFrame([{
            "": dot(s["state"]), "stage": s["name"], "state": s["state"],
            "minutes": s["minutes"], "exit": s["exit_code"]} for s in job["stages"]]),
            hide_index=True, width="stretch")
        st.code(job.get("log") or "(no output yet)", language="text")
        if live and state not in ("starting", "running"):
            st.rerun(scope="app")

    watch()
