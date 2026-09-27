"""The Phase 2 panel: reasons and notices for a run whose decisions are ready.

Presentation only. Phase 2 itself is creditsurv.phase2.explain_run -- started in the
background with phase2.launch_background (the runner large uploads use), or for one
applicant on demand -- and this panel only reads the progress it writes. Closing the
tab does not stop it; opening the run again picks the progress up.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from creditsurv import phase2
from creditsurv.batch import BatchError


def _fmt_seconds(seconds) -> str:
    if seconds is None:
        return "working it out"
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def render_phase2(run_dir, cfg, *, summary: dict, launcher=phase2.launch_background,
                  explain_one=None) -> None:
    """Start, choose, watch, or explain one applicant.

    ``explain_one(row_id)`` runs Phase 2 for one applicant in this process with the
    already-loaded model; None hides the button (a page that cannot score).
    """
    run_dir = Path(run_dir)
    n_rejected = int(summary.get("n_rejected", 0) or 0)
    if not n_rejected:
        return
    st.subheader("Reasons and notices")
    prog = phase2.read_progress(run_dir)
    state = prog.get("state", "")
    started_key = f"phase2_started_{run_dir.name}"
    limit = int(summary.get("explain_confirm_above")
                or getattr(cfg.decision, "explain_confirm_above", 1000))

    # A run small enough to explain without asking starts on its own, once.
    if state == "not started" and not st.session_state.get(started_key):
        _start(run_dir, "all", None, launcher, started_key)

    if state == "awaiting choice":
        _choice(run_dir, cfg, n_rejected, limit, launcher, started_key)
    elif state in ("running", "not started") or (
            st.session_state.get(started_key) and state not in (
                "completed", "skipped", "failed_checks", "stopped", "failed")):
        _watch(run_dir, n_rejected)
    elif state in ("stopped", "failed"):
        st.warning(f"**Phase 2 stopped before it finished** ({prog.get('done', 0):,} of "
                   f"{prog.get('target', n_rejected):,} explained"
                   + (f"; {prog['error']}" if prog.get("error") else "") + "). What "
                   "was finished is kept; resuming continues from the next applicant, "
                   "with the same reasons an uninterrupted run gives.")
        if st.button("Resume Phase 2", key=f"resume_{run_dir.name}"):
            mode = (prog.get("mode") or "all").split(" ")[0]
            _start(run_dir, mode if mode in phase2.MODES else "all",
                   int(prog["mode"].split()[-1]) if mode == "sample" else None,
                   launcher, started_key, rerun=True)
    elif state == "completed":
        st.success(f"Phase 2 finished: {summary.get('n_explained', 0):,} of "
                   f"{n_rejected:,} rejected applicants explained, "
                   f"{summary.get('n_notices', 0):,} notices"
                   + (f" ({summary['phase2_mode']})" if summary.get("phase2_mode") not in
                      (None, "", "all") else "")
                   + (f", in {_fmt_seconds(summary.get('phase2_seconds'))}."
                      if summary.get("phase2_seconds") else "."))
    elif state == "skipped":
        st.warning("Phase 2 was skipped: no reasons and no notices. The run is stamped "
                   "not for lending decisions.")
    elif state == "failed_checks":
        st.error("Phase 2 failed its checks; the notices were withheld. See "
                 "RUN_FAILED_CHECKS.txt in the run folder.")

    if explain_one is not None and int(summary.get("n_reasons_pending", 0) or 0):
        _on_demand(run_dir, explain_one)


def _start(run_dir, mode, sample_n, launcher, started_key, rerun=True) -> None:
    try:
        launcher(run_dir, mode, sample_n)
    except Exception as exc:                        # LockHeld, a refused run...
        st.error(f"**Phase 2 could not be started.** {exc}")
        return
    st.session_state[started_key] = True
    if rerun:
        st.rerun()


def _choice(run_dir, cfg, n, limit, launcher, started_key) -> None:
    st.warning(f"**{n:,} rejected applicants** is more than the "
               f"{limit:,} explained without asking "
               f"(`decision.explain_confirm_above`). Explaining all of them would take "
               f"about **{_fmt_seconds(phase2.estimate_seconds(n, cfg))}**. The "
               f"decisions above are final either way.")
    with st.form(f"phase2_choice_{run_dir.name}", border=True):
        choice = st.radio("Phase 2", ["Explain all", "Explain a random sample",
                                      "Skip"], index=1, key=f"p2_choice_{run_dir.name}",
                          captions=[
                              f"every applicant gets reasons and a notice (about "
                              f"{_fmt_seconds(phase2.estimate_seconds(n, cfg))})",
                              "a seeded random sample; the rest are marked, and the run "
                              "is stamped not for lending decisions",
                              "no reasons, no notices; stamped not for lending decisions"])
        sample_n = st.number_input("Sample size", 1, n, min(500, n), 50,
                                   key=f"p2_n_{run_dir.name}")
        st.caption(f"A sample of {int(sample_n):,} takes about "
                   f"{_fmt_seconds(phase2.estimate_seconds(int(sample_n), cfg))}.")
        go = st.form_submit_button("Start Phase 2", type="primary")
    if go:
        mode = {"Explain all": "all", "Explain a random sample": "sample",
                "Skip": "skip"}[choice]
        _start(run_dir, mode, int(sample_n) if mode == "sample" else None, launcher,
               started_key)


def _watch(run_dir, n_rejected) -> None:
    st.caption("Phase 2 runs as its own process: leave this page and come back, and "
               "it picks up here. Reasons are written into rejected_applicants.csv "
               "and the notice zip as they are generated.")

    @st.fragment(run_every=2)
    def watch():
        prog = phase2.read_progress(run_dir)
        state = prog.get("state", "")
        target = int(prog.get("target") or n_rejected)
        done = int(prog.get("done") or 0)
        if state in ("not started", ""):
            st.markdown("🔵 **Starting Phase 2...** (loading the model in the background "
                        "process)")
        else:
            st.progress(min(done / max(target, 1), 1.0),
                        text=f"{done:,} of {target:,} rejected applicants explained")
            rate = prog.get("rate_per_min")
            st.markdown(
                f"🔵 **{state}**"
                + (f" · {rate:,.1f} per minute" if rate else "")
                + (f" · about {_fmt_seconds(prog.get('eta_seconds'))} left"
                   if state == "running" else ""))
            recent = phase2.recent_results(run_dir, 5)
            if recent:
                st.dataframe(pd.DataFrame([{
                    "row": r["row_id"], "applicant": r["applicant_id"],
                    "first reason": (r["reasons"][0]["reason"] if r["reasons"] else
                                     r["status"])} for r in recent]),
                    hide_index=True, width="stretch")
        if state not in ("running", "not started", ""):
            st.rerun(scope="app")                  # finished: redraw with the result

    watch()


def _on_demand(run_dir, explain_one) -> None:
    with st.expander("Explain one applicant now (a few seconds)"):
        pending = phase2.pending_rows(run_dir, 500)
        if pending.empty:
            st.caption("No row is waiting for reasons.")
            return
        risk = next((c for c in pending.columns if c.startswith("pd_") and c != "pd_12m"),
                    "pd_12m")
        options = pending["row_id"].astype(int).tolist()
        labels = dict(zip(options, [f"{a} (row {r}, risk {p})" for a, r, p in
                                    zip(pending["applicant_id"], pending["row_id"],
                                        pending[risk])]))
        row = st.selectbox("Rejected applicant with reasons pending", options,
                           format_func=lambda r: labels[r], key=f"od_{run_dir.name}")
        if st.button("Explain this applicant", key=f"od_go_{run_dir.name}"):
            try:
                with st.spinner("Explaining..."):
                    explain_one(int(row))
            except BatchError as exc:
                st.error(f"**{exc.message}**" + (f"\n\n{exc.fix}" if exc.fix else ""))
                return
            st.session_state[f"od_done_{run_dir.name}"] = int(row)
        shown = st.session_state.get(f"od_done_{run_dir.name}")
        if shown is not None:
            res, text = phase2.result_for(run_dir, shown)
            if res is not None:
                st.markdown(f"**Applicant {res['applicant_id']}** — "
                            + ("; ".join(r["reason"] for r in res["reasons"])
                               or res["status"]))
                if text:
                    st.code(text, language="text")
                    st.caption("This notice is in adverse_action_notices.zip now. Its "
                               "internal record is in internal/, never in the notice.")
