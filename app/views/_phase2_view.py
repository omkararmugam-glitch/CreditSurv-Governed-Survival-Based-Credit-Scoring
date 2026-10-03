"""The Phase 2 panel: reasons and notices for a run whose decisions are ready.

Presentation only. Phase 2 is started through the API (POST /runs/{id}/phase2,
which launches it in the background runner) and followed through its status; one
applicant can be explained on demand (POST /runs/{id}/explain-rows). Closing the
tab does not stop it; opening the run again picks the progress up.
"""

from __future__ import annotations

import streamlit as st

from _client import ApiError, api, frame
from _common import fmt_seconds

MODES = ("all", "sample", "skip")


def render_phase2(run_id: str, meta: dict, *, summary: dict) -> None:
    """Start, choose, watch, or explain one applicant."""
    n_rejected = int(summary.get("n_rejected", 0) or 0)
    if not n_rejected:
        return
    st.subheader("Reasons and notices")
    prog = api().get(f"/runs/{run_id}/status")["phase2"]
    state = prog.get("state") or ""
    started_key = f"phase2_started_{run_id}"
    d = meta["decision"]
    limit = int(summary.get("explain_confirm_above") or d["explain_confirm_above"])

    if state == "awaiting choice":
        _choice(run_id, d, n_rejected, limit, started_key)
    elif state in ("running", "not started") or (
            st.session_state.get(started_key) and state not in (
                "completed", "skipped", "failed_checks", "stopped", "failed")):
        if state == "not started" and not st.session_state.get(started_key):
            # Small enough not to ask, and not started by the API on its own
            # (a run scored with phase2=defer): start it once, as before.
            _start(run_id, "all", None, started_key)
        _watch(run_id, n_rejected)
    elif state in ("stopped", "failed"):
        st.warning(f"**Phase 2 stopped before it finished** ({prog.get('done') or 0:,} "
                   f"of {prog.get('target') or n_rejected:,} explained"
                   + (f"; {prog['error']}" if prog.get("error") else "") + "). What "
                   "was finished is kept; resuming continues from the next applicant, "
                   "with the same reasons an uninterrupted run gives.")
        if st.button("Resume Phase 2", key=f"resume_{run_id}"):
            mode = (prog.get("mode") or "all").split(" ")[0]
            _start(run_id, mode if mode in MODES else "all",
                   int(prog["mode"].split()[-1]) if mode == "sample" else None,
                   started_key)
    elif state == "completed":
        st.success(f"Phase 2 finished: {summary.get('n_explained', 0):,} of "
                   f"{n_rejected:,} rejected applicants explained, "
                   f"{summary.get('n_notices', 0):,} notices"
                   + (f" ({summary['phase2_mode']})" if summary.get("phase2_mode") not in
                      (None, "", "all") else "")
                   + (f", in {fmt_seconds(summary.get('phase2_seconds'))}."
                      if summary.get("phase2_seconds") else "."))
    elif state == "skipped":
        st.warning("Phase 2 was skipped: no reasons and no notices. The run is stamped "
                   "not for lending decisions.")
    elif state == "failed_checks":
        st.error("Phase 2 failed its checks; the notices were withheld. See "
                 "RUN_FAILED_CHECKS.txt in the run folder.")

    if int(summary.get("n_reasons_pending", 0) or 0):
        _on_demand(run_id)


def _start(run_id: str, mode: str, sample_n, started_key: str) -> None:
    try:
        api().post(f"/runs/{run_id}/phase2", {"mode": mode, "sample_n": sample_n})
    except ApiError as exc:
        if exc.status != 409:                     # 409: already running -- fine
            st.error(f"**Phase 2 could not be started.** {exc.message}")
            return
    st.session_state[started_key] = True
    st.rerun()


def _choice(run_id, d, n, limit, started_key) -> None:
    each = float(d.get("explain_seconds_each") or 2.1)
    st.warning(f"**{n:,} rejected applicants** is more than the {limit:,} explained "
               f"without asking (`decision.explain_confirm_above`). Explaining all of "
               f"them would take about **{fmt_seconds(n * each)}**. The decisions "
               f"above are final either way.")
    with st.form(f"phase2_choice_{run_id}", border=True):
        choice = st.radio("Phase 2", ["Explain all", "Explain a random sample", "Skip"],
                          index=1, key=f"p2_choice_{run_id}", captions=[
                              f"every applicant gets reasons and a notice (about "
                              f"{fmt_seconds(n * each)})",
                              "a seeded random sample; the rest are marked, and the run "
                              "is stamped not for lending decisions",
                              "no reasons, no notices; stamped not for lending decisions"])
        sample_n = st.number_input("Sample size", 1, n, min(500, n), 50,
                                   key=f"p2_n_{run_id}")
        st.caption(f"A sample of {int(sample_n):,} takes about "
                   f"{fmt_seconds(int(sample_n) * each)}.")
        go = st.form_submit_button("Start Phase 2", type="primary")
    if go:
        mode = {"Explain all": "all", "Explain a random sample": "sample",
                "Skip": "skip"}[choice]
        _start(run_id, mode, int(sample_n) if mode == "sample" else None, started_key)


def _watch(run_id: str, n_rejected: int) -> None:
    st.caption("Phase 2 runs as its own process: leave this page and come back, and "
               "it picks up here. Reasons are written into rejected_applicants.csv "
               "and the notice zip as they are generated.")

    @st.fragment(run_every=2)
    def watch():
        prog = api().get(f"/runs/{run_id}/status")["phase2"]
        state = prog.get("state") or ""
        target = int(prog.get("target") or n_rejected)
        done = int(prog.get("done") or 0)
        if state in ("not started", ""):
            st.markdown("🔵 **Starting Phase 2...** (loading the model in the background "
                        "process)")
        else:
            st.progress(min(done / max(target, 1), 1.0),
                        text=f"{done:,} of {target:,} rejected applicants explained")
            rate = prog.get("rate_per_min")
            st.markdown(f"🔵 **{state}**"
                        + (f" · {rate:,.1f} per minute" if rate else "")
                        + (f" · about {fmt_seconds(prog.get('eta_seconds'))} left"
                           if state == "running" else ""))
            if prog.get("recent"):
                import pandas as pd
                st.dataframe(pd.DataFrame([{
                    "row": r["row_id"], "applicant": r["applicant_id"],
                    "first reason": r["first_reason"]} for r in prog["recent"]]),
                    hide_index=True, width="stretch")
        if state not in ("running", "not started", ""):
            st.rerun(scope="app")

    watch()


def _on_demand(run_id: str) -> None:
    with st.expander("Explain one applicant now (a few seconds)"):
        try:
            pending = frame(api().get(f"/runs/{run_id}/pending-rows", limit=500)["rows"])
        except ApiError as exc:
            exc.show()
            return
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
                           format_func=lambda r: labels[r], key=f"od_{run_id}")
        if st.button("Explain this applicant", key=f"od_go_{run_id}"):
            try:
                with st.spinner("Explaining..."):
                    out = api().post(f"/runs/{run_id}/explain-rows",
                                     {"row_ids": [int(row)]})
            except ApiError as exc:
                exc.show()
                return
            st.session_state[f"od_done_{run_id}"] = out["results"][0]
        shown = st.session_state.get(f"od_done_{run_id}")
        if shown and shown.get("result"):
            res = shown["result"]
            st.markdown(f"**Applicant {res['applicant_id']}** — "
                        + ("; ".join(r["reason"] for r in res["reasons"])
                           or res["status"]))
            if shown.get("notice_text"):
                st.code(shown["notice_text"], language="text")
                st.caption("This notice is in adverse_action_notices.zip now. Its "
                           "internal record is in internal/, never in the notice.")
