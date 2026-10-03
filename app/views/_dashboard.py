"""The result view for one scoring run, drawn from the API.

Separated from the pages so a test can draw every panel and every chart of a real
run (tests/test_app.py). Read-only: everything shown is what the run wrote, as the
API returns it (GET /runs/{id}); nothing is recomputed here.

Drawn in the order it becomes useful: :func:`render_headline` (stamps, checks,
counts, approval rate and the first rows of decisions) needs only the run's summary
and its first rows, so it appears the moment Phase 1 finishes; the Phase 2 panel
goes between; :func:`render_details` (charts, profile, drift, downloads) follows.
Downloads fetch nothing until clicked.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from _client import ApiError, api, run_view
from _common import (ACCENT, AMBER, APPROVE, MUTED, REJECT, badge, fmt_seconds,
                     kpi_row, status_box)

DRIFT_KIND = {"stable": "done", "moderate": "moderate", "large": "failed",
              "unknown": "moderate", "insufficient": "pending"}

# Any chart where a small group could otherwise look as alarming as a large one
# carries its own counts. A 100% rejection rate over three applicants and over
# three thousand are not the same finding, and a bar cannot tell them apart.
COUNT_RULE = ("Counts are shown beside each bar: a share computed over a handful "
              "of applicants is not comparable with one over thousands.")


def load(run_id: str) -> "object":
    """The run, as the result view needs it."""
    return run_view(api().get(f"/runs/{run_id}"))


def _bar_with_counts(data: pd.DataFrame, *, y: str, x: str, colour, title=None,
                     label_format: str = ",.0f", sort: str = "-x"):
    """A horizontal bar chart with the count printed at the end of each bar."""
    base = alt.Chart(data)
    encode = dict(y=alt.Y(f"{y}:N", sort=sort, title=None),
                  x=alt.X(f"{x}:Q", title=title))
    bars = base.mark_bar(**({"color": colour} if isinstance(colour, str) else {})).encode(
        **encode, **({} if isinstance(colour, str) else {"color": colour}),
        tooltip=list(data.columns))
    labels = base.mark_text(align="left", dx=3, color=MUTED, fontSize=10).encode(
        **encode, text=alt.Text(f"{x}:Q", format=label_format))
    return bars + labels


# The chart surface, for the 1px ring that keeps two stacked fills apart. The
# approve/reject pair is a green and a red, which no choice of green separates for
# deuteranopia (measured: the best ~6.5 of a target 8 in OKLab dE x100), so every
# chart that puts them together carries a second encoding -- a legend, a ring
# between the fills, and a printed count -- rather than relying on the hues.
SURFACE = "#fcfcfb"
DECISION_SCALE = alt.Scale(domain=["approve", "reject"], range=[APPROVE, REJECT])


def risk_chart(risk: pd.DataFrame, threshold: float):
    """The predicted-risk histogram with the decision threshold marked."""
    hist = alt.Chart(risk).mark_bar().encode(
        x=alt.X("bin_start:Q", title="default probability",
                scale=alt.Scale(domain=[0, 1])),
        x2="bin_end:Q",
        y=alt.Y("applicants:Q", title="applicants"),
        color=alt.Color("decision:N", scale=DECISION_SCALE, title=None),
        tooltip=["bin_start", "bin_end", "applicants"])
    rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(
        color=MUTED, strokeDash=[6, 4]).encode(x="t:Q")
    return hist + rule


def decision_share_chart(groups: pd.DataFrame, *, order=None):
    """Approve/reject share per level, with each level's total printed beside it.

    ``groups`` is ``group, decision, applicants``. The share is what the eye
    compares and the total is what makes it mean something: 100% rejected over
    three applicants and over three thousand are not the same finding.

    ``order`` is the level order down the axis -- a list for a dimension that has
    one (income bands belong in band order, not in size order), and otherwise the
    levels by how many applicants they hold. Never Altair's ``sort="-x"``: the x
    channel is stacked to 100%, so every bar ends at 1.0 and sorting by it puts the
    levels in an order that means nothing.
    """
    groups = groups.assign(group=groups["group"].astype("string"))
    # observed=True because the income band arrives as an ordered Categorical: with
    # the default, groupby adds a row for every band the file had nobody in, and the
    # label layer then prints a 0 beside a bar that is not there.
    totals = groups.groupby("group", as_index=False, observed=True)["applicants"].sum()
    totals = totals.rename(columns={"applicants": "total"})
    if order is None:
        order = totals.sort_values("total", ascending=False)["group"].tolist()
    order = [str(g) for g in order]
    shares = groups.merge(totals, on="group")
    axis = alt.Y("group:N", sort=order, title=None)
    bars = alt.Chart(shares).mark_bar(stroke=SURFACE, strokeWidth=1).encode(
        y=axis,
        x=alt.X("applicants:Q", stack="normalize", title="share",
                axis=alt.Axis(format="%")),
        color=alt.Color("decision:N", scale=DECISION_SCALE, title=None),
        tooltip=["group", "decision", "applicants", "total"])
    labels = alt.Chart(totals).mark_text(
        align="left", dx=4, color=MUTED, fontSize=10).encode(
        y=axis, x=alt.value(0), text=alt.Text("total:Q", format=",.0f"))
    return bars + labels


def reasons_chart(reasons: pd.DataFrame):
    """The most-cited rejection reasons, longest bar first."""
    return _bar_with_counts(reasons, y="reason", x="times cited", colour=REJECT)


def render(view, meta: dict, *, middle=None) -> None:
    """Draw the whole result view: the headline first, then ``middle`` (the
    Phase 2 panel), then the details."""
    render_headline(view, meta)
    if middle is not None:
        middle()
    render_details(view, meta)


def render_failed(view) -> None:
    """A run that stopped before it finished: why, and the checks that say so.
    Its decisions may not be used, so none are shown."""
    err = view.error or {}
    if view.state == "failed checks":
        failed = view.checks[view.checks["status"] == "FAIL"] if not view.checks.empty \
            else view.checks
        st.error(f"**Run checks: {len(failed)} of {len(view.checks)} FAILED.** The run is "
                 f"not marked finished and no notices will be produced for it. Nothing "
                 f"from it may be used.", icon=":material/gpp_bad:")
        for _, row in failed.iterrows():
            st.markdown(f"- **{row['check']}** — {row['detail']}")
        if err.get("fix"):
            st.caption(err["fix"])
        st.dataframe(view.checks, hide_index=True, width="stretch")
        if view.cleaning_sentences or not view.cleaning_table.empty:
            with st.expander("Cleaning report"):
                st.dataframe(view.cleaning_table, hide_index=True, width="stretch")
    else:
        st.error(f"**{err.get('message') or 'This run stopped before its decisions '
                                           'were written.'}**"
                 + (f"\n\n{err['fix']}" if err.get("fix") else ""),
                 icon=":material/error:")
        if err.get("detail") and err.get("detail") != err.get("message"):
            with st.expander("Details"):
                st.code(err["detail"], language="text")


def render_headline(view, meta: dict) -> None:
    """What is known the moment Phase 1 finishes."""
    s = view.summary
    text = meta.get("text", {})
    _phase2_banner(view, text)

    # Said first and loudest: a run that may not be used to decide anything.
    if s.get("for_lending_decisions") is False:
        st.error(f"**NOT FOR LENDING DECISIONS.** "
                 f"{s.get('not_for_lending_reasons', '')}. Every output file and every "
                 f"notice of this run carries the same stamp.", icon=":material/block:")

    p1 = float(s.get("phase1_seconds") or 0.0)
    st.success(f"Decisions ready in {p1:.1f}s "
               f"({p1 / max(s['n_rows'], 1) * 1000:.2f}s per 1,000 applicants). "
               f"Run `{view.run_id}`.")
    pending_n = int(s.get("n_reasons_pending", 0) or 0)
    if pending_n:
        st.info(f"**Reasons pending for {pending_n:,} of {s['n_rejected']:,} rejected "
                f"applicants.** Their rows say \"{text.get('reasons_pending')}\" until "
                f"Phase 2 reaches them; decisions are final either way.")

    checks = view.checks
    if not checks.empty:
        n_fail = int((checks["status"] == "FAIL").sum())
        n_over = int((checks["status"] == "OVERRIDDEN").sum())
        line = (f"**Run checks: {int((checks['status'] == 'PASS').sum())} of "
                f"{len(checks)} passed**"
                + (f", {n_over} overridden" if n_over else "")
                + (f", {n_fail} FAILED" if n_fail else "")
                + " — verified from the files this run wrote (Checks tab).")
        status_box("failed" if n_fail else "overridden" if n_over else "done", line)
    else:
        st.warning("**Run checks: not recorded.** This run predates the post-run "
                   "checks; treat its outputs as unverified.")

    for w in view.warnings:
        st.warning(w)
    if s.get("required_rule") == "provisional":
        st.warning("This model has no ablation table, so the required columns were "
                   "the provisional hand-picked list rather than measured costs: "
                   + str(s.get("required_features", "")))
    if s.get("degraded_coverage"):
        st.error(f"**Degraded run:** only {s['features_present']} of "
                 f"{s['features_expected']} model features were in this file "
                 f"({s['feature_coverage']:.0%}). The scores are weaker than the "
                 f"model's published performance.")
    _render_filled_features(s)
    if int(s.get("n_duplicates_removed") or 0):
        st.info(f"**{int(s['n_duplicates_removed']):,} duplicate applicant row(s) were "
                f"removed** ({s.get('duplicates_removed_by_rule', '')}): "
                f"{int(s.get('n_rows_in_file') or 0):,} rows in the file, "
                f"{s['n_rows']:,} applicants scored, each decided once.")

    if view.drift is not None:
        label = ("NOT ASSESSED" if view.drift.status == "insufficient"
                 else view.drift.colour.upper())
        status_box(DRIFT_KIND.get(view.drift.status, "pending"),
                   f"**Data drift: {label}** — {view.drift.headline}")

    missing_reasons = int(s.get("n_rejected_without_reasons", 0) or 0)
    if missing_reasons:
        st.error(f"**{missing_reasons:,} of {s['n_rejected']:,} rejected applicants "
                 f"have no adverse-action reasons.** Those rows are marked "
                 f"\"{text.get('cap_note')}\" in every output file. This happens only "
                 f"when a run is stopped before it finishes; re-running resumes where "
                 f"it stopped.")
    pending = int(s.get("n_pending_manual_review", 0) or 0)
    if pending:
        st.warning(f"**{pending:,} rejected applicant(s) are pending manual review:** "
                   f"no adverse factor could be stated in Regulation B wording, so no "
                   f"notice was issued. Their drivers are in "
                   f"`internal/internal_review_flags.csv`.")

    status = s.get("model_registry_status", "not recorded")
    approved = bool(s.get("model_approved"))
    st.info(f"Decision rule: reject at a {s['horizon_months']}-month default "
            f"probability of **{s['threshold']:.0%}** or higher. This is a policy "
            f"choice, not a model output"
            + ("" if s.get("threshold_is_published", True) else
               f" — and **not** the published {s.get('published_threshold', 0):.0%}")
            + f". Model `{s['model_tag']}` ({s['model']}), registry status "
            f"**{status}**" + (" (approved)." if approved else
                               " — **not approved for lending decisions**."))

    flagged = s.get("n_fair_lending_flagged")
    if flagged is not None:
        share = float(s.get("fair_lending_flag_share", 0.0) or 0.0)
        limit = float(s.get("fair_lending_review_share", 0.05) or 0.05)
        msg = (f"{int(flagged):,} of {int(s.get('n_explained', 0)):,} explained "
               f"rejections ({share:.1%}) had a non-disclosable feature among the "
               f"strongest adverse drivers"
               + (f" ({s['fair_lending_flag_features']})"
                  if s.get("fair_lending_flag_features") else "")
               + f"; {int(s.get('n_top_driver_not_disclosable', 0)):,} as the single "
                 f"strongest. Review threshold: {limit:.0%}.")
        if s.get("fair_lending_review_required"):
            st.error(f"**Fair-lending review required.** {msg} The stated reasons "
                     f"exclude these features, so the notices cannot show it: the "
                     f"detail is in `internal/internal_review_flags.csv`.",
                     icon=":material/balance:")
        else:
            st.caption(f"Fair-lending monitor: {msg}")

    mean_pd = s.get(f"mean_pd_{s['horizon_months']}m", float("nan"))
    kpi_row([("Applicants", f"{s['n_rows']:,}"),
             ("Approved", f"{s['n_approved']:,}"),
             ("Rejected", f"{s['n_rejected']:,}"),
             ("Approval rate", f"{s['approval_rate']:.1%}"),
             (f"Average {s['horizon_months']}m risk", f"{mean_pd:.1%}")])
    if s["n_rejected"]:
        st.caption(
            f"Reasons generated for {s['n_explained']:,} of {s['n_rejected']:,} "
            f"rejected applicants by **{s.get('explainer', 'survshap')}**"
            + (f" — {missing_reasons:,} without reasons." if missing_reasons else
               f" — {pending_n:,} pending." if pending_n else " (all of them).")
            + (f"  Profiled and drift-checked on a random sample of "
               f"{s['profiled_rows']:,} of {s['n_rows']:,} rows."
               if s.get("profiled_rows", 0) < s["n_rows"] else ""))
    if s.get("features_derived"):
        st.caption(f"Derived from other columns: {s['features_derived']}.")
    if s.get("features_missing_optional"):
        st.caption(f"Scored without (optional, measured cost in the Data profile "
                   f"tab): {s['features_missing_optional']}.")

    st.markdown(f"**First decisions** (first {len(view.preview)} of "
                f"{s['n_rows']:,} rows; every row is in the downloads)")
    st.dataframe(view.preview, hide_index=True, width="stretch")


NEWLINE = chr(10)


def _render_filled_features(s) -> None:
    """Name every feature whose value was supplied rather than read.

    Its own box, and named one by one, because the alternative was what hid this:
    a coverage percentage and a concordance cost, neither of which can say which
    applicant attribute was invented or which way it moved the risk (FINDINGS 7o).
    """
    filled = str(s.get("features_filled_from_training") or "")
    if filled:
        lines = "".join(
            f"{NEWLINE}- **{part.split(' = ')[0].strip()}** — absent from the file, "
            f"scored with the training value `{part.split(' = ')[-1].strip()}`"
            for part in filled.split(";") if part.strip())
        st.error(
            f"**{int(s.get('n_features_filled_from_training') or 0)} feature(s) were "
            f"filled with a training value, not read from this file.** The model "
            f"never saw them missing in training, so leaving them absent would have "
            f"rested on an unlearned default rather than on degradation."
            + lines
            + f"{NEWLINE}{NEWLINE}Do not treat these applicants' results as "
              f"fully reliable for "
              "these attributes. Supplying the real columns is the fix; the "
              "substitution is recorded per feature in the run summary.")
    unfilled = [c.strip() for c in
                str(s.get("features_unlearned_missing") or "").split(";") if c.strip()]
    unfilled = [c for c in unfilled if c not in filled]
    if unfilled and str(s.get("unlearned_missing_action") or "fill") != "block":
        st.error(
            f"**{len(unfilled)} feature(s) absent from this file have no learned "
            f"route for being missing and no training value to stand in:** "
            f"{', '.join(unfilled)}. These rows rest on the model's default split "
            f"direction for them, which is an unmeasured constant.")


def render_details(view, meta: dict) -> None:
    """Charts, reasons, profile, drift and downloads: drawn after the headline."""
    s = view.summary
    agg = view.aggregates

    left, right = st.columns(2)
    with left:
        st.markdown("**Predicted risk**")
        if agg is not None and not agg.risk.empty:
            st.altair_chart(risk_chart(agg.risk, s["threshold"]),
                            use_container_width=True)
    with right:
        by = agg.groups if agg is not None else pd.DataFrame()
        if not by.empty:
            st.markdown(f"**Decisions by loan {agg.group_column}**")
            st.altair_chart(decision_share_chart(by), use_container_width=True)
            st.caption(f"Totals per {agg.group_column} shown beside each bar. "
                       + COUNT_RULE)

    top = agg.reasons.head(8) if agg is not None else pd.DataFrame()
    if not top.empty:
        st.markdown("**Most common rejection reasons** (Regulation B wording)")
        st.altair_chart(reasons_chart(top), use_container_width=True)

    tab_checks, tab_clean, tab_profile, tab_drift = st.tabs(
        ["Checks", "Cleaning report", "Data profile", "Drift check"])

    with tab_checks:
        if view.checks.empty:
            st.caption("No checks recorded for this run.")
        else:
            st.caption("Computed after every file was written, by reading the files "
                       "back. A FAIL on a blocking check stops the run before it is "
                       "marked finished; OVERRIDDEN means the operator chose it and "
                       "the run is stamped not for lending decisions.")
            st.dataframe(view.checks, hide_index=True, width="stretch")

    with tab_clean:
        st.caption(
            f"Cleaning policy `{s.get('cleaning_policy_version')}`, using values "
            f"fitted on {int(s.get('cleaning_values_fitted_rows', 0) or 0):,} training "
            f"rows" + ("" if s.get("cleaning_values_from_model_bundle") else
                       " (from the model's sidecar file)") + ".")
        for line in view.cleaning_sentences:
            st.markdown(f"- {line}")
        st.dataframe(view.cleaning_table, hide_index=True, width="stretch")

    with tab_profile:
        prof = view.profile
        o = prof.get("overview", {})
        kpi_row([("Rows", f"{o.get('rows', 0):,}"),
                 ("Columns", f"{o.get('columns', 0):,}"),
                 ("Numeric", f"{o.get('numeric_columns', 0):,}"),
                 ("Categorical", f"{o.get('categorical_columns', 0):,}")])
        if s.get("profiled_rows", 0) < s["n_rows"]:
            st.caption(f"Computed on a random sample of {s['profiled_rows']:,} of "
                       f"{s['n_rows']:,} rows, drawn across the whole file.")
        if s.get("optional_missing_cost"):
            st.warning(f"Measured accuracy cost of the optional features absent "
                       f"from this file: {s['optional_missing_cost']}")
            st.caption("That cost is concordance, which measures ranking. It does "
                       "not measure how far an absence moves the *level* of "
                       "predicted risk, which is what the unlearned-default rule "
                       "above covers (FINDINGS 7o).")
        if s.get("unlearned_rule_note"):
            st.caption(str(s["unlearned_rule_note"]))
        miss = prof.get("missing", pd.DataFrame())
        if not miss.empty:
            st.markdown("**Missing values by column**")
            affected = miss[miss["missing_share"] > 0]
            if affected.empty:
                st.caption("No missing values in any model feature.")
            else:
                st.altair_chart(_bar_with_counts(affected.head(25), y="column",
                                                 x="missing", colour=ACCENT,
                                                 title="rows missing"),
                                use_container_width=True)
                st.caption(COUNT_RULE)
            st.dataframe(miss, hide_index=True, width="stretch", height=240)
        for label, key, height in (("**Numeric summary**", "numeric", 260),
                                   ("**Category counts**", "categorical", 260),
                                   ("**Outliers** (counts only; nothing was "
                                    "altered)", "outliers", 240)):
            frame = prof.get(key, pd.DataFrame())
            if not frame.empty:
                st.markdown(label)
                st.dataframe(frame, hide_index=True, width="stretch", height=height)

    with tab_drift:
        dr = view.drift
        if dr is None:
            st.caption("No drift check for this run.")
        else:
            st.markdown(f"**{dr.headline}**")
            if dr.status == "insufficient":
                st.caption("Nothing is wrong with the file; there is simply not "
                           "enough of it to tell whether it resembles the training "
                           "population.")
            st.caption("Numeric features: population stability index against the "
                       "training distribution (below 0.10 stable, 0.10-0.25 "
                       "moderate, above 0.25 large). Categorical: total variation "
                       "distance on the same bands.")
            shifted = dr.table[dr.table["status"].isin(["large", "moderate", "unknown"])]
            if not shifted.empty:
                st.altair_chart(alt.Chart(shifted.head(20)).mark_bar().encode(
                    y=alt.Y("feature:N", sort="-x", title=None),
                    x=alt.X("score:Q", title="PSI / TVD"),
                    color=alt.Color("status:N", scale=alt.Scale(
                        domain=["large", "moderate", "unknown", "stable"],
                        range=[REJECT, AMBER, MUTED, APPROVE]), title=None),
                    tooltip=list(dr.table.columns)), use_container_width=True)
            st.dataframe(dr.table, hide_index=True, width="stretch", height=320)

    downloads(view.run_id, s, meta)

    with st.expander("Run details"):
        st.json(s, expanded=False)
        st.caption("Also in the run folder as provenance.json, with SHA-256 hashes of "
                   "the upload, the model and every output file.")


# ------------------------------------------------------------- Phase 2 live --

def _live(snap: dict):
    """How often a view redraws itself: while something can change without the
    operator doing anything, and not at all otherwise."""
    return 3 if snap.get("state") in ("running", "not started", "finishing") else None


def _snapshot(run_id: str) -> dict:
    return api().get(f"/runs/{run_id}/status")["phase2"]["snapshot"]


def _phase2_banner(view, text: dict) -> None:
    """At the top of the results for as long as the files are a partial snapshot,
    with live counts; gone once Phase 2 is done."""
    snap = view.status.get("phase2", {}).get("snapshot") or {}
    if not snap.get("partial"):
        return

    @st.fragment(run_every=_live(snap))
    def banner():
        now = _snapshot(view.run_id)
        if not now.get("partial"):
            st.rerun(scope="app")
        counts = f"{now['done']:,} of {now['target']:,} explained"
        if now["state"] == "running":
            eta = now.get("eta_seconds")
            head = (f"Phase 2 running: {counts}, "
                    + (f"~{fmt_seconds(eta)} remaining" if eta is not None
                       else "estimating time remaining") + ".")
        elif now["state"] == "not started":
            head = f"Phase 2 starting: {counts}."
        else:
            head = f"Phase 2 {now['state']}: {counts}."
        if now["state"] in ("running", "not started", "awaiting choice", "finishing"):
            st.warning(f"**{head}** Downloads are disabled until it finishes: "
                       f"{now['pending']:,} rejected applicants still say "
                       f"\"{text.get('reasons_pending')}\" and have no notice yet. They "
                       f"switch on by themselves once this banner is gone.",
                       icon=":material/hourglass_top:")
            return
        st.warning(f"**{head}** Every download on this page is a **partial snapshot**: "
                   f"{now['pending']:,} rejected applicants still say "
                   f"\"{text.get('reasons_pending')}\" and have no notice yet. Decisions "
                   f"are final.", icon=":material/hourglass_top:")

    banner()


def downloads(run_id: str, s: dict, meta: dict) -> None:
    """The download buttons, redrawn while Phase 2 runs so their names always say
    what they hold. The API holds them back while Phase 2 is still writing."""
    listing = api().get(f"/runs/{run_id}/files")

    @st.fragment(run_every=_live(listing["snapshot"]) if listing["snapshot"].get("partial")
                 else None)
    def buttons():
        now = api().get(f"/runs/{run_id}/files")
        if listing["snapshot"].get("partial") and not now["snapshot"].get("partial"):
            st.rerun(scope="app")
        _download_buttons(run_id, s, now, meta)

    buttons()


SHOWN = ("scored_applicants.csv", "approved_applicants.csv", "rejected_applicants.csv",
         "adverse_action_notices.zip", "internal/internal_review_flags.csv",
         "validation_checks.csv", "cleaning_report.csv", "data_drift.csv",
         "run_summary.csv")


def _download_buttons(run_id: str, s: dict, listing: dict, meta: dict) -> None:
    st.subheader("Downloads")
    snap, locked = listing["snapshot"], listing["locked"]
    if locked:
        st.info(f"**Downloads open when Phase 2 finishes** ({snap['done']:,} of "
                f"{snap['target']:,} rejected applicants explained, {snap['pending']:,} "
                f"to go). They switch on by themselves, with the complete files.",
                icon=":material/lock_clock:")
    elif snap.get("partial"):
        st.warning(f"**Partial snapshot:** Phase 2 is {snap['state']} and "
                   f"{snap['pending']:,} rejected applicants have no reasons yet. Files "
                   f"downloaded now are named `..._PARTIAL_{snap['pending']}_pending` and "
                   f"the ZIP carries `{meta['text']['partial_snapshot']}`.")
    files = {f["name"]: f for f in listing["files"]}
    counts = {"approved_applicants.csv": f" ({s['n_approved']:,} rows)",
              "rejected_applicants.csv": f" ({s['n_rejected']:,} rows)",
              "adverse_action_notices.zip": f" ({int(s.get('n_notices', 0) or 0):,} "
                                            f"notices)"}
    cols = st.columns(2)
    for i, name in enumerate(SHOWN):
        f = files.get(name)
        with cols[i % 2]:
            if f is None:
                st.caption(f"{name} — not produced"
                           + (" (no rejected applicants)" if "notice" in name else ""))
                continue
            short = name.rsplit("/", 1)[-1]
            partial = snap.get("partial") and not locked
            st.download_button(
                f"⬇ {name}" + (" (partial)" if partial else ""),
                (lambda n=name: api().raw(f"/runs/{run_id}/files/{n}")),
                file_name=(short if not snap.get("partial") else
                           f"{short.rpartition('.')[0]}_PARTIAL_{snap['pending']}_pending"
                           f".{short.rpartition('.')[2]}"),
                mime="application/zip" if name.endswith(".zip") else "text/csv",
                width="stretch", key=f"dl_{run_id}_{name}", on_click="ignore",
                disabled=locked)
            label = f["label"] + counts.get(name, "")
            if f["internal"]:
                st.markdown(badge("failed", "INTERNAL") + f" <span style='color:{MUTED};"
                            f"font-size:.85rem'>{label} · {f['size_bytes'] / 1e6:,.2f} "
                            f"MB</span>", unsafe_allow_html=True)
            else:
                st.caption(f"{label} · {f['size_bytes'] / 1e6:,.2f} MB")
    st.download_button(
        "⬇ Download all (ZIP)" + (f" — PARTIAL, {snap['pending']:,} reasons pending"
                                  if snap.get("partial") and not locked else ""),
        (lambda: api().raw(f"/runs/{run_id}/bundle")),
        file_name=(f"{run_id}_PARTIAL_{snap['pending']}_reasons_pending.zip"
                   if snap.get("partial") else f"{run_id}.zip"),
        mime="application/zip", type="primary", key=f"dl_all_{run_id}",
        on_click="ignore", disabled=locked)


def safe_load(run_id: str):
    """:func:`load`, or the API's refusal drawn and None."""
    try:
        return load(run_id)
    except ApiError as exc:
        exc.show()
        return None
