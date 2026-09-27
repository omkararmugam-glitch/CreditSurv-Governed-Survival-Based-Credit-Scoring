"""The result dashboard for one scoring run.

Separated from the upload page so it can be rendered in a test from a finished run
on disk. A NameError in a tab that only appears after a 9-minute upload is not
something to discover by uploading; ``tests/test_app.py`` renders every panel and
every chart here against a real run directory.

Read-only: everything shown comes from what the run wrote.

Drawn in the order it becomes useful: :func:`render_headline` (stamps, checks,
counts, approval rate and the first rows of decisions) needs only the run's summary
and its first rows, so it appears the moment Phase 1 finishes; the Phase 2 panel
goes between; :func:`render_details` (charts, profile, drift, downloads) follows.
Downloads read nothing until clicked -- a run of a 450 MB file has over a gigabyte
of outputs, and reading or zipping them on every rerun is what made the page slow.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from _common import ACCENT, APPROVE, MUTED, REJECT, rel
from creditsurv.batch import CAP_NOTE, bundle_zip
from creditsurv.run_checks import REASONS_PENDING

DRIFT_BOX = {"stable": st.success, "moderate": st.warning, "large": st.error,
             "unknown": st.warning, "insufficient": st.info}

# Any chart where a small group could otherwise look as alarming as a large one
# carries its own counts. A 100% rejection rate over three applicants and over
# three thousand are not the same finding, and a bar cannot tell them apart.
COUNT_RULE = ("Counts are shown beside each bar: a share computed over a handful "
              "of applicants is not comparable with one over thousands.")


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


def render(result, cfg=None, *, middle=None) -> None:
    """Draw the whole result view: the headline first, then ``middle`` (the
    Phase 2 panel, on the upload page), then the details."""
    render_headline(result)
    if middle is not None:
        middle()
    render_details(result)


def render_headline(result) -> None:
    """What is known the moment Phase 1 finishes: nothing here waits for reasons,
    charts, the profile or any file to be read in full."""
    s = result.summary
    scored = result.scored

    # Said first and loudest: a run that may not be used to decide anything.
    if s.get("for_lending_decisions") is False:
        st.error(f"**NOT FOR LENDING DECISIONS.** "
                 f"{s.get('not_for_lending_reasons', '')}. Every output file and every "
                 f"notice of this run carries the same stamp.",
                 icon=":material/block:")

    p1 = float(s.get("phase1_seconds") or result.seconds or 0.0)
    st.success(
        f"Decisions ready in {p1:.1f}s "
        f"({p1 / max(s['n_rows'], 1) * 1000:.2f}s per 1,000 applicants). "
        f"Files saved to `{rel(result.run_dir)}`.")
    pending_n = int(s.get("n_reasons_pending", 0) or 0)
    if pending_n:
        st.info(f"**Reasons pending for {pending_n:,} of {s['n_rejected']:,} rejected "
                f"applicants.** Their rows say \"{REASONS_PENDING}\" until Phase 2 "
                f"reaches them; decisions are final either way.")

    checks = getattr(result, "checks", None)
    if checks is not None and not checks.empty:
        n_fail = int((checks["status"] == "FAIL").sum())
        n_over = int((checks["status"] == "OVERRIDDEN").sum())
        line = (f"**Run checks: {int((checks['status'] == 'PASS').sum())} of "
                f"{len(checks)} passed**"
                + (f", {n_over} overridden" if n_over else "")
                + (f", {n_fail} FAILED" if n_fail else "")
                + " — verified from the files this run wrote (Checks tab).")
        (st.error if n_fail else st.warning if n_over else st.success)(line)
    else:
        st.warning("**Run checks: not recorded.** This run predates the post-run "
                   "checks; treat its outputs as unverified.")

    for w in result.report.warnings:
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

    if result.drift is not None:
        label = ("NOT ASSESSED" if result.drift.status == "insufficient"
                 else result.drift.colour.upper())
        DRIFT_BOX[result.drift.status](f"**Data drift: {label}** — "
                                       + result.drift.headline())

    # A declined applicant with no stated reasons is a compliance gap, so it is
    # said here rather than left to a column in the CSV.
    missing_reasons = int(s.get("n_rejected_without_reasons", 0) or 0)
    if missing_reasons:
        st.error(f"**{missing_reasons:,} of {s['n_rejected']:,} rejected applicants "
                 f"have no adverse-action reasons.** Those rows are marked "
                 f"\"{CAP_NOTE}\" in every output file. This happens only when a run "
                 f"is stopped before it finishes; re-running resumes where it "
                 f"stopped.")

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

    # Fair-lending monitoring, on every run.
    flagged = s.get("n_fair_lending_flagged")
    if flagged is not None:
        share = float(s.get("fair_lending_flag_share", 0.0) or 0.0)
        limit = float(s.get("fair_lending_review_share", 0.05) or 0.05)
        text = (f"{int(flagged):,} of {int(s.get('n_explained', 0)):,} explained "
                f"rejections ({share:.1%}) had a non-disclosable feature among the "
                f"strongest adverse drivers"
                + (f" ({s['fair_lending_flag_features']})"
                   if s.get("fair_lending_flag_features") else "")
                + f"; {int(s.get('n_top_driver_not_disclosable', 0)):,} as the single "
                  f"strongest. Review threshold: {limit:.0%}.")
        if s.get("fair_lending_review_required"):
            st.error(f"**Fair-lending review required.** {text} The stated reasons "
                     f"exclude these features, so the notices cannot show it: the "
                     f"detail is in `internal/internal_review_flags.csv`.",
                     icon=":material/balance:")
        else:
            st.caption(f"Fair-lending monitor: {text}")

    # ----------------------------------------------------------- headline --
    c = st.columns(5)
    c[0].metric("Applicants", f"{s['n_rows']:,}")
    c[1].metric("Approved", f"{s['n_approved']:,}")
    c[2].metric("Rejected", f"{s['n_rejected']:,}")
    c[3].metric("Approval rate", f"{s['approval_rate']:.1%}")
    mean_pd = s.get(f"mean_pd_{s['horizon_months']}m", float("nan"))
    c[4].metric(f"Average {s['horizon_months']}m risk", f"{mean_pd:.1%}")
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

    st.markdown(f"**First decisions** (first {min(len(scored), 25)} of "
                f"{s['n_rows']:,} rows; every row is in the downloads)")
    st.dataframe(scored.head(25), hide_index=True, width="stretch")


def render_details(result) -> None:
    """Charts, reasons, profile, drift and downloads: drawn after the headline."""
    s = result.summary
    agg = result.aggregates
    checks = getattr(result, "checks", None)

    # ------------------------------------------------------------- charts --
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
                    domain=["approve", "reject"], range=[APPROVE, REJECT]),
                    title=None),
                tooltip=["bin_start", "bin_end", "applicants"])
            rule = alt.Chart(pd.DataFrame({"t": [s["threshold"]]})).mark_rule(
                color=MUTED, strokeDash=[6, 4]).encode(x="t:Q")
            st.altair_chart(hist + rule, use_container_width=True)
    with right:
        by = agg.group_frame() if agg is not None else pd.DataFrame()
        if not by.empty:
            st.markdown(f"**Decisions by loan {agg.group_column}**")
            totals = by.groupby("group", as_index=False)["applicants"].sum()
            totals = totals.rename(columns={"applicants": "total"})
            shares = by.merge(totals, on="group")
            chart = alt.Chart(shares).mark_bar().encode(
                y=alt.Y("group:N", sort="-x", title=None),
                x=alt.X("applicants:Q", stack="normalize", title="share"),
                color=alt.Color("decision:N", scale=alt.Scale(
                    domain=["approve", "reject"], range=[APPROVE, REJECT]),
                    title=None),
                tooltip=["group", "decision", "applicants", "total"])
            labels = alt.Chart(totals).mark_text(
                align="left", dx=4, color=MUTED, fontSize=10).encode(
                y=alt.Y("group:N", sort="-x", title=None),
                x=alt.value(0),
                text=alt.Text("total:Q", format=",.0f"))
            st.altair_chart(chart + labels, use_container_width=True)
            st.caption(f"Totals per {agg.group_column} shown beside each bar. "
                       + COUNT_RULE)

    top = agg.reason_frame().head(8) if agg is not None else pd.DataFrame()
    if not top.empty:
        st.markdown("**Most common rejection reasons** (Regulation B wording)")
        st.altair_chart(_bar_with_counts(top, y="reason", x="times cited",
                                         colour=REJECT),
                        use_container_width=True)

    # --------------------------------------------------------------- tabs --
    tab_checks, tab_clean, tab_profile, tab_drift = st.tabs(
        ["Checks", "Cleaning report", "Data profile", "Drift check"])

    with tab_checks:
        if checks is None or checks.empty:
            st.caption("No checks recorded for this run.")
        else:
            st.caption("Computed after every file was written, by reading the files "
                       "back. A FAIL on a blocking check stops the run before it is "
                       "marked finished; OVERRIDDEN means the operator chose it and "
                       "the run is stamped not for lending decisions.")
            st.dataframe(checks, hide_index=True, width="stretch")

    with tab_clean:
        cr = result.clean_report
        st.caption(
            f"Cleaning policy `{s.get('cleaning_policy_version')}`, using values "
            f"fitted on {s.get('cleaning_values_fitted_rows', 0):,} training rows"
            + ("" if s.get("cleaning_values_from_model_bundle") else
               " (re-fitted from this model's training split because the bundle "
               "carries none)") + ".")
        if cr is not None:
            for line in cr.plain_english():
                st.markdown(f"- {line}")
            st.dataframe(cr.to_frame(), hide_index=True, width="stretch")

    with tab_profile:
        prof = result.profile or {}
        o = prof.get("overview", {})
        cols = st.columns(4)
        cols[0].metric("Rows", f"{o.get('rows', 0):,}")
        cols[1].metric("Columns", f"{o.get('columns', 0):,}")
        cols[2].metric("Numeric", f"{o.get('numeric_columns', 0):,}")
        cols[3].metric("Categorical", f"{o.get('categorical_columns', 0):,}")
        if s.get("profiled_rows", 0) < s["n_rows"]:
            st.caption(f"Computed on a random sample of {s['profiled_rows']:,} of "
                       f"{s['n_rows']:,} rows, drawn across the whole file.")
        cost = s.get("optional_missing_cost")
        if cost:
            st.warning(f"Measured accuracy cost of the optional features absent "
                       f"from this file: {cost}")
        miss = prof.get("missing", pd.DataFrame())
        if not miss.empty:
            st.markdown("**Missing values by column**")
            affected = miss[miss["missing_share"] > 0]
            if affected.empty:
                st.caption("No missing values in any model feature.")
            else:
                st.altair_chart(
                    _bar_with_counts(affected.head(25), y="column", x="missing",
                                     colour=ACCENT, title="rows missing"),
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
        dr = result.drift
        if dr is None:
            st.caption("No drift check for this run.")
        else:
            st.markdown(f"**{dr.headline()}**")
            if dr.status == "insufficient":
                st.caption("Nothing is wrong with the file; there is simply not "
                           "enough of it to tell whether it resembles the training "
                           "population.")
            st.caption("Numeric features: population stability index against the "
                       "training distribution (below 0.10 stable, 0.10-0.25 "
                       "moderate, above 0.25 large). Categorical: total variation "
                       "distance on the same bands.")
            table = dr.table
            shifted = table[table["status"].isin(["large", "moderate", "unknown"])]
            if not shifted.empty:
                st.altair_chart(alt.Chart(shifted.head(20)).mark_bar().encode(
                    y=alt.Y("feature:N", sort="-x", title=None),
                    x=alt.X("score:Q", title="PSI / TVD"),
                    color=alt.Color("status:N", scale=alt.Scale(
                        domain=["large", "moderate", "unknown", "stable"],
                        range=[REJECT, "#d9a02b", MUTED, APPROVE]), title=None),
                    tooltip=list(table.columns)), use_container_width=True)
            st.dataframe(table, hide_index=True, width="stretch", height=320)

    # ---------------------------------------------------------- downloads --
    st.subheader("Downloads")
    labels = {
        "scored_applicants.csv": "Every applicant, with risk, decision and top reasons",
        "approved_applicants.csv": f"Approved only ({s['n_approved']:,} rows)",
        "rejected_applicants.csv": f"Rejected only ({s['n_rejected']:,} rows), with "
                                   "Regulation B reasons and fair-lending flags",
        "adverse_action_notices.zip": f"{s['n_notices']:,} applicant notices — "
                                      "only what the applicant is given",
        "internal/internal_review_flags.csv": "INTERNAL — never send to applicants: "
                                              "fair-lending flags, drivers, "
                                              "attributions",
        "validation_checks.csv": "Pass/fail of every post-run check",
        "cleaning_report.csv": "What cleaning did: per rule and per column",
        "data_drift.csv": "Per-feature drift of this file against the training data",
        "run_summary.csv": "One row describing this run, for traceability",
    }
    cols = st.columns(2)
    for i, (name, label) in enumerate(labels.items()):
        path = result.files.get(name)
        with cols[i % 2]:
            if path is None or not path.exists():
                st.caption(f"{name} — not produced"
                           + (" (no rejected applicants)" if "notice" in name else ""))
                continue
            # A callable: the file is read when the button is clicked, not on
            # every rerun of the page.
            st.download_button(
                f"⬇ {name}", (lambda p=path: p.read_bytes()),
                file_name=name.rsplit("/", 1)[-1],
                mime="application/zip" if name.endswith(".zip") else "text/csv",
                width="stretch", key=f"dl_{name}", on_click="ignore")
            st.caption(label + f" · {path.stat().st_size / 1e6:,.1f} MB")
    st.download_button("⬇ Download all (ZIP)", (lambda: bundle_zip(result)),
                       file_name=f"{result.run_dir.name}.zip",
                       mime="application/zip", type="primary", key="dl_all",
                       on_click="ignore")

    with st.expander("Run details"):
        st.json(s, expanded=False)
        st.caption("Also written to the run folder as provenance.json, with SHA-256 "
                   "hashes of the upload, the model and every output file.")
