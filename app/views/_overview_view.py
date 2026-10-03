"""The Overview page's filter panel and its three sections, drawn from the API.

Separated from the page so a test can draw every panel against a real run
(tests/test_app.py). Read-only, and nothing here computes a figure: the API sums the
history from every run's ``run_summary.csv`` and serves the per-run panels from the
``aggregates.json`` that run wrote, including the order an ordinal dimension's levels
belong in. A chart on this page can always be traced to one of those files.

Three sections, top to bottom:

1. :func:`render_history` -- volume and outcomes across every run in the filter.
2. :func:`render_run` -- the selected run's own decisions, in the same chart style
   the Results viewer uses (the builders live in ``_dashboard``, so the two pages
   cannot drift apart). Score applicants draws this same function for the run that
   just finished, under the heading "What happened": one per-run panel, two pages,
   so the same run can never be shown with two sets of numbers.
3. :func:`render_insights` -- four short cards, each a template with real numbers
   substituted. No sentence here is written from anything but a number already on
   the page.
"""

from __future__ import annotations

import html

import altair as alt
import pandas as pd
import streamlit as st

from _common import (ACCENT, AMBER, APPROVE, MUTED, REJECT, badge, card_row,
                     colour, dot, kpi_row, when)
from _dashboard import (COUNT_RULE, SURFACE, decision_share_chart, reasons_chart,
                        risk_chart)

# The monitoring dimensions a run records, in display order: the title to give each
# one, and which of three things it is to the model that scored the run. The
# distinction is the point of the panel, not decoration -- "decisions by state" and
# "decisions by purpose" are claims of a different kind, and only one of them is
# about something the model read.
#
#   feature  the model scored this column, so the chart is about one of its inputs.
#   banded   the model scored the underlying number, not these buckets. It sees
#            income as a continuous value; the bands exist to read the outcome by
#            (features.encoders.income_band, Stage 3's segment report).
#   outside  not a model input at all. State is excluded on purpose (FINDINGS 7c,
#            7l), which is exactly why outcomes are reported by it: a model that
#            cannot see geography can still decide unevenly across it.
SEGMENTS = {
    "purpose": ("loan purpose", "feature"),
    "income_band": ("income band", "banded"),
    "addr_state": ("state", "outside"),
}
# Values cleaning changed, as opposed to values it only noticed. "Needed cleaning"
# has to mean something was altered, or the number is just a count of columns.
CHANGED_BY_CLEANING = ("unreadable_number", "text_coerced", "imputed", "clipped",
                       "pooled_to_other")


# ----------------------------------------------------------- the filter panel --

def filter_panel(runs: list[dict], history: dict, finished_states=()) -> dict:
    """The panel on the left, and the choices it returns.

    In the sidebar so it stays visible while the page scrolls, under the app's own
    navigation. The date range is ``st.date_input``, which Streamlit already
    provides -- no calendar component is added for this page.
    """
    with st.sidebar:
        st.markdown("### Filters")

        # Every run is offered, newest first, so a refused or failed one can still
        # be looked at -- but the default is the newest *finished* run, because a run
        # that failed its checks has no decisions and would open section 2 empty.
        options = [r["run_id"] for r in runs]
        labels = {r["run_id"]: f"{dot(r.get('state'))} {when(r.get('created_at'))} · "
                              f"{r.get('source_file') or r['run_id']}"
                  for r in runs}
        finished = set(finished_states or ())
        default = next((i for i, r in enumerate(runs)
                        if r.get("state") in finished), 0)
        run_id = None
        if options:
            run_id = st.selectbox(
                "Run", options, index=default,
                format_func=lambda i: labels.get(i, i),
                help="The run shown in sections 2 and 3. Newest first; the most "
                     "recent finished run is the default.",
                key="ov_run")
        else:
            st.caption("No run yet.")

        st.markdown("###### History window")
        first, last = history.get("first_day"), history.get("last_day")
        bounds = _date_bounds(first, last)
        dates = st.date_input(
            "Run date from / to", value=bounds, min_value=bounds[0],
            max_value=bounds[1],
            help="Filters section 1 only. Section 2 shows the run picked above, "
                 "whatever this window is.",
            key="ov_dates")
        date_from, date_to = _unpack_dates(dates, bounds)

        tags = history.get("model_tags") or []
        model = st.selectbox("Model", ["All models", *tags], index=0,
                             help="Show only runs scored with one model tag.",
                             key="ov_model")
        lending = st.toggle(
            "Only runs for lending decisions", value=True,
            help="Leaves out runs stamped not for lending decisions (an unapproved "
                 "model or an unpublished threshold): their rates answer a different "
                 "question.",
            key="ov_lending")
        st.caption("Read-only: this page only reads finished runs. No scoring "
                   "happens here.")
    return {"run_id": run_id, "date_from": date_from, "date_to": date_to,
            "model_tag": None if model == "All models" else model,
            "for_lending_only": lending}


def _date_bounds(first, last):
    """The widest window the history can show, as two dates."""
    from datetime import date, timedelta
    try:
        lo = date.fromisoformat(str(first)[:10])
    except (TypeError, ValueError):
        lo = date.today() - timedelta(days=30)
    try:
        hi = date.fromisoformat(str(last)[:10])
    except (TypeError, ValueError):
        hi = date.today()
    if hi < lo:
        hi = lo
    return lo, hi


def _unpack_dates(value, bounds):
    """``st.date_input`` returns one date while a range is half-chosen."""
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            return str(value[0]), str(value[1])
        if len(value) == 1:
            return str(value[0]), str(bounds[1])
        return str(bounds[0]), str(bounds[1])
    return str(value), str(bounds[1])


# ------------------------------------------------ section 1: all runs to date --

def render_history(h: dict, meta: dict | None = None) -> None:
    """Volume and outcomes across every run the filter kept."""
    st.subheader("Processing history")
    rate = h.get("approval_rate_overall")
    kpi_row([
        ("Applicants decided", f"{h['applicants_processed']:,}",
         "Applicants whose decision is final: Phase 1 complete and every blocking "
         "check passed, summed over those runs' run_summary.csv. A run that failed "
         "its checks is not counted at all. Decided is not the same as explained -- "
         "the line under the chart gives the reason coverage."),
        ("Runs in window", f"{h['total_runs']:,}",
         f"Every run in this window, whatever its state. {h['finished_runs']:,} of "
         "them finished and are counted in the other figures. The unfiltered total "
         "is below, under the model."),
        ("Total rejected", f"{h['total_rejected']:,}",
         "Applicants declined, summed over the same finished runs."),
        ("Overall approval rate", "—" if rate is None else f"{rate:.1%}",
         "Approved / decided over the window, weighted by applicants."),
        ("Last 7 days", f"{h['applicants_last_7_days']:,}",
         "Applicants decided in runs made in the last 7 days."),
        ("Last 30 days", f"{h['applicants_last_30_days']:,}",
         "Applicants decided in runs made in the last 30 days."),
    ])

    series = pd.DataFrame(h.get("series") or [])
    if series.empty:
        st.info("No finished run in this window. Widen the date range, or clear the "
                "model filter.")
        return

    series["label"] = series["created_at"].map(when)
    series["file"] = series["source_file"].fillna("")
    # The axis is the run date and each run is its own bar: grouping on the date and
    # offsetting within it gives both, where a date on the x channel alone would
    # merge every run of one day into a single bar.
    series["tick"] = series["label"].str.slice(0, 6)
    st.markdown("**Applicants processed per run** — oldest first")
    bars = alt.Chart(series).mark_bar(color=ACCENT, stroke=SURFACE,
                                      strokeWidth=1).encode(
        x=alt.X("tick:N", sort=None, title=None,
                axis=alt.Axis(labelAngle=-40, labelOverlap="greedy")),
        xOffset=alt.XOffset("run_id:N", sort=None),
        y=alt.Y("n_rows:Q", title="applicants"),
        tooltip=[alt.Tooltip("label:N", title="when"),
                 alt.Tooltip("file:N", title="file"),
                 alt.Tooltip("model_tag:N", title="model"),
                 alt.Tooltip("n_rows:Q", title="applicants", format=","),
                 alt.Tooltip("n_approved:Q", title="approved", format=","),
                 alt.Tooltip("n_rejected:Q", title="rejected", format=",")])
    st.altair_chart(bars, use_container_width=True)
    st.caption(f"One bar per finished run, labelled by run date; hover for the file "
               f"and the counts. {h['files_processed']:,} distinct input file(s) "
               f"across {h['finished_runs']:,} run(s). Runs that failed a blocking "
               f"check are not here and are not counted above.")
    _reason_coverage(h)
    _rate_trends(series, meta or {})


def _reason_coverage(h: dict) -> None:
    """How much of the decided volume has its reasons written.

    Decisions and reasons finish separately: a run counted above may have final,
    checked decisions and no Regulation B reasons yet. Saying so here is the point --
    "applicants decided" would otherwise read as "applicants processed end to end",
    which it is not.
    """
    rejected = int(h.get("total_rejected") or 0)
    if not rejected:
        return
    explained = int(h.get("rejected_explained") or 0)
    pending = int(h.get("reasons_pending") or 0)
    missing = int(h.get("rejected_without_reasons") or 0)
    waiting = int(h.get("runs_awaiting_reasons") or 0)
    if not pending and not missing:
        st.caption(f"Reasons are written for all {explained:,} rejection(s) in this "
                   f"window.")
        return
    st.warning(
        f"**{pending + missing:,} of {rejected:,} rejection(s) in this window have no "
        f"reasons yet** ({(pending + missing) / rejected:.0%}), across {waiting:,} run"
        f"(s) whose decisions are final but whose Phase 2 has not finished. The "
        f"applicant counts above are decisions, not completed notices"
        + (f"; {missing:,} were left without reasons by a cap, a sample or a skip, "
           f"which is a gap rather than a state." if missing else "."),
        icon=":material/pending_actions:")


def _rate_trends(series: pd.DataFrame, meta: dict) -> None:
    """How the approval rate and the fair-lending flag share moved over the window.

    The same two trends the Overview carried before this page was rebuilt, now over
    the filtered runs rather than the last twenty, so the panel moves them too.
    """
    limit = float((meta.get("decision") or {}).get("fair_lending_review_share") or 0.05)
    left, right = st.columns(2)
    with left:
        st.markdown("**Approval rate per run**")
        rates = series.dropna(subset=["approval_rate"])
        if rates.empty:
            st.caption("No run in this window recorded an approval rate.")
        else:
            base = alt.Chart(rates).encode(
                x=alt.X("run_id:N", sort=None, title=None,
                        axis=alt.Axis(labels=False, ticks=False)),
                y=alt.Y("approval_rate:Q", title="approval rate",
                        axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("label:N", title="when"),
                         alt.Tooltip("file:N", title="file"),
                         alt.Tooltip("approval_rate:Q", format=".1%"),
                         alt.Tooltip("n_rows:Q", title="applicants", format=",")])
            st.altair_chart(base.mark_line(color=APPROVE, strokeWidth=2)
                            + base.mark_point(color=APPROVE, filled=True, size=55,
                                              stroke=SURFACE, strokeWidth=2),
                            use_container_width=True)
    with right:
        st.markdown("**Fair-lending flag share** — explained rejections with a "
                    "non-disclosable top driver")
        flags = series.dropna(subset=["fair_lending_flag_share"])
        if flags.empty:
            st.caption("No run in this window has explained rejections yet.")
        else:
            base = alt.Chart(flags).encode(
                x=alt.X("run_id:N", sort=None, title=None,
                        axis=alt.Axis(labels=False, ticks=False)),
                y=alt.Y("fair_lending_flag_share:Q", title="flag share",
                        axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("label:N", title="when"),
                         alt.Tooltip("fair_lending_flag_share:Q", format=".1%"),
                         alt.Tooltip("n_explained:Q", title="explained", format=",")])
            rule = alt.Chart(pd.DataFrame({"y": [limit]})).mark_rule(
                color=REJECT, strokeDash=[6, 4]).encode(y="y:Q")
            st.altair_chart(base.mark_line(color=AMBER, strokeWidth=2)
                            + base.mark_point(color=AMBER, filled=True, size=55,
                                              stroke=SURFACE, strokeWidth=2)
                            + rule, use_container_width=True)
            st.caption(f"Dashed line: the {limit:.0%} review threshold "
                       f"(`decision.fair_lending_review_share`).")


# ----------------------- section 2 here, "What happened" on Score applicants --

def render_run(view, meta: dict, *, heading: str = "This run") -> None:
    """One run's own decisions: counts, risk, mix, segments and data quality.

    The single per-run panel in the app. Overview draws it as its section 2 for the
    run picked in the filter, and Score applicants draws it for the run that just
    finished (as "What happened") -- one function, called from two pages, so the
    two can never show different numbers for the same run. Everything comes from
    that run's summary and the ``aggregates.json`` it wrote; nothing is computed
    here, and ``heading`` is the only thing either caller changes.
    """
    s = view.summary
    agg = view.aggregates
    n = int(s.get("n_rows") or 0)
    approved = int(s.get("n_approved") or 0)
    rejected = int(s.get("n_rejected") or 0)
    explained = int(s.get("n_explained") or 0)
    flag_share = s.get("fair_lending_flag_share")

    st.subheader(heading)
    st.caption(f"{s.get('source_file') or view.run_id} · "
               f"{when(s.get('created_at'))} · model `{s.get('model_tag')}` · "
               f"reject at {float(s.get('threshold') or 0):.0%} over "
               f"{int(s.get('horizon_months') or 0)} months")
    # The same stamp, in the same words, wherever this panel is drawn: a run scored
    # with an unapproved model or an unpublished threshold may not be used to decide
    # anything, and the numbers below are exactly what must not be acted on.
    if not s.get("for_lending_decisions", True):
        st.error("**NOT FOR LENDING DECISIONS.** "
                 + str(s.get("not_for_lending_reasons") or "")
                 + ". Every output file and every notice of this run carries the "
                   "same stamp.", icon=":material/block:")

    kpi_row([
        ("Applicants", f"{n:,}"),
        ("Approved", f"{approved:,}"),
        ("Rejected", f"{rejected:,}"),
        ("Approval rate", f"{(approved / n):.1%}" if n else "—"),
        ("Mean risk", f"{float(s.get('mean_pd_36m') or 0):.1%}",
         f"Mean predicted default probability over "
         f"{int(s.get('horizon_months') or 36)} months."),
        ("Fair-lending flags",
         "—" if not explained or flag_share is None else f"{float(flag_share):.1%}",
         "Share of explained rejections whose strongest adverse drivers included a "
         "feature that must not be disclosed as a reason."),
    ])

    # --- risk distribution and the approve/reject mix ---
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Predicted risk**")
        if agg is not None and not agg.risk.empty:
            st.altair_chart(risk_chart(agg.risk, s["threshold"]),
                            use_container_width=True)
            st.caption("Dashed line: the decision threshold. Bars left of it were "
                       "approved.")
        else:
            st.caption("Not recorded for this run.")
    with right:
        st.markdown("**Approval mix**")
        _donut(approved, rejected)

    # --- reasons and purpose ---
    tag = str(s.get("model_tag") or "")
    left, right = st.columns(2)
    with left:
        top = agg.reasons.head(8) if agg is not None else pd.DataFrame()
        st.markdown("**Most common rejection reasons** (Regulation B wording)")
        if top.empty:
            st.caption(_reasons_pending_note(s, rejected))
        else:
            st.altair_chart(reasons_chart(top), use_container_width=True)
    with right:
        _segment_chart(agg, "purpose", tag)

    # --- income band and state ---
    left, right = st.columns(2)
    with left:
        _segment_chart(agg, "income_band", tag)
    with right:
        _segment_chart(agg, "addr_state", tag)

    _quality_card(view, meta)


def _reasons_pending_note(s: dict, rejected: int) -> str:
    """What stands in for the reasons chart while there are no reasons to chart.

    Decisions and reasons finish separately, so this panel can be complete in every
    other respect and still have nothing to put here. It says which of the two it
    is -- waiting on Phase 2, or a run that will never have reasons -- with the
    count, rather than one sentence covering both.
    """
    if not rejected:
        return "No applicant was rejected in this run, so there are no reasons."
    pending = int(s.get("n_reasons_pending") or 0)
    missing = int(s.get("n_rejected_without_reasons") or 0)
    if pending:
        return (f"**Reasons pending** for {pending:,} of {rejected:,} rejected "
                f"applicant(s): Phase 2 writes them after the decisions, and this "
                f"chart fills in by itself as it finishes. Decisions are final "
                f"either way.")
    if missing:
        return (f"{missing:,} of {rejected:,} rejected applicant(s) have no reasons, "
                f"and no further are coming: Phase 2 was skipped, sampled or stopped.")
    return "No reasons yet: Phase 2 writes them after the decisions."


def _donut(approved: int, rejected: int) -> None:
    """Approve/reject as a part-to-whole ring, with both slices labelled.

    Two slices is the weakest case for a ring -- the pair of numbers beside it says
    as much -- so the labels carry the share and the count, and they are also what
    makes the green/red pair readable without relying on the hues.
    """
    total = approved + rejected
    if not total:
        st.caption("No decisions in this run.")
        return
    data = pd.DataFrame({
        "decision": ["approve", "reject"],
        "applicants": [approved, rejected],
        "share": [approved / total, rejected / total]})
    base = alt.Chart(data).encode(
        theta=alt.Theta("applicants:Q", stack=True),
        color=alt.Color("decision:N",
                        scale=alt.Scale(domain=["approve", "reject"],
                                        range=[APPROVE, REJECT]), title=None),
        tooltip=[alt.Tooltip("decision:N"),
                 alt.Tooltip("applicants:Q", format=","),
                 alt.Tooltip("share:Q", format=".1%")])
    ring = base.mark_arc(innerRadius=58, outerRadius=92, stroke=SURFACE,
                         strokeWidth=2)
    labels = base.mark_text(radius=116, fontSize=11, color=MUTED).encode(
        text=alt.Text("share:Q", format=".1%"))
    st.altair_chart(ring + labels, use_container_width=True)
    st.caption(f"{approved:,} approved · {rejected:,} rejected · {total:,} decided.")


def _segment_chart(agg, column: str, model_tag: str = "") -> None:
    """Approve/reject share by one monitoring dimension."""
    title, kind = SEGMENTS[column]
    st.markdown(f"**Decisions by {title}**")
    if agg is None or not getattr(agg, "segment_columns", None):
        st.caption("Not recorded for this run: it was scored before the run output "
                   "carried decisions by segment. Re-score the file to see it.")
        return
    if column not in agg.segment_columns:
        st.caption(f"This file had no column for {title}, so the run recorded none. "
                   f"Nothing is inferred to fill the chart.")
        return
    rows = agg.segments
    if rows is None or rows.empty or "column" not in rows.columns:
        st.caption("Nothing recorded.")
        return
    rows = rows[rows["column"] == column][["group", "decision", "applicants"]]
    if rows.empty:
        st.caption("Nothing recorded.")
        return
    # An ordinal dimension keeps the order the run recorded for it (income bands in
    # band order, whatever their sizes); anything else is ordered by size, which
    # decision_share_chart does by default.
    order = None
    declared = (getattr(agg, "segment_order", None) or {}).get(column)
    if declared:
        present = set(rows["group"].astype("string"))
        order = [b for b in declared if b in present]
        order += sorted(present - set(order))
    if column == "addr_state":
        top = (rows.groupby("group")["applicants"].sum()
               .sort_values(ascending=False).head(12).index)
        rows = rows[rows["group"].isin(set(top))]
    st.altair_chart(decision_share_chart(rows, order=order),
                    use_container_width=True)
    tag = f"`{model_tag}`" if model_tag else "this model"
    if kind == "outside":
        note = (f"**{tag} does not use {title}.** It is excluded on purpose "
                f"(FINDINGS 7c, 7l), so this is fair-lending monitoring of where "
                f"decisions landed, not an account of what drove them. ")
    elif kind == "banded":
        note = (f"{tag} sees income as a number, not as these bands. The bands are "
                f"for reading the outcome, and are the ones Stage 3's segment report "
                f"uses. ")
    else:
        note = ""
    if column == "addr_state":
        note = "Twelve largest states in this file. " + note
    st.caption(note + COUNT_RULE)


def _quality_card(view, meta: dict) -> None:
    """Data quality and drift for this run, as one panel of plain statements."""
    s = view.summary
    st.markdown("**Data quality and drift**")
    drift = view.drift
    kpi_row([
        ("Features present", f"{int(s.get('features_present') or 0)} of "
                             f"{int(s.get('features_expected') or 0)}"),
        ("Values cleaned", f"{_cleaned_values(view):,}",
         "Values the cleaning step changed: unreadable numbers, text coerced to a "
         "number, imputed, clipped or pooled into 'other'."),
        ("Rows out of range", f"{int(s.get('rows_out_of_range') or 0):,}",
         "Rows with at least one value outside the range seen in training. Flagged, "
         "never dropped."),
        ("Drift", str(s.get("drift_status") or "not run").title(),
         "Distribution distance from the model's training data, per feature, "
         "summarised to one band."),
    ])
    bits = []
    if drift is not None:
        bits.append(drift.headline)
    if s.get("features_filled_from_training"):
        bits.append(f"**Filled with a training value, not read from the file:** "
                    f"{s['features_filled_from_training']}")
    if s.get("optional_missing_cost"):
        bits.append(f"Measured cost of the absent optional features: "
                    f"{s['optional_missing_cost']}")
    for line in bits:
        st.caption(line)


def _cleaned_values(view) -> int:
    table = view.cleaning_table
    if table is None or table.empty or "rule" not in table.columns:
        return 0
    hit = table[table["rule"].isin(CHANGED_BY_CLEANING)]
    return int(pd.to_numeric(hit.get("count"), errors="coerce").fillna(0).sum())


# ----------------------------------------------------- section 3: the insights --

def _card(state: str, headline: str, number: str, detail: str) -> str:
    c = colour(state)
    return (f"<div style='border:1px solid #d7dce5;border-left:4px solid {c};"
            f"border-radius:8px;padding:.7rem .9rem;height:100%;'>"
            f"<div style='font-weight:650;font-size:.95rem;margin-bottom:.15rem;'>"
            f"{html.escape(headline)}</div>"
            f"<div style='color:{c};font-weight:700;font-size:1.45rem;"
            f"line-height:1.1;'>{html.escape(number)}</div>"
            f"<div style='color:{MUTED};font-size:.8rem;margin-top:.2rem;'>"
            f"{html.escape(detail)}</div></div>")


def insight_cards(view, history: dict, meta: dict, registry: dict) -> list[tuple]:
    """Four findings about this run, each a template filled with a real number.

    Returned rather than drawn so a test can assert on the numbers. Every value
    comes from the run's summary, its cleaning report or the registry -- there is no
    sentence here that is not a number with wording around it.
    """
    s = view.summary
    n = int(s.get("n_rows") or 0)
    approved = int(s.get("n_approved") or 0)
    explained = int(s.get("n_explained") or 0)
    cards = []

    # 1. this run's approval rate against the history the filter kept
    rate = (approved / n) if n else None
    overall = history.get("approval_rate_overall")
    if rate is None or overall is None:
        cards.append(("wait", "Approval rate: nothing to compare with", "—",
                      "No other finished run in the history window."))
    else:
        points = (rate - overall) * 100
        word = "higher than" if points > 0.05 else (
            "lower than" if points < -0.05 else "in line with")
        cards.append(("done" if abs(points) < 5 else "overridden",
                      f"Approval rate is {word} the history average",
                      f"{points:+.1f} pts",
                      f"{rate:.1%} this run against {overall:.1%} over "
                      f"{history.get('finished_runs', 0):,} run(s)."))

    # 2. fair lending, from the share the notices themselves recorded
    share = s.get("fair_lending_flag_share")
    limit = float((meta.get("decision") or {}).get("fair_lending_review_share") or 0.05)
    if not explained:
        cards.append(("wait", "Fair lending: not yet measured", "—",
                      "Reasons are written in Phase 2; the share needs them."))
    elif not float(share or 0):
        cards.append(("done", "No fair-lending concern", "0.0%",
                      f"None of {explained:,} explained rejection(s) had a "
                      f"non-disclosable feature among its strongest drivers."))
    else:
        over = float(share) >= limit
        cards.append(("failed" if over else "overridden",
                      f"{float(share):.1%} of rejections flagged for review",
                      f"{float(share):.1%}",
                      f"{int(s.get('n_fair_lending_flagged') or 0):,} of "
                      f"{explained:,} explained; review threshold {limit:.0%}."))

    # 3. data quality, from the cleaning report
    cleaned = _cleaned_values(view)
    filled = str(s.get("features_filled_from_training") or "")
    if filled:
        cards.append(("failed", "A feature was filled, not read from the file",
                      f"{int(s.get('n_features_filled_from_training') or 0)}",
                      f"{filled}. The model never saw these missing in training "
                      f"(FINDINGS 7o)."))
    elif cleaned:
        cards.append(("overridden", "Values needed cleaning", f"{cleaned:,}",
                      f"Changed by the cleaning step across {n:,} applicant(s); "
                      f"every change is in the cleaning report."))
    else:
        cards.append(("done", "Data quality: clean", "0",
                      f"No value was changed by cleaning in {n:,} applicant(s)."))

    # 4. the model, from the registry
    tag = str(s.get("model_tag") or "")
    rec = next((m for m in (registry.get("models") or []) if m.get("tag") == tag), {})
    status = str(rec.get("status") or s.get("model_registry_status") or "unknown")
    if status == "approved":
        cards.append(("done", "Model: approved and in use", tag or "—",
                      "Approved in the registry for lending decisions."))
    else:
        rules = rec.get("rules") or []
        passed = sum(1 for r in rules if r.get("passed"))
        cards.append(("overridden" if status == "candidate" else "failed",
                      f"Model: {status}, not approved", tag or "—",
                      f"{passed} of {len(rules)} approval rule(s) pass."
                      if rules else "See the Model registry page."))
    return cards


def render_insights(cards: list[tuple]) -> None:
    st.subheader("Insights")
    st.caption("Each card states one fact about this run with the number behind it. "
               "Nothing here is written commentary: the wording is a fixed template "
               "and the values come from the run's own files.")
    card_row([_card(state, headline, number, detail)
              for state, headline, number, detail in cards])
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    st.caption(" ".join([badge("done", "as expected"), badge("overridden", "worth a look"),
                         badge("failed", "needs attention"),
                         badge("pending", "not measured yet")]),
               unsafe_allow_html=True)
