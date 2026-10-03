"""The dashboard, drawn page by page against a real API (in process, no server).

Two APIs are used: one over the project itself (read-only pages against the real
registry, runs and research results), and one over the stub model from test_batch,
for everything that needs a run to exist. Every page is a client of the API; the
last test fails if one imports the pipeline instead.
"""

from __future__ import annotations

import ast
import time

import pytest

from creditsurv.provenance import PROJECT_ROOT

pytest.importorskip("streamlit")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

VIEWS = PROJECT_ROOT / "app" / "views"


@pytest.fixture(autouse=True)
def _paths(monkeypatch):
    monkeypatch.syspath_prepend(str(VIEWS))
    monkeypatch.chdir(PROJECT_ROOT)


@pytest.fixture
def project_api():
    """The API over the project's own config, runs and registry. Read-only use."""
    import _client
    from creditsurv.api.app import create_app

    client = TestClient(create_app(watch_code=False))
    _client.use(client)
    yield client
    _client.use(None)


def _stub(tmp_path):
    import _client
    import test_api as ta

    cfg = _stub_cfg(tmp_path)
    ctx = _stub_ctx(cfg)
    client = ta._make(cfg, ctx, tmp_path)
    _client.use(client)
    return client


def _stub_cfg(tmp_path):
    import test_batch as tb
    return tb.Config(paths=tb.Paths(data_dir=tmp_path / "data",
                                    models_dir=tmp_path / "models",
                                    figures_dir=tmp_path / "figures",
                                    tables_dir=tmp_path / "tables",
                                    registry=tmp_path / "models.yaml"),
                     decision=tb.DecisionConfig(
        model_tag="stub", horizon_months=36, reject_at_or_above=0.45,
        explain_nsamples=2 * len(tb.NUMERIC + tb.CATEGORICAL), explain_n_background=8,
        max_explained=3, background_rows=200, explain_workers=1))


def _stub_ctx(cfg):
    import numpy as np
    import test_batch as tb
    train = tb._training_frame()
    dm = tb.build_design_matrix(train, tb.SPEC, flavour="gbm")
    model_path = tb._write_dummy_model(cfg)
    tb.write_registry(cfg, model_path)
    return tb.ScoringContext(
        cfg=cfg, model_tag="stub", model_name="discrete_hazard", model=tb.StubModel(),
        spec=tb.SPEC, bundle={"artefacts": {"gbm_columns": list(dm.X.columns)}},
        model_path=model_path, background=dm.X, reference=train,
        clean_values=tb.fit_values(train, tb.SPEC, source="stub.parquet"),
        policy=tb.policy_from_config(cfg),
        times=np.array([6.0, 12.0, 24.0, 36.0]), data_source=tb._dummy_source(cfg))


@pytest.fixture
def stub(tmp_path):
    import _client
    client = _stub(tmp_path)
    yield client
    _client.use(None)


def _scored(client, n=30, **body) -> str:
    import test_api as ta
    import test_batch as tb
    run_id, _ = ta.run_file(client, tb._upload(n), **body)
    return run_id


def _run(page: str) -> AppTest:
    at = AppTest.from_file(str(VIEWS / page), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _run_in_app(page: str) -> AppTest:
    """A page opened through app.py, as a user reaches it: page links resolve only
    inside the app's navigation."""
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py"), default_timeout=180)
    at.run()
    at.switch_page(f"views/{page}")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _text(at) -> str:
    return " ".join(str(x.value) for kind in ("markdown", "caption", "info", "success",
                                              "warning", "error")
                    for x in getattr(at, kind))


# ------------------------------------------------- every page, real project --

def test_entrypoint_opens_on_score_applicants(project_api):
    """The first screen is the upload page, not the history dashboard."""
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Upload applicant dataset" in u.label for u in at.get("file_uploader"))
    assert not any(m.label == "Runs to date" for m in at.metric)   # not Overview


@pytest.mark.parametrize("page", ["overview.py", "home.py", "pipeline.py", "results.py",
                                  "model_registry.py", "retrain.py", "findings.py"])
def test_every_page_renders_against_the_project(project_api, page):
    _run_in_app(page)


def test_header_says_where_the_api_runs(project_api):
    from creditsurv.environment import runtime_label
    at = _run("findings.py")
    assert any(f"API on {runtime_label()}" in m.value for m in at.markdown)


def test_the_overview_numbers_are_the_run_history(project_api):
    """Nothing invented: the page's counts are the API's, which are the files'."""
    ov = project_api.get("/overview").json()
    at = _run("overview.py")
    metric = {m.label: m.value for m in at.metric}
    assert metric["Runs to date"] == f"{ov['total_runs']:,}"
    approved = ov["models"]["approved"]
    assert metric["Approved model"] == (approved[0]["tag"] if approved else "none")


def test_a_page_says_so_when_the_api_is_down():
    import _client
    import httpx
    _client.use(httpx.Client(base_url="http://127.0.0.1:9"))       # nothing listens
    try:
        at = AppTest.from_file(str(VIEWS / "overview.py"), default_timeout=60)
        at.run()
        assert not at.exception
        assert any("not reachable" in e.value for e in at.error)
    finally:
        _client.use(None)


def test_retrain_page_defaults_are_safe(project_api):
    at = _run_in_app("retrain.py")
    assert at.radio[0].value == "Small"
    assert at.checkbox[0].value is False                   # --overwrite off
    assert any(b.label == "Start run" for b in at.button)


def test_retrain_page_refuses_primary_tag(project_api):
    at = _run_in_app("retrain.py")
    at.text_input[0].set_value("full").run()
    assert any("Tag refused" in e.value for e in at.error)
    assert not any(b.label == "Start run" for b in at.button)


def test_retrain_page_follows_only_retraining_runs(project_api):
    at = _run_in_app("retrain.py")
    for sb in at.selectbox:
        if sb.label == "Run":
            assert not any("explain" in o or "drifted" in o for o in sb.options)


def test_home_shows_only_the_drop_zone_before_upload(stub):
    at = _run("home.py")
    assert len(at.get("file_uploader")) == 1
    assert any("default probability" in c.value for c in at.caption)
    assert not at.dataframe and not at.button


def test_home_background_switch_is_off_by_default(stub):
    at = _run("home.py")
    labels = [c.label for c in at.checkbox]
    assert at.checkbox[labels.index("Always run in the background")].value is False


# ------------------------------------------------- a run, from upload to result --

def test_an_upload_held_by_the_api_survives_reruns_and_scores(stub):
    """The upload lives in the API; the page holds only its id. Rerunning the page
    (any click) or leaving and coming back never loses it -- the bug of the file
    that disappeared on click."""
    import test_batch as tb
    up = stub.post("/uploads", files={"file": ("applicants.csv", tb._upload(30),
                                               "text/csv")}).json()
    at = AppTest.from_file(str(VIEWS / "home.py"), default_timeout=180)
    at.session_state["upload_id"] = up["upload_id"]
    at.session_state["upload_name"] = up["name"]
    at.session_state["upload_size"] = up["size_bytes"]
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Working on **applicants.csv**" in i.value for i in at.info)
    end = time.time() + 60
    while stub.app_.state.scorer.active and time.time() < end:
        time.sleep(0.1)
    for _ in range(3):                                   # clicks, reruns
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["upload_id"] == up["upload_id"]
    assert any("Working on **applicants.csv**" in i.value for i in at.info)
    assert any("Decisions ready" in s.value for s in at.success)
    runs = stub.get("/runs").json()["runs"]
    assert len(runs) == 1                                # scored once, not per rerun


def test_dashboard_renders_every_panel_and_tab(stub):
    """The whole result view of a finished run, every tab and chart, drawn from the
    API. The test that would have caught the NameError in the Data profile tab."""
    run_id = _scored(stub)

    def script(views, run_id):
        import sys
        sys.path.insert(0, views)
        import streamlit as st
        from _common import meta
        from _dashboard import load, render
        render(load(run_id), meta())
        st.session_state["rendered"] = True

    at = AppTest.from_function(script, default_timeout=180,
                               kwargs={"views": str(VIEWS), "run_id": run_id})
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["rendered"] is True
    assert len(at.tabs) >= 3
    assert len(at.metric) >= 5
    assert len(at.dataframe) >= 4
    assert len(at.get("vega_lite_chart")) >= 3
    labels = [b.label for b in at.get("download_button")]
    assert "⬇ Download all (ZIP)" in labels
    assert any("scored_applicants.csv" in b for b in labels)
    text = _text(at)
    assert "Cleaning policy" in text
    assert "population stability index" in text
    assert "Counts are shown beside each bar" in text
    assert "INTERNAL" in text                           # the internal file is marked


# ------------------------ "What happened", on the page that did the scoring --
# The per-run panel is one function (_overview_view.render_run) drawn by two pages.
# These tests are what would fail if a second copy of it were ever written for the
# Score applicants page, or if the panel stopped appearing there on its own.

def _home_scored(stub, n=30) -> AppTest:
    """The Score applicants page as it stands the moment a real run has finished."""
    import test_batch as tb
    up = stub.post("/uploads", files={"file": ("applicants.csv", tb._upload(n),
                                               "text/csv")}).json()
    at = AppTest.from_file(str(VIEWS / "home.py"), default_timeout=180)
    at.session_state["upload_id"] = up["upload_id"]
    at.session_state["upload_name"] = up["name"]
    at.session_state["upload_size"] = up["size_bytes"]
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    end = time.time() + 120
    while stub.app_.state.scorer.active and time.time() < end:
        time.sleep(0.1)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _first_metrics(at) -> dict:
    """Each card by its label, the first time the page draws it."""
    out: dict = {}
    for m in at.metric:
        out.setdefault(m.label, m.value)
    return out


def test_the_score_page_shows_the_run_panel_when_the_run_finishes(stub):
    """No extra click and no change of page: the run finishes and the panel is
    there, with the cards and charts Overview's section 2 has."""
    at = _home_scored(stub)
    assert any("Decisions ready" in s.value for s in at.success)
    assert "What happened" in [str(h.value) for h in at.subheader]
    metric = _first_metrics(at)
    for label in OV_SECTION_2:
        assert label in metric, label
    headings = {str(m.value) for m in at.markdown}
    for chart in ("**Predicted risk**", "**Approval mix**",
                  "**Most common rejection reasons** (Regulation B wording)",
                  "**Data quality and drift**"):
        assert chart in headings, chart
    # additive: the downloads are still on the page, where they were
    labels = [b.label for b in at.get("download_button")]
    assert "⬇ Download all (ZIP)" in labels or any("PARTIAL" in b for b in labels)


def test_the_run_panel_shows_one_run_the_same_way_on_both_pages(stub):
    """Same run, same numbers. Two implementations would be a bug, so there is one:
    both pages call _overview_view.render_run."""
    at_home = _home_scored(stub)
    run_id = at_home.session_state["run_id"]
    at_ov = _run("overview.py")
    assert at_ov.selectbox("ov_run").value == run_id
    home, overview = _first_metrics(at_home), _first_metrics(at_ov)
    for label in OV_SECTION_2:
        assert home[label] == overview[label], (label, home[label], overview[label])


def test_the_score_page_draws_none_of_the_run_panel_itself():
    """Read, not rendered: the Score applicants page imports the panel and builds no
    part of it, so the two pages cannot drift apart."""
    home = (VIEWS / "home.py").read_text(encoding="utf-8")
    assert "from _overview_view import render_run" in home
    assert 'render_run(view, meta, heading="What happened")' in home
    for own in ("import altair", "risk_chart", "decision_share_chart", "reasons_chart",
                "mark_arc", "kpi_row("):
        assert own not in home, own


def test_the_run_panel_renders_with_reasons_pending_and_the_rest_of_it_intact(stub):
    """Phase 2 still writing: the reasons chart is the only thing missing, and what
    stands in its place says how many are pending. The section needs no polling of
    its own -- the Phase 2 panel and the partial-snapshot banner on the same page
    rerun the whole app when it finishes (st.rerun(scope="app")), which redraws
    this with the reasons in it."""
    run_id = _scored(stub)

    def script(views, run_id):
        import sys
        sys.path.insert(0, views)
        import pandas as pd
        from _common import meta
        from _dashboard import load
        from _overview_view import render_run
        view = load(run_id)
        # the run as it stands while Phase 2 is still working: decisions final,
        # aggregates written, no reasons in them yet
        view.aggregates.reasons = pd.DataFrame()
        view.summary["n_reasons_pending"] = view.summary["n_rejected"]
        render_run(view, meta(), heading="What happened")

    at = AppTest.from_function(script, default_timeout=180,
                               kwargs={"views": str(VIEWS), "run_id": run_id})
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert "What happened" in [str(h.value) for h in at.subheader]
    assert any("Reasons pending" in c.value for c in at.caption)
    assert any("fills in by itself" in c.value for c in at.caption)
    metric = _first_metrics(at)
    for label in OV_SECTION_2:                      # every card still there
        assert label in metric, label
    headings = {str(m.value) for m in at.markdown}
    assert "**Predicted risk**" in headings and "**Approval mix**" in headings
    assert "**Data quality and drift**" in headings


def test_the_run_panel_says_reasons_are_pending_rather_than_charting_nothing(stub):
    """Phase 2 finishes after the decisions, so the panel can be complete in every
    other respect with no reasons to chart. It says which, with the count."""
    import _overview_view as ov
    pending = ov._reasons_pending_note(
        {"n_reasons_pending": 7, "n_rejected_without_reasons": 0}, 9)
    assert "Reasons pending" in pending and "7 of 9" in pending
    assert "fills in by itself" in pending
    assert "no applicant was rejected" in ov._reasons_pending_note({}, 0).lower()
    stopped = ov._reasons_pending_note({"n_rejected_without_reasons": 4}, 9)
    assert "4 of 9" in stopped and "no further are coming" in stopped


def test_a_failed_run_shows_why_and_no_decisions(stub):
    import io
    import pandas as pd
    import test_api as ta
    import test_batch as tb
    df = pd.read_csv(io.BytesIO(tb._upload()))
    df["dti"] = df["dti"].astype(object)
    df.loc[1:, "dti"] = "see attached"
    run_id, _ = ta.run_file(stub, df.to_csv(index=False).encode())

    def script(views, run_id):
        import sys
        sys.path.insert(0, views)
        from _dashboard import load, render_failed
        render_failed(load(run_id))

    at = AppTest.from_function(script, default_timeout=120,
                               kwargs={"views": str(VIEWS), "run_id": run_id})
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("FAILED" in e.value for e in at.error)
    assert "input_quality" in _text(at)
    assert not at.get("download_button")


def test_pipeline_page_shows_every_stage_of_a_run(stub):
    run_id = _scored(stub)
    at = AppTest.from_file(str(VIEWS / "pipeline.py"), default_timeout=180)
    at.session_state["run_id"] = run_id
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    for label in ("Check file", "Clean", "Score", "Decide", "Drift", "Run checks",
                  "Explain", "Notices"):
        assert f">{label}<" in html, label
    assert "✓ done" in html
    assert any("[Score] done" in c.value for c in at.code)       # the live log


def test_results_viewer_lists_runs_and_opens_one(stub):
    run_id = _scored(stub)
    at = AppTest.from_file(str(VIEWS / "results.py"), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert run_id in set(at.dataframe[0].value["run"])
    assert any("Decisions ready" in s.value for s in at.success)


def test_overview_is_an_empty_state_before_the_first_run(stub):
    """Nothing scored yet: a sentence and a way to the upload page, not a row of
    zeroed cards and six empty charts."""
    at = _run("overview.py")
    said = _text(at)
    assert "No runs yet" in said
    assert "Score applicants" in said
    assert not at.metric                      # no card at all, so none can be cut
    assert "Processing history" not in said   # and none of the sections it heads
    assert "This run" not in said


# ------------------------------------------------------- the KPI cards' width --

def test_no_page_draws_a_metric_outside_the_wrapping_kpi_row():
    """``st.metric`` renders its label and its value with the front end's truncate
    flag, so either is cut to an ellipsis once the card is narrower than the text
    ("Appli...", "34..."). Columns cannot prevent that -- they only stack below
    640 px, and otherwise keep shrinking. Every card therefore goes through
    ``_common.kpi_row``, which sizes each one to its own text and wraps the row."""
    stray = [f"{path.name}:{i}"
             for path in sorted(VIEWS.glob("*.py")) if path.name != "_common.py"
             for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
             if ".metric(" in line]
    assert not stray, stray


def test_kpi_width_fits_the_label_and_the_value():
    import _common

    # the longest label on the history row, with its help "?" beside it
    assert (_common.kpi_width("Overall approval rate", "73.5%", has_help=True)
            >= _common.KPI_MIN_WIDTH)
    # a value too long for the floor widens its own card instead of being cut
    assert (_common.kpi_width("Approved model", "full_applicant_nogeo")
            > _common.kpi_width("Approved model", "none"))
    # the value is set in 2.25rem, so it is what decides the width, not the label
    assert _common.kpi_width("Runs", "1") == _common.KPI_MIN_WIDTH


@pytest.mark.parametrize("page", ["overview.py", "pipeline.py", "results.py"])
def test_every_card_a_page_draws_is_wide_enough_for_its_text(project_api, page,
                                                             monkeypatch):
    """Rendered, not read: every card on the page is bordered and carries a pixel
    width at least as wide as the text it holds."""
    import streamlit as st

    import _common

    drawn = []
    real = st.metric

    def spy(label, value, *a, **kw):
        drawn.append((label, str(value), kw.get("width"), kw.get("border")))
        return real(label, value, *a, **kw)

    monkeypatch.setattr(st, "metric", spy)
    _run_in_app(page)
    assert drawn, page
    for label, value, width, border in drawn:
        assert border is True, label
        assert isinstance(width, int), (label, width)
        assert width >= _common.KPI_MIN_WIDTH, (label, width)
        assert width >= _common.text_px(value, _common.KPI_VALUE_PX), (label, value)
        assert width >= _common.text_px(label, _common.KPI_LABEL_PX), label


def test_overview_counts_the_stub_run(stub):
    _scored(stub)
    at = _run("overview.py")
    metric = {m.label: m.value for m in at.metric}
    assert metric["Runs to date"] == "1"
    assert metric["Approved model"] == "stub"


def test_options_come_back_after_the_page_was_left():
    """Streamlit drops a page's widget keys when another page is shown; remember()
    restores them from the copy keep() made."""
    def page():
        import streamlit as st
        from _common import keep, remember

        if st.session_state.get("_leave"):
            st.session_state.pop("_leave")
            st.session_state.pop("t", None)
            st.stop()
        remember("t", 0.25)
        st.slider("t", 0.05, 0.60, step=0.01, key="t")
        keep("t")

    at = AppTest.from_function(page)
    at.run()
    at.slider[0].set_value(0.4).run()
    at.session_state["_leave"] = True
    at.run()
    at.run()
    assert at.slider[0].value == 0.4


# ------------------------------------------- the Overview page's three sections --

OV_SECTION_1 = ("Applicants decided", "Runs in window", "Total rejected",
                "Overall approval rate", "Last 7 days", "Last 30 days")
OV_SECTION_2 = ("Applicants", "Approved", "Rejected", "Approval rate", "Mean risk",
                "Fair-lending flags", "Features present", "Values cleaned",
                "Rows out of range", "Drift")


def _overview(project_api):
    at = _run("overview.py")
    return at, {m.label: m.value for m in at.metric}


def test_overview_draws_the_filter_panel_and_three_sections(project_api):
    """The page is a filter panel plus history, this run, and the insight cards."""
    at, metric = _overview(project_api)
    assert [s.label for s in at.selectbox] == ["Run", "Model"]
    assert [d.label for d in at.date_input] == ["Run date from / to"]
    assert any("lending decisions" in t.label for t in at.toggle)
    for label in OV_SECTION_1 + OV_SECTION_2:
        assert label in metric, label
    headings = {str(m.value) for m in at.markdown}
    for chart in ("**Applicants processed per run** — oldest first", "**Predicted risk**",
                  "**Approval mix**", "**Decisions by loan purpose**",
                  "**Decisions by income band**", "**Decisions by state**",
                  "**Data quality and drift**"):
        assert chart in headings, chart


def test_overview_separates_decided_from_explained(project_api):
    """Decisions and reasons finish separately, so the volume tile must not read as
    "processed end to end": a run with final decisions and no reasons yet is counted
    in the volume, and the page says how much of it is still waiting."""
    h = project_api.get("/overview").json()["history"]
    at, metric = _overview(project_api)
    assert "Applicants decided" in metric          # not "processed"
    said = " ".join(str(w.value) for w in at.warning) + " ".join(
        str(c.value) for c in at.caption)
    short = h["reasons_pending"] + h["rejected_without_reasons"]
    if short:
        assert f"{short:,} of {h['total_rejected']:,} rejection(s)" in said
        assert "decisions, not completed notices" in said
    else:
        assert "Reasons are written for all" in said


def test_overview_volume_excludes_a_run_that_failed_its_checks(project_api):
    """A run that failed a blocking check has no decisions to count."""
    runs = project_api.get("/runs", params={"limit": 5000}).json()["runs"]
    failed = [r for r in runs if r["state"] == "failed checks" and r.get("n_rows")]
    if not failed:
        pytest.skip("no failed-checks run in this project's history")
    h = project_api.get("/overview").json()["history"]
    assert h["total_runs"] > h["finished_runs"]
    ids = {r["run_id"] for r in runs if r["state"] in
           set(project_api.get("/meta").json()["finished_states"])}
    assert not ({r["run_id"] for r in failed} & ids)
    # and their applicants are not in the volume
    assert h["applicants_processed"] == sum(
        int(r["n_rows"] or 0) for r in runs
        if r["run_id"] in ids and r.get("for_lending_decisions") in (True, "True"))


def test_overview_metric_labels_are_unique(project_api):
    """Three sections on one page, each with counts and a rate: a repeated label
    would make two different scopes read as the same number."""
    at, _ = _overview(project_api)
    labels = [m.label for m in at.metric]
    assert len(labels) == len(set(labels)), [l for l in labels if labels.count(l) > 1]


def test_overview_insight_cards_state_a_number_each(project_api):
    """Four cards, each a template with a real value -- never free text."""
    at, _ = _overview(project_api)
    cards = [str(m.value) for m in at.markdown if "border-left:4px solid" in str(m.value)]
    assert len(cards) == 4, len(cards)
    for card in cards:
        assert "font-size:1.45rem" in card            # the value slot is filled


def test_overview_names_the_state_chart_as_monitoring_only(project_api):
    """The approved model excludes geography on purpose; the chart must not read as
    an account of a driver (FINDINGS 7c, 7l)."""
    at, _ = _overview(project_api)
    said = " ".join(str(c.value) for c in at.caption)
    run_id = at.selectbox("ov_run").value
    tag = project_api.get(f"/runs/{run_id}").json()["summary"]["model_tag"]
    assert f"`{tag}` does not use state" in said      # the run's own model, named
    assert "fair-lending monitoring" in said
    # and the income band says the different thing it has to say: the model read the
    # number, not the bucket
    assert "sees income as a number, not as these bands" in said


def test_overview_filters_the_history_section(project_api):
    """The panel drives section 1: one day counts fewer runs than the whole window.

    The dates stay inside the widget's own min/max, which come from the first and
    last run on disk -- a value outside them is not a filter a user can choose.
    """
    import datetime as dt

    h = project_api.get("/overview").json()["history"]
    if h["first_day"] == h["last_day"]:
        pytest.skip("every run is on one day, so narrowing cannot show anything")
    at = _run("overview.py")
    before = {m.label: m.value for m in at.metric}
    day = dt.date.fromisoformat(h["first_day"])
    at.date_input("ov_dates").set_value((day, day)).run()
    assert not at.exception, [e.value for e in at.exception]
    after = {m.label: m.value for m in at.metric}
    def count(d, key):
        return int(d[key].replace(",", ""))

    assert count(after, "Runs in window") < count(before, "Runs in window")
    assert count(after, "Applicants decided") <= count(before, "Applicants decided")
    # section 2 is untouched: it shows the run the selector picked, not the window
    assert after["Applicants"] == before["Applicants"]


def test_overview_filters_the_run_section(project_api):
    """The run selector drives section 2, and leaves section 1 where it was."""
    at = _run("overview.py")
    options = at.selectbox("ov_run").options
    if len(options) < 2:
        pytest.skip("needs two runs to switch between")
    first = {m.label: m.value for m in at.metric}
    at.selectbox("ov_run").select(options[1]).run()
    assert not at.exception, [e.value for e in at.exception]
    second = {m.label: m.value for m in at.metric}
    assert second["Applicants decided"] == first["Applicants decided"]


def test_overview_defaults_to_a_finished_run(project_api):
    """A run that failed its checks has no decisions, so it is never the default."""
    at = _run("overview.py")
    chosen = at.selectbox("ov_run").value
    states = {r["run_id"]: r["state"] for r in project_api.get("/runs").json()["runs"]}
    finished = set(project_api.get("/meta").json()["finished_states"])
    if any(s in finished for s in states.values()):
        assert states.get(chosen) in finished, (chosen, states.get(chosen))


def test_overview_numbers_are_the_history_and_the_run(project_api):
    """Nothing invented: section 1 is the API's history, section 2 the run's summary."""
    ov = project_api.get("/overview").json()
    h = ov["history"]
    at, metric = _overview(project_api)
    assert metric["Applicants decided"] == f"{h['applicants_processed']:,}"
    assert metric["Total rejected"] == f"{h['total_rejected']:,}"
    assert metric["Runs to date"] == f"{ov['total_runs']:,}"      # unfiltered
    run_id = at.selectbox("ov_run").value
    s = project_api.get(f"/runs/{run_id}").json()["summary"]
    assert metric["Applicants"] == f"{int(s['n_rows']):,}"
    assert metric["Approved"] == f"{int(s['n_approved']):,}"
    assert metric["Mean risk"] == f"{float(s['mean_pd_36m']):.1%}"


# ------------------------------------------------ the dashboard is a client --

PIPELINE_MODULES = {"batch", "phase2", "registry", "runner", "evidence_jobs",
                    "pipeline", "cleaning", "schema_match", "derive", "plan", "status",
                    "history", "stages", "run_checks", "explain", "models", "features",
                    "config", "api", "drift", "eda"}
"""Everything that scores, explains, governs or reads run folders. A page reaches it
only through the API."""


def test_no_page_imports_the_pipeline():
    offenders = []
    for path in list(VIEWS.glob("*.py")) + [PROJECT_ROOT / "app" / "app.py"]:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for n in names:
                parts = n.split(".")
                if parts[0] == "creditsurv" and len(parts) > 1 \
                        and parts[1] in PIPELINE_MODULES:
                    offenders.append(f"{path.name}: {n}")
    assert not offenders, offenders
