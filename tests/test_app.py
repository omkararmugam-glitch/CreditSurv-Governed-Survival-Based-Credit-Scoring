"""Every page renders without an exception. Read-only: no test here presses
"Start run", so no stage is launched."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from creditsurv.provenance import PROJECT_ROOT

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

VIEWS = PROJECT_ROOT / "app" / "views"


@pytest.fixture(autouse=True)
def _paths(monkeypatch):
    monkeypatch.syspath_prepend(str(VIEWS))
    monkeypatch.chdir(PROJECT_ROOT)


def _run(page: str) -> AppTest:
    at = AppTest.from_file(str(VIEWS / page), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _run_in_app(page: str) -> AppTest:
    """A page opened through app.py, as a user reaches it: page links to other
    pages resolve only inside the app's navigation."""
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py"), default_timeout=120)
    at.run()
    at.switch_page(f"views/{page}")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_entrypoint_loads_home_page():
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Upload applicant dataset" in w.label for w in at.get("file_uploader"))


@pytest.mark.parametrize("page", ["home.py", "overview.py", "results.py",
                                  "findings.py"])
def test_read_only_pages_render(page):
    _run(page)


def test_header_says_where_the_app_runs():
    """A Windows and a WSL instance of the app look identical otherwise."""
    from creditsurv.environment import runtime_label

    at = _run("findings.py")
    assert any(f"Running on {runtime_label()}" in m.value for m in at.markdown)


def test_run_page_defaults_are_safe():
    at = _run_in_app("run.py")
    assert at.radio[0].value == "Small"
    assert at.checkbox[0].value is False                   # --overwrite off
    start = next(b for b in at.button if b.label == "Start run")
    assert start is not None


def test_run_page_refuses_primary_tag():
    at = _run_in_app("run.py")
    at.text_input[0].set_value("full").run()
    assert any("Tag refused" in e.value for e in at.error)
    assert not any(b.label == "Start run" for b in at.button)


def test_home_shows_only_the_drop_zone_before_upload():
    at = _run("home.py")
    assert len(at.get("file_uploader")) == 1
    # The decision rule is stated, and no pipeline controls are on the main page.
    assert any("default probability" in c.value for c in at.caption)
    assert not at.dataframe and not at.button


def test_home_background_switch_is_off_by_default():
    at = _run("home.py")
    labels = [c.label for c in at.checkbox]
    assert "Always run in the background" in labels
    assert at.checkbox[labels.index("Always run in the background")].value is False


# ------------------------------------------------ the finished-run dashboard --

def _finished_run(tmp_path, *, status="approved", allow_unapproved=False,
                  threshold=None):
    """A real run directory, produced by the stub-model fixtures in test_batch."""
    import test_batch as tb

    cfg = tb.Config(paths=tb.Paths(data_dir=tmp_path / "data",
                                   models_dir=tmp_path / "models",
                                   figures_dir=tmp_path / "figures",
                                   tables_dir=tmp_path / "tables",
                                   registry=tmp_path / "models.yaml"),
                    decision=tb.DecisionConfig(
                        model_tag="stub", horizon_months=36,
                        reject_at_or_above=0.45,
                        explain_nsamples=2 * len(tb.NUMERIC + tb.CATEGORICAL),
                        explain_n_background=8, max_explained=3,
                        background_rows=200, explain_workers=1))
    train = tb._training_frame()
    dm = tb.build_design_matrix(train, tb.SPEC, flavour="gbm")
    model_path = tb._write_dummy_model(cfg)
    tb.write_registry(cfg, model_path, status=status)
    ctx = tb.ScoringContext(
        cfg=cfg, model_tag="stub", model_name="discrete_hazard", model=tb.StubModel(),
        spec=tb.SPEC, bundle={"artefacts": {"gbm_columns": list(dm.X.columns)}},
        model_path=model_path, background=dm.X, reference=train,
        clean_values=tb.fit_values(train, tb.SPEC, source="stub.parquet"),
        policy=tb.policy_from_config(cfg),
        times=np.array([6.0, 12.0, 24.0, 36.0]), data_source=tb._dummy_source(cfg))
    return tb.run_batch(tb._upload(30), "applicants.csv", cfg, ctx=ctx,
                        runs_dir=tmp_path / "runs", threshold=threshold,
                        allow_unapproved_model=allow_unapproved)


def test_dashboard_renders_every_panel_and_tab(tmp_path):
    """Draws the whole result view of a finished run, including every tab and chart.

    This is the test that would have caught the NameError in the Data profile tab:
    that panel only appears after a scoring run, so nothing short of rendering it
    finds a missing constant.
    """
    result = _finished_run(tmp_path)
    run_dir = result.run_dir

    def script(views, src, run):
        import sys
        sys.path.insert(0, views)
        sys.path.insert(0, src)
        import streamlit as st
        from creditsurv.batch import load_result
        from _dashboard import render

        render(load_result(run))
        st.session_state["rendered"] = True

    at = AppTest.from_function(
        script, default_timeout=180,
        kwargs={"views": str(VIEWS), "src": str(PROJECT_ROOT / "src"),
                "run": str(run_dir)})
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["rendered"] is True

    # Every panel of the view is present.
    assert len(at.tabs) >= 3                                  # the three tabs
    assert len(at.metric) >= 5                                # headline cards
    assert len(at.dataframe) >= 4                             # preview + tab tables
    assert len(at.get("vega_lite_chart")) >= 3          # risk, purpose, reasons
    labels = [b.label for b in at.get("download_button")]
    assert "⬇ Download all (ZIP)" in labels
    assert any("scored_applicants.csv" in b for b in labels)
    text = " ".join(str(m.value) for m in at.markdown) + " ".join(
        str(c.value) for c in at.caption)
    assert "Cleaning policy" in text                          # cleaning tab body
    assert "population stability index" in text                # drift tab body
    assert "Counts are shown beside each bar" in text          # the small-group rule


def test_dashboard_survives_a_run_with_no_rejections(tmp_path):
    """The panels that describe rejections must not assume there are any."""
    import test_batch as tb

    result = _finished_run(tmp_path)
    # Rewrite the run's summary so every applicant was approved, then re-render.
    import json
    prov = result.run_dir / "provenance.json"
    payload = json.loads(prov.read_text(encoding="utf-8"))
    payload["summary"].update(n_rejected=0, n_approved=payload["summary"]["n_rows"],
                              n_explained=0, n_notices=0,
                              n_rejected_without_reasons=0, approval_rate=1.0)
    prov.write_text(json.dumps(payload), encoding="utf-8")
    (result.run_dir / "adverse_action_notices.zip").unlink(missing_ok=True)
    run_dir = result.run_dir

    def script(views, src, run):
        import sys
        sys.path.insert(0, views)
        sys.path.insert(0, src)
        from creditsurv.batch import load_result
        from _dashboard import render

        render(load_result(run))

    at = AppTest.from_function(
        script, default_timeout=180,
        kwargs={"views": str(VIEWS), "src": str(PROJECT_ROOT / "src"),
                "run": str(run_dir)})
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("not produced" in str(c.value) for c in at.caption)


# ------------------------------------------------ surviving a page change --

class _FakeUpload:
    def __init__(self, data: bytes, name="a.csv", file_id="f1"):
        self._data, self.name, self.size, self.file_id = data, name, len(data), file_id

    def getbuffer(self):
        return memoryview(self._data)


def test_held_upload_is_read_back_from_disk(tmp_path):
    from _common import HeldUpload

    held = HeldUpload(_FakeUpload(b"id,x\n1,2\n"), held_dir=tmp_path)
    assert held.getvalue() == b"id,x\n1,2\n"
    assert held.head(4) == b"id,x"
    held.discard()
    assert not held.path.exists()


def test_options_come_back_after_the_page_was_left():
    """Streamlit drops a page's widget keys when another page is shown; remember()
    restores them from the copy keep() made."""
    def page():
        import streamlit as st
        from _common import keep, remember

        if st.session_state.get("_leave"):          # stands in for another page
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


def test_run_page_follows_only_pipeline_runs():
    at = _run_in_app("run.py")
    for sb in at.selectbox:
        if sb.label == "Run":
            assert not any("explain" in o or "drifted" in o for o in sb.options)
