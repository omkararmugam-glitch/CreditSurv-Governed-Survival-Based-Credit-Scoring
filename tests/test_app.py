"""Every page renders without an exception. Read-only: no test here presses
"Start run", so no stage is launched."""

from __future__ import annotations

import sys

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


def test_entrypoint_loads_home_page():
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Upload applicant dataset" in w.label for w in at.get("file_uploader"))


@pytest.mark.parametrize("page", ["home.py", "overview.py", "results.py",
                                  "findings.py"])
def test_read_only_pages_render(page):
    _run(page)


def test_run_page_defaults_are_safe():
    at = _run("run.py")
    assert at.radio[0].value == "Small"
    assert at.checkbox[0].value is False                   # --overwrite off
    start = next(b for b in at.button if b.label == "Start run")
    assert start is not None


def test_run_page_refuses_primary_tag():
    at = _run("run.py")
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
