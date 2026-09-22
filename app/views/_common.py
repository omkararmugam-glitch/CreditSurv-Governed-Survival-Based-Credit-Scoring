"""Helpers shared by the pages: palette, header, config. Read-only."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from creditsurv.config import load_config
from creditsurv.provenance import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# One palette for every chart in the app; the interface colours are in
# .streamlit/config.toml.
APPROVE = "#2e7d5b"
REJECT = "#b3402f"
ACCENT = "#2f5d8a"
MUTED = "#6b7280"

STATE_ICON = {"verified": "🟢", "unverified": "🟠", "changed": "🔴", "not run": "⚪",
              "pending": "⚪", "running": "🔵", "starting": "🔵", "completed": "🟢",
              "failed": "🔴", "interrupted": "🟠"}


@st.cache_resource
def paths():
    return load_config(CONFIG_PATH).paths


def page_header(title: str, subtitle: str = "") -> None:
    """The identity every page shares, above its own content."""
    st.markdown(
        f"<div style='border-bottom:1px solid #d7dce5;margin-bottom:1.1rem;'>"
        f"<div style='color:{ACCENT};font-weight:700;letter-spacing:.14em;"
        f"font-size:.72rem;text-transform:uppercase;'>creditsurv &nbsp;·&nbsp; "
        f"survival-based credit scoring</div>"
        f"<h1 style='margin:.1rem 0 .3rem 0;font-size:1.9rem;'>{title}</h1>"
        f"<p style='color:{MUTED};margin:0 0 .9rem 0;'>{subtitle}</p></div>",
        unsafe_allow_html=True)


def rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(p)
