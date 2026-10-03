"""What every page shares: the palette, the status colours, the header, and the
settings the pages keep across a rerun. Presentation only -- the pipeline is behind
the API (_client.py)."""

from __future__ import annotations

import html
import math
from pathlib import Path

import streamlit as st

from _client import ApiError, api, base_url, overridden

ROOT = Path(__file__).resolve().parents[2]

# One palette for every chart and every badge in the app; the interface colours are
# in .streamlit/config.toml.
APPROVE = "#2e7d5b"
REJECT = "#b3402f"
ACCENT = "#2f5d8a"
MUTED = "#6b7280"
AMBER = "#b7791f"

# ----------------------------------------------------------- status colours --
# Every state any page shows maps to one of five meanings, drawn the same way
# everywhere: green = done / passed, red = failed / refused, blue = working,
# amber = needs a look (overridden, stopped, waiting on a choice), grey = not yet.
_KIND = {
    "done": "ok", "pass": "ok", "passed": "ok", "finished": "ok", "completed": "ok",
    "verified": "ok", "approved": "ok", "stable": "ok",
    "failed": "bad", "fail": "bad", "failed checks": "bad", "refused": "bad",
    "changed": "bad", "large": "bad", "deprecated": "bad",
    "running": "work", "starting": "work", "explaining": "work", "finishing": "work",
    "queued": "work",
    "overridden": "warn", "interrupted": "warn", "phase 2 stopped": "warn",
    "reasons pending": "warn", "awaiting phase 2 choice": "warn", "awaiting choice": "warn",
    "unverified": "warn", "moderate": "warn", "candidate": "warn", "stopped": "warn",
    "not started": "wait", "pending": "wait", "not run": "wait", "skipped": "wait",
    "benchmark": "wait", "incomplete": "wait", "insufficient": "wait", "unknown": "warn",
}
KIND_COLOUR = {"ok": APPROVE, "bad": REJECT, "work": ACCENT, "warn": AMBER, "wait": MUTED}
KIND_DOT = {"ok": "🟢", "bad": "🔴", "work": "🔵", "warn": "🟠", "wait": "⚪"}


def kind(state) -> str:
    return _KIND.get(str(state or "").strip().lower(), "wait")


def dot(state) -> str:
    """The one-character status mark used in tables."""
    return KIND_DOT[kind(state)]


def colour(state) -> str:
    return KIND_COLOUR[kind(state)]


def badge(state, text: str | None = None) -> str:
    """An inline status pill, HTML. Use with st.markdown(..., unsafe_allow_html=True)."""
    c = colour(state)
    return (f"<span style='display:inline-block;border:1px solid {c};color:{c};"
            f"background:{c}14;border-radius:999px;padding:.05rem .55rem;"
            f"font-size:.78rem;font-weight:600;white-space:nowrap;'>"
            f"{html.escape(str(text if text is not None else state))}</span>")


def status_box(state, message: str) -> None:
    """A whole-width message in the colour of its state."""
    {"ok": st.success, "bad": st.error, "work": st.info, "warn": st.warning,
     "wait": st.info}[kind(state)](message)


# -------------------------------------------------------------- the header --

@st.cache_data(ttl=30, show_spinner=False)
def _meta(url: str) -> dict:
    return api().get("/meta")          # a failure raises, and is not cached


def meta() -> dict:
    """The API's settings and fixed wording; stops the page if the API is down."""
    try:
        return api().get("/meta") if overridden() else _meta(base_url())
    except ApiError as exc:
        exc.show()
        st.stop()


def page_header(title: str, subtitle: str = "") -> dict:
    """The identity every page shares, above its own content. Says where the API
    runs -- scoring works only where lightgbm loads -- and returns /meta."""
    m = meta()
    where = m.get("runtime", "?")
    tint = APPROVE if not m.get("blocked_imports") else REJECT
    st.markdown(
        f"<div style='border-bottom:1px solid #d7dce5;margin-bottom:1.1rem;'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"gap:.5rem;flex-wrap:wrap;'>"
        f"<div style='color:{ACCENT};font-weight:700;letter-spacing:.14em;"
        f"font-size:.72rem;text-transform:uppercase;'>creditsurv &nbsp;·&nbsp; "
        f"survival-based credit scoring</div>"
        f"<div title='Where the API that does the work is running' style='color:{tint};"
        f"border:1px solid {tint};border-radius:999px;padding:.05rem .6rem;"
        f"font-size:.72rem;font-weight:600;white-space:nowrap;'>"
        f"API on {html.escape(where)}</div></div>"
        f"<h1 style='margin:.1rem 0 .3rem 0;font-size:1.9rem;'>{html.escape(title)}</h1>"
        f"<p style='color:{MUTED};margin:0 0 .9rem 0;'>{html.escape(subtitle)}</p></div>",
        unsafe_allow_html=True)
    if m.get("blocked_imports"):
        # The API's own explanation (Smart App Control, and what still works).
        st.error(m.get("blocked_message") or (
            "The API cannot score here: " + ", ".join(m["blocked_imports"])
            + " cannot be loaded on this machine. Start it in WSL with run_linux.ps1."),
            icon=":material/gpp_bad:")
    code_freshness()
    return m


# -------------------------------------------- what survives leaving the page --
# Streamlit forgets a page's widgets as soon as another page is shown. The choices
# made on a page are kept in session state here, so coming back shows the same
# ones. The upload itself and the run are held by the API; the page keeps only ids.

def remember(key: str, default=None):
    """Give widget ``key`` back the value it had before the page was left. Call it
    before drawing the widget (which then reads st.session_state[key]), and
    :func:`keep` after, which copies the value into a key Streamlit does not clear."""
    shadow = f"_kept_{key}"
    if key not in st.session_state:
        st.session_state[key] = st.session_state.get(shadow, default)
    return st.session_state[key]


def keep(*keys: str) -> None:
    for k in keys:
        if k in st.session_state:
            st.session_state[f"_kept_{k}"] = st.session_state[k]


# ------------------------------------------------ code sync in the WSL copy --

def _session_id() -> str:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        return ctx.session_id if ctx else "bare"
    except Exception:
        return "bare"


def code_freshness(root: Path = ROOT, restart=None) -> None:
    """In the WSL copy: say when Windows has newer code, and sync it where safe.

    The rules are :func:`creditsurv.wsl_sync.auto_sync`, the same ones the
    server-side watcher (started in app.py) applies every 20 s: no background job
    or Phase 2 running, the API not scoring, and no browser session with work
    open. This is the dashboard server keeping its own code current; it touches no
    pipeline code.
    """
    import json
    import time

    from creditsurv.wsl_sync import (auto_sync, note_session, stale_files,
                                     sync_blockers, windows_source)

    source = windows_source(root)
    if source is None:
        return
    open_work = bool(st.session_state.get("upload_id") or st.session_state.get("run_id"))
    note_session(_session_id(), open_work, root)
    try:
        note = json.loads((root / "outputs" / "logs" / "last_code_sync.json")
                          .read_text(encoding="utf-8"))
        if time.time() - note.get("at", 0) < 180:
            st.success(f"Synced {note['files']} newer file(s) from Windows at "
                       f"{note['when']} and restarted on the new code.",
                       icon=":material/sync:")
    except (OSError, ValueError, KeyError):
        pass
    stale = stale_files(root, source)
    if not stale:
        return
    examples = ", ".join(stale[:3]) + (f" and {len(stale) - 3} more" if len(stale) > 3
                                       else "")
    busy = sync_blockers(root)
    if busy:
        st.warning(f"**Windows has newer code** than this app is running ({examples}). "
                   f"Not syncing while {', '.join(busy)} runs. It syncs by itself once "
                   f"that is done.", icon=":material/sync_problem:")
        return
    if open_work:
        st.warning(f"**Windows has newer code** than this app is running ({examples}). "
                   f"Syncing restarts the dashboard (every open tab reconnects); your run "
                   f"is held by the API and stays.", icon=":material/sync_problem:")
        if not st.button("Sync from Windows and restart", key="sync_code_now"):
            return
    with st.spinner(f"Windows has newer code ({examples}); syncing it into {root}..."):
        what = auto_sync(root, open_work=False, restart=restart, source=source)
    if what.startswith("failed"):
        st.error("**The sync from Windows failed**, so this app is still serving the "
                 "older code.", icon=":material/sync_problem:")
        st.code(what[-3000:], language="text")
    elif what.startswith("blocked"):
        st.warning(f"Not synced: {what}.", icon=":material/sync_problem:")


# ------------------------------------------------------------ the KPI cards --
# ``st.metric`` never wraps: the front end renders both the label and the value with
# ``truncate`` set, so anything wider than the card is cut to an ellipsis. Six
# ``st.columns`` on a laptop leave each card about 150 px, which is narrower than
# most of this app's labels -- hence "Appli...", "Overa...", "34...". Columns cannot
# fix that, because they only stack below 640 px; they just keep shrinking.
#
# So every KPI row in the app is drawn by :func:`kpi_row`: a wrapping horizontal
# container whose cards each carry a pixel width wide enough for their own text.
# A card in a horizontal container with a pixel width gets ``flex: 0 0 Npx``, which
# cannot shrink, and the row wraps onto a second line once the window runs out.
# Fewer cards per row, never a cut label or a cut number.

KPI_LABEL_PX = 14.0      # theme fontSizes.sm -- the metric label
KPI_VALUE_PX = 36.0      # theme fontSizes.metricValueFontSize (2.25rem)
KPI_CARD_PADDING = 56    # the bordered card's own padding, both sides
KPI_HELP_PX = 26         # the "?" a help tooltip adds beside the label
KPI_MIN_WIDTH = 240      # the floor, so an ordinary row stays even
KPI_STEP = 20            # widths round up to this, so near-equal cards come out equal

# Per-character widths as a fraction of the font size, rounded up rather than
# measured: too wide only costs a card per row, too narrow puts the ellipsis back.
_WIDE_CHARS = set("ABCDEFGHJKLMNOPQRSTUVWXYZmwMW%&@")
_NARROW_CHARS = set("ijltfrI.,:;'`|!()[] ")


def text_px(text, font_px: float) -> float:
    """A deliberately generous width for one line of the interface font."""
    em = 0.0
    for ch in str(text):
        em += 0.36 if ch in _NARROW_CHARS else 0.86 if ch in _WIDE_CHARS else 0.62
    return em * font_px


def kpi_width(label, value, has_help: bool = False,
              min_width: int = KPI_MIN_WIDTH) -> int:
    """How wide a metric card has to be for this label and this value to fit."""
    need = max(text_px(label, KPI_LABEL_PX) + (KPI_HELP_PX if has_help else 0),
               text_px(value, KPI_VALUE_PX)) + KPI_CARD_PADDING
    return max(int(min_width), int(math.ceil(need / KPI_STEP) * KPI_STEP))


def kpi_row(items, min_width: int = KPI_MIN_WIDTH) -> None:
    """One row of KPI cards that wraps rather than truncating.

    ``items`` are ``(label, value)`` or ``(label, value, help)``. Every card is
    sized to its own text, so a long value widens that card alone and the row
    wraps; the rest stay at ``min_width`` and the row reads as a row.
    """
    with st.container(horizontal=True, wrap=True, gap="small"):
        for label, value, *rest in items:
            tip = rest[0] if rest else None
            st.metric(label, value, help=tip, border=True,
                      width=kpi_width(label, value, bool(tip), min_width))


def card_row(cards: list[str], width: int = 300) -> None:
    """The same wrapping row for the hand-drawn HTML cards (the insights).

    Their text wraps, so they were never cut the way a metric is -- but four of
    them in four columns leaves each one about 270 px, which is a headline over
    three lines. Same row, same rule: a width each, and wrap.
    """
    with st.container(horizontal=True, wrap=True, gap="small"):
        for card in cards:
            with st.container(width=int(width)):
                st.markdown(card, unsafe_allow_html=True)


def nav_link(page: str, label: str, icon: str | None = None) -> None:
    """A link to another page of the app; plain text where the page is drawn on its
    own (a test), since page links resolve only inside app.py's navigation."""
    try:
        st.page_link(page, label=label, icon=icon)
    except Exception:
        st.caption(f"→ {label} (in the menu)")


def fmt_seconds(seconds) -> str:
    if seconds is None:
        return "working it out"
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def when(stamp) -> str:
    """``2026-09-28T17:36:35`` -> ``28 Sep 17:36``."""
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(stamp)).strftime("%d %b %H:%M")
    except ValueError:
        return str(stamp or "")
