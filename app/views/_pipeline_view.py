"""The scoring pipeline drawn as stages that change state as a run moves.

Check -> Clean -> Score -> Decide -> Drift -> Run checks -> Explain -> Notices, each
a box coloured by its state (pending, running, done, failed, skipped), from the
API's GET /runs/{id}/status -- which reads the stage log the run itself writes.
Used by the Run pipeline page and, while a file is being scored, by Score
applicants.
"""

from __future__ import annotations

import html

import streamlit as st

from _client import ApiError, api
from _common import MUTED, colour, fmt_seconds, kind

ICON = {"ok": "✓", "bad": "✕", "work": "●", "warn": "!", "wait": "○"}
LIVE = ("running", "starting", "explaining")


def stage_boxes(stages: list[dict]) -> None:
    """The stages side by side, wrapping on a narrow screen."""
    cells = []
    for i, s in enumerate(stages):
        state = s.get("state", "pending")
        k = kind(state)
        c = colour(state)
        pulse = "animation:csPulse 1.4s ease-in-out infinite;" if k == "work" else ""
        secs = f" · {s['seconds']:.1f}s" if s.get("seconds") else ""
        msg = html.escape(str(s.get("message") or ""))[:140]
        cells.append(
            f"<div style='flex:1 1 118px;min-width:118px;border:1.5px solid {c};"
            f"border-radius:10px;padding:.55rem .6rem;background:{c}0f;{pulse}'>"
            f"<div style='font-size:.7rem;color:{MUTED};letter-spacing:.06em;'>"
            f"STEP {i + 1}</div>"
            f"<div style='font-weight:700;color:#1c2430;'>{html.escape(s['label'])}</div>"
            f"<div style='color:{c};font-weight:600;font-size:.82rem;'>"
            f"{ICON[k]} {html.escape(state)}{secs}</div>"
            f"<div style='color:{MUTED};font-size:.74rem;line-height:1.25;"
            f"margin-top:.2rem;'>{msg}</div></div>")
        if i < len(stages) - 1:
            cells.append(f"<div style='align-self:center;color:{MUTED};'>→</div>")
    st.markdown(
        "<style>@keyframes csPulse{0%{opacity:1}50%{opacity:.62}100%{opacity:1}}</style>"
        f"<div style='display:flex;gap:.35rem;flex-wrap:wrap;margin:.3rem 0 .8rem 0;'>"
        + "".join(cells) + "</div>", unsafe_allow_html=True)


def phase2_bar(status: dict) -> None:
    p2 = status.get("phase2") or {}
    target = int(p2.get("target") or 0)
    if not target or p2.get("state") in (None, "", "not needed"):
        return
    done = int(p2.get("done") or 0)
    rate = p2.get("rate_per_min")
    st.progress(min(done / max(target, 1), 1.0),
                text=f"Phase 2: {done:,} of {target:,} rejected applicants explained"
                     + (f" · {rate:,.1f}/min" if rate else "")
                     + (f" · about {fmt_seconds(p2.get('eta_seconds'))} left"
                        if p2.get("state") == "running" else ""))


def is_live(status: dict) -> bool:
    p2 = (status.get("phase2") or {}).get("state")
    return status.get("state") in LIVE or p2 in ("running", "not started", "finishing")


def follow(run_id: str, *, log_lines: int = 60, on_finish=None,
           show_log: bool = True) -> dict | None:
    """Draw the run's stages, Phase 2 progress and live log; redraw every 2 s while
    the run moves, then once more when it stops (``on_finish``: rerun the app)."""
    try:
        first = api().get(f"/runs/{run_id}/status", log_lines=log_lines)
    except ApiError as exc:
        exc.show()
        return None

    @st.fragment(run_every=2 if is_live(first) else None)
    def draw():
        try:
            s = api().get(f"/runs/{run_id}/status", log_lines=log_lines)
        except ApiError as exc:
            exc.show()
            return
        stage_boxes(s["stages"])
        phase2_bar(s)
        if show_log:
            st.caption("Live output" + (", refreshed every 2 s" if is_live(s) else "")
                       + ":")
            st.code(s.get("log") or "(no output yet)", language="text", height=260)
        if is_live(first) and not is_live(s):
            if on_finish:
                on_finish()
            else:
                st.rerun(scope="app")

    draw()
    return first
