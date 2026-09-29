"""Helpers shared by the pages: palette, header, config. Read-only."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from creditsurv.config import load_config
from creditsurv.environment import runtime_label
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


# ------------------------------------------------ things worth keeping loaded --
# Loading a model means unpickling the bundle and reading its whole training file
# to draw the SHAP background: ~2-3 s and a ~3 GB peak each time. Without a cache
# the upload page did it twice per file (preview, then scoring). With one, it is
# done once per model per server, and reused by every upload and every click.

@st.cache_resource(max_entries=2, show_spinner="Loading the model (once; it stays "
                                               "in memory for later uploads)...")
def _cached_context(config_path: str, tag: str, model_name: str, key: tuple):
    from creditsurv.batch import load_context

    return load_context(load_config(config_path), tag, model_name)


def cached_context(tag: str, model_name: str, config_path=CONFIG_PATH):
    """The loaded model, from memory when it is already there.

    Keyed on :func:`creditsurv.batch.context_key` -- the model file, its cleaning
    values, its training data and the settings -- so replacing any of them loads
    afresh rather than serving a stale model.
    """
    from creditsurv.batch import context_key

    cfg = load_config(config_path)
    return _cached_context(str(config_path), tag, model_name,
                           context_key(cfg, tag, model_name))


@st.cache_data(max_entries=4, show_spinner=False)
def model_statuses(config_path: str, models: tuple, registry_mtime: int) -> dict:
    """Registry status per model for the model picker, re-read when the registry
    file changes -- not on every click."""
    from creditsurv.registry import assess

    cfg = load_config(config_path)
    return {m: assess(m, cfg).label for m in models}


@st.cache_resource(max_entries=4, show_spinner=False)
def _cached_result(run_dir: str, provenance_mtime: int):
    from creditsurv.batch import load_result

    return load_result(Path(run_dir))


def cached_result(run_dir):
    """A finished run's result, read once and re-read only when the run writes
    (Phase 2 updates provenance.json at each checkpoint)."""
    prov = Path(run_dir) / "provenance.json"
    return _cached_result(str(run_dir), prov.stat().st_mtime_ns if prov.exists() else 0)


# ------------------------------------------- what survives leaving the page --
# Streamlit forgets a page's widgets -- the uploader's file included -- as soon as
# another page is shown. The upload and the choices made around it are therefore
# kept in session state here, so coming back shows the same file and result.

HELD_DIR = PROJECT_ROOT / "outputs" / "runs" / "_uploads"


class HeldUpload:
    """An upload copied to disk once, read back on every rerun. Offers the parts of
    Streamlit's UploadedFile the page uses, without keeping a 450 MB file in memory
    for as long as the session lasts."""

    def __init__(self, up, held_dir: Path = HELD_DIR):
        import re

        self.name, self.size, self.file_id = up.name, up.size, up.file_id
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{up.file_id}_{Path(up.name).name}")
        held_dir.mkdir(parents=True, exist_ok=True)
        self.path = held_dir / safe[:120]
        with open(self.path, "wb") as fh:
            fh.write(up.getbuffer())

    def getvalue(self) -> bytes:
        return self.path.read_bytes()

    def head(self, n: int) -> bytes:
        with open(self.path, "rb") as fh:
            return fh.read(n)

    def discard(self) -> None:
        self.path.unlink(missing_ok=True)


def hold_upload(up) -> "HeldUpload | None":
    """The file this session is working on: a new upload replaces the held one
    (and deletes its copy); no upload keeps what was held."""
    held = st.session_state.get("held_upload")
    if up is not None and (held is None or held.file_id != up.file_id
                           or not held.path.exists()):
        if held is not None:
            held.discard()
        held = st.session_state["held_upload"] = HeldUpload(up)
    return held


def drop_upload() -> None:
    held = st.session_state.pop("held_upload", None)
    if held is not None:
        held.discard()
    st.session_state.pop("mapping_base", None)


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


def page_header(title: str, subtitle: str = "") -> None:
    """The identity every page shares, above its own content.

    Carries a label saying where the server runs, because a Windows instance and a
    WSL instance of this app look identical and only one of them can score.
    """
    where = runtime_label()
    tint = APPROVE if where == "Linux (WSL)" else REJECT if where == "Windows" else MUTED
    st.markdown(
        f"<div style='border-bottom:1px solid #d7dce5;margin-bottom:1.1rem;'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
        f"<div style='color:{ACCENT};font-weight:700;letter-spacing:.14em;"
        f"font-size:.72rem;text-transform:uppercase;'>creditsurv &nbsp;·&nbsp; "
        f"survival-based credit scoring</div>"
        f"<div title='The machine this app is running on' style='color:{tint};"
        f"border:1px solid {tint};border-radius:999px;padding:.05rem .6rem;"
        f"font-size:.72rem;font-weight:600;white-space:nowrap;'>"
        f"Running on {where}</div></div>"
        f"<h1 style='margin:.1rem 0 .3rem 0;font-size:1.9rem;'>{title}</h1>"
        f"<p style='color:{MUTED};margin:0 0 .9rem 0;'>{subtitle}</p></div>",
        unsafe_allow_html=True)
    code_freshness()


SYNC_NOTE = PROJECT_ROOT / "outputs" / "logs" / "last_code_sync.json"


def _session_id() -> str:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        return ctx.session_id if ctx else "bare"
    except Exception:
        return "bare"


def code_freshness(root: Path = PROJECT_ROOT, restart=None) -> None:
    """In the WSL copy: say when Windows has newer code, and sync it where safe.

    The rules are :func:`creditsurv.wsl_sync.auto_sync`, the same ones the
    server-side watcher (started in app.py) applies every 20 s: no background job
    or Phase 2 running, and no browser session with work open. This page reports
    its own session's state for the watcher, syncs straight away when it is safe,
    and otherwise says why not and offers a button. Never serves stale code
    silently.
    """
    import json
    import time

    from creditsurv.wsl_sync import (auto_sync, note_session, stale_files,
                                     sync_blockers, windows_source)

    source = windows_source(root)
    if source is None:
        return
    open_work = bool(st.session_state.get("upload_runs") or st.session_state.get("open_run")
                     or st.session_state.get("held_upload"))
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
                   f"Not syncing while {', '.join(busy)} runs: it would finish on old "
                   f"code with new code arriving under it. It syncs by itself once "
                   f"that is done; until then this page serves the code it started "
                   f"with.", icon=":material/sync_problem:")
        return
    if open_work:
        st.warning(f"**Windows has newer code** than this app is running ({examples}). "
                   f"Syncing restarts the app (every open tab reconnects); your run "
                   f"stays on disk (Open a previous run).", icon=":material/sync_problem:")
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


def rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(p)
