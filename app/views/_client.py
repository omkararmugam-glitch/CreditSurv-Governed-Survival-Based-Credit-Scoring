"""The dashboard's only way to the pipeline: HTTP calls to the creditsurv API.

No page imports the scoring, explanation, registry or runner code; they ask the
API, which calls it (tests/test_app.py fails if a page imports it). The API's
address is ``CREDITSURV_API_URL`` (default http://127.0.0.1:8000). A test can hand
:func:`use` an in-process client instead, so every page is drawn against a real
API without a server.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import httpx
import pandas as pd
import streamlit as st

__all__ = ["ApiError", "api", "use", "overridden", "frame", "base_url", "run_view",
           "API_DOWN_FIX"]

API_DOWN_FIX = ("Start the API and the dashboard together: "
                "powershell -ExecutionPolicy Bypass -File .\\run_linux.ps1 "
                "(or, where lightgbm is not blocked: python -m uvicorn "
                "creditsurv.api.app:app --port 8000).")

_OVERRIDE: dict = {}


def use(client) -> None:
    """Point the dashboard at ``client`` (an httpx.Client, e.g. FastAPI's
    TestClient); ``None`` goes back to the real address."""
    if client is None:
        _OVERRIDE.pop("client", None)
    else:
        _OVERRIDE["client"] = client


def overridden() -> bool:
    return "client" in _OVERRIDE


def base_url() -> str:
    return os.environ.get("CREDITSURV_API_URL", "http://127.0.0.1:8000").rstrip("/")


class ApiError(Exception):
    """A refusal or failure, with the API's own message, detail and fix."""

    def __init__(self, status: int, message: str, detail: str = "", fix: str = "",
                 body: dict | None = None):
        super().__init__(message)
        self.status, self.message, self.detail, self.fix = status, message, detail, fix
        self.body = body or {}

    def show(self) -> None:
        """Draw it the one way every page draws a refusal."""
        st.error(f"**{self.message}**" + (f"\n\n{self.fix}" if self.fix else ""),
                 icon=":material/error:")
        if self.detail and self.detail != self.message:
            with st.expander("Details"):
                st.code(self.detail, language="text")


class Api:
    def __init__(self, client: httpx.Client):
        self.client = client

    def _call(self, method: str, path: str, **kw):
        try:
            r = self.client.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise ApiError(0, f"The creditsurv API is not reachable at {base_url()}.",
                           repr(exc), API_DOWN_FIX) from exc
        if r.status_code >= 400:
            try:
                body = r.json()
            except ValueError:
                body = {"message": r.text[:500]}
            if isinstance(body.get("detail"), list):          # FastAPI validation
                body = {"message": "The request was not accepted.",
                        "detail": str(body["detail"])}
            raise ApiError(r.status_code, body.get("message") or f"HTTP {r.status_code}",
                           str(body.get("detail") or ""), body.get("fix") or "", body)
        return r

    def get(self, path: str, **params):
        return self._call("GET", path, params=params or None).json()

    def post(self, path: str, json: dict | None = None, **kw):
        return self._call("POST", path, json=json, **kw).json()

    def delete(self, path: str):
        return self._call("DELETE", path).json()

    def raw(self, path: str) -> bytes:
        return self._call("GET", path).content

    def upload(self, name: str, data) -> dict:
        return self._call("POST", "/uploads",
                          files={"file": (name, data, "text/csv")}).json()


@st.cache_resource(show_spinner=False)
def _http(url: str) -> httpx.Client:
    # Long read timeout: checking a file's columns loads the model the first time.
    return httpx.Client(base_url=url, timeout=httpx.Timeout(10.0, read=600.0))


def api() -> Api:
    client = _OVERRIDE.get("client")
    return Api(client if client is not None else _http(base_url()))


def frame(payload: dict | None) -> pd.DataFrame:
    """A frame the API sent in pandas' split orientation, rebuilt exactly."""
    if not payload:
        return pd.DataFrame()
    return pd.DataFrame(payload.get("data") or [], columns=payload.get("columns") or [])


def run_view(detail: dict) -> SimpleNamespace:
    """The run the result view draws, from GET /runs/{id}: the same fields the
    dashboard used to read from the run folder, now from the API."""
    drift = detail.get("drift")
    cleaning = detail.get("cleaning") or {}
    agg = detail.get("aggregates")
    prof = detail.get("profile") or {}
    return SimpleNamespace(
        run_id=detail["run_id"], state=detail["state"], error=detail.get("error"),
        summary=detail.get("summary") or {}, files=detail.get("files") or [],
        status=detail.get("status") or {}, validation=detail.get("validation") or {},
        warnings=detail.get("warnings") or [],
        checks=frame(detail.get("checks")), preview=frame(detail.get("preview")),
        cleaning_sentences=cleaning.get("sentences") or [],
        cleaning_table=frame(cleaning.get("table")),
        drift=None if not drift else SimpleNamespace(
            status=drift["status"], colour=drift["colour"], headline=drift["headline"],
            table=frame(drift["table"])),
        aggregates=None if not agg else SimpleNamespace(
            group_column=agg["group_column"], risk=frame(agg["risk"]),
            groups=frame(agg["groups"]), reasons=frame(agg["reasons"]),
            segment_columns=list(agg.get("segment_columns") or []),
            segment_order=dict(agg.get("segment_order") or {}),
            segments=frame(agg.get("segments"))),
        profile={"overview": prof.get("overview") or {},
                 **{k: frame(prof.get(k)) for k in
                    ("missing", "numeric", "categorical", "outliers")}})
