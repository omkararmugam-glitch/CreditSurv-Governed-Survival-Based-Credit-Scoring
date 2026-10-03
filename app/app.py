"""Streamlit front end for creditsurv: a client of the creditsurv API.

The pages call the API over HTTP (app/views/_client.py) and import none of the
scoring, explanation, registry or runner code; the API calls those, the same
functions scripts/06_score_upload.py calls. Start both together:

    powershell -ExecutionPolicy Bypass -File .\\run_linux.ps1        # in WSL

or, where lightgbm is not blocked, in two terminals from the project root:

    python -m uvicorn creditsurv.api.app:app --host 127.0.0.1 --port 8000
    python -m streamlit run app/app.py

``CREDITSURV_API_URL`` points the dashboard at an API elsewhere.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "app" / "views"):     # page helpers (+ wsl_sync)
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.chdir(ROOT)

import streamlit as st  # noqa: E402

st.set_page_config(page_title="creditsurv", layout="wide")

# In the WSL copy, keep this dashboard's own code up to date with edits made on
# Windows: a watcher thread (once per server process) syncs and restarts when that
# is safe. Anywhere else it does nothing.
from creditsurv.wsl_sync import start_watcher  # noqa: E402

start_watcher(ROOT)

VIEWS = ROOT / "app" / "views"
# Everyday use on top; retraining and research below it, where they cannot be
# reached by accident. The app opens on Score applicants, which is the work: before
# a file has been uploaded this session there is nothing of this session to monitor,
# and Overview -- a dashboard over runs already finished -- is one click away.
st.navigation({
    "Operations": [
        st.Page(str(VIEWS / "home.py"), title="Score applicants",
                icon=":material/upload_file:", default=True),
        st.Page(str(VIEWS / "overview.py"), title="Overview",
                icon=":material/monitoring:"),
        st.Page(str(VIEWS / "pipeline.py"), title="Run pipeline",
                icon=":material/account_tree:"),
        st.Page(str(VIEWS / "results.py"), title="Results viewer", icon=":material/table:"),
        st.Page(str(VIEWS / "model_registry.py"), title="Model registry",
                icon=":material/verified:"),
    ],
    "Research": [
        st.Page(str(VIEWS / "retrain.py"), title="Retrain models (research)",
                icon=":material/model_training:"),
        st.Page(str(VIEWS / "findings.py"), title="FINDINGS", icon=":material/science:"),
    ],
}).run()
