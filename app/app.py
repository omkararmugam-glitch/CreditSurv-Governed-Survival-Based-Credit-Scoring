"""Streamlit front end for the creditsurv pipeline.

A thin layer over the existing scripts: it runs them only through
``creditsurv.plan`` (the same commands ``run_holdout.ps1`` runs) and
``creditsurv.runner``, and reads results the scripts wrote. It contains no
modelling code, never passes ``--overwrite`` unless the user ticks it, and every
stage still stamps its own provenance.

Launch from the project root:

    .venv\\Scripts\\python.exe -m streamlit run app/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "app" / "views"):     # package + page helpers
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.chdir(ROOT)                    # config paths are relative to the project root

import streamlit as st  # noqa: E402

st.set_page_config(page_title="creditsurv", layout="wide")

# Checked before any page loads a model, so a blocked native library produces a
# sentence rather than an OSError from inside pickle.load. The pages that only read
# results still work, so this warns and continues rather than stopping the app.
from creditsurv.environment import blocked_imports, policy_block_message  # noqa: E402

_blocked = blocked_imports()
if _blocked:
    st.error(policy_block_message(_blocked), icon=":material/gpp_bad:")

VIEWS = ROOT / "app" / "views"
# Everyday use on top; retraining and research below it, where they cannot be
# reached by accident.
st.navigation({
    "Score applicants": [
        st.Page(str(VIEWS / "home.py"), title="Score applicants", icon=":material/upload_file:",
                default=True),
    ],
    "Advanced": [
        st.Page(str(VIEWS / "overview.py"), title="Overview", icon=":material/monitoring:"),
        st.Page(str(VIEWS / "run.py"), title="Run pipeline", icon=":material/play_arrow:"),
        st.Page(str(VIEWS / "results.py"), title="Results viewer", icon=":material/table:"),
        st.Page(str(VIEWS / "findings.py"), title="FINDINGS", icon=":material/science:"),
    ],
}).run()
