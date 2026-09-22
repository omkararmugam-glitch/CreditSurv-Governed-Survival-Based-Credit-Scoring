"""FINDINGS viewer: the research log as written, plus what changed since the last
commit. Read-only; the only writer of FINDINGS.md is scripts/05_report.py."""

import re

import streamlit as st

from _common import page_header
from creditsurv.provenance import PROJECT_ROOT
from creditsurv.status import findings_diff

page_header("FINDINGS", "The research log, including the negative results.")
path = PROJECT_ROOT / "FINDINGS.md"
if not path.exists():
    st.error("FINDINGS.md not found.")
    st.stop()
text = path.read_text(encoding="utf-8")

d = findings_diff(PROJECT_ROOT)
if not d["available"]:
    st.caption("git not available; cannot compare with the last commit.")
elif not d["diff"]:
    st.success("No changes to FINDINGS.md since the last commit.")
elif d["above_s6"]:
    st.error(f"FINDINGS.md changed ABOVE section 6 (committed heading at line {d['h6']}). "
             "The primary-result sections were touched. Review before committing.")
else:
    st.warning(f"FINDINGS.md has uncommitted changes, all in section 6 (at or below "
               f"committed line {d['h6']}); sections 0-5 untouched.")
if d.get("diff"):
    with st.expander("git diff FINDINGS.md"):
        st.code(d["diff"], language="diff")

# Split on level-2 headings so a long log can be read one section at a time.
parts = re.split(r"(?m)^(?=## )", text)
preamble, sections = parts[0], parts[1:]
titles = ["All"] + [s.splitlines()[0].lstrip("# ").strip() for s in sections]
pick = st.selectbox("Section", titles)
# Keep-block markers are HTML comments; markdown hides them, so show where they are.
mark = lambda s: re.sub(r"<!-- (/?)keep:([\w-]+) -->",  # noqa: E731
                        lambda m: f"\n> *{'end of ' if m.group(1) else ''}protected "
                                  f"block `{m.group(2)}` (never rewritten by 05_report)*\n",
                        s)
if pick == "All":
    st.markdown(mark(text))
else:
    st.markdown(mark(sections[titles.index(pick) - 1]))
