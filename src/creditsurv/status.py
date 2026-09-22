"""Read-only views of pipeline state, for the Streamlit UI.

Nothing here runs a stage or writes a file. Stage status is derived from the
result files the scripts themselves write and from their provenance stamps, using
the same :func:`creditsurv.provenance.verify_stamp` as ``check_provenance.py``.
The pre-flight check predicts which stages would be refused for existing outputs
using the same :func:`find_existing_outputs` the scripts' guards call, with the
same directories and patterns -- ``tests/test_ui_backend.py`` checks that every
prediction matches what the scripts actually do.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .provenance import file_fingerprint, find_existing_outputs, verify_stamp

__all__ = ["StageStatus", "RESULT_STAGES", "discover_tags", "stage_status", "overview",
           "preflight", "fingerprint_cache", "figures_for_tag", "findings_diff"]

# stage key -> (label, result-file template, which tag it uses: "base" or "strat")
RESULT_STAGES: dict[str, tuple[str, str, str]] = {
    "02": ("Stage 2  survival models", "02_metrics_{tag}.json", "base"),
    "03": ("Stage 3  naive vs SurvSHAP(t)", "03_explain_{tag}.json", "base"),
    "03c": ("Stage 3c noise floor", "03c_noise_floor_{tag}.json", "base"),
    "03s": ("Stage 3  grade-stratified", "03_explain_{tag}.json", "strat"),
    "03b": ("Stage 3b grade G bootstrap", "03_segment_bootstrap_{tag}.json", "strat"),
    "04": ("Stage 4  selection-bias diagnostic", "04_reject_inference_{tag}.json", "base"),
}

_TAG_PATTERNS = [re.compile(r"^02_metrics_(.+)\.json$"),
                 re.compile(r"^03_explain_(.+)\.json$"),
                 re.compile(r"^03c_noise_floor_(.+)\.json$"),
                 re.compile(r"^04_reject_inference_(.+)\.json$")]


@dataclass
class StageStatus:
    key: str
    label: str
    tag: str
    file: str
    state: str          # "verified" | "unverified" | "changed" | "not run"
    detail: str
    run_time: str | None
    commit: str | None

    @property
    def colour(self) -> str:
        return {"verified": "green", "unverified": "orange", "changed": "red",
                "not run": "gray"}[self.state]


def fingerprint_cache():
    """A memoised fingerprint function keyed on (path, size, mtime), so a file used
    by many results is hashed once per status refresh."""
    cache: dict = {}

    def fp(path):
        p = Path(path)
        try:
            st = p.stat()
        except OSError:
            return file_fingerprint(p)
        key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
        if key not in cache:
            cache[key] = file_fingerprint(p)
        return cache[key]
    return fp


def discover_tags(tables_dir: Path) -> list[str]:
    """Base tags with at least one stage result (``*_strat`` folded into its base)."""
    tags: set[str] = set()
    for p in Path(tables_dir).glob("*.json"):
        for rx in _TAG_PATTERNS:
            m = rx.match(p.name)
            if m:
                t = m.group(1)
                tags.add(t[: -len("_strat")] if t.endswith("_strat") else t)
    return sorted(tags)


def _local(ts_utc: str) -> str:
    try:
        return datetime.fromisoformat(ts_utc).astimezone().strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(ts_utc)


def stage_status(tables_dir: Path, base_tag: str, key: str, fp=None) -> StageStatus:
    label, template, which = RESULT_STAGES[key]
    tag = base_tag if which == "base" else f"{base_tag}_strat"
    path = Path(tables_dir) / template.format(tag=tag)
    if not path.exists():
        return StageStatus(key, label, tag, path.name, "not run",
                           "no result file for this tag", None, None)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return StageStatus(key, label, tag, path.name, "changed",
                           f"result file unreadable: {exc}", None, None)
    stamp = payload.get("provenance") if isinstance(payload, dict) else None
    if not stamp:
        mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        return StageStatus(key, label, tag, path.name, "unverified",
                           "produced before provenance stamping; inputs cannot be "
                           "verified (file date shown)", mtime, None)
    rows = verify_stamp(stamp, fingerprint=fp)
    bad = [r for r in rows if r["status"] in ("changed", "missing")]
    code = stamp.get("code") or {}
    commit = (code.get("commit") or "")[:10] or None
    if bad:
        detail = "; ".join(f"{r['name']} {r['status']}" for r in bad)
        return StageStatus(key, label, tag, path.name, "changed",
                           f"inputs no longer match: {detail}",
                           _local(stamp.get("created_at")), commit)
    dirty = " (code was modified at run time)" if code.get("code_dirty") else ""
    return StageStatus(key, label, tag, path.name, "verified",
                       f"{len(rows)} recorded file(s) verified{dirty}",
                       _local(stamp.get("created_at")), commit)


def overview(tables_dir: Path) -> dict[str, list[StageStatus]]:
    """Every discovered tag -> status of each result stage."""
    fp = fingerprint_cache()
    return {t: [stage_status(tables_dir, t, k, fp) for k in RESULT_STAGES]
            for t in discover_tags(tables_dir)}


def preflight(plan, paths) -> dict[str, list[Path]]:
    """Existing files that would make each stage in ``plan`` refuse to run.

    Mirrors each script's guard exactly: same directories, same patterns.
    ``paths`` is a :class:`creditsurv.config.Paths`.
    """
    out: dict[str, list[Path]] = {}
    tables, figures, models = paths.tables_dir, paths.figures_dir, paths.models_dir
    for st in plan.stages:
        if st.key == "02":
            hits = find_existing_outputs([tables, figures, models], "02", st.tag)
        elif st.key in ("03", "03s"):
            hits = find_existing_outputs([tables, figures], "03", st.tag)
        elif st.key == "03c":
            p = tables / f"03c_noise_floor_{st.tag}.json"
            hits = [p] if p.exists() else []
        elif st.key == "03b":
            p = tables / f"03_segment_bootstrap_{st.tag}.json"
            hits = [p] if p.exists() else []
        elif st.key == "04":
            hits = find_existing_outputs([tables, figures], "04", st.tag)
        else:
            hits = []                        # 05 and the provenance check write no
        if hits:                             # guarded outputs
            out[st.key] = hits
    return out


def figures_for_tag(figures_dir: Path, tag: str) -> list[Path]:
    """PNGs ending exactly in ``_<tag>.png`` (so ``full`` excludes ``full_strat``)."""
    return sorted(p for p in Path(figures_dir).glob("*.png") if p.name.endswith(f"_{tag}.png"))


def findings_diff(root: Path) -> dict:
    """``git diff FINDINGS.md`` against HEAD, with the wrapper's section-6 check:
    any hunk starting above the committed ``## 6.`` heading touched sections 0-5."""
    import subprocess

    def git(*args):
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=30)
        return r.stdout if r.returncode == 0 else None

    diff = git("--no-pager", "diff", "--", "FINDINGS.md")
    if diff is None:
        return {"available": False, "diff": "", "above_s6": False, "h6": None}
    head = git("show", "HEAD:FINDINGS.md") or ""
    h6 = next((i for i, line in enumerate(head.splitlines(), 1)
               if line.startswith("## 6. ")), None)
    starts = [int(m.group(1)) for m in re.finditer(
        r"^@@ -(\d+)", git("--no-pager", "diff", "-U0", "--", "FINDINGS.md") or "", re.M)]
    return {"available": True, "diff": diff, "h6": h6,
            "above_s6": bool(h6) and any(s < h6 for s in starts)}
