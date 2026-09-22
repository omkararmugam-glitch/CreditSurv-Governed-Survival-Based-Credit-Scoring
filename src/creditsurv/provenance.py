"""Overwrite protection and provenance stamps for pipeline outputs.

Two separate safety mechanisms, both used by every stage script:

**Overwrite guards.** A script refuses to start if any output it would write
already exists, unless it is given ``--overwrite``. The check runs *before* any
data is loaded, so a refused run costs nothing. The files listed include outputs
that are not version-controlled (``outputs/models``, ``outputs/data``), which,
unlike tables and figures, cannot be recovered from git once replaced.

**Provenance stamps.** Every results JSON carries a ``provenance`` block
recording exactly which code, config and input files produced it: the git commit
(and whether the code was modified at run time), SHA-256 hashes of every input
file and of any model the stage wrote, and the library versions. Overwrites
cannot be prevented everywhere, but with the stamp they can be *detected*
afterwards -- ``scripts/check_provenance.py`` recomputes the hashes and reports any
result whose inputs no longer match what is on disk.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "PROJECT_ROOT",
    "OVERWRITE_REFUSED",
    "file_fingerprint",
    "find_existing_outputs",
    "guard_outputs",
    "git_state",
    "build_stamp",
    "verify_stamp",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OVERWRITE_REFUSED = 4
"""Exit code a script returns when it refuses to overwrite existing outputs."""

_CODE_PATHS = ("src", "scripts", "config", "pyproject.toml")


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def file_fingerprint(path: str | Path, *, chunk: int = 1 << 20) -> dict:
    """Size, modification time and SHA-256 of a file, hashed in streaming chunks.

    A missing file is recorded as such rather than raising, so a stamp can still be
    written when an optional input is absent.
    """
    p = Path(path)
    if not p.exists():
        return {"path": _rel(p), "exists": False}
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    st = p.stat()
    return {
        "path": _rel(p),
        "exists": True,
        "bytes": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "sha256": h.hexdigest(),
    }


def find_existing_outputs(dirs, stage: str, tag: str) -> list[Path]:
    """Existing files named ``{stage}_*_{tag}.*`` in any of ``dirs``.

    The pattern needs ``_{tag}.`` immediately before the extension, so tag
    ``full`` matches ``03_explain_full.json`` but not ``03_explain_full_strat.json``
    -- a variant run never blocks, or is blocked by, the run it derives from.
    """
    hits: set[Path] = set()
    for d in dirs:
        d = Path(d)
        if d.exists():
            hits.update(p for p in d.glob(f"{stage}_*_{tag}.*") if p.is_file())
    return sorted(hits)


def guard_outputs(existing, overwrite: bool, *, script: str) -> int | None:
    """Refuse to proceed if outputs exist, unless ``overwrite`` is set.

    Returns :data:`OVERWRITE_REFUSED` (for the caller to return as its exit code)
    when refusing, otherwise ``None``. With ``overwrite`` set it prints what will
    be replaced, so the replacement is never silent.
    """
    existing = [Path(p) for p in existing if Path(p).exists()]
    if not existing:
        return None
    untracked = [p for p in existing
                 if _rel(p).startswith(("outputs/models/", "outputs/data/"))]
    listing = "\n".join(f"    {_rel(p)}" for p in existing[:15])
    more = f"\n    ... and {len(existing) - 15} more" if len(existing) > 15 else ""
    if not overwrite:
        msg = (
            f"ERROR: {script} would overwrite {len(existing)} existing output file(s):\n"
            f"{listing}{more}\n"
            "Nothing has been run. Re-run with --overwrite to replace them, or use a "
            "different --tag to keep both."
        )
        if untracked:
            msg += (f"\n{len(untracked)} of these are in outputs/models or outputs/data, "
                    "which are NOT in git and cannot be recovered once replaced.")
        print(msg, file=sys.stderr)
        return OVERWRITE_REFUSED
    print(f"--overwrite: replacing {len(existing)} existing output file(s):\n"
          f"{listing}{more}")
    return None


def git_state(root: str | Path = PROJECT_ROOT) -> dict | None:
    """Commit hash and whether any *code* was modified relative to it.

    Only code paths count as dirty. Outputs change during a run by design, and
    counting them would mark every stamp dirty.
    """
    def run(*args: str) -> str | None:
        try:
            r = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                               text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    top = run("rev-parse", "--show-toplevel")
    if top is None or Path(top).resolve() != Path(root).resolve():
        return None
    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--", *_CODE_PATHS) or ""
    dirty = [line[3:] for line in status.splitlines() if line.strip()]
    return {"commit": commit, "code_dirty": bool(dirty), "dirty_files": dirty[:50]}


def _package_versions() -> dict:
    out = {}
    for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "lightgbm", "shap",
                 "lifelines", "scipy"):
        try:
            from importlib.metadata import version

            out[name] = version(name)
        except Exception:
            out[name] = None
    return out


def build_stamp(
    *,
    stage: str,
    inputs: dict[str, str | Path],
    outputs: dict[str, str | Path] | None = None,
    config_path: str | Path | None = None,
    args: dict | None = None,
) -> dict:
    """The ``provenance`` block embedded in a stage's results JSON."""
    return {
        "stage": stage,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": git_state(),
        "config": file_fingerprint(config_path) if config_path else None,
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in (args or {}).items()},
        "inputs": {k: file_fingerprint(v) for k, v in inputs.items()},
        "outputs": {k: file_fingerprint(v) for k, v in (outputs or {}).items()},
        "environment": {
            "python": platform.python_version(),
            "packages": _package_versions(),
        },
    }


def verify_stamp(stamp: dict, fingerprint=None) -> list[dict]:
    """Re-hash every file a stamp recorded and report what changed.

    Returns one row per recorded file: ``status`` is ``ok``, ``changed`` (the
    content hash differs -- the file was overwritten or edited), ``missing``, or
    ``was_missing`` (it was already absent when the stamp was written).

    ``fingerprint`` may be passed to reuse hashes across many stamps that share an
    input (the UI hashes a 166 MB file once per refresh, not once per result).
    """
    fingerprint = fingerprint or file_fingerprint
    rows = []
    for kind in ("inputs", "outputs"):
        for name, fp in (stamp.get(kind) or {}).items():
            path = PROJECT_ROOT / fp["path"] if not Path(fp["path"]).is_absolute() \
                else Path(fp["path"])
            if not fp.get("exists", False):
                status = "was_missing"
            elif not path.exists():
                status = "missing"
            else:
                status = "ok" if fingerprint(path)["sha256"] == fp["sha256"] \
                    else "changed"
            rows.append({"kind": kind, "name": name, "path": fp["path"],
                         "status": status})
    return rows
