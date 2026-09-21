"""Detect results whose inputs have been overwritten since they were produced.

Every stage's results JSON carries a ``provenance`` block with SHA-256 hashes of
the files it read and wrote. This script re-hashes those files and reports any
that no longer match -- i.e. a result that now describes data or a model that no
longer exists on disk. Read-only unless ``--write-baseline`` is given.

Results produced before provenance stamping existed carry no stamp. For those,
``--write-baseline`` records the current hashes of the untracked artefacts
(``outputs/models``, ``outputs/data``) in ``outputs/provenance_baseline.json``,
which is committed to git; later runs of this script compare against it.

Usage
-----
    python scripts/check_provenance.py                   # verify, exit 1 on any change
    python scripts/check_provenance.py --write-baseline  # record untracked-artefact hashes
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditsurv.config import load_config  # noqa: E402
from creditsurv.provenance import PROJECT_ROOT, file_fingerprint, verify_stamp  # noqa: E402

NOT_RESULTS = {"05_report_state.json"}   # bookkeeping, not a stage result


def _baseline_path(cfg) -> Path:
    """Next to the tables/figures dirs, so a test config can redirect it."""
    return cfg.paths.tables_dir.parent / "provenance_baseline.json"


def _untracked_artefacts(cfg) -> list[Path]:
    files = sorted(cfg.paths.models_dir.glob("*.pkl"))
    files += sorted(cfg.paths.data_dir.glob("*.parquet"))
    files += sorted(cfg.paths.data_dir.glob("*.csv"))
    return files


def write_baseline(cfg) -> int:
    files = _untracked_artefacts(cfg)
    print(f"hashing {len(files)} untracked artefacts (the raw CSVs take a while)...")
    manifest = {
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": (
            "State of artefacts that are NOT version-controlled, recorded when "
            "provenance stamping was introduced. It fixes what these files are now; "
            "it does not by itself prove which of them any earlier result was computed "
            "from. Results produced after this point carry their own stamps."
        ),
        "files": [file_fingerprint(f) for f in files],
    }
    baseline = _baseline_path(cfg)
    baseline.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for fp in manifest["files"]:
        print(f"  {fp['sha256'][:12]}  {fp['bytes'] / 1e6:9.1f} MB  {fp['path']}")
    print(f"wrote {baseline}")
    return 0


def verify(cfg) -> int:
    problems = 0
    stamped, unstamped = [], []
    for path in sorted(cfg.paths.tables_dir.glob("*.json")):
        if path.name in NOT_RESULTS:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        stamp = payload.get("provenance") if isinstance(payload, dict) else None
        (stamped if stamp else unstamped).append((path, stamp))

    print(f"stamped results: {len(stamped)}")
    for path, stamp in stamped:
        rows = verify_stamp(stamp)
        bad = [r for r in rows if r["status"] in {"changed", "missing"}]
        code = stamp.get("code") or {}
        commit = (code.get("commit") or "no-git")[:10]
        dirty = " (code modified at run time)" if code.get("code_dirty") else ""
        state = "OK" if not bad else "CHANGED"
        print(f"  [{state:7}] {path.name}  commit {commit}{dirty}")
        for r in bad:
            problems += 1
            print(f"            {r['status'].upper()}: {r['kind']} '{r['name']}' "
                  f"-> {r['path']}")

    if unstamped:
        print(f"\nunstamped results (produced before provenance stamping): "
              f"{len(unstamped)}")
        for path, _ in unstamped:
            print(f"  [ ----- ] {path.name}")
        print("  Their inputs cannot be verified individually. The untracked-artefact "
              "baseline below is the check that covers them.")

    baseline = _baseline_path(cfg)
    if baseline.exists():
        manifest = json.loads(baseline.read_text(encoding="utf-8"))
        print(f"\nbaseline of untracked artefacts (recorded "
              f"{manifest['recorded_at']}): {len(manifest['files'])} files")
        for fp in manifest["files"]:
            p = PROJECT_ROOT / fp["path"]
            if not p.exists():
                status = "MISSING"
            else:
                status = "OK" if file_fingerprint(p)["sha256"] == fp["sha256"] \
                    else "CHANGED"
            if status != "OK":
                problems += 1
            print(f"  [{status:7}] {fp['path']}")
    else:
        print("\nno baseline recorded; run with --write-baseline to create one.")

    print(f"\n{'no overwrites detected' if not problems else f'{problems} problem(s)'}")
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    return write_baseline(cfg) if args.write_baseline else verify(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
