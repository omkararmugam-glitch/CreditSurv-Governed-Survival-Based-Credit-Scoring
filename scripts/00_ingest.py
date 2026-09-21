"""Stage 1a -- stream both Lending Club CSVs into Parquet.

Run once. Everything downstream reads the Parquet output, so the expensive parse
of ~1.6 GB of messy CSV happens a single time.

Usage
-----
    python scripts/00_ingest.py --accepted-csv C:/data/lc/accepted.csv \
                               --rejected-csv C:/data/lc/rejected.csv
    python scripts/00_ingest.py --config config/config.yaml
    python scripts/00_ingest.py --row-limit 50000      # smoke test first
    python scripts/00_ingest.py --skip-rejected        # accepted only

Paths may be given either on the command line (which wins) or in the config
file under ``paths.accepted_csv`` / ``paths.rejected_csv``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditsurv.config import load_config  # noqa: E402
from creditsurv.io.loaders import ingest_accepted, ingest_rejected  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--accepted-csv", default=None,
                    help="path to accepted_2007_to_2018Q4.csv (overrides config)")
    ap.add_argument("--rejected-csv", default=None,
                    help="path to rejected_2007_to_2018Q4.csv (overrides config)")
    ap.add_argument("--row-limit", type=int, default=None,
                    help="cap rows read from each file (smoke test)")
    ap.add_argument("--skip-accepted", action="store_true")
    ap.add_argument("--skip-rejected", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    overrides = {}
    if args.accepted_csv:
        overrides["accepted_csv"] = Path(args.accepted_csv).expanduser()
    if args.rejected_csv:
        overrides["rejected_csv"] = Path(args.rejected_csv).expanduser()
    if overrides:
        cfg = dataclasses.replace(cfg, paths=dataclasses.replace(cfg.paths, **overrides))

    for label, p in (("accepted_csv", cfg.paths.accepted_csv),
                     ("rejected_csv", cfg.paths.rejected_csv)):
        if p is not None and not p.exists():
            print(f"ERROR: {label} does not exist: {p}", file=sys.stderr)
            return 2

    cfg.paths.ensure_dirs()
    row_limit = args.row_limit if args.row_limit is not None else cfg.ingest.row_limit
    audits: dict[str, dict] = {}

    if not args.skip_accepted:
        if cfg.paths.accepted_csv is None:
            print(
                "ERROR: no accepted-loans CSV given.\n"
                "  Either pass it on the command line:\n"
                "      --accepted-csv C:/data/lending-club/accepted_2007_to_2018Q4.csv\n"
                "  or set paths.accepted_csv in config/config.yaml (it is a YAML\n"
                "  file to edit, not something to paste into the shell).",
                file=sys.stderr,
            )
            return 2
        print(f"[accepted] reading {cfg.paths.accepted_csv}")
        audits["accepted"] = ingest_accepted(
            cfg.paths.accepted_csv,
            cfg.paths.accepted_parquet,
            extended=cfg.ingest.extended_features,
            chunksize=cfg.ingest.chunksize,
            row_limit=row_limit,
        )
        a = audits["accepted"]
        print(f"[accepted] {a['n_rows_written']:,} rows -> {a['output']} "
              f"({a['output_size_mb']} MB)")
        if a["columns_requested_but_absent"]:
            print(f"[accepted] columns absent from this extract: "
                  f"{a['columns_requested_but_absent']}")

    if not args.skip_rejected:
        if cfg.paths.rejected_csv is None:
            print("WARNING: paths.rejected_csv is not set; skipping.", file=sys.stderr)
        else:
            print(f"[rejected] reading {cfg.paths.rejected_csv}")
            audits["rejected"] = ingest_rejected(
                cfg.paths.rejected_csv,
                cfg.paths.rejected_parquet,
                row_limit=row_limit,
            )
            r = audits["rejected"]
            print(f"[rejected] {r['n_rows_written']:,} rows -> {r['output']} "
                  f"({r['output_size_mb']} MB)")

    out = cfg.paths.tables_dir / "00_ingest_audit.json"
    out.write_text(json.dumps(audits, indent=2), encoding="utf-8")
    print(f"audit written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
