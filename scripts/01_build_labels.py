"""Stage 1b -- build the survival target and the development sample.

Reads the Parquet written by ``00_ingest.py``, applies the event/censoring rules
from ``config.label``, and writes:

* ``accepted_labeled.parquet``  -- full labelled population
* ``dev_sample.parquet``        -- stratified subset for fast iteration
* ``01_label_audit.json``       -- every count needed for FINDINGS.md

Usage
-----
    python scripts/01_build_labels.py --config config/config.yaml
    python scripts/01_build_labels.py --late-as-event      # sensitivity run
    python scripts/01_build_labels.py --event-lag 5        # sensitivity run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditsurv.config import load_config  # noqa: E402
from creditsurv.io.loaders import stratified_sample  # noqa: E402
from creditsurv.provenance import build_stamp, guard_outputs  # noqa: E402
from creditsurv.labeling.survival_target import (  # noqa: E402
    LabelConfig,
    build_survival_target,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--late-as-event", action="store_true",
                    help="sensitivity: count Late (31-120 days) as an event")
    ap.add_argument("--event-lag", type=int, default=None,
                    help="sensitivity: months added to event times")
    ap.add_argument("--include-policy-exceptions", action="store_true")
    ap.add_argument("--tag", default=None,
                    help="suffix for output filenames, for sensitivity runs")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing existing outputs for this tag")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.paths.ensure_dirs()

    # Checked before anything is loaded, so a refused run costs nothing.
    tag = f"_{args.tag}" if args.tag else ""
    labeled_path = cfg.paths.data_dir / f"accepted_labeled{tag}.parquet"
    dev_path = cfg.paths.data_dir / f"dev_sample{tag}.parquet"
    audit_path = cfg.paths.tables_dir / f"01_label_audit{tag}.json"
    sensitivity = (args.late_as_event or args.include_policy_exceptions
                   or args.event_lag is not None)
    if sensitivity and not args.tag and not args.overwrite and labeled_path.exists():
        print("NOTE: a sensitivity flag was given without --tag. Without a tag this "
              "run would REPLACE the primary labelled dataset that every later stage "
              "reads. Use --tag <name> to write the variant alongside it.",
              file=sys.stderr)
    refused = guard_outputs([labeled_path, dev_path, audit_path], args.overwrite,
                            script="01_build_labels.py")
    if refused:
        return refused

    if not cfg.paths.accepted_parquet.exists():
        print(f"ERROR: {cfg.paths.accepted_parquet} not found. "
              f"Run scripts/00_ingest.py first.", file=sys.stderr)
        return 2

    label_kwargs = dict(cfg.label)
    if args.late_as_event:
        label_kwargs["late_31_120_is_event"] = True
    if args.include_policy_exceptions:
        label_kwargs["include_policy_exceptions"] = True
    if args.event_lag is not None:
        label_kwargs["event_lag_months"] = args.event_lag
    label_cfg = LabelConfig(**label_kwargs)

    print(f"reading {cfg.paths.accepted_parquet}")
    df = pd.read_parquet(cfg.paths.accepted_parquet)
    raw_gb = df.memory_usage(deep=True).sum() / 1e9

    # Parquet stores these compactly but pandas expands them to Python str
    # objects, which is what makes the frame ~2.3 GB. Every string column in the
    # allowlist is low-cardinality (<=1000 distinct), so category dtype is
    # lossless here and cuts memory by roughly 5x. On a 15 GB machine that is
    # the difference between comfortable and swapping.
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    for col in obj_cols:
        if col == "id":
            continue  # unique per row; category would only add overhead
        df[col] = df[col].astype("category")
    print(f"  {len(df):,} rows x {df.shape[1]} columns "
          f"({raw_gb:.2f} GB -> {df.memory_usage(deep=True).sum() / 1e9:.2f} GB "
          f"after categorising {len(obj_cols) - 1} string columns)")

    labeled, audit = build_survival_target(df, label_cfg)
    labeled["issue_year"] = labeled["issue_month"].dt.year.astype("int16")

    print(f"  retained {audit['n_retained']:,} of {audit['n_input']:,} rows")
    print(f"  data cutoff: {audit['data_cutoff']} ({audit['cutoff_source']})")
    print(f"  event rate: {audit['event_rate']:.4f}")
    print(f"  median duration: {audit['duration_months_median']:.0f} months")
    for reason, count in sorted(audit["drop_counts"].items(), key=lambda kv: -kv[1]):
        print(f"    dropped {count:>9,}  {reason}")

    labeled.to_parquet(labeled_path, index=False)
    print(f"wrote {labeled_path}")

    dev = stratified_sample(
        labeled,
        cfg.sample.dev_sample_size,
        by=cfg.sample.stratify_by,
        seed=cfg.sample.seed,
    )
    dev.to_parquet(dev_path, index=False)
    print(f"wrote {dev_path} ({len(dev):,} rows, "
          f"event rate {dev['event'].mean():.4f})")

    audit["dev_sample"] = {
        "n": int(len(dev)),
        "event_rate": float(dev["event"].mean()),
        "stratify_by": list(cfg.sample.stratify_by),
        "seed": cfg.sample.seed,
    }
    audit["provenance"] = build_stamp(
        stage="01_build_labels",
        inputs={"accepted_raw": cfg.paths.accepted_parquet},
        outputs={"labeled": labeled_path, "dev_sample": dev_path},
        config_path=args.config,
        args=vars(args),
    )
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"audit written to {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
