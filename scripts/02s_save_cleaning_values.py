"""Stage 2s -- fit a model's cleaning values on its FULL training split and save them.

Cleaning values are the imputation medians, the range bounds and the category lists
that :mod:`creditsurv.cleaning` reapplies when scoring an upload. They belong to the
model, not to the file being scored, so they are fitted once on the whole training
split the model was fitted on and then only ever read.

Stage 2 saves them inside new bundles. This script adds them to a bundle trained
before that existed, **without retraining and without touching the model file**:
the values go in a sidecar, ``02_cleaning_values_<tag>.json``, so the model's
SHA-256 stays what every existing result recorded for it.

    python scripts/02s_save_cleaning_values.py --model-tag full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditsurv.cleaning import fit_values, policy_from_config  # noqa: E402
from creditsurv.config import load_config  # noqa: E402
from creditsurv.features.build import add_derived_features  # noqa: E402
from creditsurv.pipeline import load_model_bundle, resolve_data_source  # noqa: E402
from creditsurv.provenance import build_stamp, guard_outputs  # noqa: E402


DERIVATION_INPUTS: frozenset[str] = frozenset({
    "fico_range_low", "fico_range_high", "emp_length", "installment", "annual_inc",
    "loan_amnt"})
"""Raw columns :func:`add_derived_features` needs. Read even when the spec does not
name them, because the features that depend on them do."""


def sidecar_path(models_dir: Path, tag: str) -> Path:
    return Path(models_dir) / f"02_cleaning_values_{tag}.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg.paths.ensure_dirs()
    out = sidecar_path(cfg.paths.models_dir, args.model_tag)
    refused = guard_outputs([out], args.overwrite,
                            script="02s_save_cleaning_values.py")
    if refused:
        return refused

    bundle, model_path = load_model_bundle(cfg.paths.models_dir, args.model_tag)
    if bundle.get("cleaning_values"):
        print(f"note: the {args.model_tag} bundle already carries cleaning values; "
              f"writing the sidecar anyway so both agree.")
    spec = bundle["spec"]
    src = resolve_data_source(bundle, cfg, args.model_tag)
    if not Path(src).exists():
        print(f"ERROR: {src} is not on disk, so the training split cannot be read. "
              f"Retrain the model to get cleaning values.", file=sys.stderr)
        return 2

    # A model trained with --with-derived has features the parquet does not store:
    # fico_midpoint, the income ratios, emp_length_years. Asking pyarrow for them by
    # name fails ("No match for fico_midpoint"), so the raw inputs are read and the
    # derived columns are computed here, exactly as training computed them.
    cols = list(spec.all_columns)
    print(f"reading {src}")
    stored = set(pq.read_schema(src).names)
    to_read = sorted((set(cols) | DERIVATION_INPUTS) & stored)
    df = add_derived_features(pd.read_parquet(src, columns=to_read))
    for c in [c for c in df.columns if df[c].dtype == object]:
        df[c] = df[c].astype("category")
    computed = sorted(set(cols) - stored)
    if computed:
        print(f"computed from raw columns rather than read: {', '.join(computed)}")
    absent = [c for c in cols if c not in df.columns]
    if absent:
        print(f"ERROR: {absent} are features of the {args.model_tag} model but are "
              f"neither stored in {Path(src).name} nor derivable from it, so its "
              f"cleaning values cannot be fitted. Nothing has been written.",
              file=sys.stderr)
        return 2
    train_idx = bundle["train_idx"].intersection(df.index)
    if len(train_idx) == 0:
        print("ERROR: the bundle's recorded training index does not match this data "
              "file, so its training split cannot be reconstructed. Retrain.",
              file=sys.stderr)
        return 2
    train = df.loc[train_idx]
    print(f"fitting on the FULL training split: {len(train):,} rows "
          f"({len(bundle['train_idx']):,} recorded)")

    policy = policy_from_config(cfg)
    t0 = time.perf_counter()
    values = fit_values(train, spec, policy=policy, source=src)
    seconds = time.perf_counter() - t0

    payload = {
        "model_tag": args.model_tag,
        "fitted_on": "full training split of the saved model",
        "fitted_rows": values.fitted_rows,
        "seconds": round(seconds, 1),
        "values": values.to_dict(),
        "policy": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in vars(policy).items()},
        "provenance": build_stamp(
            stage="02s_save_cleaning_values",
            inputs={"data": src, "model": model_path},
            outputs={"cleaning_values": out},
            config_path=args.config,
            args={"model_tag": args.model_tag}),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"written {out.name}: {len(values.ranges)} numeric ranges, "
          f"{len(values.categories)} category lists, {len(values.medians)} medians, "
          f"from {values.fitted_rows:,} rows in {seconds:.1f}s")
    print("The model file was not modified, so its hash and every result that "
          "recorded it stay valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
