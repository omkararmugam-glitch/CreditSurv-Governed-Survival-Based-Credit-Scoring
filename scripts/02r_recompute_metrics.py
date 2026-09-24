"""Stage 2r -- recompute a model's metrics from the saved bundle, without retraining.

For a model whose metrics JSON was lost. Everything the metric suite needs is in
the bundle: both fitted models, the train/test indices, the data source, and the
Cox standardisation, fill values and column list. Scoring the saved model on its own
test split reproduces concordance, time-dependent AUC, IBS and calibration exactly,
because none of them depends on anything from training beyond the fitted model --
the IPCW censoring model is fitted from observed outcomes on the evaluation sample
and never sees a prediction.

**What this cannot recover, and says so in its output:**

* the original provenance stamp -- the git commit, the input file hashes and the
  library versions as they were at training time;
* the early-stopping history and the best iteration the run chose;
* the wall-clock timings.

Those existed only in the deleted file. The output is therefore written under a
``_recomputed`` name with ``recomputed: true`` and an explicit list of the missing
fields, so it can never be mistaken for the file Stage 2 wrote.

    python scripts/02r_recompute_metrics.py --model-tag holdout_devsample
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditsurv.config import load_config  # noqa: E402
from creditsurv.features.build import build_design_matrix  # noqa: E402
from creditsurv.models.evaluate import CensoringModel, evaluate_survival  # noqa: E402
from creditsurv.pipeline import load_model_bundle, resolve_data_source  # noqa: E402
from creditsurv.provenance import build_stamp, guard_outputs  # noqa: E402
from creditsurv.reporting.tables import (  # noqa: E402
    model_comparison_table,
    to_markdown,
    write_json,
    write_table,
)

UNRECOVERABLE = [
    "provenance stamp of the original training run (git commit, input file "
    "SHA-256 hashes, library versions at train time)",
    "early-stopping history and the best iteration chosen during training",
    "wall-clock timings of the original run",
    "the validation split used for early stopping",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--sample", type=int, default=0,
                    help="evaluate on a random sample of the test split "
                         "(0 = the whole split, which is what the original did)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg.paths.ensure_dirs()
    tag = args.model_tag
    out_json = cfg.paths.tables_dir / f"02_metrics_{tag}_recomputed.json"
    out_csv = cfg.paths.tables_dir / f"02_model_comparison_{tag}_recomputed.csv"
    refused = guard_outputs([out_json, out_csv], args.overwrite,
                            script="02r_recompute_metrics.py")
    if refused:
        return refused

    bundle, model_path = load_model_bundle(cfg.paths.models_dir, tag)
    art, spec = bundle["artefacts"], bundle["spec"]
    src = resolve_data_source(bundle, cfg, tag)
    print(f"model {tag} ({model_path.name})")
    print(f"recorded data source: {src}")
    print(f"recorded split: {bundle.get('split')}")

    cols = list(spec.all_columns) + ["duration_months", "event"]
    df = pd.read_parquet(src, columns=cols)
    for c in [c for c in df.columns if df[c].dtype == object]:
        df[c] = df[c].astype("category")
    test_idx = bundle["test_idx"].intersection(df.index)
    test = df.loc[test_idx]
    if args.sample and args.sample < len(test):
        test = test.sample(args.sample, random_state=cfg.model.seed)
    print(f"test rows: {len(test):,} of {len(test_idx):,} recorded")

    duration = test["duration_months"].to_numpy()
    event = test["event"].to_numpy()
    times = np.array(cfg.model.eval_horizons_months, dtype=float)
    censoring = CensoringModel(duration, event)

    results = []
    t0 = time.perf_counter()
    if "cox" in art:
        dm = build_design_matrix(test, spec, flavour="cox",
                                 standardisation=art.get("cox_standardisation"),
                                 fill_values=art.get("cox_fill_values"),
                                 reference_columns=art.get("cox_columns"))
        cox = art["cox"]
        results.append(evaluate_survival(
            model_name="cox", split_name="test",
            survival=cox.predict_survival(dm.X, times), times=times,
            duration=duration, event=event, censoring=censoring,
            risk=cox.predict_risk(dm.X), calibration_at=float(times[-1])))
        print(f"  cox: C={results[-1].concordance:.4f}")
    if "discrete_hazard" in art:
        dg = build_design_matrix(test, spec, flavour="gbm")
        dh = art["discrete_hazard"]
        results.append(evaluate_survival(
            model_name="discrete_hazard", split_name="test",
            survival=dh.predict_survival(dg.X, times), times=times,
            duration=duration, event=event, censoring=censoring,
            calibration_at=float(times[-1])))
        print(f"  discrete_hazard: C={results[-1].concordance:.4f}")
    if not results:
        print("ERROR: the bundle holds no fitted model to evaluate.", file=sys.stderr)
        return 2

    comparison = model_comparison_table(results)
    write_table(comparison, out_csv)
    print()
    print(to_markdown(comparison))

    dh = art.get("discrete_hazard")
    payload = {
        "recomputed": True,
        "recomputed_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "why": ("the original 02_metrics_%s.json was lost before it was committed; "
                "these numbers come from evaluating the saved model, not from a "
                "training run" % tag),
        "unrecoverable_fields": UNRECOVERABLE,
        "tag": tag,
        "source": str(src),
        "data_source": str(src),
        "n_rows": int(len(df)),
        "n_train": int(len(bundle["train_idx"])),
        "n_test": int(len(test)),
        "n_test_recorded": int(len(test_idx)),
        "evaluated_on_sample": bool(args.sample and args.sample < len(test_idx)),
        "split_scheme": (bundle.get("split") or {}).get("scheme"),
        "oot_cutoff": (bundle.get("split") or {}).get("oot_cutoff"),
        "train_years": (bundle.get("split") or {}).get("train_years"),
        "test_years": (bundle.get("split") or {}).get("test_years"),
        "time_bin_months": getattr(dh, "time_bin_months", None),
        "negative_subsample": getattr(dh, "negative_subsample", None),
        "num_boost_round": getattr(dh, "num_boost_round", None),
        "trees_kept": (dh.booster.num_trees() if dh is not None
                       and getattr(dh, "booster", None) is not None else None),
        "features": {"numeric": list(spec.numeric),
                     "categorical": list(spec.categorical)},
        "eval_horizons": [int(t) for t in times],
        "results": [r.summary() for r in results],
        "seconds_to_recompute": round(time.perf_counter() - t0, 1),
        "provenance": build_stamp(
            stage="02_metrics_recomputed",
            inputs={"data": src, "model": model_path},
            outputs={"metrics": out_json, "comparison": out_csv},
            config_path=args.config,
            args={"model_tag": tag, "sample": args.sample}),
    }
    write_json(payload, out_json)
    print(f"\nWritten {out_json.name} and {out_csv.name}, both marked recomputed.")
    print("Not recovered by this run:")
    for item in UNRECOVERABLE:
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
