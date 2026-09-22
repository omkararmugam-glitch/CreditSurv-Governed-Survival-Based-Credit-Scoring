"""Stage 3 support -- measure SurvSHAP(t)'s coalition-sampling noise floor.

The Stage 3(a) comparison is judged against how closely SurvSHAP(t) agrees with
*itself* across independent coalition draws. That floor is model-specific, so it
must be measured on the model being explained; one measured on another model does
not carry over. Explains the same borrowers twice with the background held fixed,
varying only the KernelSHAP draw.

Output is named ``03c_noise_floor_<tag>.json`` -- deliberately not ``03_...`` --
so it does not trip ``03_explain.py``'s overwrite guard for the same tag.

Usage
-----
    python scripts/03c_noise_floor.py --tag holdout --model-tag holdout
    python scripts/03c_noise_floor.py --tag holdout --nsamples 600 --n-explain 40
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
from creditsurv.explain.compare import coalition_noise_floor  # noqa: E402
from creditsurv.features.build import build_design_matrix  # noqa: E402
from creditsurv.pipeline import load_model_bundle, resolve_data_source  # noqa: E402
from creditsurv.provenance import build_stamp, guard_outputs  # noqa: E402
from creditsurv.reporting.tables import write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--tag", default="dev", help="suffix for the output filename")
    ap.add_argument("--model-tag", default=None, help="defaults to --tag")
    ap.add_argument("--model", default="discrete_hazard",
                    choices=["discrete_hazard", "cox"])
    ap.add_argument("--n-explain", type=int, default=40)
    ap.add_argument("--nsamples", type=int, default=None,
                    help="default: config explain.kernel_nsamples, the value Stage 3 "
                         "itself uses -- the floor must be measured at that setting")
    ap.add_argument("--n-background", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing an existing noise-floor result for this --tag")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.paths.ensure_dirs()
    model_tag = args.model_tag or args.tag
    out_path = cfg.paths.tables_dir / f"03c_noise_floor_{args.tag}.json"
    refused = guard_outputs([out_path], args.overwrite, script="03c_noise_floor.py")
    if refused:
        return refused

    try:
        bundle, model_path = load_model_bundle(cfg.paths.models_dir, model_tag)
        src = resolve_data_source(bundle, cfg, model_tag)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    artefacts, spec = bundle["artefacts"], bundle["spec"]
    if args.model not in artefacts:
        print(f"ERROR: model {args.model!r} not in {model_path}", file=sys.stderr)
        return 2
    model = artefacts[args.model]

    df = pd.read_parquet(src)
    for col in [c for c in df.columns if df[c].dtype == object and c != "id"]:
        df[col] = df[col].astype("category")
    test = df.loc[bundle["test_idx"].intersection(df.index)]
    if args.model == "cox":
        dm = build_design_matrix(test, spec, flavour="cox",
                                 standardisation=artefacts.get("cox_standardisation"),
                                 fill_values=artefacts.get("cox_fill_values"),
                                 reference_columns=artefacts.get("cox_columns"))
    else:
        dm = build_design_matrix(test, spec, flavour="gbm")

    nsamples = args.nsamples or cfg.explain.kernel_nsamples
    n_background = args.n_background or cfg.explain.n_background
    rng = np.random.default_rng(cfg.explain.seed)
    pick = np.sort(rng.choice(len(dm.X), min(args.n_explain, len(dm.X)), replace=False))
    times = np.array(cfg.model.eval_horizons_months, dtype=float)
    print(f"model={args.model} ({model_path.name}), data={src.name}, "
          f"{len(pick)} borrowers, nsamples={nsamples}, background={n_background}")

    t0 = time.time()
    floor = coalition_noise_floor(model, dm.X.iloc[pick], dm.X, times,
                                  nsamples=nsamples, n_background=n_background,
                                  seed=cfg.explain.seed)
    elapsed = time.time() - t0
    for k, v in floor.items():
        print(f"  {k}: {v}")
    print(f"done in {elapsed:.1f}s")

    write_json({**floor, "model": args.model, "model_tag": model_tag,
                "seconds_total": round(elapsed, 1),
                "provenance": build_stamp(stage="03c_noise_floor",
                                          inputs={"model": model_path, "data": src},
                                          config_path=args.config, args=vars(args))},
               out_path)
    print(f"results -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
