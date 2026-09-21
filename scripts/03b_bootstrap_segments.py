"""Stage 3(c) follow-up -- bootstrap confidence intervals on segment rank shifts.

Reads the per-borrower attributions persisted by ``03_explain.py`` and resamples
borrowers within each grade to put intervals on the rank shifts seen in the
highest-risk grades. Cheap: no model evaluation, just resampling means.

Usage
-----
    python scripts/03b_bootstrap_segments.py --tag full_strat
    python scripts/03b_bootstrap_segments.py --tag full_strat --n-boot 10000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditsurv.config import load_config  # noqa: E402
from creditsurv.explain.segments import bootstrap_segment_ranks  # noqa: E402
from creditsurv.reporting.tables import to_markdown, write_json  # noqa: E402

# Which (target grade, features) to test. These are the features whose rank in F or
# G fell outside the whole A-E range in the stratified point estimates.
CHECKS = {
    "G": ["annual_inc", "mths_since_recent_inq", "percent_bc_gt_75", "dti"],
    "F": ["percent_bc_gt_75", "annual_inc", "dti"],
}
REFERENCE = ["A", "B", "C", "D", "E"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--tag", default="full_strat")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--segment", default="grade")
    args = ap.parse_args()

    cfg = load_config(args.config)
    path = cfg.paths.tables_dir / f"03_per_borrower_importance_{args.tag}.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found. Re-run 03_explain.py; per-borrower "
              f"attributions are only persisted by the current version.",
              file=sys.stderr)
        return 2

    df = pd.read_parquet(path)
    seg_col = f"_seg_{args.segment}"
    feats = [c for c in df.columns if not c.startswith("_")]
    seg = df[seg_col]
    print(f"{len(df)} borrowers, {len(feats)} features, "
          f"per-level counts {seg.value_counts().sort_index().to_dict()}")

    results = {}
    for target, features in CHECKS.items():
        if target not in set(seg):
            continue
        out = bootstrap_segment_ranks(
            df[feats], seg, target=target, reference_levels=REFERENCE,
            features=features, n_boot=args.n_boot, seed=cfg.explain.seed,
        )
        print(f"\n=== grade {target} vs pooled {'+'.join(REFERENCE)} "
              f"({args.n_boot} stratified bootstrap replicates, 95% CI) ===")
        view = out[["feature", "rank_target", "rank_target_lo", "rank_target_hi",
                    "rank_reference", "rank_shift", "rank_shift_lo", "rank_shift_hi",
                    "shift_ci_excludes_zero", "share_ratio", "share_ratio_lo",
                    "share_ratio_hi", "p_outside_reference_range",
                    "reference_rank_range"]]
        print(to_markdown(view, floatfmt="{:.2f}"))
        results[target] = out.to_dict(orient="records")

    write_json({"tag": args.tag, "n_boot": args.n_boot, "reference": REFERENCE,
                "results": results},
               cfg.paths.tables_dir / f"03_segment_bootstrap_{args.tag}.json")
    print(f"\nwrote 03_segment_bootstrap_{args.tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
