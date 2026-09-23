"""Score an applicant file from the command line, or in the background for the app.

The dashboard runs this through :mod:`creditsurv.runner` -- the same detached-process
machinery the pipeline page uses -- so a large upload keeps going while the browser
is elsewhere, and the page tails this script's output. It is equally usable on its
own:

    python scripts/06_score_upload.py --file applicants.csv
    python scripts/06_score_upload.py --file applicants.csv --model-tag holdout \
        --threshold 0.25 --max-explained 500

All of the work is in creditsurv.batch; this file only parses arguments, prints
progress in the wrapper's wording, and turns a failure into an exit code:

    0  finished          2  bad arguments or missing input
    3  the file or the model was refused (the message says which, and how to fix it)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditsurv.batch import BatchError, run_batch  # noqa: E402
from creditsurv.config import load_config  # noqa: E402

STEP_LABELS = {"check": "Checking file", "clean": "Cleaning data",
               "profile": "Profiling data", "score": "Scoring applicants",
               "explain": "Explaining decisions", "files": "Preparing files"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help="CSV of applicants to score")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--model-tag", default=None, help="default: decision.model_tag")
    ap.add_argument("--model", default=None, choices=["discrete_hazard", "cox"])
    ap.add_argument("--threshold", type=float, default=None,
                    help="reject at or above this default probability "
                         "(default: decision.reject_at_or_above)")
    ap.add_argument("--max-explained", type=int, default=None,
                    help="rejected applicants to explain, ~2.7s each "
                         "(default: decision.max_explained)")
    ap.add_argument("--chunk-rows", type=int, default=None,
                    help="rows per block; peak memory follows this, not the file "
                         "size (default: decision.chunk_rows)")
    ap.add_argument("--explainer", default=None,
                    choices=["survshap", "treeshap", "auto"],
                    help="which explainer writes the reasons "
                         "(default: decision.bulk_explainer)")
    ap.add_argument("--run-dir", default=None,
                    help="write the outputs here instead of a new timestamped "
                         "folder under outputs/runs (used by the dashboard)")
    args = ap.parse_args(argv)

    src = Path(args.file)
    if not src.exists():
        print(f"ERROR: {src} not found.", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    started = time.perf_counter()
    name = src.name
    if name.startswith("input_"):        # the dashboard saves the upload as input_<name>
        name = name[len("input_"):]

    def progress(step: str, state: str, message: str = "") -> None:
        label = STEP_LABELS.get(step, step)
        if state == "running":
            print(f"[{label}] started", flush=True)
        elif state == "done":
            print(f"[{label}] done   {message}", flush=True)
        else:
            print(f"[{label}] FAILED {message}", flush=True)

    try:
        result = run_batch(src, name, cfg, model_tag=args.model_tag,
                           model_name=args.model, threshold=args.threshold,
                           max_explained=args.max_explained,
                           chunk_rows=args.chunk_rows, explainer=args.explainer,
                           run_dir=Path(args.run_dir) if args.run_dir else None,
                           progress=progress)
    except BatchError as exc:
        print(f"\nFAILED: {exc.message}", file=sys.stderr)
        if exc.fix:
            print(f"How to fix it: {exc.fix}", file=sys.stderr)
        print(f"Details: {exc.detail}", file=sys.stderr)
        return 3

    s = result.summary
    print(f"\nScored {s['n_rows']:,} applicants in "
          f"{time.perf_counter() - started:.1f}s: "
          f"{s['n_approved']:,} approved, {s['n_rejected']:,} rejected "
          f"({s['approval_rate']:.1%} approval rate) at a "
          f"{s['horizon_months']}-month threshold of {s['threshold']:.0%}.")
    print(f"Data drift: {s['drift_status']} "
          f"({s['drift_features_large']} large, {s['drift_features_moderate']} moderate).")
    print(f"Feature coverage: {s['features_present']} of {s['features_expected']}"
          + ("  DEGRADED" if s["degraded_coverage"] else ""))
    print(f"Notices written: {s['n_notices']:,} of {s['n_rejected']:,} rejected "
          f"applicants, explained with {s['explainer']}.")
    if s.get("n_rejected_without_reasons"):
        print(f"WARNING: {s['n_rejected_without_reasons']:,} rejected applicant(s) "
              f"have NO reasons (cap {s['max_explained']}); those rows say so.")
    print(f"\nOutputs in {result.run_dir}:")
    for name in sorted(result.files):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
