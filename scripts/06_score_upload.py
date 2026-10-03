"""Score an applicant file from the command line, or in the background for the app.

Two phases, the same two functions every caller uses (creditsurv.batch.score_file,
then creditsurv.phase2.explain_run):

  Phase 1  check, clean, score, decide, drift, write the decision files and check
           them -- minutes for any file size. Rejected rows say "reasons pending".
  Phase 2  reasons and adverse-action notices, written in as they are generated,
           resumable if interrupted.

    python scripts/06_score_upload.py --file applicants.csv
    python scripts/06_score_upload.py --file big.csv --phase2 defer     # decisions only
    python scripts/06_score_upload.py --explain outputs/runs/<run>                  # all
    python scripts/06_score_upload.py --explain outputs/runs/<run> --phase2 sample --sample-n 500
    python scripts/06_score_upload.py --explain outputs/runs/<run> --phase2 skip
    python scripts/06_score_upload.py --explain outputs/runs/<run> --row 17 --row 42

With ``--phase2 auto`` (the default) a file with at most
``decision.explain_confirm_above`` rejected applicants goes straight on to Phase 2;
a larger one stops after Phase 1 and prints the three choices, since explaining it
can take hours and the choice is the operator's. The dashboard runs this script
through :mod:`creditsurv.runner` for large uploads and for every Phase 2, so the
work outlives the browser tab.

Exit codes:

    0  finished (or decisions written and Phase 2 awaiting a choice)
    2  bad arguments or missing input
    3  the file or the model was refused, or the run failed its own checks (the
       message says which, and how to fix it)

Only a model approved in config/models.yaml scores for lending decisions.
``--allow-unapproved-model`` is the explicit override: the run proceeds and every
output is stamped "not for lending decisions".
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditsurv import batch  # noqa: E402
from creditsurv.batch import BatchError  # noqa: E402
from creditsurv.config import load_config  # noqa: E402
from creditsurv.environment import (blocked_imports,  # noqa: E402
                                    policy_block_message)
from creditsurv.stages import StageLog  # noqa: E402

STEP_LABELS = {"check": "Checking file", "clean": "Cleaning data",
               "profile": "Profiling data", "score": "Scoring applicants",
               "decide": "Deciding", "checks": "Run checks",
               "explain": "Explaining decisions", "files": "Preparing files"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--file", help="CSV of applicants to score (Phase 1, then Phase 2)")
    what.add_argument("--explain", metavar="RUN_DIR",
                      help="run Phase 2 on a scored run folder")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--model-tag", default=None, help="default: decision.model_tag")
    ap.add_argument("--model", default=None, choices=["discrete_hazard", "cox"])
    ap.add_argument("--threshold", type=float, default=None,
                    help="reject at or above this default probability "
                         "(default: decision.reject_at_or_above)")
    ap.add_argument("--phase2", default="auto",
                    choices=["auto", "all", "sample", "skip", "defer"],
                    help="what Phase 2 does: auto (all, unless there are more than "
                         "decision.explain_confirm_above rejected applicants, then "
                         "stop and ask), all, sample (with --sample-n), skip (stamped "
                         "not for lending decisions), defer (decisions only)")
    ap.add_argument("--sample-n", type=int, default=None,
                    help="how many rejected applicants a sample explains")
    ap.add_argument("--row", type=int, action="append", default=None,
                    help="with --explain: explain this row_id now, repeatable")
    ap.add_argument("--max-explained", type=int, default=None,
                    help="explain only the first N rejected rows (an explicit cap; "
                         "the run is stamped not for lending decisions)")
    ap.add_argument("--chunk-rows", type=int, default=None,
                    help="rows per block; peak memory follows this, not the file "
                         "size (default: decision.chunk_rows)")
    ap.add_argument("--explainer", default=None,
                    choices=["survshap", "treeshap", "auto"],
                    help="which explainer writes the reasons "
                         "(default: decision.bulk_explainer)")
    ap.add_argument("--map", action="append", default=None, metavar="COLUMN=FEATURE",
                    help="a confirmed column mapping, repeatable. Without any, the "
                         "high-confidence proposals are used and reported.")
    ap.add_argument("--run-dir", default=None,
                    help="write the outputs here instead of a new timestamped "
                         "folder under outputs/runs (used by the dashboard)")
    ap.add_argument("--allow-unapproved-model", action="store_true",
                    help="score with a model the registry has not approved; every "
                         "output is stamped NOT FOR LENDING DECISIONS")
    args = ap.parse_args(argv)

    if args.phase2 == "sample" and not args.sample_n:
        print("ERROR: --phase2 sample needs --sample-n.", file=sys.stderr)
        return 2
    target = Path(args.file or args.explain)
    if not target.exists():
        print(f"ERROR: {target} not found.", file=sys.stderr)
        return 2

    blocked = blocked_imports()
    if blocked:
        print("ERROR: this machine cannot score applicants: Windows is blocking "
              + ", ".join(c.name for c in blocked) + ".", file=sys.stderr)
        print(policy_block_message(blocked), file=sys.stderr)
        return 6

    cfg = load_config(args.config)
    started = time.perf_counter()

    def progress(step: str, state: str, message: str = "") -> None:
        label = STEP_LABELS.get(step, step)
        if state == "running":
            print(f"[{label}] {'started' if not message else message}", flush=True)
        elif state == "done":
            print(f"[{label}] done   {message}", flush=True)
        else:
            print(f"[{label}] FAILED {message}", flush=True)

    def refused(exc: BatchError) -> int:
        print(f"\nFAILED: {exc.message}", file=sys.stderr)
        if exc.fix:
            print(f"How to fix it: {exc.fix}", file=sys.stderr)
        print(f"Details: {exc.detail}", file=sys.stderr)
        checks = Path(exc.run_dir) / "validation_checks.csv" if exc.run_dir else None
        if checks and checks.exists():
            print(f"Checks: {checks}", file=sys.stderr)
        return 3

    model, model_path, run_dir = None, None, None
    # With --run-dir (how the API and the dashboard hand a large file to the
    # background runner) the steps are also kept in <run>/stages.json, which the
    # pipeline view reads; printing is unchanged either way.
    stages = StageLog(Path(args.run_dir), also=progress) if args.file and args.run_dir \
        else None
    try:
        if args.file:
            name = target.name
            if name.startswith("input_"):    # the dashboard saves uploads as input_<name>
                name = name[len("input_"):]
            if stages:
                stages("check", "running", "loading the model")
            ctx = batch.load_context(cfg, args.model_tag, args.model)
            result = batch.score_file(
                target, name, cfg, threshold=args.threshold, chunk_rows=args.chunk_rows,
                explainer=args.explainer,
                mapping=dict(pair.split("=", 1) for pair in args.map) if args.map else None,
                run_dir=Path(args.run_dir) if args.run_dir else None,
                progress=stages or progress, ctx=ctx,
                allow_unapproved_model=args.allow_unapproved_model)
            if stages:
                stages.finish()
            model, model_path, run_dir = ctx.model, ctx.model_path, result.run_dir
            s = result.summary
            print(f"\nPHASE 1 done in {s['phase1_seconds']:.1f}s: {s['n_rows']:,} "
                  f"applicants, {s['n_approved']:,} approved, {s['n_rejected']:,} "
                  f"rejected. Decisions are in {run_dir}.", flush=True)
            mode = args.phase2
            if not s["n_rejected"]:
                mode = "none"
            elif mode == "auto":
                limit = int(cfg.decision.explain_confirm_above)
                if s["n_rejected"] > limit:
                    _print_choice(run_dir, s["n_rejected"], limit, cfg)
                    return _report(batch.load_result(run_dir), started)
                mode = "all"
            if mode in ("defer", "none"):
                return _report(result, started)
        else:
            run_dir = target
            mode = "all" if args.phase2 in ("auto", "defer") else args.phase2
        cap = args.max_explained
        phase2 = batch.explain_run(
            run_dir, cfg, mode=mode, sample_n=args.sample_n,
            limit=cap if cap and cap > 0 else None, only=args.row,
            model=model, model_path=model_path, progress=progress)
    except BatchError as exc:
        if stages and stages.state.get("finished") is None:
            stages.fail(exc.message, exc.detail, exc.fix)
        return refused(exc)
    return _report(batch.load_result(phase2.run_dir), started)


def _print_choice(run_dir: Path, n: int, limit: int, cfg) -> None:
    from creditsurv.phase2 import estimate_seconds

    hours = estimate_seconds(n, cfg) / 3600
    me = "python scripts/06_score_upload.py --explain"
    print(f"\nPHASE 2 NOT STARTED: {n:,} rejected applicants is more than "
          f"decision.explain_confirm_above ({limit:,}); explaining all of them would "
          f"take about {hours:.1f} h. Choose one:")
    print(f"  {me} \"{run_dir}\"                                   # all")
    print(f"  {me} \"{run_dir}\" --phase2 sample --sample-n 500    # a random sample")
    print(f"  {me} \"{run_dir}\" --phase2 skip                     # none; not for lending")


def _report(result, started: float) -> int:
    s = result.summary
    if not s["for_lending_decisions"]:
        print(f"\n*** NOT FOR LENDING DECISIONS: {s['not_for_lending_reasons']} ***")
    print(f"\nModel {s['model_tag']}: registry status {s['model_registry_status']}"
          f"{'' if s['model_approved'] else ' (NOT approved; overridden)'}.")
    print("Checks on the written outputs:")
    for c in result.checks.itertuples():
        print(f"  {c.status:<10} {c.check:<34} {c.detail}")
    print(f"\nScored {s['n_rows']:,} applicants: {s['n_approved']:,} approved, "
          f"{s['n_rejected']:,} rejected ({s['approval_rate']:.1%} approval rate) at a "
          f"{s['horizon_months']}-month threshold of {s['threshold']:.0%}. "
          f"Phase 1 {s.get('phase1_seconds') or 0:.1f}s"
          + (f", Phase 2 {s['phase2_seconds']:.1f}s" if s.get("phase2_seconds") else "")
          + f"; {time.perf_counter() - started:.1f}s in this process.")
    print(f"Data drift: {s['drift_status']} "
          f"({s['drift_features_large']} large, {s['drift_features_moderate']} moderate).")
    print(f"Feature coverage: {s['features_present']} of {s['features_expected']}"
          + ("  DEGRADED" if s["degraded_coverage"] else ""))
    if s.get("schema_message"):
        print(s["schema_message"])
    print(f"Reasons: {s['n_explained']:,} of {s['n_rejected']:,} rejected applicants "
          f"explained with {s['explainer']}; {s.get('n_reasons_pending', 0):,} pending; "
          f"{s['n_notices']:,} notices written.")
    if s.get("n_rejected_without_reasons"):
        print(f"WARNING: {s['n_rejected_without_reasons']:,} rejected applicant(s) "
              f"have NO reasons ({s.get('phase2_mode') or 'not chosen'}); those rows "
              f"say so.")
    if s.get("n_pending_manual_review"):
        print(f"WARNING: {s['n_pending_manual_review']:,} rejected applicant(s) had no "
              f"disclosable reason: no notice issued, marked pending manual review.")
    print(f"Fair-lending monitor: {s['n_fair_lending_flagged']:,} of "
          f"{s['n_explained']:,} explained rejections "
          f"({s['fair_lending_flag_share']:.1%}) had a non-disclosable top adverse "
          f"driver" + (f" [{s['fair_lending_flag_features']}]"
                       if s['fair_lending_flag_features'] else "")
          + (f".  REVIEW REQUIRED: above the {s['fair_lending_review_share']:.0%} "
             f"review share." if s["fair_lending_review_required"] else "."))
    print(f"\nOutputs in {result.run_dir}:")
    for name in sorted(result.files):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
