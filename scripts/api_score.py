"""Score a file through the running API, end to end, and time every step.

The third way to use the pipeline, beside the dashboard and 06_score_upload.py:
nothing here scores anything -- it uploads, starts the run, polls, and reads back
what the API reports, exactly as the dashboard does. What it adds is a timing
table: the time spent in each HTTP call, in each pipeline stage (from the run's own
stage log), in Phase 2, and whether the model came from the API's cache.

    python scripts/api_score.py --file applicants.csv
    python scripts/api_score.py --file applicants.csv --api http://127.0.0.1:8000 --json out.json

Exit codes: 0 finished, 3 refused or failed its checks (the message says which),
2 the API could not be reached.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True)
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--model-tag", default=None)
    ap.add_argument("--allow-unapproved-model", action="store_true")
    ap.add_argument("--phase2", default="auto", choices=["auto", "defer"])
    ap.add_argument("--timeout", type=float, default=7200, help="seconds to wait")
    ap.add_argument("--json", default=None, help="write the timings here as well")
    args = ap.parse_args(argv)

    client = httpx.Client(base_url=args.api, timeout=httpx.Timeout(30.0, read=900.0))
    calls: list[dict] = []

    def call(method: str, path: str, **kw):
        t = time.perf_counter()
        try:
            r = client.request(method, path, **kw)
        except httpx.HTTPError as exc:
            print(f"ERROR: the API at {args.api} did not answer: {exc!r}", file=sys.stderr)
            raise SystemExit(2)
        calls.append({"call": f"{method} {path.split('?')[0]}",
                      "seconds": round(time.perf_counter() - t, 3),
                      "status": r.status_code})
        return r

    started = time.perf_counter()
    health = call("GET", "/health").json()
    print(f"API on {health['runtime']}; models loaded: {health['models']['loaded'] or 'none'}")

    path = Path(args.file)
    with open(path, "rb") as fh:
        r = call("POST", "/uploads", files={"file": (path.name, fh, "text/csv")})
    if r.status_code >= 400:
        print(f"REFUSED at upload: {r.json().get('message')}", file=sys.stderr)
        return 3
    up = r.json()
    check = call("POST", f"/uploads/{up['upload_id']}/check",
                 json={"model_tag": args.model_tag}).json()
    print(f"Schema check: {len(check['features']) - len(check['optional_missing']) - len(check['required_missing'])}"
          f" of {len(check['features'])} features present or derived; "
          f"renamed {check['mapping'] or 'none'}; derived {list(check['derived']) or 'none'}; "
          f"required missing {check['required_missing'] or 'none'}; "
          f"optional missing {check['optional_missing'] or 'none'}")

    r = call("POST", "/runs", json={"upload_id": up["upload_id"],
                                    "model_tag": args.model_tag,
                                    "mapping": check["mapping"] or None,
                                    "allow_unapproved_model": args.allow_unapproved_model,
                                    "phase2": args.phase2})
    run_id = r.json()["run_id"]
    print(f"Run {run_id} ({'background runner' if r.json()['background'] else 'in the API'})")

    polls, last_line = 0, ""
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        s = call("GET", f"/runs/{run_id}/status").json()
        polls += 1
        line = "  ".join(f"{x['label']}:{x['state']}" for x in s["stages"])
        if line != last_line:
            print(f"  {time.perf_counter() - started:7.1f}s  {line}")
            last_line = line
        p2 = s["phase2"].get("state")
        if s["state"] in ("refused", "failed checks", "interrupted", "finished",
                          "awaiting phase 2 choice", "phase 2 stopped") or (
                s["state"] == "reasons pending" and args.phase2 == "defer"):
            break
        time.sleep(1.0)
    wall = time.perf_counter() - started
    detail = call("GET", f"/runs/{run_id}", params={"preview_rows": 0}).json()
    health2 = call("GET", "/health").json()
    s = detail["summary"]
    stages = {x["key"]: x for x in detail["status"]["stages"]}

    out = {"run_id": run_id, "state": detail["state"], "error": detail.get("error"),
           "wall_seconds": round(wall, 2), "polls": polls,
           "phase1_seconds": s.get("phase1_seconds"),
           "phase2_seconds": s.get("phase2_seconds"),
           "seconds_by_step": s.get("seconds_by_step"),
           "stage_seconds": {k: v.get("seconds") for k, v in stages.items()},
           "model_loads_before": health["models"]["loads"],
           "model_loads_after": health2["models"]["loads"],
           "model_load_seconds": health2["models"]["load_seconds"],
           "http": calls,
           "summary": {k: s.get(k) for k in (
               "n_rows_in_file", "n_rows", "n_duplicates_removed", "n_approved",
               "n_rejected", "approval_rate", "n_explained", "n_notices",
               "n_pending_manual_review", "fair_lending_flag_share",
               "n_fair_lending_flagged", "fair_lending_review_required",
               "for_lending_decisions", "failed_checks", "drift_status",
               "features_present", "features_expected")}}
    checks = detail.get("checks") or {}
    out["checks"] = [dict(zip(checks.get("columns", []), row))
                     for row in checks.get("data", [])]

    print(f"\n{detail['state'].upper()} in {wall:.1f}s wall")
    if detail.get("error"):
        print(f"  {detail['error'].get('message')}")
    for c in out["checks"]:
        print(f"  {c.get('status'):<10} {c.get('check'):<34} {str(c.get('detail'))[:110]}")
    print("\nTime per pipeline step (from the run's own record):")
    for k, v in (s.get("seconds_by_step") or {}).items():
        print(f"  {k:<22} {v:8.2f} s")
    print(f"  {'phase 2':<22} {float(s.get('phase2_seconds') or 0):8.2f} s")
    print("\nHTTP calls (client side):")
    agg: dict[str, list] = {}
    for c in calls:
        agg.setdefault(c["call"], []).append(c["seconds"])
    for k, v in agg.items():
        print(f"  {k:<40} x{len(v):<4} total {sum(v):7.2f} s  max {max(v):6.2f} s")
    print(f"\nModel loads in the API: {out['model_loads_before']} before, "
          f"{out['model_loads_after']} after "
          f"({'loaded for this run' if out['model_loads_after'] > out['model_loads_before'] else 'served from the cache'}).")
    print("Summary: " + json.dumps(out["summary"]))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0 if detail["state"] in ("finished", "reasons pending",
                                    "awaiting phase 2 choice") else 3


if __name__ == "__main__":
    raise SystemExit(main())
