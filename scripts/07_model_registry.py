"""The model registry, from the command line: register, check, approve, deprecate.

    python scripts/07_model_registry.py show
    python scripts/07_model_registry.py rules --model-tag full_applicant_nogeo
    python scripts/07_model_registry.py approve --model-tag full_applicant_nogeo \
        --by "Your Name" --findings 7l
    python scripts/07_model_registry.py register --model-tag holdout_x --status benchmark
    python scripts/07_model_registry.py set-status --model-tag full --status deprecated \
        --reason "..."

Reads JSON and YAML only -- it never unpickles a model -- so it runs on Windows even
where Smart App Control blocks lightgbm. The evidence it checks (03d, 03e, cleaning
values) is produced in WSL; copy it back first (run_linux.ps1 -CopyBack), then run
this on Windows, where config/ is the source of truth.

Exit codes: 0 done; 2 bad arguments; 4 approval refused (a rule failed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditsurv.config import load_config  # noqa: E402
from creditsurv.explain.adverse_action import NOT_DISCLOSABLE  # noqa: E402
from creditsurv.provenance import PROJECT_ROOT  # noqa: E402
from creditsurv.registry import (APPROVAL_RULES, STATUSES, ModelRecord,  # noqa: E402
                                 Registry, RegistryError, approve_model,
                                 evaluate_rules, load_registry, registry_path,
                                 sha256_of)

DEFECT_7C = {"id": "7c", "blocking": True,
             "summary": "no credit score or employment length: default_spec dropped "
                        "fico_range_* before fico_midpoint existed (FINDINGS 7c)"}


def record_from_metrics(tag: str, cfg, status: str, reason: str = "") -> ModelRecord:
    """A registry entry built from what Stage 2 recorded about the model."""
    metrics_path = cfg.paths.tables_dir / f"02_metrics_{tag}.json"
    if not metrics_path.exists():
        raise SystemExit(f"ERROR: {metrics_path} not found; the registry records what "
                         f"Stage 2 wrote about a model, so train it first.")
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    model_path = cfg.paths.models_dir / f"02_models_{tag}.pkl"
    features = {"numeric": list(m["features"].get("numeric", [])),
                "categorical": list(m["features"].get("categorical", []))}
    every = features["numeric"] + features["categorical"]
    defects = [] if "fico_midpoint" in every else [dict(DEFECT_7C)]
    metrics = {}
    for r in m.get("results", []):
        metrics[r["model"]] = {k: r[k] for k in
                               ("split", "n", "event_rate", "concordance", "ibs",
                                "auc_12m", "auc_36m") if k in r}
    training = {k: m.get(k) for k in
                ("data_source", "source", "split_scheme", "oot_cutoff", "n_train",
                 "n_test", "time_bin_months", "negative_subsample", "with_lc_grade",
                 "with_derived", "dropped_by_request", "added_by_request")
                if m.get(k) is not None}
    training.setdefault("data_source", str(m.get("source", "")).replace("\\", "/"))
    training.pop("source", None)
    return ModelRecord(
        tag=tag, status=status, status_reason=reason,
        summary=f"{len(features['numeric'])} numeric + "
                f"{len(features['categorical'])} categorical features",
        model_file=model_path.as_posix(), model_sha256=sha256_of(model_path) or "",
        features=features, training=training, known_defects=defects,
        non_disclosable_features=sorted(set(every) & set(NOT_DISCLOSABLE)),
        validation_metrics=metrics)


def _open(cfg) -> Registry:
    try:
        return load_registry(cfg)
    except RegistryError:
        return Registry(path=registry_path(cfg), models={})


def _print_rules(results) -> None:
    for r in results:
        print(f"  {'PASS' if r.passed else 'FAIL'}  {r.rule:<26} {r.detail}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["show", "rules", "approve", "register",
                                        "set-status", "check"])
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--model-tag")
    ap.add_argument("--status", choices=STATUSES)
    ap.add_argument("--reason", default="")
    ap.add_argument("--by", help="who approves (approve)")
    ap.add_argument("--findings", help="FINDINGS section recording the decision")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    cfg = load_config(PROJECT_ROOT / args.config if not Path(args.config).is_absolute()
                      else args.config)
    reg = _open(cfg)
    d = cfg.decision
    settings = (int(d.explain_nsamples), int(d.explain_n_background))

    if args.command in ("show", "check"):
        if not reg.models:
            print(f"ERROR: no registry at {reg.path}", file=sys.stderr)
            return 2
        for tag, rec in sorted(reg.models.items()):
            flag = " <- decision.model_tag" if tag == d.model_tag else ""
            print(f"{tag:<26} {rec.status:<11} {rec.status_reason[:70]}{flag}")
        return 0

    if not args.model_tag:
        print("ERROR: --model-tag is required.", file=sys.stderr)
        return 2
    tag = args.model_tag

    if args.command == "register":
        if tag in reg.models:
            print(f"ERROR: {tag} is already registered; use set-status.", file=sys.stderr)
            return 2
        status = args.status or "benchmark"
        if status == "approved":
            print("ERROR: register never approves; use approve.", file=sys.stderr)
            return 2
        reg.models[tag] = record_from_metrics(tag, cfg, status, args.reason)
        print(f"registered {tag} as {status} -> {reg.save()}")
        return 0

    rec = reg.get(tag)
    if rec is None:
        print(f"ERROR: {tag} is not registered.", file=sys.stderr)
        return 2

    if args.command == "set-status":
        if args.status in (None, "approved"):
            print("ERROR: set-status takes candidate, benchmark or deprecated; "
                  "approval goes through approve.", file=sys.stderr)
            return 2
        if not args.reason:
            print("ERROR: --reason is required: a status change says why.",
                  file=sys.stderr)
            return 2
        rec.status, rec.status_reason, rec.approval = args.status, args.reason, {}
        print(f"{tag} -> {args.status} ({reg.save()})")
        return 0

    print(f"Approval rules for {tag} (scoring settings nsamples={settings[0]}, "
          f"n_background={settings[1]}):")

    if args.command == "rules":
        results = evaluate_rules(cfg, rec)
        _print_rules(results)
        failed = [r for r in results if not r.passed]
        print("\nAll rules pass." if not failed else
              f"\n{len(failed)} rule(s) fail; {tag} cannot be approved yet.")
        for rule, text in APPROVAL_RULES.items():
            print(f"  {rule}: {text}")
        return 0

    # approve -- the same function the Model registry page's Approve button runs
    if not (args.by and args.findings):
        print("ERROR: approve needs --by and --findings: an approval names who made "
              "it and where the decision is recorded.", file=sys.stderr)
        return 2
    outcome = approve_model(cfg, tag, by=args.by, findings=args.findings,
                            note=args.note, registry=reg)
    _print_rules(outcome.results)
    if not outcome.approved:
        print(f"\nREFUSED: {outcome.refusal}", file=sys.stderr)
        return 4
    print(f"\nAPPROVED: {tag} ({reg.path}). Explainers: "
          f"{', '.join(rec.approval['explainers'])}; settings "
          f"{rec.approval['explain_settings']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
