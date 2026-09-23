"""Stage 3d -- does TreeSHAP state the same reasons as SurvSHAP(t)?

Runs the comparison pre-registered in FINDINGS section 7. Both explainers are
passed through the same ``build_adverse_action_notice``, so only the attributions
differ: the Regulation B filtering, the geography exclusion and the wording are
identical.

The sample is applicants the configured threshold **declines**, since those are
the ones a notice is written for. Two numbers decide it, and they were fixed
before this script was run:

    E1  same top stated reason           >= 90% of applicants
    E2  mean top-4 reason-set overlap    >= 0.75

A third number is reported for interpretation only: SurvSHAP(t) against itself
under two seeds, which is the best any stand-in could do. It cannot rescue a
failed bar.

    python scripts/03d_explainer_validation.py --model-tag full --n-explain 1000
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
from creditsurv.explain.adverse_action import build_adverse_action_notice  # noqa: E402
from creditsurv.explain.survshap import explain_survshap  # noqa: E402
from creditsurv.explain.tree_shap import explain_tree_shap  # noqa: E402
from creditsurv.features.build import build_design_matrix  # noqa: E402
from creditsurv.pipeline import load_model_bundle, resolve_data_source  # noqa: E402
from creditsurv.provenance import build_stamp, guard_outputs  # noqa: E402
from creditsurv.reporting.tables import to_markdown, write_json, write_table  # noqa: E402

E1_BAR = 0.90
E2_BAR = 0.75


def _reasons(expl, obs: int, horizon: int, model_name: str) -> list[str]:
    notice = build_adverse_action_notice(expl, obs=obs, horizon_months=horizon,
                                         model_name=model_name)
    return [r.reason for r in notice.reasons]


def _agreement(a: list[list[str]], b: list[list[str]]) -> dict:
    top1 = [x[:1] == y[:1] and bool(x) for x, y in zip(a, b)]
    overlap = [len(set(x[:4]) & set(y[:4])) / 4.0 for x, y in zip(a, b)]
    both_empty = sum(1 for x, y in zip(a, b) if not x and not y)
    return {
        "n": len(a),
        "top1_agreement": float(np.mean(top1)) if a else float("nan"),
        "mean_top4_overlap": float(np.mean(overlap)) if a else float("nan"),
        "median_top4_overlap": float(np.median(overlap)) if a else float("nan"),
        "exact_top4_set_match": float(np.mean([o == 1.0 for o in overlap])) if a else 0.0,
        "no_reasons_either_side": both_empty,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--tag", default="explainer")
    ap.add_argument("--model-tag", default="full")
    ap.add_argument("--n-explain", type=int, default=1000,
                    help="declined applicants compared (pre-registered: >= 1000)")
    ap.add_argument("--n-ceiling", type=int, default=300,
                    help="subset used for the SurvSHAP-against-itself reference")
    ap.add_argument("--threshold", type=float, default=None,
                    help="default: decision.reject_at_or_above")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg.paths.ensure_dirs()
    out_json = cfg.paths.tables_dir / f"03d_explainer_validation_{args.tag}.json"
    out_csv = cfg.paths.tables_dir / f"03d_explainer_per_applicant_{args.tag}.csv"
    refused = guard_outputs([out_json, out_csv], args.overwrite,
                            script="03d_explainer_validation.py")
    if refused:
        return refused

    threshold = (args.threshold if args.threshold is not None
                 else cfg.decision.reject_at_or_above)
    horizon = int(cfg.decision.horizon_months)

    bundle, model_path = load_model_bundle(cfg.paths.models_dir, args.model_tag)
    if "discrete_hazard" not in bundle["artefacts"]:
        print("ERROR: TreeSHAP needs the discrete-hazard model; this bundle has none.",
              file=sys.stderr)
        return 2
    model = bundle["artefacts"]["discrete_hazard"]
    src = resolve_data_source(bundle, cfg, args.model_tag)
    print(f"model {args.model_tag}; data {src}")

    cols = list(bundle["spec"].all_columns) + ["duration_months", "event"]
    df = pd.read_parquet(src, columns=cols)
    for c in [c for c in df.columns if df[c].dtype == object]:
        df[c] = df[c].astype("category")
    test = df.loc[bundle["test_idx"].intersection(df.index)]
    print(f"test split: {len(test):,} loans")

    # Declined applicants only: a notice is written for those.
    rng = np.random.default_rng(cfg.explain.seed)
    pool = test.sample(min(len(test), 60_000), random_state=cfg.explain.seed)
    dm_pool = build_design_matrix(pool, bundle["spec"], flavour="gbm")
    pd_h = 1.0 - model.predict_survival(dm_pool.X, np.array([float(horizon)]))[:, 0]
    declined = np.flatnonzero(pd_h >= threshold)
    print(f"declined at {threshold:.2f}: {len(declined):,} of {len(pool):,} "
          f"({len(declined) / len(pool):.1%})")
    if len(declined) < args.n_explain:
        print(f"ERROR: only {len(declined)} declined applicants available, "
              f"{args.n_explain} requested.", file=sys.stderr)
        return 2
    pick = np.sort(rng.choice(declined, args.n_explain, replace=False))
    X = dm_pool.X.iloc[pick]
    background = dm_pool.X
    times = np.array(cfg.model.eval_horizons_months, dtype=float)

    # ---------------------------------------------------------- TreeSHAP ----
    t0 = time.perf_counter()
    tree = explain_tree_shap(model, X, horizon_months=float(horizon), times=times)
    tree_seconds = time.perf_counter() - t0
    tree_reasons = [_reasons(tree, i, horizon, "treeshap") for i in range(len(X))]
    print(f"TreeSHAP: {tree_seconds:.1f}s for {len(X):,} applicants "
          f"({tree_seconds / len(X) * 1000:.2f} ms each)")

    # -------------------------------------------------------- SurvSHAP(t) ----
    t0 = time.perf_counter()
    np.random.seed(cfg.explain.seed)
    surv = explain_survshap(model, X, background, times,
                            nsamples=cfg.explain.kernel_nsamples,
                            n_background=cfg.explain.n_background,
                            seed=cfg.explain.seed)
    surv_seconds = time.perf_counter() - t0
    surv_reasons = [_reasons(surv, i, horizon, "survshap") for i in range(len(X))]
    print(f"SurvSHAP(t): {surv_seconds:.1f}s for {len(X):,} applicants "
          f"({surv_seconds / len(X):.2f} s each)")

    result = _agreement(surv_reasons, tree_reasons)
    e1 = result["top1_agreement"] >= E1_BAR
    e2 = result["mean_top4_overlap"] >= E2_BAR

    # ------------------------------------------- reference ceiling only ----
    ceiling = {}
    n_ceiling = min(args.n_ceiling, len(X))
    if n_ceiling:
        sub = X.iloc[:n_ceiling]
        np.random.seed(cfg.explain.seed + 1)
        surv_b = explain_survshap(model, sub, background, times,
                                  nsamples=cfg.explain.kernel_nsamples,
                                  n_background=cfg.explain.n_background,
                                  seed=cfg.explain.seed + 1)
        reasons_b = [_reasons(surv_b, i, horizon, "survshap") for i in range(n_ceiling)]
        ceiling = _agreement(surv_reasons[:n_ceiling], reasons_b)
        print(f"reference ceiling (SurvSHAP vs itself, {n_ceiling} applicants): "
              f"top-1 {ceiling['top1_agreement']:.1%}, "
              f"mean top-4 overlap {ceiling['mean_top4_overlap']:.2f}")

    per_applicant = pd.DataFrame({
        "row_index": X.index.to_numpy(),
        "pd_at_horizon": np.round(pd_h[pick], 4),
        "survshap_top1": [r[0] if r else "" for r in surv_reasons],
        "treeshap_top1": [r[0] if r else "" for r in tree_reasons],
        "top1_same": [bool(a[:1] == b[:1] and a) for a, b in
                      zip(surv_reasons, tree_reasons)],
        "top4_overlap": [len(set(a[:4]) & set(b[:4])) / 4.0 for a, b in
                         zip(surv_reasons, tree_reasons)],
        "survshap_reasons": ["; ".join(r) for r in surv_reasons],
        "treeshap_reasons": ["; ".join(r) for r in tree_reasons],
    })
    write_table(per_applicant, out_csv)

    verdict = "PASS" if (e1 and e2) else "FAIL"
    payload = {
        "tag": args.tag,
        "model_tag": args.model_tag,
        "model": "discrete_hazard",
        "n_applicants": int(len(X)),
        "threshold": float(threshold),
        "horizon_months": horizon,
        "nsamples": cfg.explain.kernel_nsamples,
        "n_background": cfg.explain.n_background,
        "bars": {"E1_top1_agreement": E1_BAR, "E2_mean_top4_overlap": E2_BAR},
        "treeshap_vs_survshap": result,
        "E1_pass": bool(e1),
        "E2_pass": bool(e2),
        "verdict": verdict,
        "reference_ceiling_survshap_vs_itself": ceiling,
        "seconds": {"treeshap_total": round(tree_seconds, 2),
                    "treeshap_per_applicant_ms": round(tree_seconds / len(X) * 1000, 3),
                    "survshap_total": round(surv_seconds, 1),
                    "survshap_per_applicant_s": round(surv_seconds / len(X), 3)},
        "projected_100k_rejected": {
            "treeshap_minutes": round(tree_seconds / len(X) * 100_000 / 60, 1),
            "survshap_hours": round(surv_seconds / len(X) * 100_000 / 3600, 1)},
        "provenance": build_stamp(
            stage="03d_explainer_validation",
            inputs={"data": src, "model": model_path},
            outputs={"result": out_json, "per_applicant": out_csv},
            config_path=args.config,
            args={"tag": args.tag, "model_tag": args.model_tag,
                  "n_explain": args.n_explain, "n_ceiling": args.n_ceiling,
                  "threshold": threshold}),
    }
    write_json(payload, out_json)

    print("\n=== pre-registered bar ===")
    print(to_markdown(pd.DataFrame([
        {"criterion": "E1 same top reason", "bar": f">= {E1_BAR:.0%}",
         "observed": f"{result['top1_agreement']:.1%}",
         "result": "PASS" if e1 else "FAIL"},
        {"criterion": "E2 mean top-4 overlap", "bar": f">= {E2_BAR:.2f}",
         "observed": f"{result['mean_top4_overlap']:.2f}",
         "result": "PASS" if e2 else "FAIL"},
    ])))
    print(f"\nVERDICT: {verdict}")
    if ceiling:
        print(f"(reference only: SurvSHAP against itself scores "
              f"{ceiling['top1_agreement']:.1%} / {ceiling['mean_top4_overlap']:.2f} "
              f"on the same measures; this cannot change the verdict.)")
    print(f"\n100,000 declined applicants would take "
          f"{payload['projected_100k_rejected']['treeshap_minutes']:.1f} min with "
          f"TreeSHAP and {payload['projected_100k_rejected']['survshap_hours']:.1f} h "
          f"with SurvSHAP(t).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
