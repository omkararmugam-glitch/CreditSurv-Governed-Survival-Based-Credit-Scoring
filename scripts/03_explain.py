"""Stage 3 -- SurvSHAP(t), the naive-SHAP comparison, adverse action, segments.

Answers the three Stage 3 questions and writes every result to disk:

  (a) does naive SHAP disagree with SurvSHAP(t) on THIS dataset? (measured)
  (b) an ECOA / Reg B adverse-action notice for one applicant
  (c) is feature importance stable across borrower segments?

Usage
-----
    python scripts/03_explain.py                      # defaults from config
    python scripts/03_explain.py --n-explain 200      # faster
    python scripts/03_explain.py --model cox
    python scripts/03_explain.py --tag full
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditsurv.config import load_config  # noqa: E402
from creditsurv.explain.adverse_action import build_adverse_action_notice  # noqa: E402
from creditsurv.explain.compare import compare_explanations  # noqa: E402
from creditsurv.explain.naive_shap import explain_naive_shap  # noqa: E402
from creditsurv.explain.segments import analyse_segment_stability  # noqa: E402
from creditsurv.explain.survshap import check_efficiency, explain_survshap  # noqa: E402
from creditsurv.features.build import add_derived_features, build_design_matrix  # noqa: E402
from creditsurv.provenance import (  # noqa: E402
    build_stamp,
    find_existing_outputs,
    guard_outputs,
)
from creditsurv.reporting import figures as figs  # noqa: E402
from creditsurv.reporting.tables import to_markdown, write_json, write_table  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--tag", default="dev",
                    help="suffix for OUTPUT filenames")
    ap.add_argument("--model-tag", default=None,
                    help="tag of the Stage 2 model to load; defaults to --tag. Set it "
                         "separately to write a variant run under a new output tag "
                         "without overwriting an earlier run's results.")
    ap.add_argument("--model", default="discrete_hazard",
                    choices=["discrete_hazard", "cox"])
    ap.add_argument("--n-explain", type=int, default=None)
    ap.add_argument("--n-background", type=int, default=None)
    ap.add_argument("--nsamples", type=int, default=None)
    ap.add_argument("--at-month", type=float, default=36.0,
                    help="horizon for naive SHAP and for the notice")
    ap.add_argument("--stratify-by", default=None,
                    help="column to stratify the explained borrowers by, with EQUAL "
                         "allocation per level (e.g. 'grade'). Random sampling "
                         "mirrors the population, so rare-but-important strata such "
                         "as grades F and G are missed; equal allocation makes them "
                         "measurable at no extra runtime.")
    ap.add_argument("--per-stratum", type=int, default=36,
                    help="borrowers per stratum when --stratify-by is set")
    ap.add_argument("--skip-naive", action="store_true",
                    help="skip the naive-SHAP comparison. Use when the comparison is "
                         "already settled from another run; halves the runtime, since "
                         "naive SHAP costs about as much as SurvSHAP(t) itself.")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing existing outputs for this --tag")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.paths.ensure_dirs()

    model_tag = args.model_tag or args.tag

    # Checked before the model or data is loaded. This is the guard that would
    # have prevented the earlier near-miss, where a variant run was launched with
    # the tag of the settled comparison result.
    refused = guard_outputs(
        find_existing_outputs([cfg.paths.tables_dir, cfg.paths.figures_dir],
                              "03", args.tag),
        args.overwrite, script="03_explain.py")
    if refused:
        return refused

    model_path = cfg.paths.models_dir / f"02_models_{model_tag}.pkl"
    if not model_path.exists():
        print(f"ERROR: {model_path} not found. Run scripts/02_train_models.py first.",
              file=sys.stderr)
        return 2
    with open(model_path, "rb") as fh:
        bundle = pickle.load(fh)

    artefacts, spec = bundle["artefacts"], bundle["spec"]
    if args.model not in artefacts:
        print(f"ERROR: model {args.model!r} not in {model_path}. "
              f"Available: {[k for k in artefacts if not k.endswith(('columns','standardisation','fill_values'))]}",
              file=sys.stderr)
        return 2
    model = artefacts[args.model]

    src = (cfg.paths.labeled_parquet if model_tag.startswith("full")
           else cfg.paths.dev_sample_parquet)
    df = pd.read_parquet(src)
    for col in [c for c in df.columns if df[c].dtype == object and c != "id"]:
        df[col] = df[col].astype("category")

    test_idx = bundle["test_idx"].intersection(df.index)
    test = df.loc[test_idx]

    n_explain = args.n_explain or cfg.explain.n_explain
    n_background = args.n_background or cfg.explain.n_background
    nsamples = args.nsamples or cfg.explain.kernel_nsamples

    flavour = "cox" if args.model == "cox" else "gbm"
    if flavour == "cox":
        dm = build_design_matrix(
            test, spec, flavour="cox",
            standardisation=artefacts.get("cox_standardisation"),
            fill_values=artefacts.get("cox_fill_values"),
            reference_columns=artefacts.get("cox_columns"),
        )
    else:
        dm = build_design_matrix(test, spec, flavour="gbm")

    rng = np.random.default_rng(cfg.explain.seed)
    strata_counts: dict[str, int] = {}
    if args.stratify_by:
        if args.stratify_by not in test.columns:
            print(f"ERROR: --stratify-by {args.stratify_by!r} is not a column in the "
                  f"data.", file=sys.stderr)
            return 2
        # Equal allocation per level, not proportional: the whole point is to make
        # rare strata measurable, so they must not inherit their population share.
        labels = test[args.stratify_by].astype(str).to_numpy()
        positions = np.arange(len(dm.X))
        chosen = []
        for level in sorted(pd.unique(labels)):
            avail = positions[labels == level]
            take = min(args.per_stratum, len(avail))
            if take == 0:
                continue
            chosen.append(rng.choice(avail, take, replace=False))
            strata_counts[str(level)] = int(take)
        pick = np.sort(np.concatenate(chosen))
        n_explain = len(pick)
        print(f"stratified by {args.stratify_by!r}: "
              f"{strata_counts} -> {n_explain} borrowers")
        thin = {k: v for k, v in strata_counts.items() if v < args.per_stratum}
        if thin:
            print(f"  note: these strata had fewer than {args.per_stratum} available "
                  f"in the test split and were taken in full: {thin}")
    else:
        n_explain = min(n_explain, len(dm.X))
        pick = np.sort(rng.choice(len(dm.X), n_explain, replace=False))

    X_explain = dm.X.iloc[pick]
    times = np.array(cfg.model.eval_horizons_months, dtype=float)

    print(f"model={args.model}  explaining {n_explain:,} borrowers  "
          f"nsamples={nsamples}  background={n_background}  "
          f"times={[int(t) for t in times]}")

    # ---------------- SurvSHAP(t) ----------------
    # shap draws coalitions from numpy's *global* RNG, which nothing else seeds, so
    # without this two runs of identical config give slightly different
    # attributions. Seeding it makes a run reproducible end to end.
    np.random.seed(cfg.explain.seed)
    t0 = time.time()
    surv = explain_survshap(
        model, X_explain, dm.X, times,
        nsamples=nsamples, n_background=n_background, seed=cfg.explain.seed,
    )
    elapsed = time.time() - t0
    print(f"SurvSHAP(t) done in {elapsed:.1f}s ({elapsed / n_explain:.2f}s/borrower)")

    # Persist per-borrower attributions. Every segment statistic is a mean over
    # borrowers, and without the per-borrower values there is no way to put an
    # interval on it -- only a point estimate. Stored time-integrated (the same
    # quantity segment_importance averages) plus the raw phi array.
    span = max(float(times[-1] - times[0]), 1e-12)
    per_borrower = pd.DataFrame(
        np.trapezoid(np.abs(surv.phi), times, axis=2) / span,
        columns=list(surv.feature_names),
    )
    meta = test.iloc[pick].reset_index()
    for col in ("grade", "purpose", "term_months", "annual_inc"):
        if col in meta.columns:
            per_borrower[f"_seg_{col}"] = meta[col].astype(str).to_numpy()
    per_borrower["_row_index"] = meta.iloc[:, 0].to_numpy()
    per_borrower.to_parquet(
        cfg.paths.tables_dir / f"03_per_borrower_importance_{args.tag}.parquet",
        index=False,
    )
    np.save(cfg.paths.tables_dir / f"03_per_borrower_phi_{args.tag}.npy", surv.phi)
    print(f"per-borrower attributions saved ({per_borrower.shape[0]} borrowers)")

    eff = check_efficiency(surv)
    worst = float(eff["max_abs_residual"].max())
    print(f"\nefficiency axiom: worst residual {worst:.2e} -> "
          f"{'PASS' if worst < 1e-6 else 'FAIL'}")
    print(to_markdown(eff))
    write_table(eff, cfg.paths.tables_dir / f"03_efficiency_check_{args.tag}.csv")

    importance = surv.importance()
    write_table(
        importance.rename("importance").reset_index().rename(columns={"index": "feature"}),
        cfg.paths.tables_dir / f"03_survshap_importance_{args.tag}.csv",
    )
    print("\nSurvSHAP(t) global importance (top 15):")
    print(to_markdown(importance.head(15).rename("importance").reset_index()
                      .rename(columns={"index": "feature"})))

    figs.plot_survshap_curves(
        surv, cfg.paths.figures_dir / f"03_survshap_curves_{args.tag}.png", obs=0
    )

    # ---------------- (a) naive vs survival ----------------
    cmp = None
    if args.skip_naive:
        print("\n=== (a) naive SHAP comparison SKIPPED (--skip-naive) ===")
        print("  Reusing the settled result from the unstratified run. Naive SHAP "
              "costs about\n  as much as SurvSHAP(t) itself, so skipping it halves "
              "this run's cost.")
    else:
        print(f"\n=== (a) naive SHAP vs SurvSHAP(t) at {args.at_month:.0f} months ===")
        naive = explain_naive_shap(
            model, X_explain, dm.X, at_month=args.at_month,
            nsamples=nsamples, n_background=n_background, seed=cfg.explain.seed,
        )
        cmp = compare_explanations(surv, naive)
        for key, value in cmp.summary().items():
            print(f"  {key}: {value}")
        write_table(cmp.per_feature.reset_index(drop=True),
                    cfg.paths.tables_dir / f"03_shap_comparison_{args.tag}.csv")
        figs.plot_importance_comparison(
            cmp.survshap_importance, cmp.naive_importance,
            cfg.paths.figures_dir / f"03_importance_comparison_{args.tag}.png",
        )

    # ---------------- (b) adverse action ----------------
    print("\n=== (b) adverse-action notice (ECOA / Reg B) ===")
    # Pick the highest-risk explained borrower: a notice only makes sense for
    # someone who would actually be declined.
    k = int(np.argmin(np.abs(times - args.at_month)))
    riskiest = int(np.argmax(1.0 - surv.prediction[:, k]))
    score_col = "fico_midpoint"
    derived = add_derived_features(test.iloc[pick])
    score = (
        float(derived[score_col].iloc[riskiest])
        if score_col in derived.columns and pd.notna(derived[score_col].iloc[riskiest])
        else None
    )
    notice = build_adverse_action_notice(
        surv, obs=riskiest, horizon_months=int(args.at_month),
        credit_score=score, model_name=args.model,
    )
    text = notice.render()
    print(text)
    notice_path = cfg.paths.tables_dir / f"03_adverse_action_notice_{args.tag}.txt"
    notice_path.write_text(text, encoding="utf-8")
    write_json(notice.to_dict(),
               cfg.paths.tables_dir / f"03_adverse_action_{args.tag}.json")

    # ---------------- (c) segment stability ----------------
    print("\n=== (c) explanation stability across segments ===")
    seg_results = {}
    explained = add_derived_features(test.iloc[pick]).reset_index(drop=True)
    candidates = {
        "grade": explained["grade"].astype(str) if "grade" in explained else None,
        "purpose": explained["purpose"].astype(str) if "purpose" in explained else None,
        "income_band": (
            explained["income_band"].astype(str) if "income_band" in explained else None
        ),
        "term": (
            explained["term_months"].astype(str) if "term_months" in explained else None
        ),
    }
    for name, series in candidates.items():
        if series is None or series.nunique() < 2:
            continue
        st = analyse_segment_stability(
            surv, series, segment_name=name, k=5,
            min_size=max(10, n_explain // 40),
        )
        if st.importance.empty:
            print(f"  [{name}] no level met the minimum size; skipped")
            continue
        seg_results[name] = st.summary()
        print(f"\n  [{name}] {st.verdict()}")
        write_table(st.importance.reset_index().rename(columns={"index": "feature"}),
                    cfg.paths.tables_dir / f"03_segment_importance_{name}_{args.tag}.csv")
        figs.plot_segment_heatmap(
            st.importance,
            cfg.paths.figures_dir / f"03_segment_heatmap_{name}_{args.tag}.png",
            title=f"Importance by {name}",
        )

    payload = {
        "model": args.model,
        "tag": args.tag,
        "model_tag": model_tag,
        "n_explained": int(n_explain),
        "stratified_by": args.stratify_by,
        "strata_counts": strata_counts,
        "nsamples": int(nsamples),
        "n_background": int(surv.n_background),
        "times": [int(t) for t in times],
        "seconds_total": round(elapsed, 1),
        "seconds_per_borrower": round(elapsed / n_explain, 3),
        "efficiency_worst_residual": worst,
        "efficiency_pass": bool(worst < 1e-6),
        "global_importance": {k: float(v) for k, v in importance.items()},
        "naive_vs_survshap": (
            cmp.summary() if cmp is not None
            else {"status": "skipped",
                  "reason": "settled in the unstratified run; see 03_explain_full.json"}
        ),
        "adverse_action": notice.to_dict(),
        "segment_stability": seg_results,
    }
    payload["provenance"] = build_stamp(
        stage="03_explain",
        inputs={"model": model_path, "data": src},
        outputs={"per_borrower_importance": cfg.paths.tables_dir
                 / f"03_per_borrower_importance_{args.tag}.parquet"},
        config_path=args.config,
        args=vars(args),
    )
    out = cfg.paths.tables_dir / f"03_explain_{args.tag}.json"
    write_json(payload, out)
    print(f"\nresults -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
