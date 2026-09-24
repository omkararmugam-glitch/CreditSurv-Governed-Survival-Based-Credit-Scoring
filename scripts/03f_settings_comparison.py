"""Stage 3f -- are cheaper SurvSHAP(t) settings good enough for bulk runs?

Runs the comparison pre-registered in FINDINGS section 7a. Candidates are compared
against the **stored** full-settings reasons from stage 3d, so the 42-minute
reference run is not repeated:

    S1  same top stated reason           >= 90% of applicants
    S2  mean top-4 reason-set overlap    >= 0.85

Read against the 95.3% / 0.94 agreement SurvSHAP(t) has with itself at full
settings (7.1): a candidate cannot be asked to beat the method's own
reproducibility. The cheapest candidate passing both is adopted for bulk runs only.

Explanations here are seeded per applicant, as bulk runs are, so a candidate's
result does not depend on batch membership or on the number of workers.

    python scripts/03f_settings_comparison.py --model-tag full
    python scripts/03f_settings_comparison.py --model-tag full --workers 8
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
from creditsurv.explain.parallel import explain_rows_parallel, suggest_workers  # noqa: E402
from creditsurv.features.build import build_design_matrix  # noqa: E402
from creditsurv.pipeline import load_feature_frame, load_model_bundle, resolve_data_source  # noqa: E402
from creditsurv.provenance import build_stamp, guard_outputs  # noqa: E402
from creditsurv.reporting.tables import to_markdown, write_json, write_table  # noqa: E402

S1_BAR = 0.90
S2_BAR = 0.85
CEILING = {"top1": 0.953, "top4": 0.94}     # measured in FINDINGS 7.1

CANDIDATES = [
    ("C1", 300, 100),
    ("C2", 600, 50),
    ("C3", 300, 50),
    ("C4", 146, 25),
]


def _reasons(expl, obs: int, horizon: int) -> list[str]:
    return [r.reason for r in build_adverse_action_notice(
        expl, obs=obs, horizon_months=horizon, model_name="survshap").reasons]


def _agreement(reference: list[list[str]], candidate: list[list[str]]) -> dict:
    top1 = [bool(a) and a[:1] == b[:1] for a, b in zip(reference, candidate)]
    overlap = [len(set(a[:4]) & set(b[:4])) / 4.0 for a, b in zip(reference, candidate)]
    return {"n": len(reference),
            "top1_agreement": float(np.mean(top1)),
            "mean_top4_overlap": float(np.mean(overlap)),
            "exact_top4_set_match": float(np.mean([o == 1.0 for o in overlap]))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--model-tag", default="full")
    ap.add_argument("--tag", default="settings")
    ap.add_argument("--reference-tag", default="explainer",
                    help="stage 3d run whose full-settings reasons are the reference")
    ap.add_argument("--n-explain", type=int, default=0,
                    help="applicants compared; 0 uses every row of the reference")
    ap.add_argument("--workers", type=int, default=0, help="0 = choose automatically")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg.paths.ensure_dirs()
    out_json = cfg.paths.tables_dir / f"03f_settings_{args.tag}.json"
    out_csv = cfg.paths.tables_dir / f"03f_settings_{args.tag}.csv"
    refused = guard_outputs([out_json, out_csv], args.overwrite,
                            script="03f_settings_comparison.py")
    if refused:
        return refused

    ref_path = (cfg.paths.tables_dir
                / f"03d_explainer_per_applicant_{args.reference_tag}.csv")
    if not ref_path.exists():
        print(f"ERROR: {ref_path.name} not found. Run "
              f"scripts/03d_explainer_validation.py first: its full-settings reasons "
              f"are the reference this compares against.", file=sys.stderr)
        return 2
    reference_table = pd.read_csv(ref_path)
    if args.n_explain:
        reference_table = reference_table.head(args.n_explain)
    row_index = reference_table["row_index"].to_numpy()
    reference = [str(r).split("; ") if isinstance(r, str) and r else []
                 for r in reference_table["survshap_reasons"]]
    print(f"reference: {len(reference):,} applicants from {ref_path.name} "
          f"at nsamples={cfg.explain.kernel_nsamples}, "
          f"n_background={cfg.explain.n_background}")

    horizon = int(cfg.decision.horizon_months)
    bundle, model_path = load_model_bundle(cfg.paths.models_dir, args.model_tag)
    model = bundle["artefacts"]["discrete_hazard"]
    src = resolve_data_source(bundle, cfg, args.model_tag)
    df = load_feature_frame(src, bundle["spec"], verbose=True)
    rows = df.loc[df.index.intersection(pd.Index(row_index))]
    if len(rows) != len(reference):
        print(f"ERROR: {len(rows)} of {len(reference)} referenced rows found in "
              f"{src.name}; the reference was built from a different data file.",
              file=sys.stderr)
        return 2
    dm = build_design_matrix(rows, bundle["spec"], flavour="gbm")
    background = build_design_matrix(
        df.loc[bundle["train_idx"].intersection(df.index)].sample(
            min(20_000, len(bundle["train_idx"])), random_state=cfg.explain.seed),
        bundle["spec"], flavour="gbm").X
    times = np.array(cfg.model.eval_horizons_months, dtype=float)
    ids = [str(i) for i in dm.X.index]
    workers = args.workers or suggest_workers(len(dm.X))
    print(f"comparing {len(CANDIDATES)} candidates on {workers} workers\n")

    results = []
    for name, nsamples, n_background in CANDIDATES:
        t0 = time.perf_counter()
        expl = explain_rows_parallel(model, dm.X, background, times, row_ids=ids,
                                     nsamples=nsamples, n_background=n_background,
                                     seed=cfg.explain.seed, workers=workers)
        seconds = time.perf_counter() - t0
        candidate = [_reasons(expl, i, horizon) for i in range(len(dm.X))]
        agree = _agreement(reference, candidate)
        s1 = agree["top1_agreement"] >= S1_BAR
        s2 = agree["mean_top4_overlap"] >= S2_BAR
        results.append({
            "candidate": name, "nsamples": nsamples, "n_background": n_background,
            "relative_cost": round(nsamples * n_background
                                   / (cfg.explain.kernel_nsamples
                                      * cfg.explain.n_background), 3),
            "seconds_per_applicant": round(seconds / len(dm.X), 3),
            **{k: round(v, 4) for k, v in agree.items() if k != "n"},
            "S1_pass": bool(s1), "S2_pass": bool(s2),
            "verdict": "PASS" if (s1 and s2) else "FAIL"})
        print(f"{name}: nsamples={nsamples} bg={n_background} -> top-1 "
              f"{agree['top1_agreement']:.1%}, top-4 {agree['mean_top4_overlap']:.2f}, "
              f"{seconds / len(dm.X):.2f}s each -> "
              f"{results[-1]['verdict']}", flush=True)

    table = pd.DataFrame(results)
    write_table(table, out_csv)
    passing = table[table["verdict"] == "PASS"].sort_values("relative_cost")
    adopt = passing.iloc[0]["candidate"] if len(passing) else None

    payload = {
        "tag": args.tag, "model_tag": args.model_tag,
        "reference": {"file": ref_path.name, "n": len(reference),
                      "nsamples": cfg.explain.kernel_nsamples,
                      "n_background": cfg.explain.n_background},
        "bars": {"S1_top1_agreement": S1_BAR, "S2_mean_top4_overlap": S2_BAR},
        "self_agreement_ceiling": CEILING,
        "candidates": results,
        "cheapest_passing": adopt,
        "workers": workers,
        "provenance": build_stamp(
            stage="03f_settings_comparison",
            inputs={"data": src, "model": model_path, "reference": ref_path},
            outputs={"result": out_json, "table": out_csv},
            config_path=args.config,
            args={"model_tag": args.model_tag, "n_explain": len(reference),
                  "workers": workers}),
    }
    write_json(payload, out_json)

    print("\n=== pre-registered bar (FINDINGS 7a) ===")
    print(to_markdown(table[["candidate", "nsamples", "n_background",
                             "top1_agreement", "mean_top4_overlap",
                             "seconds_per_applicant", "verdict"]]))
    print(f"\nceiling for reference only: SurvSHAP against itself scores "
          f"{CEILING['top1']:.1%} / {CEILING['top4']:.2f} at full settings.")
    if adopt:
        row = passing.iloc[0]
        print(f"\nCHEAPEST PASSING: {adopt} (nsamples={row['nsamples']}, "
              f"n_background={row['n_background']}, "
              f"{row['relative_cost']:.2f}x the cost). To adopt it for bulk runs, set "
              f"explain_nsamples and explain_n_background under decision: in "
              f"config/config.yaml -- the research stages keep the full settings.")
    else:
        print("\nNO CANDIDATE PASSED. Bulk runs keep the full settings; record the "
              "negative result in FINDINGS 7a.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
