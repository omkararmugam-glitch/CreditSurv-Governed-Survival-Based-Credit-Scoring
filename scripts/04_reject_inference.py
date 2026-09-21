"""Stage 4 -- selection-bias diagnostic, then correction only if it is warranted.

The order matters and is the point of the stage: the diagnostic runs first, its
thresholds were fixed in config before any result was seen, and the correction is
applied only if the gate allows. If the gate blocks, that is reported as the
finding rather than worked around.

Whatever the gate decides, Stage 3's explanations are recomputed before and after
so that any shift in feature importance is measured. That comparison is the
project's own question; none of the reviewed literature tests it.

Usage
-----
    python scripts/04_reject_inference.py
    python scripts/04_reject_inference.py --tag full
    python scripts/04_reject_inference.py --force-correction   # override the gate
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
from creditsurv.explain.survshap import explain_survshap  # noqa: E402
from creditsurv.features.build import build_design_matrix  # noqa: E402
from creditsurv.io import schema as sch  # noqa: E402
from creditsurv.models.discrete_hazard import DiscreteTimeHazardModel  # noqa: E402
from creditsurv.models.evaluate import CensoringModel, evaluate_survival  # noqa: E402
from creditsurv.reject_inference.correction import fit_reweighting  # noqa: E402
from creditsurv.reject_inference.diagnostics import (  # noqa: E402
    parse_rejected_dti,
    run_selection_diagnostic,
)
from creditsurv.provenance import (  # noqa: E402
    build_stamp,
    find_existing_outputs,
    guard_outputs,
)
from creditsurv.reporting import figures as figs  # noqa: E402
from creditsurv.reporting.tables import (  # noqa: E402
    explanation_shift_table,
    to_markdown,
    write_json,
    write_table,
)

# Features comparable across both files. Grades come from schema.COMMON_FEATURE_MAP.
COMPARABILITY = {
    "loan_amnt": "good",
    "emp_length_years": "good",
    "score": "partial",       # FICO band vs Lending Club Risk_Score
    "dti": "partial",         # different definitions, self-reported on reject side
}


def _prepare_common_frames(accepted: pd.DataFrame, rejected: pd.DataFrame, cfg):
    """Align both populations onto the handful of genuinely common features."""
    from creditsurv.features.encoders import parse_emp_length

    acc = pd.DataFrame(index=accepted.index)
    acc["loan_amnt"] = pd.to_numeric(accepted["loan_amnt"], errors="coerce")
    acc["dti"] = pd.to_numeric(accepted["dti"], errors="coerce").clip(0, 100)
    acc["score"] = pd.to_numeric(accepted["fico_range_low"], errors="coerce")
    acc["emp_length_years"] = parse_emp_length(accepted["emp_length"])

    rej = pd.DataFrame(index=rejected.index)
    rej["loan_amnt"] = pd.to_numeric(rejected["loan_amnt"], errors="coerce")
    rej["dti"] = parse_rejected_dti(rejected["dti_raw"], clip_upper=100.0)
    rej["score"] = pd.to_numeric(rejected["risk_score"], errors="coerce")
    rej["emp_length_years"] = parse_emp_length(rejected["emp_length"])
    return acc, rej


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--n-explain", type=int, default=200,
                    help="borrowers explained before/after; KernelSHAP is the cost")
    ap.add_argument("--force-correction", action="store_true",
                    help="apply the correction even if the gate blocks it (recorded)")
    ap.add_argument("--dti-clip", type=float, default=100.0)
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing existing outputs for this --tag")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.paths.ensure_dirs()
    d = cfg.diagnostic

    # Checked before either population is loaded, so a refused run costs nothing.
    refused = guard_outputs(
        find_existing_outputs([cfg.paths.tables_dir, cfg.paths.figures_dir],
                              "04", args.tag),
        args.overwrite, script="04_reject_inference.py")
    if refused:
        return refused

    def stamp() -> dict:
        # Built at write time. The model is only loaded on the correction path,
        # but is recorded either way: if it is later replaced, a no-correction
        # result would otherwise look unaffected when its sibling results are not.
        return build_stamp(
            stage="04_reject_inference",
            inputs={"accepted": src, "rejected": cfg.paths.rejected_parquet,
                    "model": cfg.paths.models_dir / f"02_models_{args.tag}.pkl"},
            config_path=args.config,
            args=vars(args),
        )

    if not cfg.paths.rejected_parquet.exists():
        print(f"ERROR: {cfg.paths.rejected_parquet} not found. Run 00_ingest.py.",
              file=sys.stderr)
        return 2
    src = (cfg.paths.labeled_parquet if args.tag.startswith("full")
           else cfg.paths.dev_sample_parquet)
    if not src.exists():
        print(f"ERROR: {src} not found. Run 01_build_labels.py.", file=sys.stderr)
        return 2

    print(f"reading accepted: {src}")
    accepted = pd.read_parquet(src)
    print(f"  {len(accepted):,} rows")

    print(f"reading rejected: {cfg.paths.rejected_parquet} "
          f"(sampling {d.rejected_sample_size:,})")
    rejected = pd.read_parquet(
        cfg.paths.rejected_parquet,
        columns=["loan_amnt", "risk_score", "dti_raw", "emp_length", "application_d"],
    )
    if len(rejected) > d.rejected_sample_size:
        rejected = rejected.sample(n=d.rejected_sample_size, random_state=d.seed)
    print(f"  {len(rejected):,} rows sampled")

    acc_common, rej_common = _prepare_common_frames(accepted, rejected, cfg)
    print("\ncommon-feature coverage:")
    for col in acc_common.columns:
        print(f"  {col:<20} accepted {acc_common[col].notna().mean():6.2%}   "
              f"rejected {rej_common[col].notna().mean():6.2%}")

    # ---------------- the gate ----------------
    print("\n" + "=" * 74)
    print("SELECTION-BIAS DIAGNOSTIC (thresholds pre-registered in config)")
    print("=" * 74)
    diag = run_selection_diagnostic(
        acc_common, rej_common,
        comparability=COMPARABILITY,
        smd_notable=d.smd_notable,
        smd_substantial=d.smd_substantial,
        ks_substantial=d.ks_substantial,
        separability_auc_strong=d.separability_auc_strong,
        separability_auc_weak=d.separability_auc_weak,
        min_common_support=d.min_common_support,
        seed=d.seed,
    )
    print(to_markdown(diag.table()))
    print(f"\nseparability AUC        : {diag.separability_auc:.4f}")
    print(f"top selection drivers   : {list(diag.separability_coefficients.head(3).index)}")
    print(f"common support share    : {diag.support.get('share_in_support', float('nan')):.4f}")
    print(f"\nbias detected           : {diag.bias_detected}")
    print(f"support adequate        : {diag.support_adequate}")
    print(f"GATE DECISION           : {diag.gate_decision.upper()}")
    print(f"\n{diag.rationale()}")
    for note in diag.notes:
        print(f"  note: {note}")

    write_table(diag.table(), cfg.paths.tables_dir / f"04_diagnostic_{args.tag}.csv")

    # Propensities for the overlap figure are recomputed on the aligned frames.
    from creditsurv.reject_inference.diagnostics import separability
    _, p_acc, p_rej, _ = separability(acc_common, rej_common, seed=d.seed)
    figs.plot_propensity_overlap(
        p_acc, p_rej,
        cfg.paths.figures_dir / f"04_propensity_overlap_{args.tag}.png",
        support_range=diag.support.get("accepted_propensity_range"),
    )

    apply_correction = diag.gate_decision == "apply_correction" or args.force_correction
    if args.force_correction and diag.gate_decision != "apply_correction":
        print("\nWARNING: --force-correction overrides a blocking gate. This is "
              "recorded in the output as an override, not as a warranted correction.")

    payload = {
        "tag": args.tag,
        "n_accepted": int(len(accepted)),
        "n_rejected_sampled": int(len(rejected)),
        "dti_clip": args.dti_clip,
        "diagnostic": diag.summary(),
        "correction_applied": bool(apply_correction),
        "correction_forced": bool(args.force_correction),
    }

    # ---------------- correction + explanation shift ----------------
    if not apply_correction:
        print("\nNo correction applied. Per the spec, explanations are still "
              "compared before/after -- but with no correction there is no 'after', "
              "so the explanation-shift question is answered as: not applicable, "
              "because the gate correctly blocked the correction.")
        payload["explanation_shift"] = {
            "status": "not_applicable",
            "reason": diag.rationale(),
        }
        payload["provenance"] = stamp()
        out = cfg.paths.tables_dir / f"04_reject_inference_{args.tag}.json"
        write_json(payload, out)
        print(f"\nresults -> {out}")
        return 0

    print("\n" + "=" * 74)
    print("APPLYING REWEIGHTING CORRECTION")
    print("=" * 74)
    weights = fit_reweighting(
        acc_common, rej_common,
        unavailable_features=("annual_inc", "revol_util", "home_ownership",
                              "credit history depth"),
        seed=d.seed,
    )
    for key, value in weights.summary().items():
        print(f"  {key}: {value}")
    payload["weights"] = weights.summary()

    # Refit the model with and without weights, then re-explain both.
    model_path = cfg.paths.models_dir / f"02_models_{args.tag}.pkl"
    if not model_path.exists():
        print(f"\nERROR: {model_path} not found; run 02_train_models.py first.",
              file=sys.stderr)
        return 2
    with open(model_path, "rb") as fh:
        bundle = pickle.load(fh)
    spec = bundle["spec"]
    train_idx = bundle["train_idx"].intersection(accepted.index)
    test_idx = bundle["test_idx"].intersection(accepted.index)

    for col in [c for c in accepted.columns if accepted[c].dtype == object and c != "id"]:
        accepted[col] = accepted[col].astype("category")

    dg_tr = build_design_matrix(accepted.loc[train_idx], spec, flavour="gbm")
    dg_te = build_design_matrix(accepted.loc[test_idx], spec, flavour="gbm")
    times = np.array(cfg.model.eval_horizons_months, dtype=float)
    censoring = CensoringModel(dg_tr.duration.to_numpy(), dg_tr.event.to_numpy())

    w_series = pd.Series(weights.weights, index=accepted.index)
    results, importances = {}, {}
    for label, sample_weight in (("before", None), ("after", w_series.loc[train_idx])):
        print(f"\n[{label}] fitting discrete-time hazard model")
        model = DiscreteTimeHazardModel(
            time_bin_months=cfg.model.time_bin_months,
            max_horizon_months=60,
            num_boost_round=300,
            seed=cfg.model.seed,
        )
        t0 = time.time()
        model.fit(dg_tr.X, dg_tr.duration.to_numpy(), dg_tr.event.to_numpy(),
                  loan_weight=(None if sample_weight is None
                               else sample_weight.to_numpy(dtype=float)))
        res = evaluate_survival(
            model_name=f"discrete_hazard_{label}", split_name="test",
            survival=model.predict_survival(dg_te.X, times), times=times,
            duration=dg_te.duration.to_numpy(), event=dg_te.event.to_numpy(),
            censoring=censoring,
        )
        print(res)
        print(f"  fitted in {time.time() - t0:.1f}s")
        results[label] = res.summary()

        rng = np.random.default_rng(cfg.explain.seed)
        pick = np.sort(rng.choice(len(dg_te.X), min(args.n_explain, len(dg_te.X)),
                                  replace=False))
        expl = explain_survshap(
            model, dg_te.X.iloc[pick], dg_te.X, times,
            nsamples=cfg.explain.kernel_nsamples,
            n_background=cfg.explain.n_background,
            seed=cfg.explain.seed,
        )
        importances[label] = expl.importance()

    shift = explanation_shift_table(importances["before"], importances["after"], k=15)
    print("\n" + "=" * 74)
    print("EXPLANATION SHIFT: before vs after correction (the project's own question)")
    print("=" * 74)
    print(to_markdown(shift))
    write_table(shift, cfg.paths.tables_dir / f"04_explanation_shift_{args.tag}.csv")

    from scipy.stats import spearmanr
    feats = list(importances["before"].index)
    rho = float(spearmanr(
        importances["before"].reindex(feats).to_numpy(),
        importances["after"].reindex(feats).to_numpy(),
    ).statistic)
    top5_before = set(importances["before"].head(5).index)
    top5_after = set(importances["after"].head(5).index)
    overlap = len(top5_before & top5_after) / len(top5_before | top5_after)

    verdict = (
        "IMPORTANCE SHIFTS materially after correction"
        if rho < 0.9 or overlap < 0.6
        else "IMPORTANCE IS STABLE under correction"
    )
    print(f"\nrank correlation before/after : {rho:.4f}")
    print(f"top-5 overlap                 : {overlap:.4f}")
    print(f"VERDICT                       : {verdict}")

    payload["metrics_before_after"] = results
    payload["explanation_shift"] = {
        "status": "measured",
        "rank_correlation": round(rho, 4),
        "top5_overlap": round(overlap, 4),
        "top5_before": sorted(top5_before),
        "top5_after": sorted(top5_after),
        "verdict": verdict,
        "table": shift.to_dict(orient="records"),
    }
    payload["provenance"] = stamp()
    out = cfg.paths.tables_dir / f"04_reject_inference_{args.tag}.json"
    write_json(payload, out)
    print(f"\nresults -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
