"""Stage 2 -- fit and evaluate the Cox baseline and the discrete-time hazard model.

Reads the labelled Parquet from Stage 1, fits both models on the same split, and
writes metrics, coefficient tables, figures and the fitted models to disk.

Usage
-----
    python scripts/02_train_models.py                       # dev sample (fast)
    python scripts/02_train_models.py --full                 # full dataset
    python scripts/02_train_models.py --with-lc-grade        # benchmark variant
    python scripts/02_train_models.py --split out_of_time    # robustness check
    python scripts/02_train_models.py --skip-cox             # GBM only
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
from creditsurv.features.build import (  # noqa: E402
    build_design_matrix,
    default_spec,
    train_test_split_loans,
)
from creditsurv.models.cox import CoxModel, age_dependent_risk_profile  # noqa: E402
from creditsurv.models.discrete_hazard import (  # noqa: E402
    DiscreteTimeHazardModel,
    expansion_row_estimate,
)
from creditsurv.models.evaluate import CensoringModel, evaluate_survival  # noqa: E402
from creditsurv.provenance import (  # noqa: E402
    build_stamp,
    find_existing_outputs,
    guard_outputs,
)
from creditsurv.reporting import figures as figs  # noqa: E402
from creditsurv.reporting.tables import (  # noqa: E402
    model_comparison_table,
    to_markdown,
    write_json,
    write_table,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--full", action="store_true",
                    help="use the full labelled dataset instead of the dev sample")
    ap.add_argument("--with-lc-grade", action="store_true",
                    help="benchmark variant including LC grade / sub_grade / int_rate")
    ap.add_argument("--split", default="random", choices=["random", "out_of_time"])
    ap.add_argument("--time-bin", type=int, default=None,
                    help="discrete-hazard bin width in months (default from config)")
    ap.add_argument("--negative-subsample", type=float, default=1.0,
                    help="keep this fraction of non-event person-periods, with "
                         "compensating weights. Every event period is always kept. "
                         "Standard case-control sampling; the weights leave the "
                         "hazard estimate consistent. Use on the full dataset, "
                         "where the expansion otherwise exceeds memory.")
    ap.add_argument("--cox-max-rows", type=int, default=300_000,
                    help="cap rows used to fit Cox; it is O(n) per Newton step")
    ap.add_argument("--skip-cox", action="store_true")
    ap.add_argument("--skip-gbm", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing existing outputs for this tag, including "
                         "the fitted model that later stages were computed from")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.paths.ensure_dirs()
    tag = args.tag or ("full" if args.full else "dev")
    if args.with_lc_grade:
        tag += "_lcgrade"

    # Checked before any data is loaded, so a refused run costs nothing. Replacing
    # 02_models_<tag>.pkl silently would leave every Stage 3/4 result describing a
    # model that no longer exists.
    refused = guard_outputs(
        find_existing_outputs(
            [cfg.paths.tables_dir, cfg.paths.figures_dir, cfg.paths.models_dir],
            "02", tag),
        args.overwrite, script="02_train_models.py")
    if refused:
        return refused

    src = cfg.paths.labeled_parquet if args.full else cfg.paths.dev_sample_parquet
    if not src.exists():
        print(f"ERROR: {src} not found. Run scripts/01_build_labels.py first.",
              file=sys.stderr)
        return 2

    print(f"reading {src}")
    df = pd.read_parquet(src)
    for col in [c for c in df.columns if df[c].dtype == object and c != "id"]:
        df[col] = df[col].astype("category")
    print(f"  {len(df):,} rows  event_rate={df['event'].mean():.4f}  "
          f"{df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    with_grade = args.with_lc_grade or cfg.model.with_lc_grade
    spec = default_spec(df.columns, extended=True, with_lc_grade=with_grade)
    print(f"  features: {len(spec.numeric)} numeric + {len(spec.categorical)} categorical"
          f"{'  (INCLUDING LC grade -- benchmark variant)' if with_grade else ''}")

    train_idx, test_idx = train_test_split_loans(
        df, test_size=cfg.model.test_size, seed=cfg.model.seed, scheme=args.split
    )
    print(f"  split={args.split}  train={len(train_idx):,}  test={len(test_idx):,}")

    times = np.array(cfg.model.eval_horizons_months, dtype=float)
    results, artefacts = [], {}

    # Censoring model is fitted on TRAIN only, so IPCW weights never see test
    # outcomes.
    censoring = CensoringModel(
        df.loc[train_idx, "duration_months"].to_numpy(),
        df.loc[train_idx, "event"].to_numpy(),
    )

    # ---------------- Cox ----------------
    if not args.skip_cox:
        cox_train = train_idx
        if len(cox_train) > args.cox_max_rows:
            rng = np.random.default_rng(cfg.model.seed)
            cox_train = pd.Index(
                rng.choice(train_idx.to_numpy(), args.cox_max_rows, replace=False)
            )
            print(f"\n[cox] subsampling train to {len(cox_train):,} rows "
                  f"(--cox-max-rows); Cox is O(n) per Newton step")
        else:
            print("\n[cox] fitting")

        dm_tr = build_design_matrix(df.loc[cox_train], spec, flavour="cox")
        dm_te = build_design_matrix(
            df.loc[test_idx], spec, flavour="cox",
            standardisation=dm_tr.standardisation,
            fill_values=dm_tr.fill_values,
            reference_columns=list(dm_tr.X.columns),
        )
        print(f"[cox] design matrix {dm_tr.X.shape}, dropped {len(dm_tr.dropped)} columns")
        t0 = time.time()
        cox = CoxModel(penalizer=0.01).fit(dm_tr.X, dm_tr.duration, dm_tr.event)
        print(f"[cox] fitted in {time.time() - t0:.1f}s")

        res = evaluate_survival(
            model_name="cox", split_name="test",
            survival=cox.predict_survival(dm_te.X, times), times=times,
            duration=dm_te.duration.to_numpy(), event=dm_te.event.to_numpy(),
            censoring=censoring, risk=cox.predict_risk(dm_te.X), calibration_at=24.0,
        )
        print(res)
        results.append(res)

        coefs = cox.coefficient_table()
        write_table(coefs, cfg.paths.tables_dir / f"02_cox_coefficients_{tag}.csv")
        print("\n[cox] strongest hazard ratios:")
        print(to_markdown(coefs.head(10)[["feature", "hazard_ratio", "hr_lower_95",
                                          "hr_upper_95", "p"]]))

        age = age_dependent_risk_profile(cox, dm_te.X.iloc[[0]], np.arange(1, 61))
        write_table(age, cfg.paths.tables_dir / f"02_age_dependent_risk_{tag}.csv")
        print("\n[cox] age-dependent risk for a single fixed borrower profile:")
        print(to_markdown(age))

        figs.plot_survival_curves(
            cox.predict_survival(dm_te.X.iloc[:6], np.arange(1, 61)),
            np.arange(1, 61),
            cfg.paths.figures_dir / f"02_cox_survival_curves_{tag}.png",
            title="Cox: predicted survival curves",
        )
        if not res.calibration.empty:
            figs.plot_calibration(
                res.calibration, cfg.paths.figures_dir / f"02_cox_calibration_{tag}.png"
            )
        artefacts["cox"] = cox
        artefacts["cox_columns"] = list(dm_tr.X.columns)
        artefacts["cox_standardisation"] = dm_tr.standardisation
        artefacts["cox_fill_values"] = dm_tr.fill_values

    # ---------------- discrete-time hazard ----------------
    if not args.skip_gbm:
        time_bin = args.time_bin or cfg.model.time_bin_months
        dg_tr = build_design_matrix(df.loc[train_idx], spec, flavour="gbm")
        dg_te = build_design_matrix(df.loc[test_idx], spec, flavour="gbm")

        est = expansion_row_estimate(dg_tr.duration.to_numpy(), time_bin, 60)
        kept = int(est * args.negative_subsample)
        note = (f", ~{kept:,} kept at negative_subsample={args.negative_subsample}"
                if args.negative_subsample < 1.0 else "")
        print(f"\n[gbm] time_bin={time_bin}mo -> ~{est:,} person-period rows{note}")
        # 74 float32 columns, and LightGBM builds its own binned copy on top.
        est_gb = kept * 74 * 4 / 1e9
        print(f"[gbm] estimated ~{est_gb:.1f} GB for the expansion")
        if est_gb > 3.0:
            print(f"[gbm] WARNING: that may not fit alongside the feature matrix. "
                  f"Consider --time-bin {time_bin * 2} or --negative-subsample "
                  f"{max(0.1, round(args.negative_subsample / 2, 2))}")

        t0 = time.time()
        dh = DiscreteTimeHazardModel(
            time_bin_months=time_bin,
            max_horizon_months=60,
            negative_subsample=args.negative_subsample,
            num_boost_round=600,
            seed=cfg.model.seed,
        ).fit(
            dg_tr.X, dg_tr.duration.to_numpy(), dg_tr.event.to_numpy(),
            valid=(dg_te.X, dg_te.duration.to_numpy(), dg_te.event.to_numpy()),
        )
        print(f"[gbm] fitted on {dh.training_rows_:,} person-periods in "
              f"{time.time() - t0:.1f}s")

        res = evaluate_survival(
            model_name="discrete_hazard", split_name="test",
            survival=dh.predict_survival(dg_te.X, times), times=times,
            duration=dg_te.duration.to_numpy(), event=dg_te.event.to_numpy(),
            censoring=censoring, calibration_at=24.0,
        )
        print(res)
        results.append(res)

        imp = dh.feature_importance()
        write_table(imp, cfg.paths.tables_dir / f"02_gbm_importance_{tag}.csv")
        print("\n[gbm] top gain importance:")
        print(to_markdown(imp.head(10)))

        figs.plot_survival_curves(
            dh.predict_survival(dg_te.X.iloc[:6], np.arange(1, 61)),
            np.arange(1, 61),
            cfg.paths.figures_dir / f"02_gbm_survival_curves_{tag}.png",
            title="Discrete-time hazard: predicted survival curves",
        )
        if not res.calibration.empty:
            figs.plot_calibration(
                res.calibration, cfg.paths.figures_dir / f"02_gbm_calibration_{tag}.png"
            )
        artefacts["discrete_hazard"] = dh
        artefacts["gbm_columns"] = list(dg_tr.X.columns)

    # ---------------- comparison + persistence ----------------
    comparison = model_comparison_table(results)
    write_table(comparison, cfg.paths.tables_dir / f"02_model_comparison_{tag}.csv")
    print("\n=== MODEL COMPARISON ===")
    print(to_markdown(comparison))

    if results:
        figs.plot_time_dependent_auc(
            {r.model: r.auc_table for r in results},
            cfg.paths.figures_dir / f"02_time_dependent_auc_{tag}.png",
        )
    figs.plot_km_by_group(
        df.loc[test_idx, "duration_months"].to_numpy(),
        df.loc[test_idx, "event"].to_numpy(),
        df.loc[test_idx, "grade"] if "grade" in df.columns
        else df.loc[test_idx, "term_months"],
        cfg.paths.figures_dir / f"02_km_by_grade_{tag}.png",
        title="Observed Kaplan-Meier by loan grade",
    )

    payload = {
        "source": str(src),
        "tag": tag,
        "n_rows": int(len(df)),
        "split_scheme": args.split,
        "time_bin_months": int(args.time_bin or cfg.model.time_bin_months),
        "negative_subsample": float(args.negative_subsample),
        "with_lc_grade": bool(with_grade),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "eval_horizons": [int(t) for t in times],
        "features": {
            "numeric": list(spec.numeric),
            "categorical": list(spec.categorical),
        },
        "results": [r.summary() for r in results],
    }

    # Model first, so the metrics can record the hash of the exact model they
    # describe; Stage 3/4 stamps record the same hash for the model they loaded.
    model_path = cfg.paths.models_dir / f"02_models_{tag}.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump({"artefacts": artefacts, "spec": spec,
                     "train_idx": train_idx, "test_idx": test_idx}, fh)
    payload["provenance"] = build_stamp(
        stage="02_train_models",
        inputs={"data": src},
        outputs={"model": model_path},
        config_path=args.config,
        args=vars(args),
    )
    write_json(payload, cfg.paths.tables_dir / f"02_metrics_{tag}.json")
    print(f"\nmodels -> {model_path}")
    print(f"metrics -> {cfg.paths.tables_dir / f'02_metrics_{tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
