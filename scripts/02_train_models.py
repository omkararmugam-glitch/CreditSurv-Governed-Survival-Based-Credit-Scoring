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
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditsurv.cleaning import (  # noqa: E402
    clean,
    fit_values,
    policy_from_config,
)
from creditsurv.config import load_config  # noqa: E402
from creditsurv.features.build import (  # noqa: E402
    FeatureSpec,
    build_design_matrix,
    add_derived_features,
    default_spec,
    train_test_split_loans,
    validation_split,
)
from creditsurv.models.cox import CoxModel, age_dependent_risk_profile  # noqa: E402
from creditsurv.models.discrete_hazard import (  # noqa: E402
    DiscreteTimeHazardModel,
    expansion_row_estimate,
)
from creditsurv.models.evaluate import CensoringModel, evaluate_survival  # noqa: E402
from creditsurv.io import schema as sch  # noqa: E402
from creditsurv.pipeline import encode_data_source  # noqa: E402
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


RESERVED_TAGS: dict[str, dict] = {
    # A tag whose results other sections cite by name. Writing something else under
    # one is how outputs/models/02_models_holdout.pkl came to hold a 200k dev-sample
    # monthly-bin model for a day, with FINDINGS section 6 reserved for the
    # pre-registered full-data run (see FINDINGS 7d).
    "holdout": {
        "full": (True, "the pre-registered holdout is trained on the full dataset "
                       "(--full)"),
        "time_bin": (3, "the pre-registered holdout uses quarterly bins "
                        "(--time-bin 3)"),
        "negative_subsample": (0.4, "the pre-registered holdout subsamples negatives "
                                    "at 0.4 (--negative-subsample 0.4)"),
        "split": ("out_of_time", "the holdout is an out-of-time split "
                                 "(--split out_of_time)"),
        "oot_cutoff": (2016, "the pre-registered cutoff is 2016 (--oot-cutoff 2016)"),
    },
    "full": {
        "full": (True, "the 'full' tag is the full-dataset primary model (--full)"),
        "time_bin": (3, "the primary full run uses quarterly bins (--time-bin 3)"),
        "negative_subsample": (0.4, "the primary full run subsamples negatives at 0.4 "
                                    "(--negative-subsample 0.4)"),
    },
}
"""Tags that may only be written by the run they were registered for, with the
setting each one requires and the sentence to print when it does not match."""


def check_reserved_tag(tag: str, args) -> list[str]:
    """Settings that disagree with a reserved tag's registered design.

    Returns one message per mismatch, empty when the tag is free or the run matches.
    A derived tag such as ``holdout_lcgrade`` is not reserved: only the exact name
    is, because that is the name other sections cite.
    """
    required = RESERVED_TAGS.get(tag)
    if not required:
        return []
    problems = []
    for name, (expected, why) in required.items():
        actual = getattr(args, name, None)
        if name == "full":
            actual = bool(actual)
        if actual != expected:
            problems.append(f"--{name.replace('_', '-')} is {actual!r}, expected "
                            f"{expected!r}: {why}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--full", action="store_true",
                    help="use the full labelled dataset instead of the dev sample")
    ap.add_argument("--with-derived", action="store_true",
                    help="include the derived origination features -- fico_midpoint, "
                         "emp_length_years, installment_to_income, loan_to_income, "
                         "log_annual_inc -- by building the feature spec AFTER they "
                         "are computed. Without this flag the spec is built from the "
                         "raw columns, so the derived features are absent (see "
                         "FINDINGS section 7c). Off by default: turning it on changes "
                         "the model's inputs, so it belongs to a new --tag.")
    ap.add_argument("--add-features", default=None,
                    help="comma-separated columns to ADD to the spec, for features "
                         "the default lists omit. 'term_months' is the case this "
                         "exists for: loan term is known before any decision but is "
                         "in no schema list (FINDINGS 7c). Leakage-checked, and each "
                         "added name is recorded in the metrics JSON.")
    ap.add_argument("--drop-features", default=None,
                    help="comma-separated features to remove from the spec, e.g. "
                         "'installment,installment_to_income' for a model that has "
                         "not seen the lender's assigned rate, or 'addr_state' for "
                         "one that has not seen geography. Dropped names are recorded "
                         "in the metrics JSON.")
    ap.add_argument("--with-lc-grade", action="store_true",
                    help="benchmark variant including LC grade / sub_grade / int_rate")
    ap.add_argument("--split", default="random", choices=["random", "out_of_time"])
    ap.add_argument("--oot-cutoff", type=int, default=None,
                    help="with --split out_of_time: first issue year of the test "
                         "period (e.g. 2016 trains on <=2015, tests on >=2016). "
                         "Required; the quantile-based default landed on 2018.")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="share of TRAINING loans held out for early stopping")
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
    if args.split == "out_of_time" and args.oot_cutoff is None:
        print("ERROR: --split out_of_time requires an explicit --oot-cutoff YEAR. "
              "The implicit quantile-based cutoff would put the boundary in the "
              "wrong year.", file=sys.stderr)
        return 2
    if args.oot_cutoff is not None and args.split != "out_of_time":
        print("ERROR: --oot-cutoff only applies with --split out_of_time.",
              file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    cfg.paths.ensure_dirs()
    tag = args.tag or ("full" if args.full else "dev")
    if args.with_lc_grade:
        tag += "_lcgrade"

    # Checked before anything is read or written: a reserved tag is a name other
    # sections cite, so it may only be written by the run it was registered for.
    mismatches = check_reserved_tag(tag, args)
    if mismatches:
        print(f"ERROR: {tag!r} is a reserved tag and this run does not match its "
              f"registered settings:", file=sys.stderr)
        for problem in mismatches:
            print(f"  {problem}", file=sys.stderr)
        print("Nothing has been run. Use a different --tag, or run the registered "
              "settings.", file=sys.stderr)
        return 2

    # Checked before any data is loaded, so a refused run costs nothing. Replacing
    # 02_models_<tag>.pkl silently would leave every Stage 3/4 result describing a
    # model that no longer exists.
    refused = guard_outputs(
        find_existing_outputs(
            [cfg.paths.tables_dir, cfg.paths.figures_dir, cfg.paths.models_dir],
            "02", tag)
        # The cleaning report is written by this stage too, so it is guarded here.
        + find_existing_outputs([cfg.paths.tables_dir], "00", tag),
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
    # The spec is normally built from the raw columns, which is why the derived
    # features are not in it. --with-derived builds it from the derived frame
    # instead; the design matrix derives them either way, so the only difference is
    # whether the spec names them.
    spec_columns = (add_derived_features(df.head(1)).columns if args.with_derived
                    else df.columns)
    spec = default_spec(spec_columns, extended=True, with_lc_grade=with_grade)
    added_by_request: list[str] = []
    if args.add_features:
        wanted = [c.strip() for c in args.add_features.split(",") if c.strip()]
        absent = [c for c in wanted if c not in df.columns]
        if absent:
            print(f"ERROR: --add-features names {absent}, which are not columns in "
                  f"{src.name}. Nothing has been run.", file=sys.stderr)
            return 2
        already = [c for c in wanted if c in spec.all_columns]
        if already:
            print(f"ERROR: --add-features names {already}, already in the spec. "
                  f"Nothing has been run.", file=sys.stderr)
            return 2
        # A leakage check before anything else: this flag is the one way a column
        # outside the curated lists can reach a model.
        sch.assert_no_leakage(list(spec.all_columns) + wanted)
        numeric_adds = [c for c in wanted
                        if pd.api.types.is_numeric_dtype(df[c])]
        categorical_adds = [c for c in wanted if c not in numeric_adds]
        spec = FeatureSpec(
            numeric=tuple(spec.numeric) + tuple(numeric_adds),
            categorical=tuple(spec.categorical) + tuple(categorical_adds),
            structural_missing=spec.structural_missing)
        added_by_request = wanted
        print(f"  added by request: {', '.join(wanted)}"
              + (f" (numeric: {', '.join(numeric_adds)})" if numeric_adds else "")
              + (f" (categorical: {', '.join(categorical_adds)})"
                 if categorical_adds else ""))

    dropped_by_request: list[str] = []
    if args.drop_features:
        wanted = [c.strip() for c in args.drop_features.split(",") if c.strip()]
        unknown = [c for c in wanted if c not in spec.all_columns]
        if unknown:
            print(f"ERROR: --drop-features names {unknown}, which are not in the "
                  f"feature spec. Nothing has been run.", file=sys.stderr)
            return 2
        dropped_by_request = wanted
        spec = FeatureSpec(
            numeric=tuple(c for c in spec.numeric if c not in wanted),
            categorical=tuple(c for c in spec.categorical if c not in wanted),
            structural_missing=tuple(c for c in spec.structural_missing
                                     if c not in wanted))
        print(f"  dropped by request: {', '.join(wanted)}")
    print(f"  features: {len(spec.numeric)} numeric + {len(spec.categorical)} categorical"
          f"{'  (INCLUDING LC grade -- benchmark variant)' if with_grade else ''}")

    train_idx, test_idx = train_test_split_loans(
        df, test_size=cfg.model.test_size, seed=cfg.model.seed, scheme=args.split,
        out_of_time_cutoff=args.oot_cutoff,
    )

    # --- cleaning (creditsurv/cleaning.py: the same module the dashboard uses) ---
    # Values are fitted on the TRAINING split only and saved in the model bundle,
    # so scoring an upload later reapplies these numbers instead of learning its
    # own. Under the default v1-parity policy nothing is altered -- the report
    # records what was seen, and tests/test_cleaning.py pins that no value moves.
    policy = policy_from_config(cfg)
    clean_values = fit_values(df.loc[train_idx], spec, policy=policy, source=src)
    # Row exclusions happen in Stage 1 (they need the outcome), so their counts are
    # read in here rather than recomputed, keeping one report per run.
    audit_file = cfg.paths.tables_dir / "01_label_audit.json"
    label_audit = None
    if audit_file.exists():
        try:
            label_audit = json.loads(audit_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            label_audit = None
    df, clean_report, _ = clean(df, spec, clean_values, policy=policy,
                               label_audit=label_audit)
    print(f"  cleaning: policy {policy.version}, values fitted on "
          f"{clean_values.fitted_rows:,} training rows")
    for line in clean_report.plain_english()[:6]:
        print(f"    {line}")
    if policy.changes_model_inputs():
        print("  WARNING: this cleaning policy alters model inputs; results are NOT "
              "comparable with v1-parity runs.")
    write_json({"cleaning": clean_report.to_dict(),
                "values": clean_values.to_dict(),
                "policy": {k: (list(v) if isinstance(v, tuple) else v)
                           for k, v in vars(policy).items()}},
               cfg.paths.tables_dir / f"00_cleaning_report_{tag}.json")
    print(f"  split={args.split}  train={len(train_idx):,}  test={len(test_idx):,}")
    years = {}
    if "issue_year" in df.columns:
        years = {
            "train_years": sorted(int(y) for y in df.loc[train_idx, "issue_year"].unique()),
            "test_years": sorted(int(y) for y in df.loc[test_idx, "issue_year"].unique()),
        }
        print(f"  train years {years['train_years'][0]}-{years['train_years'][-1]}, "
              f"test years {years['test_years'][0]}-{years['test_years'][-1]}")
    if args.split == "out_of_time":
        overlap = set(years.get("train_years", [])) & set(years.get("test_years", []))
        if overlap:
            print(f"ERROR: out-of-time split has years on both sides: {sorted(overlap)}",
                  file=sys.stderr)
            return 2

    times = np.array(cfg.model.eval_horizons_months, dtype=float)
    results, artefacts, by_vintage = [], {}, {}
    early_stopping = None

    # IPCW censoring model. On a random split train and test share one censoring
    # distribution, so it is fitted on train and the test set stays untouched. On
    # an out-of-time split they do not -- 6% of 2007-15 loans are still performing
    # versus 60% of 2016-18 -- and train-fitted weights would under-weight holdout
    # cases by up to 23x. There it is fitted on the evaluation set (standard for
    # Uno-type estimators; it uses follow-up times only, never predictions).
    ipcw_on = "test" if args.split == "out_of_time" else "train"
    ipcw_idx = test_idx if ipcw_on == "test" else train_idx
    censoring = CensoringModel(
        df.loc[ipcw_idx, "duration_months"].to_numpy(),
        df.loc[ipcw_idx, "event"].to_numpy(),
    )
    print(f"  IPCW censoring model fitted on the {ipcw_on} set")

    def by_year(name, surv, risk, dur, evt):
        """Per-vintage metrics on an out-of-time test set, each with its own
        censoring model, restricted to horizons that vintage can actually reach."""
        if args.split != "out_of_time" or "issue_year" not in df.columns:
            return
        yrs = df.loc[test_idx, "issue_year"].to_numpy()
        out = {}
        for y in sorted(set(yrs.tolist())):
            m = yrs == y
            reach = times[times < dur[m].max()]
            if len(reach) < 1:
                continue
            k = np.isin(times, reach)
            cm = CensoringModel(dur[m], evt[m])
            r = evaluate_survival(
                model_name=name, split_name=f"test_{int(y)}", survival=surv[m][:, k],
                times=reach, duration=dur[m], event=evt[m], censoring=cm,
                risk=None if risk is None else risk[m],
            )
            out[str(int(y))] = r.summary()
            print(f"  [{name}] vintage {int(y)}: n={int(m.sum()):,} "
                  f"C={r.concordance:.4f}, horizons up to {int(reach.max())}m")
        by_vintage[name] = out

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

        cox_surv = cox.predict_survival(dm_te.X, times)
        cox_risk = cox.predict_risk(dm_te.X)
        res = evaluate_survival(
            model_name="cox", split_name="test",
            survival=cox_surv, times=times,
            duration=dm_te.duration.to_numpy(), event=dm_te.event.to_numpy(),
            censoring=censoring, risk=cox_risk, calibration_at=24.0,
        )
        print(res)
        results.append(res)
        by_year("cox", cox_surv, cox_risk, dm_te.duration.to_numpy(),
                dm_te.event.to_numpy())

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
        fit_idx, val_idx = validation_split(train_idx, frac=args.val_frac,
                                            seed=cfg.model.seed)
        print(f"\n[gbm] early stopping validates on {len(val_idx):,} held-out "
              f"TRAINING loans ({args.val_frac:.0%}); fitting on {len(fit_idx):,}")
        dg_tr = build_design_matrix(df.loc[fit_idx], spec, flavour="gbm")
        dg_val = build_design_matrix(df.loc[val_idx], spec, flavour="gbm")
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
            valid=(dg_val.X, dg_val.duration.to_numpy(), dg_val.event.to_numpy()),
        )
        n_trees = dh.booster.num_trees()
        print(f"[gbm] fitted on {dh.training_rows_:,} person-periods in "
              f"{time.time() - t0:.1f}s; {n_trees} trees "
              f"(best iteration {dh.booster.best_iteration}, cap 600)")
        early_stopping = {"validation": f"random {args.val_frac:.0%} of training loans",
                          "n_fit": int(len(fit_idx)), "n_val": int(len(val_idx)),
                          "trees": int(n_trees),
                          "best_iteration": int(dh.booster.best_iteration)}

        gbm_surv = dh.predict_survival(dg_te.X, times)
        res = evaluate_survival(
            model_name="discrete_hazard", split_name="test",
            survival=gbm_surv, times=times,
            duration=dg_te.duration.to_numpy(), event=dg_te.event.to_numpy(),
            censoring=censoring, calibration_at=24.0,
        )
        print(res)
        results.append(res)
        by_year("discrete_hazard", gbm_surv, None, dg_te.duration.to_numpy(),
                dg_te.event.to_numpy())

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
        "oot_cutoff": args.oot_cutoff,
        **years,
        "ipcw_fitted_on": ipcw_on,
        "early_stopping": early_stopping,
        "data_source": encode_data_source(src),
        "results_by_vintage": by_vintage,
        "time_bin_months": int(args.time_bin or cfg.model.time_bin_months),
        "negative_subsample": float(args.negative_subsample),
        "with_lc_grade": bool(with_grade),
        "with_derived": bool(args.with_derived),
        "added_by_request": added_by_request,
        "dropped_by_request": dropped_by_request,
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
                     "train_idx": train_idx, "test_idx": test_idx,
                     # Fitted on training rows only; the dashboard reapplies these.
                     "cleaning_values": clean_values.to_dict(),
                     "cleaning_policy": {k: (list(v) if isinstance(v, tuple) else v)
                                         for k, v in vars(policy).items()},
                     # Stages 3 and 4 read this instead of guessing from the tag.
                     "data_source": encode_data_source(src),
                     "split": {"scheme": args.split, "oot_cutoff": args.oot_cutoff,
                               **years}}, fh)
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
