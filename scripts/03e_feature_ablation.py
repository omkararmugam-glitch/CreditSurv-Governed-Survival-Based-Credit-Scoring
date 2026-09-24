"""Stage 3e -- what does the model lose when a feature is missing?

Measures, rather than assumes, which uploaded columns a scoring run actually needs.
For each feature and each logical group of features, the holdout is scored with
that feature set to **missing** -- not to a guessed value -- and the loss in
Harrell's C and in 12-month AUC is recorded against the intact model.

Setting a feature to missing is exactly what a scoring run does when a column is
absent from the file, so these numbers are the cost of that absence rather than a
general importance measure: a feature can matter a great deal and still cost little
when absent, because the trees route around it through correlated columns.

Only the discrete-hazard model is ablated. It takes missing values natively, which
is what makes the question well posed; the Cox path median-fills, so a missing
column there is an imputed column and a different question.

    python scripts/03e_feature_ablation.py --model-tag holdout --sample 50000
    python scripts/03e_feature_ablation.py --model-tag holdout --sample 0   # all rows
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
from creditsurv.features.build import build_design_matrix  # noqa: E402
from creditsurv.models.evaluate import (  # noqa: E402
    CensoringModel,
    concordance_index,
    cumulative_dynamic_auc,
)
from creditsurv.pipeline import load_model_bundle, resolve_data_source  # noqa: E402
from creditsurv.provenance import build_stamp, guard_outputs  # noqa: E402
from creditsurv.reporting.tables import to_markdown, write_json, write_table  # noqa: E402

# Logical groups: columns that arrive together in a real file, so their joint
# absence is the case worth measuring. A file either has the bureau's recency
# block or it does not.
GROUPS: dict[str, tuple[str, ...]] = {
    # Reported separately because three of its four members are excluded from the
    # model by the section 0 decision, so the group reduces to installment.
    "lender_pricing": ("grade", "sub_grade", "int_rate", "installment"),
    "loan_structure": ("loan_amnt", "installment"),
    "income_and_burden": ("annual_inc", "dti"),
    "recency_months": ("mths_since_last_delinq", "mths_since_last_record",
                       "mths_since_last_major_derog", "mths_since_recent_bc",
                       "mths_since_recent_bc_dlq", "mths_since_recent_inq",
                       "mths_since_recent_revol_delinq", "mo_sin_old_il_acct",
                       "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op",
                       "mo_sin_rcnt_tl"),
    "delinquency_and_public_record": ("delinq_2yrs", "delinq_amnt", "pub_rec",
                                      "pub_rec_bankruptcies", "tax_liens",
                                      "acc_now_delinq", "chargeoff_within_12_mths",
                                      "collections_12_mths_ex_med",
                                      "num_accts_ever_120_pd", "num_tl_30dpd",
                                      "num_tl_90g_dpd_24m", "pct_tl_nvr_dlq"),
    "utilisation": ("revol_util", "bc_util", "percent_bc_gt_75", "bc_open_to_buy",
                    "revol_bal", "total_rev_hi_lim", "total_bc_limit"),
    "balances_and_limits": ("tot_cur_bal", "avg_cur_bal", "tot_hi_cred_lim",
                            "total_bal_ex_mort", "total_il_high_credit_limit",
                            "tot_coll_amt"),
    "account_counts": ("open_acc", "total_acc", "num_sats", "num_actv_bc_tl",
                       "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
                       "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
                       "mort_acc", "acc_open_past_24mths", "num_tl_op_past_12m"),
    "inquiries": ("inq_last_6mths", "mths_since_recent_inq"),
    "geography": ("addr_state",),
    "application_descriptors": ("purpose", "home_ownership", "verification_status",
                                "application_type", "initial_list_status"),
}


def _ablate(X: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    """A copy of ``X`` with ``features`` set to missing, exactly as an absent
    column arrives: NaN for numerics, an unset category for categoricals, and the
    matching ``<col>_missing`` indicator raised where the model has one."""
    out = X.copy()
    for col in features:
        if col not in out.columns:
            continue
        if str(out[col].dtype) == "category":
            out[col] = pd.Categorical([None] * len(out),
                                      categories=out[col].cat.categories)
        else:
            out[col] = np.nan
        indicator = f"{col}_missing"
        if indicator in out.columns:
            out[indicator] = 1
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--tag", default=None, help="output tag (default: --model-tag)")
    ap.add_argument("--model-tag", default="holdout")
    ap.add_argument("--sample", type=int, default=50_000,
                    help="test rows used; 0 uses the whole test split")
    ap.add_argument("--groups-only", action="store_true",
                    help="skip the per-feature pass and measure groups only")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg.paths.ensure_dirs()
    tag = args.tag or args.model_tag
    out_json = cfg.paths.tables_dir / f"03e_ablation_{tag}.json"
    out_features = cfg.paths.tables_dir / f"03e_ablation_features_{tag}.csv"
    out_groups = cfg.paths.tables_dir / f"03e_ablation_groups_{tag}.csv"
    refused = guard_outputs([out_json, out_features, out_groups], args.overwrite,
                            script="03e_feature_ablation.py")
    if refused:
        return refused

    bundle, model_path = load_model_bundle(cfg.paths.models_dir, args.model_tag)
    if "discrete_hazard" not in bundle["artefacts"]:
        print("ERROR: this bundle has no discrete-hazard model; the Cox path "
              "median-fills, which is a different question.", file=sys.stderr)
        return 2
    model = bundle["artefacts"]["discrete_hazard"]
    spec = bundle["spec"]
    src = resolve_data_source(bundle, cfg, args.model_tag)
    print(f"model {args.model_tag}; data {src}")

    cols = list(spec.all_columns) + ["duration_months", "event"]
    df = pd.read_parquet(src, columns=cols)
    for c in [c for c in df.columns if df[c].dtype == object]:
        df[c] = df[c].astype("category")
    test = df.loc[bundle["test_idx"].intersection(df.index)]
    if args.sample and args.sample < len(test):
        test = test.sample(args.sample, random_state=cfg.model.seed)
    print(f"test rows used: {len(test):,}")

    dm = build_design_matrix(test, spec, flavour="gbm")
    duration = test["duration_months"].to_numpy()
    event = test["event"].to_numpy()
    times = np.array(cfg.model.eval_horizons_months, dtype=float)
    # Out-of-time split: the censoring distribution is the evaluation sample's own,
    # and it depends on outcomes only, never on the model, so it is fitted once and
    # reused for every ablation.
    censoring = CensoringModel(duration, event)

    def score(X: pd.DataFrame) -> tuple[float, float]:
        survival = model.predict_survival(X, times)
        c = concordance_index(duration, event, 1.0 - survival[:, -1])
        auc = cumulative_dynamic_auc(survival, times, duration, event, censoring)
        row = auc.loc[auc["time"] == 12.0]
        return c, (float(row["auc"].iloc[0]) if len(row) else float("nan"))

    t0 = time.perf_counter()
    base_c, base_auc12 = score(dm.X)
    per_pass = time.perf_counter() - t0
    print(f"intact model: concordance {base_c:.4f}, 12m AUC {base_auc12:.4f} "
          f"({per_pass:.1f}s per pass)")

    features = [c for c in list(spec.numeric) + list(spec.categorical)
                if c in dm.X.columns or c in spec.categorical]
    n_passes = len(GROUPS) + (0 if args.groups_only else len(features))
    print(f"{n_passes} ablation passes to run, roughly "
          f"{n_passes * per_pass / 60:.1f} min\n")

    rows = []
    if not args.groups_only:
        for i, col in enumerate(features, start=1):
            c, auc12 = score(_ablate(dm.X, (col,)))
            rows.append({"scope": "feature", "name": col, "n_features": 1,
                         "concordance": c, "concordance_drop": base_c - c,
                         "auc_12m": auc12, "auc_12m_drop": base_auc12 - auc12})
            if i % 10 == 0 or i == len(features):
                print(f"  features {i}/{len(features)}")

    group_rows = []
    for name, members in GROUPS.items():
        present = tuple(m for m in members if m in dm.X.columns or m in spec.categorical)
        if not present:
            continue
        c, auc12 = score(_ablate(dm.X, present))
        absent = [m for m in members if m not in present]
        group_rows.append({"scope": "group", "name": name,
                          "n_features": len(present),
                          "features": ", ".join(present),
                          "not_model_features": ", ".join(absent),
                          "concordance": c, "concordance_drop": base_c - c,
                          "auc_12m": auc12, "auc_12m_drop": base_auc12 - auc12})
        print(f"  group {name}: C {c:.4f} (-{base_c - c:.4f}), "
              f"12m AUC {auc12:.4f} (-{base_auc12 - auc12:.4f})")

    features_table = (pd.DataFrame(rows).sort_values("concordance_drop",
                                                     ascending=False)
                      .reset_index(drop=True) if rows else pd.DataFrame())
    groups_table = (pd.DataFrame(group_rows).sort_values("concordance_drop",
                                                         ascending=False)
                    .reset_index(drop=True))
    if not features_table.empty:
        write_table(features_table, out_features)
        print("\nWorst 15 features to lose (by concordance drop):")
        print(to_markdown(features_table.head(15)[
            ["name", "concordance_drop", "auc_12m_drop"]]))
    write_table(groups_table, out_groups)
    print("\nGroups, worst first:")
    print(to_markdown(groups_table[["name", "n_features", "concordance_drop",
                                    "auc_12m_drop"]]))

    payload = {
        "tag": tag,
        "model_tag": args.model_tag,
        "model": "discrete_hazard",
        "n_test_rows": int(len(test)),
        "sampled": bool(args.sample and args.sample < len(df.loc[bundle["test_idx"]
                                                                .intersection(df.index)])),
        "baseline": {"concordance": base_c, "auc_12m": base_auc12},
        "horizons": [int(t) for t in times],
        "seconds_per_pass": round(per_pass, 2),
        "features": features_table.to_dict("records") if not features_table.empty else [],
        "groups": groups_table.to_dict("records"),
        "provenance": build_stamp(
            stage="03e_feature_ablation",
            inputs={"data": src, "model": model_path},
            outputs={"result": out_json, "features": out_features,
                     "groups": out_groups},
            config_path=args.config,
            args={"model_tag": args.model_tag, "sample": args.sample,
                  "groups_only": args.groups_only}),
    }
    write_json(payload, out_json)
    print(f"\nWritten: {out_json.name}, {out_groups.name}"
          + (f", {out_features.name}" if not features_table.empty else ""))
    print("\nNo required/optional split is applied here. That split is a decision "
          "recorded in FINDINGS from these numbers, not an output of this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
