"""Inspect the ingested data. Read-only; writes nothing.

Usage
-----
    python scripts/inspect_data.py --what raw        # ingested accepted file
    python scripts/inspect_data.py --what missing    # missingness by vintage
    python scripts/inspect_data.py --what labeled    # survival target + KM curves
    python scripts/inspect_data.py --what rejected    # rejected file + overlap preview
    python scripts/inspect_data.py --what all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditsurv.config import load_config  # noqa: E402
from creditsurv.io import schema as sch  # noqa: E402
from creditsurv.labeling.survival_target import parse_month_series  # noqa: E402

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 50)


def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _load(path: Path, script: str, **kwargs) -> pd.DataFrame | None:
    """Read a Parquet file, or explain clearly why it cannot be read yet.

    A Parquet footer is only written when the file is closed, so a file still
    being produced by a running ingest raises ``ArrowInvalid`` about missing
    magic bytes. That is expected, not corruption, and says so.
    """
    if not path.exists():
        print(f"  {path} not found - run {script} first")
        return None
    try:
        return pd.read_parquet(path, **kwargs)
    except Exception as exc:  # pyarrow.lib.ArrowInvalid and friends
        if "magic bytes" in str(exc) or "Parquet" in type(exc).__name__:
            print(f"  {path.name} is incomplete ({path.stat().st_size / 1e6:.0f} MB "
                  f"written so far).\n"
                  f"  A Parquet footer is only written when the file is closed, so "
                  f"{script}\n  is most likely still running. Wait for it to finish, "
                  f"then re-run this.")
            return None
        raise


def show_raw(cfg) -> None:
    _rule("ACCEPTED (raw, post-ingest)")
    path = cfg.paths.accepted_parquet
    df = _load(path, "scripts/00_ingest.py")
    if df is None:
        return
    print(f"rows={len(df):,}  cols={df.shape[1]}  "
          f"memory={df.memory_usage(deep=True).sum() / 1e9:.2f} GB  "
          f"on disk={path.stat().st_size / 1e6:.0f} MB")

    print("\nloan_status distribution:")
    vc = df["loan_status"].value_counts(dropna=False)
    for status, n in vc.items():
        print(f"  {str(status):<52} {n:>10,}  {n / len(df):6.2%}")

    print("\nterm:")
    for t, n in df["term"].value_counts(dropna=False).items():
        print(f"  {str(t):<52} {n:>10,}  {n / len(df):6.2%}")

    issue = parse_month_series(df["issue_d"])
    print("\nloans by issue year:")
    for y, n in issue.dt.year.value_counts().sort_index().items():
        print(f"  {int(y)}  {n:>10,}  {'#' * int(60 * n / len(df))}")

    leaked = set(df.columns) & sch.LEAKAGE_COLUMNS
    print(f"\nleakage columns present: {sorted(leaked) if leaked else 'NONE (correct)'}")


def show_missing(cfg) -> None:
    _rule("MISSINGNESS")
    df = _load(cfg.paths.accepted_parquet, "scripts/00_ingest.py")
    if df is None:
        return
    miss = (df.isna().mean() * 100).sort_values(ascending=False)

    print("worst 25 columns overall (% missing):")
    for col, pct in miss.head(25).items():
        print(f"  {col:<38} {pct:6.2f}%  {'#' * int(pct / 2)}")

    # The claim worth checking: the extended bureau families were only added
    # around 2012, so they should be near-100% missing for early vintages.
    df["_year"] = parse_month_series(df["issue_d"]).dt.year
    families = {
        "num_* (bureau counts)": [c for c in df.columns if c.startswith("num_")],
        "mo_sin_* (months since)": [c for c in df.columns if c.startswith("mo_sin_")],
        "core (loan_amnt, dti, fico)": ["loan_amnt", "dti", "fico_range_low"],
    }
    print("\n% missing by issue year, by column family:")
    header = "  year  " + "".join(f"{k:>30}" for k in families)
    print(header)
    for year, grp in df.groupby("_year", observed=True):
        if pd.isna(year):
            continue
        cells = ""
        for cols in families.values():
            cols = [c for c in cols if c in grp.columns]
            cells += f"{grp[cols].isna().mean().mean() * 100:29.1f}%" if cols else f"{'-':>30}"
        print(f"  {int(year)}  {cells}")


def show_labeled(cfg) -> None:
    _rule("SURVIVAL TARGET")
    df = _load(cfg.paths.labeled_parquet, "scripts/01_build_labels.py")
    if df is None:
        return
    print(f"rows={len(df):,}  event rate={df['event'].mean():.4f}  "
          f"events={int(df['event'].sum()):,}")

    print("\noutcome breakdown:")
    for outcome, grp in df.groupby("outcome", observed=True):
        print(f"  {str(outcome):<22} n={len(grp):>9,}  "
              f"median duration={grp['duration_months'].median():5.1f}  "
              f"event={grp['event'].mean():.0f}")

    print("\nduration_months quantiles:")
    q = df["duration_months"].quantile([0, .1, .25, .5, .75, .9, .99, 1.0])
    print("  " + "  ".join(f"p{int(k * 100)}={int(v)}" for k, v in q.items()))

    print("\nevent rate by term:")
    for term, grp in df.groupby("term_months", observed=True):
        print(f"  {int(term)} months  n={len(grp):>9,}  event rate={grp['event'].mean():.4f}")

    if "grade" in df.columns:
        print("\nevent rate by LC grade (sanity check - should rise A->G):")
        for g, grp in df.groupby("grade", observed=True):
            print(f"  {g}  n={len(grp):>9,}  event rate={grp['event'].mean():.4f}  "
                  f"{'#' * int(grp['event'].mean() * 100)}")

    print("\nKaplan-Meier survival (probability of no default by month):")
    try:
        from lifelines import KaplanMeierFitter
        km = KaplanMeierFitter().fit(df["duration_months"], df["event"])
        for t in (6, 12, 18, 24, 30, 36, 48, 60):
            if t <= df["duration_months"].max():
                print(f"  S({t:>2} months) = {float(km.predict(t)):.4f}")
    except ImportError:
        print("  lifelines not installed")

    print("\nadministrative censoring by vintage (why survival framing matters):")
    for y, grp in df.groupby("issue_year", observed=True):
        admin = (grp["outcome"] == "censored_admin").mean()
        print(f"  {int(y)}  n={len(grp):>9,}  still-current={admin:6.2%}  "
              f"median obs={grp['duration_months'].median():4.0f} months")


def show_rejected(cfg) -> None:
    _rule("REJECTED APPLICATIONS")
    path = cfg.paths.rejected_parquet
    if not path.exists():
        print(f"  {path} not found - run scripts/00_ingest.py first")
        return

    # 27.6M rows. Materialising all five string columns as Python objects would
    # need roughly 8 GB, so null counts come from Parquet row-group metadata
    # (free) and only the columns actually analysed are read, as categories.
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    print(f"rows={n_rows:,}  cols={pf.metadata.num_columns}  "
          f"on disk={path.stat().st_size / 1e6:.0f} MB  "
          f"row_groups={pf.metadata.num_row_groups}")

    nulls: dict[str, int] = {}
    for rg in range(pf.metadata.num_row_groups):
        meta = pf.metadata.row_group(rg)
        for c in range(meta.num_columns):
            col = meta.column(c)
            name = col.path_in_schema
            nulls[name] = nulls.get(name, 0) + col.statistics.null_count

    print("\n% missing per column (from Parquet metadata, no full read):")
    for col, n in sorted(nulls.items(), key=lambda kv: -kv[1]):
        print(f"  {col:<20} {n / n_rows * 100:6.2f}%   ({n:,} nulls)")

    rej = pd.read_parquet(
        path, columns=["loan_amnt", "application_d", "risk_score", "dti_raw"]
    )
    for col in ("application_d", "dti_raw"):
        rej[col] = rej[col].astype("category")

    # The Stage 4 ceiling: risk_score coverage by year decides how much of the
    # rejected population is usable for the selection-bias diagnostic at all.
    app = parse_month_series(rej["application_d"])
    rej["_year"] = app.dt.year
    print("\nrisk_score coverage by application year:")
    for y, grp in rej.groupby("_year", observed=True):
        if pd.isna(y):
            continue
        cov = grp["risk_score"].notna().mean()
        print(f"  {int(y)}  n={len(grp):>10,}  coverage={cov:6.2%}  "
              f"{'#' * int(cov * 50)}")

    dti = pd.to_numeric(
        rej["dti_raw"].astype("string").str.replace("%", "", regex=False), errors="coerce"
    )
    print(f"\ndti_raw parsed: {dti.notna().mean():.2%} numeric")
    print(f"  quantiles: " + "  ".join(
        f"p{int(k * 100)}={v:,.1f}" for k, v in
        dti.quantile([.01, .25, .5, .75, .95, .99, 1.0]).items()))
    print(f"  implausible (>100%): {(dti > 100).mean():.2%}   "
          f"(>1000%): {(dti > 1000).mean():.2%}")

    acc_path = cfg.paths.accepted_parquet
    if acc_path.exists():
        print("\ncommon-feature comparison preview (accepted vs rejected):")
        acc = pd.read_parquet(acc_path, columns=["loan_amnt", "fico_range_low", "dti"])
        pairs = [
            ("loan amount", acc["loan_amnt"], rej["loan_amnt"]),
            ("score", acc["fico_range_low"], rej["risk_score"]),
            ("dti", acc["dti"], dti.clip(upper=100)),
        ]
        print(f"  {'feature':<14}{'accepted mean':>16}{'rejected mean':>16}{'SMD':>10}")
        for name, a, r in pairs:
            a, r = a.dropna(), r.dropna()
            pooled = np.sqrt((a.var() + r.var()) / 2)
            smd = (a.mean() - r.mean()) / pooled if pooled else np.nan
            print(f"  {name:<14}{a.mean():>16,.1f}{r.mean():>16,.1f}{smd:>10.3f}")
        print("  (SMD only - the full pre-registered gate runs in Stage 4)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--what", default="all",
                    choices=["raw", "missing", "labeled", "rejected", "all"])
    args = ap.parse_args()
    cfg = load_config(args.config)

    for name, fn in (("raw", show_raw), ("missing", show_missing),
                     ("labeled", show_labeled), ("rejected", show_rejected)):
        if args.what in (name, "all"):
            fn(cfg)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
