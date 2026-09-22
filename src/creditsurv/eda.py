"""Exploratory data analysis: describe a dataset, change nothing.

Every function here takes a frame and returns tables. None of them writes to the
frame, fits anything, or is used by the modelling path -- an EDA result can never
leak into a model. :mod:`scripts.01b_eda` turns these tables into CSVs, PNGs and
an HTML report; the dashboard's Data Profile tab calls the same functions on an
uploaded file, so the two always describe data the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["overview", "missing_table", "numeric_summary", "categorical_frequencies",
           "outlier_table", "correlation", "high_correlations", "target_rates",
           "vintage_rates", "notable_findings"]

HIGH_CORRELATION = 0.80
"""Pairs at or above this absolute Pearson correlation are listed. Not a rule
about what to drop: an observation for the reader."""


def overview(df: pd.DataFrame, *, date_col: str = "issue_month") -> dict:
    out = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 1),
        "numeric_columns": int(df.select_dtypes("number").shape[1]),
        "categorical_columns": int(df.select_dtypes(["category", "object", "string"]).shape[1]),
        "duplicate_ids": int(df["id"].duplicated().sum()) if "id" in df.columns else None,
    }
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        if dates.notna().any():
            out["date_range"] = [str(dates.min().date()), str(dates.max().date())]
    if "event" in df.columns:
        out["event_rate"] = round(float(pd.to_numeric(df["event"], errors="coerce").mean()), 4)
    if "duration_months" in df.columns:
        d = pd.to_numeric(df["duration_months"], errors="coerce")
        out["median_duration_months"] = float(d.median())
    return out


def missing_table(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    cols = list(columns) if columns is not None else list(df.columns)
    cols = [c for c in cols if c in df.columns]
    n = max(len(df), 1)
    rows = [{"column": c, "dtype": str(df[c].dtype), "missing": int(df[c].isna().sum()),
             "missing_share": round(float(df[c].isna().mean()), 4),
             "distinct": int(df[c].nunique(dropna=True))} for c in cols]
    return (pd.DataFrame(rows).sort_values("missing_share", ascending=False)
            .reset_index(drop=True) if rows else pd.DataFrame())


def numeric_summary(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    cols = [c for c in (columns if columns is not None else df.select_dtypes("number").columns)
            if c in df.columns]
    rows = []
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        if not v.notna().any():
            continue
        q = v.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        rows.append({"column": c, "n": int(v.notna().sum()),
                     "mean": float(v.mean()), "std": float(v.std()),
                     "min": float(v.min()), "p1": float(q.loc[0.01]),
                     "p25": float(q.loc[0.25]), "median": float(q.loc[0.5]),
                     "p75": float(q.loc[0.75]), "p99": float(q.loc[0.99]),
                     "max": float(v.max()),
                     "skew": float(v.skew()) if v.notna().sum() > 2 else np.nan})
    return pd.DataFrame(rows)


def categorical_frequencies(df: pd.DataFrame, columns=None, *, top: int = 15) -> pd.DataFrame:
    cols = [c for c in (columns if columns is not None
                        else df.select_dtypes(["category", "object", "string"]).columns)
            if c in df.columns]
    rows = []
    for c in cols:
        counts = df[c].astype("string").str.strip().value_counts(dropna=False)
        for level, n in counts.head(top).items():
            rows.append({"column": c, "level": "(missing)" if pd.isna(level) else str(level),
                         "count": int(n), "share": round(float(n) / max(len(df), 1), 4)})
        if len(counts) > top:
            rest = counts.iloc[top:].sum()
            rows.append({"column": c, "level": f"(other {len(counts) - top} levels)",
                         "count": int(rest), "share": round(float(rest) / max(len(df), 1), 4)})
    return pd.DataFrame(rows)


def outlier_table(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Counts by two conventional rules. Reported, never acted on."""
    cols = [c for c in (columns if columns is not None else df.select_dtypes("number").columns)
            if c in df.columns]
    rows = []
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        if not v.notna().any():
            continue
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        iqr = q3 - q1
        lo_i, hi_i = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        lo_p, hi_p = v.quantile(0.005), v.quantile(0.995)
        rows.append({"column": c,
                     "iqr_outliers": int(((v < lo_i) | (v > hi_i)).sum()),
                     "iqr_share": round(float(((v < lo_i) | (v > hi_i)).mean()), 4),
                     "beyond_p0.5_p99.5": int(((v < lo_p) | (v > hi_p)).sum()),
                     "max_over_p99.5": round(float(v.max() / hi_p), 2)
                     if pd.notna(hi_p) and hi_p > 0 else np.nan})
    return (pd.DataFrame(rows).sort_values("iqr_share", ascending=False)
            .reset_index(drop=True) if rows else pd.DataFrame())


def correlation(df: pd.DataFrame, columns=None, *, max_rows: int = 200_000) -> pd.DataFrame:
    cols = [c for c in (columns if columns is not None else df.select_dtypes("number").columns)
            if c in df.columns]
    frame = df[cols]
    if len(frame) > max_rows:            # correlations converge long before 2M rows
        frame = frame.sample(max_rows, random_state=0)
    return frame.apply(pd.to_numeric, errors="coerce").corr(numeric_only=True)


def high_correlations(matrix: pd.DataFrame, threshold: float = HIGH_CORRELATION) -> pd.DataFrame:
    rows = []
    cols = list(matrix.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = matrix.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                rows.append({"feature_a": a, "feature_b": b, "correlation": round(float(r), 4)})
    return (pd.DataFrame(rows).sort_values("correlation", key=lambda s: s.abs(),
                                           ascending=False).reset_index(drop=True)
            if rows else pd.DataFrame(columns=["feature_a", "feature_b", "correlation"]))


def target_rates(df: pd.DataFrame, by: str, *, event_col: str = "event",
                 min_size: int = 50) -> pd.DataFrame:
    """Default rate by segment. Segments below ``min_size`` are kept but marked,
    since a 3-loan segment with a 100% rate is noise, not a finding."""
    if by not in df.columns or event_col not in df.columns:
        return pd.DataFrame()
    g = df.groupby(df[by].astype("string").fillna("(missing)"), observed=True)[event_col]
    out = g.agg(loans="size", events="sum").reset_index(names=by)
    out["default_rate"] = (out["events"] / out["loans"]).round(4)
    if "duration_months" in df.columns:
        out["median_duration"] = (df.groupby(df[by].astype("string").fillna("(missing)"),
                                             observed=True)["duration_months"]
                                  .median().to_numpy())
    out["small_segment"] = out["loans"] < min_size
    return out.sort_values("default_rate", ascending=False).reset_index(drop=True)


def vintage_rates(df: pd.DataFrame, *, year_col: str = "issue_year",
                  event_col: str = "event") -> pd.DataFrame:
    if year_col not in df.columns:
        return pd.DataFrame()
    out = target_rates(df, year_col, event_col=event_col)
    return out.sort_values(year_col).reset_index(drop=True) if not out.empty else out


def notable_findings(*, missing: pd.DataFrame, outliers: pd.DataFrame,
                     correlations: pd.DataFrame, segments: dict[str, pd.DataFrame],
                     overview_stats: dict) -> list[str]:
    """Observations, deliberately not conclusions: each states what is in the data
    and leaves the interpretation to the reader."""
    out: list[str] = []
    if overview_stats.get("event_rate") is not None:
        out.append(f"The overall default rate is {overview_stats['event_rate']:.1%} over "
                   f"{overview_stats['rows']:,} loans.")
    if not missing.empty:
        heavy = missing[missing["missing_share"] >= 0.5]
        if not heavy.empty:
            out.append(f"{len(heavy)} column(s) are at least half missing: "
                       + ", ".join(f"{r.column} ({r.missing_share:.0%})"
                                   for r in heavy.head(6).itertuples()) + ".")
        none_missing = int((missing["missing_share"] == 0).sum())
        out.append(f"{none_missing} of {len(missing)} columns have no missing values.")
    if not correlations.empty:
        top = correlations.head(5)
        out.append(f"{len(correlations)} feature pair(s) correlate at |r| >= "
                   f"{HIGH_CORRELATION:.2f}, the strongest being "
                   + ", ".join(f"{r.feature_a}~{r.feature_b} ({r.correlation:+.2f})"
                               for r in top.itertuples()) + ".")
    if not outliers.empty:
        worst = outliers.head(3)
        out.append("Columns with the longest tails by the 1.5x IQR rule: "
                   + ", ".join(f"{r.column} ({r.iqr_share:.1%} of rows)"
                               for r in worst.itertuples()) + ".")
    for name, table in segments.items():
        if table is None or table.empty or "default_rate" not in table:
            continue
        big = table[~table["small_segment"]] if "small_segment" in table else table
        if len(big) < 2:
            continue
        hi, lo = big.iloc[0], big.iloc[-1]
        key = big.columns[0]
        out.append(f"By {name}, the default rate runs from {lo['default_rate']:.1%} "
                   f"({lo[key]}, {int(lo['loans']):,} loans) to {hi['default_rate']:.1%} "
                   f"({hi[key]}, {int(hi['loans']):,} loans).")
    if "vintage" in segments and segments["vintage"] is not None and             not segments["vintage"].empty:
        out.append("Observed default rates fall for the most recent vintages because "
                   "those loans have had less time to default (right censoring), not "
                   "necessarily because they are safer.")
    if overview_stats.get("duplicate_ids"):
        out.append(f"{overview_stats['duplicate_ids']:,} rows share an id with another row.")
    return out
