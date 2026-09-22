"""Stage 1c -- exploratory data analysis of the labelled training data.

Read-only by construction: it loads the labelled Parquet, writes tables, charts
and an HTML report under ``outputs/eda/<tag>/``, and never writes back to any data
file. Every number comes from :mod:`creditsurv.eda`, the same module the
dashboard's Data Profile tab uses, so a figure here and a figure there cannot
disagree.

Accepted-vs-rejected comparison is *read* from the Stage 4 diagnostic result
(``04_reject_inference_<tag>.json``) rather than recomputed, so this report and
FINDINGS section 4 can never quote different selection-bias numbers.

    python scripts/01b_eda.py --tag full
    python scripts/01b_eda.py --tag dev --sample 50000
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditsurv import eda  # noqa: E402
from creditsurv.config import load_config  # noqa: E402
from creditsurv.features.build import add_derived_features, default_spec  # noqa: E402
from creditsurv.provenance import (  # noqa: E402
    build_stamp,
    guard_outputs,
)
from creditsurv.reporting import figures as figs  # noqa: E402
from creditsurv.reporting.tables import write_json, write_table  # noqa: E402

SEGMENTS = ("grade", "term_months", "purpose", "income_band", "home_ownership")


def _existing(out_dir: Path) -> list[Path]:
    return sorted(p for p in out_dir.rglob("*") if p.is_file()) if out_dir.exists() else []


def _html(out_dir: Path, tag: str, stats: dict, findings: list[str],
          tables: dict[str, pd.DataFrame], charts: list[Path],
          diagnostic: dict | None) -> Path:
    def table_html(df: pd.DataFrame, max_rows: int = 40) -> str:
        if df is None or df.empty:
            return "<p class='muted'>No data.</p>"
        shown = df.head(max_rows)
        more = (f"<p class='muted'>{len(df) - max_rows:,} more rows in the CSV.</p>"
                if len(df) > max_rows else "")
        return shown.to_html(index=False, classes="tbl", float_format=lambda v: f"{v:,.4g}") + more

    cards = "".join(
        f"<div class='card'><div class='k'>{html.escape(str(k).replace('_', ' '))}</div>"
        f"<div class='v'>{html.escape(str(v))}</div></div>"
        for k, v in stats.items() if not isinstance(v, (list, dict)))
    date_range = stats.get("date_range")
    if date_range:
        cards += (f"<div class='card'><div class='k'>date range</div>"
                  f"<div class='v'>{date_range[0]} to {date_range[1]}</div></div>")

    parts = [f"""<!doctype html><html><head><meta charset="utf-8">
<title>EDA report - {html.escape(tag)}</title><style>
body{{font:15px/1.55 system-ui,Segoe UI,sans-serif;margin:0;color:#1c2430;background:#fbfbfd}}
header{{background:#2f5d8a;color:#fff;padding:1.4rem 2rem}}
header h1{{margin:.2rem 0;font-size:1.6rem}} header .sub{{opacity:.85;font-size:.9rem}}
main{{max-width:1180px;margin:0 auto;padding:1.5rem 2rem 4rem}}
h2{{margin-top:2.4rem;border-bottom:1px solid #d7dce5;padding-bottom:.3rem;font-size:1.25rem}}
.cards{{display:flex;flex-wrap:wrap;gap:.6rem;margin:1rem 0}}
.card{{background:#eef1f6;border-radius:8px;padding:.6rem .9rem;min-width:120px}}
.card .k{{font-size:.7rem;text-transform:uppercase;letter-spacing:.09em;color:#6b7280}}
.card .v{{font-size:1.15rem;font-weight:650}}
table.tbl{{border-collapse:collapse;font-size:.85rem;margin:.6rem 0;background:#fff}}
table.tbl th,table.tbl td{{border:1px solid #e3e7ee;padding:.32rem .55rem;text-align:right}}
table.tbl th{{background:#eef1f6;text-align:left}} table.tbl td:first-child{{text-align:left}}
ul.find li{{margin:.3rem 0}} .muted{{color:#6b7280;font-size:.85rem}}
img{{max-width:100%;border:1px solid #e3e7ee;border-radius:6px;background:#fff;margin:.5rem 0}}
.note{{background:#fff8e6;border-left:4px solid #d9a02b;padding:.7rem 1rem;margin:1rem 0}}
</style></head><body>
<header><div class='sub'>creditsurv &middot; exploratory data analysis</div>
<h1>Training data profile - tag {html.escape(tag)}</h1>
<div class='sub'>Read-only report. Nothing here changed the data, and no number
here feeds a model.</div></header><main>
<h2>Overview</h2><div class='cards'>{cards}</div>
<h2>Observations</h2><div class='note'>Stated as observations, not conclusions:
each line says what is in the data and leaves the interpretation open.</div>
<ul class='find'>""" + "".join(f"<li>{html.escape(f)}</li>" for f in findings) + "</ul>"]

    order = [("Missing values", "missing"), ("Numeric distributions", "numeric_summary"),
             ("Categorical frequencies", "categorical_frequencies"),
             ("Outliers", "outliers"), ("Highly correlated pairs", "high_correlations")]
    for title, key in order:
        if key in tables:
            parts.append(f"<h2>{title}</h2>{table_html(tables[key])}")

    parts.append("<h2>Default rates by segment</h2>")
    for key, table in tables.items():
        if key.startswith("default_rate_"):
            parts.append(f"<h3>{html.escape(key.replace('default_rate_', ''))}</h3>"
                         + table_html(table, 25))

    if diagnostic:
        parts.append("<h2>Accepted vs rejected applicants</h2>"
                     "<p class='muted'>Read from the Stage 4 diagnostic result, not "
                     "recomputed, so these are the same numbers FINDINGS section 4 "
                     "reports.</p>")
        comp = diagnostic.get("diagnostic", {})
        rows = comp.get("comparisons") or comp.get("features") or []
        if isinstance(rows, list) and rows:
            parts.append(table_html(pd.DataFrame(rows)))
        else:
            parts.append(f"<pre class='muted'>{html.escape(json.dumps(comp, indent=2)[:4000])}</pre>")

    parts.append("<h2>Charts</h2>")
    for c in charts:
        parts.append(f"<h3>{html.escape(c.stem)}</h3><img src='{c.name}' alt='{c.stem}'>")
    parts.append("</main></body></html>")

    path = out_dir / "index.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--tag", default="full",
                    help="'full' profiles the labelled dataset, 'dev' the sample")
    ap.add_argument("--data", default=None,
                    help="explicit Parquet path, overriding --tag")
    ap.add_argument("--sample", type=int, default=None,
                    help="profile a random sample of this many rows (charts and "
                         "correlations are unaffected in shape, only in precision)")
    ap.add_argument("--diagnostic-tag", default=None,
                    help="tag of the Stage 4 result to read accepted-vs-rejected "
                         "numbers from (default: --tag)")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing an existing report for this --tag")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg.paths.ensure_dirs()
    out_dir = Path("outputs/eda") / args.tag

    # Guarded like every other stage: an existing report is never replaced silently.
    refused = guard_outputs(_existing(out_dir), args.overwrite, script="01b_eda.py")
    if refused:
        return refused

    src = (Path(args.data) if args.data
           else (cfg.paths.labeled_parquet if args.tag.startswith("full")
                 else cfg.paths.dev_sample_parquet))
    if not src.exists():
        print(f"ERROR: {src} not found. Run scripts/01_build_labels.py first.",
              file=sys.stderr)
        return 2

    print(f"reading {src}")
    df = pd.read_parquet(src)
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=cfg.sample.seed)
        print(f"  sampled {len(df):,} rows")
    df = add_derived_features(df)          # income_band etc. are useful segments
    for col in [c for c in df.columns if df[c].dtype == object and c != "id"]:
        df[col] = df[col].astype("category")
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = default_spec(df.columns, extended=True, with_lc_grade=True)
    numeric = [c for c in spec.numeric if c in df.columns]
    print(f"  {len(df):,} rows x {df.shape[1]} columns")

    stats = eda.overview(df)
    tables: dict[str, pd.DataFrame] = {
        "missing": eda.missing_table(df),
        "numeric_summary": eda.numeric_summary(df, numeric),
        "categorical_frequencies": eda.categorical_frequencies(
            df, [c for c in spec.categorical if c in df.columns] + ["grade"]),
        "outliers": eda.outlier_table(df, numeric),
    }
    corr = eda.correlation(df, numeric)
    tables["correlation_matrix"] = corr.reset_index(names="feature")
    tables["high_correlations"] = eda.high_correlations(corr)

    segments: dict[str, pd.DataFrame] = {}
    for col in SEGMENTS:
        table = eda.target_rates(df, col)
        if not table.empty:
            segments[col] = table
            tables[f"default_rate_{col}"] = table
    vint = eda.vintage_rates(df)
    if not vint.empty:
        segments["vintage"] = vint
        tables["default_rate_vintage"] = vint

    findings = eda.notable_findings(
        missing=tables["missing"], outliers=tables["outliers"],
        correlations=tables["high_correlations"], segments=segments,
        overview_stats=stats)

    for name, table in tables.items():
        write_table(table, out_dir / f"{name}.csv")
    write_json({"overview": stats, "findings": findings}, out_dir / "overview.json")

    # ------------------------------------------------------------- charts ----
    charts: list[Path] = []
    charts.append(figs.plot_missing_share(tables["missing"], out_dir / "missing_share.png"))
    charts.append(figs.plot_numeric_distributions(
        df, numeric[:12], out_dir / "numeric_distributions.png"))
    charts.append(figs.plot_correlation_heatmap(corr, out_dir / "correlation_heatmap.png"))
    for col in ("grade", "term_months", "purpose"):
        if col in tables.get(f"default_rate_{col}", pd.DataFrame()).columns:
            charts.append(figs.plot_default_rate_bars(
                tables[f"default_rate_{col}"], col, out_dir / f"default_rate_{col}.png"))
    if {"duration_months", "event"} <= set(df.columns):
        for col in ("grade", "term_months"):
            if col in df.columns:
                charts.append(figs.plot_km_by_group(
                    df["duration_months"].to_numpy(), df["event"].to_numpy(),
                    df[col].astype(str), out_dir / f"km_by_{col}.png",
                    title=f"Kaplan-Meier survival by {col}"))
    charts = [c for c in charts if c is not None]

    diag_tag = args.diagnostic_tag or args.tag
    diag_path = cfg.paths.tables_dir / f"04_reject_inference_{diag_tag}.json"
    diagnostic = None
    if diag_path.exists():
        diagnostic = json.loads(diag_path.read_text(encoding="utf-8"))
        print(f"  accepted-vs-rejected numbers read from {diag_path.name}")
    else:
        print(f"  note: {diag_path.name} not found; the accepted-vs-rejected section "
              f"is omitted rather than computed a second way")

    report = _html(out_dir, args.tag, stats, findings, tables, charts, diagnostic)

    stamp = build_stamp(stage="01b_eda", inputs={"data": src},
                        outputs={"report": report},
                        config_path=args.config,
                        args={"tag": args.tag, "sample": args.sample,
                              "diagnostic_tag": diag_tag})
    write_json({"provenance": stamp, "overview": stats, "findings": findings},
               out_dir / "provenance.json")

    print(f"\nEDA report written: {report}")
    print(f"  {len(tables)} tables, {len(charts)} charts")
    print("\nObservations:")
    for line in findings:
        print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
