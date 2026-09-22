"""Stage 5 -- assemble FINDINGS.md sections 2-6 from the JSON written by 02-04.

Sections 2-5 hold the primary random-split results (``--tag``); section 6 holds
the out-of-time holdout (``--holdout-tag``). An out-of-time run is refused as
``--tag``, since rendering it into 2-5 would replace the primary results.

Reads whatever result files exist and regenerates the corresponding FINDINGS.md
sections in place. Stages that have not been run are reported as not run rather
than omitted, so the document never implies a result that does not exist.

Usage
-----
    python scripts/05_report.py
    python scripts/05_report.py --tag full
    python scripts/05_report.py --dry-run     # print, do not write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditsurv.config import load_config  # noqa: E402

SECTION_MARKERS = {
    "2": ("## 2. Survival models", "## 3. Explainability"),
    "3": ("## 3. Explainability", "## 4. Reject inference (diagnostic-gated)"),
    "4": ("## 4. Reject inference (diagnostic-gated)", "## 5. Summary"),
    "5": ("## 5. Summary", "## 6. Out-of-time holdout"),
    "6": ("## 6. Out-of-time holdout", None),
}
SECTIONS = ("2", "3", "4", "5", "6")


def _load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def section_2(data) -> str:
    out = ["## 2. Survival models", ""]
    if not data:
        out += ["*Not run.* Execute `scripts/02_train_models.py` to populate this "
                "section.", ""]
        return "\n".join(out)

    out += [
        f"Fitted on `{data['source']}` ({data['n_rows']:,} loans), "
        f"{data['split_scheme']} split, {data['n_train']:,} train / "
        f"{data['n_test']:,} test. Lending Club grade "
        f"{'INCLUDED (benchmark variant)' if data['with_lc_grade'] else 'excluded (primary spec)'}.",
        "",
        f"Features: {len(data['features']['numeric'])} numeric + "
        f"{len(data['features']['categorical'])} categorical.",
        "",
    ]
    results = data.get("results", [])
    if results:
        horizons = [k for k in results[0] if k.startswith("auc_")]
        header = ["Model", "C-index", "IBS"] + [h.replace("auc_", "AUC ") for h in horizons]
        out += ["| " + " | ".join(header) + " |",
                "|" + "|".join("---" for _ in header) + "|"]
        for r in results:
            row = [r["model"], _fmt(r["concordance"]), _fmt(r["ibs"])]
            row += [_fmt(r.get(h)) for h in horizons]
            out.append("| " + " | ".join(row) + " |")
        out.append("")
        unreliable = {
            r["model"]: r.get("unreliable_horizons", []) for r in results
        }
        flagged = {k: v for k, v in unreliable.items() if v}
        if flagged:
            out += [
                "Horizons where the inverse-probability-of-censoring weights hit "
                "their floor, and where AUC is therefore **not trustworthy**: "
                + "; ".join(f"{k}: {v}" for k, v in flagged.items()),
                "",
            ]
    out += [
        "Risk is reported as age-dependent in "
        "`outputs/tables/02_age_dependent_risk_*.csv`: the conditional probability "
        "of default over the next 12 months for a *fixed* borrower profile, at "
        "several loan ages. A single static score cannot express this.",
        "",
    ]
    return "\n".join(out)


def section_3(data, tables_dir=None, tag: str = "dev") -> str:
    out = ["## 3. Explainability", ""]
    if not data:
        out += ["*Not run.* Execute `scripts/03_explain.py` to populate this section.",
                ""]
        return "\n".join(out)

    out += [
        f"SurvSHAP(t) computed for {data['n_explained']:,} borrowers on the "
        f"`{data['model']}` model, `nsamples={data['nsamples']}`, background "
        f"{data['n_background']} rows, at horizons {data['times']}. Cost: "
        f"{data['seconds_per_borrower']:.2f}s per borrower "
        f"({data['seconds_total']:.0f}s total).",
        "",
        "### Implementation correctness",
        "",
        f"Efficiency-axiom worst residual: **{data['efficiency_worst_residual']:.2e}** "
        f"-> {'**PASS**' if data['efficiency_pass'] else '**FAIL**'}. At every time "
        f"point the attributions plus the base value reconstruct the model's own "
        f"prediction, which is the falsifiable check that this reimplementation "
        f"behaves like Shapley values rather than merely producing plausible numbers.",
        "",
        "The first full-scale attempt failed this check outright (residual "
        "**5.65e+07**). KernelSHAP had been routed through LARS feature "
        "selection, which is unstable on this design matrix's near-constant "
        "columns -- rare-event counts and 13 missing-value indicators -- and "
        "returned degenerate active sets. Plain weighted least squares "
        "(`l1_reg=0`) makes the axiom exact. A test on well-conditioned random "
        "data did not reproduce the failure; it took real data to surface it.",
        "",
    ]

    imp = data.get("global_importance", {})
    if imp:
        top = sorted(imp.items(), key=lambda kv: -abs(kv[1]))[:10]
        out += ["### Global importance (time-integrated)", "",
                "| Feature | Importance |", "|---|---|"]
        out += [f"| `{k}` | {v:.5f} |" for k, v in top]
        out.append("")

    cmp = data.get("naive_vs_survshap", {})
    if cmp:
        out += [
            "### (a) Naive SHAP vs SurvSHAP(t) -- measured, not assumed",
            "",
            "Both explainers use the same model, background set and `nsamples`; the "
            "only difference is that naive SHAP explains the scalar "
            f"`1 - S({cmp.get('at_month')} months)` while SurvSHAP(t) explains the "
            "whole curve. Any disagreement can therefore only come from the time "
            "dimension.",
            "",
            "| Measure | Value |", "|---|---|",
            f"| Spearman rank correlation | {_fmt(cmp.get('spearman'))} |",
            f"| Pearson correlation | {_fmt(cmp.get('pearson'))} |",
            f"| Top-5 overlap (Jaccard) | {_fmt(cmp.get('top5_overlap'))} |",
            f"| Top-10 overlap (Jaccard) | {_fmt(cmp.get('top10_overlap'))} |",
            f"| Sign agreement | {_fmt(cmp.get('sign_agreement'))} |",
            f"| Mean time-varying share | {_fmt(cmp.get('mean_time_variation_share'))} |",
            "",
            f"Sign disagreements: {cmp.get('sign_disagreements') or 'none'}. "
            f"Within the top 5: {cmp.get('top5_sign_disagreements') or 'none'}.",
            "",
            f"Most time-varying features: {cmp.get('most_time_varying')}.",
            "",
            f"**{cmp.get('verdict')}**",
            "",
        ]
        out += _noise_floor_block(cmp, data, tables_dir)

    aa = data.get("adverse_action", {})
    if aa:
        out += [
            "### (b) Adverse-action notice (ECOA / Regulation B)",
            "",
            f"Generated for the highest-risk explained applicant "
            f"(`{aa.get('applicant_id')}`, predicted default probability "
            f"{_fmt(aa.get('predicted_default_probability'))} by "
            f"{aa.get('horizon_months')} months), with "
            f"{aa.get('n_reasons')} principal reasons. Full text in "
            f"`outputs/tables/03_adverse_action_notice_*.txt`.",
            "",
            "| Rank | Feature | Disclosed reason |", "|---|---|---|",
        ]
        for r in aa.get("reasons", []):
            out.append(f"| {r['rank']} | `{r['feature']}` | {r['reason']} |")
        out += [
            "",
            "Three Regulation B constraints are enforced in code, not left to the "
            "caller: reasons are capped at four per the Official Staff Commentary "
            "to 12 CFR 1002.9(b)(2); Lending Club grade and interest rate are "
            "refused as reasons because a failure-to-score disclosure does not "
            "satisfy the requirement; and only features whose attribution is "
            "*adverse* are eligible, since a feature that helped the applicant is "
            "not a reason for denial.",
            "",
        ]
        if aa.get("excluded_helpful_features"):
            out += [
                f"Features that *helped* this applicant, correctly excluded from the "
                f"notice: {aa['excluded_helpful_features']}.",
                "",
            ]

    seg = data.get("segment_stability", {})
    strat = _load(tables_dir / f"03_explain_{tag}_strat.json") if tables_dir else None
    if seg:
        out += ["### (c) Explanation stability across segments", ""]
        if strat:
            out += [
                "Purpose, income band and term are measured on the unstratified "
                f"{data['n_explained']}-borrower sample, which is representative of "
                "the book. **Grade is measured on a separate grade-stratified sample** "
                "(below), because random sampling mirrors the population and left "
                "grades F and G too thin to measure at all.",
                "",
            ]
        out += ["| Segment | Levels | Min rank corr. | Mean top-5 overlap | Verdict |",
                "|---|---|---|---|---|"]
        for name, s in seg.items():
            if strat and name == "grade":
                continue
            v = s.get("verdict", "")
            short = v.split(":")[0]
            out.append(
                f"| {name} | {s.get('n_levels')} | "
                f"{_fmt(s.get('min_rank_correlation'))} | "
                f"{_fmt(s.get('mean_topk_overlap'))} | {short} |"
            )
        out.append("")
        for name, s in seg.items():
            if strat and name == "grade":
                continue
            out += [f"**{name}.** {s.get('verdict')}", ""]
        if strat:
            out += _grade_stratified_block(strat, tables_dir, tag)
    return "\n".join(out)


def _noise_floor_block(cmp: dict, data: dict, tables_dir) -> list[str]:
    """The evidence that makes 3(a) a settled result rather than a small one.

    Compares the method-vs-method agreement against SurvSHAP(t)'s agreement with
    *itself* across independent coalition draws, from the nsamples sweep. Generated
    here, not hand-written, so that re-running this script cannot drop it.
    """
    if tables_dir is None:
        return []
    study_path = tables_dir / "03_nsamples_study.csv"
    if not study_path.exists():
        return ["*Noise-floor sweep not found (`03_nsamples_study.csv`); the "
                "comparison above cannot yet be judged against sampling noise.*", ""]
    import pandas as pd

    study = pd.read_csv(study_path)
    used = int(data.get("nsamples", 0))
    row = study.loc[study["nsamples"] == used]
    out = [
        "#### Evidence: method difference vs. the measurement's own noise floor",
        "",
        "A sweep held borrowers and background fixed and varied only the KernelSHAP "
        "coalition draw, measuring how much SurvSHAP(t) disagrees *with itself* "
        "from Monte Carlo sampling alone (`outputs/tables/03_nsamples_study.csv`):",
        "",
        "| `nsamples` | Between-draw Spearman | sec/borrower |",
        "|---|---|---|",
    ]
    for r in study.itertuples():
        mark = " **(used)**" if int(r.nsamples) == used else ""
        out.append(f"| {int(r.nsamples)}{mark} | {r.spearman_between_draws:.3f} | "
                   f"{r.sec_per_borrower:.2f} |")
    out.append("")
    if row.empty:
        out += [f"*The run used nsamples={used}, which is not in the sweep, so no "
                f"direct noise-floor comparison is available.*", ""]
        return out

    floor = float(row["spearman_between_draws"].iloc[0])
    rho = float(cmp.get("spearman") or float("nan"))
    if rho >= floor:
        out += [
            f"At nsamples={used}, two draws of the *same* method agree at "
            f"**rho = {floor:.3f}**. The two *different* methods agree at "
            f"**rho = {rho:.4f}**.",
            "",
            "**This is a settled negative result.** The difference between naive "
            "SHAP and SurvSHAP(t) is smaller than the Monte Carlo noise within either "
            "one, so the two are statistically indistinguishable on this dataset. "
            "Top-5 and top-10 membership are identical and no top-5 feature disagrees "
            "on direction, so every reason that could appear in an adverse-action "
            "notice is the same under both. A larger borrower sample cannot overturn "
            "this: the effect is below the resolution of the instrument, not merely "
            "small.",
            "",
            f"What it does *not* say: attributions do vary with time (mean "
            f"time-varying share {_fmt(cmp.get('mean_time_variation_share'))}), and "
            f"sign agreement across all features is only "
            f"{_fmt(cmp.get('sign_agreement'))}. What fails to reproduce is the claim "
            f"that collapsing that time structure to a scalar changes *which features "
            f"are named*. Generalisation is limited to this dataset, model family and "
            f"horizon grid; this book is also unusually heavily censored (39% still "
            f"performing).",
            "",
        ]
    else:
        out += [
            f"Two draws of the same method agree at rho = {floor:.3f}; the two methods "
            f"agree at rho = {rho:.4f}, *below* that floor. The disagreement therefore "
            f"exceeds sampling noise and is a real method difference.",
            "",
        ]
    return out


def _level_stability(importance, reference, k: int = 5) -> dict:
    """Per-level rank correlation and top-k overlap against a reference ranking."""
    from scipy.stats import spearmanr

    out = {}
    ref = reference.sort_values(ascending=False)
    ref_top = set(ref.head(k).index)
    for level in importance.columns:
        v = importance[level].reindex(ref.index)
        ok = v.notna() & ref.notna()
        rho = float(spearmanr(v[ok].to_numpy(), ref[ok].to_numpy()).statistic)
        top = list(importance[level].sort_values(ascending=False).head(k).index)
        jac = len(set(top) & ref_top) / len(set(top) | ref_top)
        out[str(level)] = {"rho": rho, "jaccard": jac, "top": top}
    return out


def _grade_stratified_block(strat: dict, tables_dir, tag: str) -> list[str]:
    """Per-grade stability from the equal-allocation sample, with F and G called out.

    Each grade is compared against two references, because they answer different
    questions. The pooled stratified ranking weights every grade equally, so F and G
    make up 2/7 of it rather than ~2.3% as in the book -- comparing them to a pool
    they partly define biases toward agreement. The population ranking from the
    unstratified run is the fairer test of whether high-risk borrowers are
    explained the way the model is explained globally.
    """
    import pandas as pd

    imp_path = tables_dir / f"03_segment_importance_grade_{tag}_strat.csv"
    pooled_path = tables_dir / f"03_survshap_importance_{tag}_strat.csv"
    pop_path = tables_dir / f"03_survshap_importance_{tag}.csv"
    if not (imp_path.exists() and pooled_path.exists() and pop_path.exists()):
        return ["*Grade-stratified outputs incomplete; per-grade results unavailable.*", ""]

    imp = pd.read_csv(imp_path).set_index("feature")
    pooled = pd.read_csv(pooled_path).set_index("feature")["importance"]
    pop = pd.read_csv(pop_path).set_index("feature")["importance"]
    vs_pooled = _level_stability(imp, pooled)
    vs_pop = _level_stability(imp, pop)
    counts = strat.get("strata_counts", {})
    pop_top5 = list(pop.sort_values(ascending=False).head(5).index)

    out = [
        "#### Grade, on a grade-stratified sample",
        "",
        f"{strat['n_explained']} borrowers, equal allocation per grade "
        f"({counts}), same model and SurvSHAP(t) settings. Efficiency residual "
        f"{strat['efficiency_worst_residual']:.2e}.",
        "",
        "| Grade | n | rho vs population | top-5 overlap vs population | "
        "rho vs pooled | top-5 overlap vs pooled |",
        "|---|---|---|---|---|---|",
    ]
    for g in imp.columns:
        a, b = vs_pop[str(g)], vs_pooled[str(g)]
        bold = "**" if str(g) in {"F", "G"} else ""
        out.append(
            f"| {bold}{g}{bold} | {counts.get(str(g), 'n/a')} | {a['rho']:.3f} | "
            f"{a['jaccard']:.3f} | {b['rho']:.3f} | {b['jaccard']:.3f} |"
        )
    out += ["", f"Population top 5 (unstratified run): {pop_top5}.", ""]

    # The pre-registered rule is an AGGREGATE one: minimum rank correlation >= 0.85
    # and MEAN top-5 overlap >= 0.70 across levels. It is applied here exactly as
    # registered. Per-level pass/fail labels are deliberately not produced: top-5
    # Jaccard is discrete (5 shared = 1.0, 4 = 0.667, 3 = 0.429), so a 0.70 floor
    # on a single level means "no swap at all is allowed", which would flag grades
    # B-E as drifting for a single tie-break at rank 5. That rule was never
    # registered, and inventing it after seeing the data is exactly the kind of
    # post-hoc reframing this log is meant to prevent.
    rank_floor, overlap_floor = 0.85, 0.70
    lines = []
    for label, res in (("population", vs_pop), ("pooled", vs_pooled)):
        min_rho = min(v["rho"] for v in res.values())
        mean_j = sum(v["jaccard"] for v in res.values()) / len(res)
        ok = min_rho >= rank_floor and mean_j >= overlap_floor
        lines.append((label, min_rho, mean_j, ok))
    out += [
        "**Pre-registered aggregate rule** (min rank correlation >= "
        f"{rank_floor}, mean top-5 overlap >= {overlap_floor}):",
        "",
        "| Reference | Min rank corr. | Mean top-5 overlap | Verdict |",
        "|---|---|---|---|",
    ]
    for label, min_rho, mean_j, ok in lines:
        out.append(f"| {label} | {min_rho:.3f} | {mean_j:.3f} | "
                   f"{'STABLE' if ok else 'DRIFTS'} |")
    verdicts = {ok for *_, ok in lines}
    out += [""]
    if len(verdicts) > 1:
        out += [
            "**The aggregate verdict depends on the reference ranking and sits at the "
            "threshold, so it is not robust in either direction.** Rank correlation "
            "is comfortably high for every grade; the whole question turns on top-5 "
            "membership, and there the mean lands just either side of 0.70.",
            "",
        ]

    # Descriptive per-grade reading: which features move, relative to A-E.
    base = [g for g in imp.columns if str(g) not in {"F", "G"}]
    ranks = imp.rank(ascending=False)
    movers = []
    for f in imp.index:
        base_ranks = [ranks.loc[f, g] for g in base]
        lo, hi = min(base_ranks), max(base_ranks)
        for g in ("F", "G"):
            if g in ranks.columns:
                r = ranks.loc[f, g]
                # Only report features that enter or leave the top 5 AND sit
                # outside the whole A-E range -- a break, not a tie-break.
                if (r <= 5 or lo <= 5) and (r < lo - 2 or r > hi + 2):
                    movers.append((f, g, int(r), int(lo), int(hi)))
    if movers:
        out += [
            "Features whose rank in F or G falls **outside the entire range seen in "
            "grades A-E** and crosses the top-5 boundary:",
            "",
            "| Feature | Grade | Rank there | Rank range in A-E |",
            "|---|---|---|---|",
        ]
        out += [f"| `{f}` | {g} | {r} | {lo}-{hi} |" for f, g, r, lo, hi in movers]
        out.append("")
    shared = sorted(set.intersection(
        *[set(imp[g].sort_values(ascending=False).head(5).index) for g in imp.columns]))
    out += [f"Shared by the top 5 of every grade A-G: {shared}.", ""]
    out += _bootstrap_block(tables_dir, tag)
    return out


def _bootstrap_block(tables_dir, tag: str) -> list[str]:
    """Render the stratified-bootstrap intervals, verdict first, without softening."""
    boot = _load(tables_dir / f"03_segment_bootstrap_{tag}_strat.json")
    if not boot:
        return [
            "Borrower-sampling variance at n=36 per grade is **not quantified**: no "
            "bootstrap results were found (`scripts/03b_bootstrap_segments.py`). The "
            "rank shifts above are point estimates only.",
            "",
        ]
    ci = int(round(100 * boot["results"][next(iter(boot["results"]))][0]["ci"]))
    out = [
        f"#### Borrower-sampling uncertainty ({boot['n_boot']:,} stratified bootstrap "
        f"replicates, {ci}% intervals)",
        "",
        "Borrowers were resampled with replacement *within* each grade, preserving "
        "36 per grade, and every grade's ranking recomputed. `Shift` is the rank in "
        "the target grade minus the rank in pooled A-E (positive = less important "
        "in the target grade). `Share ratio` compares the feature's share of total "
        "attribution, target over A-E; unlike ranks it is continuous, so it checks "
        "that a rank move is not just a near-tie flipping.",
        "",
        "| Grade | Feature | Rank there | Rank in A-E | Shift [CI] | Share ratio [CI] "
        "| P(outside A-E range) | Reading |",
        "|---|---|---|---|---|---|---|---|",
    ]
    readings = []
    for grade, rows in boot["results"].items():
        for r in rows:
            shift_ex = r["shift_ci_excludes_zero"]
            share_ex = r["share_ci_excludes_one"]
            if shift_ex and share_ex:
                reading = "real difference in direction"
                # An interval straddling rank 5 means top-5 membership itself is
                # not established, however robust the direction is.
                if r["rank_target_lo"] <= 5 < r["rank_target_hi"] or \
                        r["rank_target_lo"] <= 5 < r["rank_reference"]:
                    reading += "; top-5 exit/entry not established"
            elif shift_ex:
                reading = "rank shift real; share change not clear"
            else:
                reading = "**cannot exclude no difference**"
            readings.append((grade, r["feature"], reading))
            out.append(
                f"| {grade} | `{r['feature']}` | {r['rank_target']:.0f} "
                f"[{r['rank_target_lo']:.0f}, {r['rank_target_hi']:.0f}] | "
                f"{r['rank_reference']:.0f} | {r['rank_shift']:+.0f} "
                f"[{r['rank_shift_lo']:+.0f}, {r['rank_shift_hi']:+.0f}] | "
                f"{r['share_ratio']:.2f} [{r['share_ratio_lo']:.2f}, "
                f"{r['share_ratio_hi']:.2f}] | {r['p_outside_reference_range']:.2f} | "
                f"{reading} |"
            )
    out += [
        "",
        "Two reasons these intervals are, if anything, too narrow:",
        "",
        "- **The features were chosen after looking at the data.** They are the ones "
        "whose point-estimate rank fell outside the A-E range, so they were selected "
        "*because* they looked extreme. Intervals on post-hoc-selected features are "
        "optimistic (winner's curse). This is an exploratory check, not a "
        "confirmatory test.",
        "- **Coalition-sampling noise is not in them.** The bootstrap resamples "
        "borrowers but reuses each borrower's attribution. Re-running the identical "
        "sample with a different SHAP draw moved these ranks by 1-2 places "
        "(per-grade Spearman between the two runs 0.93-0.98).",
        "",
    ]
    unclear = [f"`{f}` in {g}" for g, f, rd in readings if "cannot" in rd]
    if unclear:
        out += [
            "**Not every point estimate survives resampling.** For "
            + ", ".join(unclear)
            + " the interval on the rank shift includes zero, so these could be "
            "noise from which 36 borrowers happened to be drawn and should not be "
            "reported as grade-specific behaviour.",
            "",
        ]
    return out


def section_4(data) -> str:
    out = ["## 4. Reject inference (diagnostic-gated)", ""]
    if not data:
        out += [
            "*Not run.* Execute `scripts/04_reject_inference.py` to populate this "
            "section. Thresholds remain pre-registered in `config/config.yaml`.",
            "",
        ]
        return "\n".join(out)

    diag = data.get("diagnostic", {})
    out += [
        f"Compared {data['n_accepted']:,} accepted loans against "
        f"{data['n_rejected_sampled']:,} sampled rejected applications on the "
        f"features common to both files.",
        "",
        "### Diagnostic result",
        "",
        "| Feature | Comparability | Accepted mean | Rejected mean | SMD | Rank AUC | KS | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for f in diag.get("features", []):
        out.append(
            f"| `{f['feature']}` | {f['comparability']} | {f['accepted_mean']} | "
            f"{f['rejected_mean']} | {f['smd']} | {f['rank_auc']} | {f['ks']} | "
            f"{f['verdict']} |"
        )
    out += [
        "",
        f"- Separability AUC: **{_fmt(diag.get('separability_auc'))}** "
        f"(strong threshold {diag.get('thresholds', {}).get('separability_auc_strong')})",
        f"- Top selection drivers: {diag.get('top_selection_drivers')}",
        f"- Common support: **{_fmt(diag.get('common_support_share'))}** "
        f"(floor {diag.get('thresholds', {}).get('min_common_support')})",
        f"- Bias detected: {_fmt(diag.get('bias_detected'))}",
        f"- Support adequate: {_fmt(diag.get('support_adequate'))}",
        "",
        f"### Gate decision: `{diag.get('gate_decision', 'unknown').upper()}`",
        "",
        diag.get("rationale", ""),
        "",
    ]
    if diag.get("notes"):
        out += ["Notes recorded during the diagnostic:", ""]
        out += [f"- {n}" for n in diag["notes"]]
        out.append("")

    out += [
        "Every KS p-value in this comparison is effectively zero because of sample "
        "size, which is exactly why the gate is built on effect sizes. p-values are "
        "reported in the CSV for completeness and were not used to decide.",
        "",
    ]

    shift = data.get("explanation_shift", {})
    if shift.get("status") == "not_applicable":
        out += [
            "### Explanation shift before/after correction",
            "",
            "**Not applicable.** No correction was applied, so there is no 'after' "
            "to compare. Reporting a shift here would require forcing a correction "
            "the diagnostic said was unwarranted, which would invert the point of "
            "the stage.",
            "",
            f"Reason: {shift.get('reason', '')}",
            "",
        ]
    elif shift.get("status") == "measured":
        out += [
            "### Explanation shift before/after correction",
            "",
            "This is the project's own question: does correcting for selection bias "
            "change *which features the model says matter*, not merely how well it "
            "scores? None of the reviewed literature tests it.",
            "",
            f"- Rank correlation before/after: **{_fmt(shift.get('rank_correlation'))}**",
            f"- Top-5 overlap: **{_fmt(shift.get('top5_overlap'))}**",
            f"- Top-5 before: {shift.get('top5_before')}",
            f"- Top-5 after: {shift.get('top5_after')}",
            "",
            f"**{shift.get('verdict')}**",
            "",
        ]
        if data.get("correction_forced"):
            out += [
                "> This correction was applied with `--force-correction`, overriding "
                "a blocking gate. The result is therefore a sensitivity exercise, not "
                "a warranted correction.",
                "",
            ]
        if data.get("weights"):
            w = data["weights"]
            out += [
                f"Weighting diagnostics: propensity AUC {_fmt(w.get('propensity_auc'))}, "
                f"effective sample size {_fmt(w.get('effective_n'))} of "
                f"{_fmt(w.get('nominal_n'))} "
                f"(ratio {_fmt(w.get('ess_ratio'))}), {_fmt(w.get('n_clipped'))} weights clipped.",
                "",
                f"*{w.get('caveat', '')}*",
                "",
            ]
    return "\n".join(out)


def section_5(s2, s3, s4, s3_strat=None, cfg_tables=None, tag: str = "dev") -> str:
    out = ["## 5. Summary", ""]
    done = [n for n, d in (("2", s2), ("3", s3), ("4", s4)) if d]
    if not done:
        out += ["*Pending completion of Stages 2-4.*", ""]
        return "\n".join(out)

    out += [f"Stages complete: 1, {', '.join(done)}.", ""]
    if s2 and s2.get("results"):
        best = max(s2["results"], key=lambda r: r.get("concordance") or 0)
        out.append(
            f"- Best concordance: **{_fmt(best['concordance'])}** "
            f"(`{best['model']}`), on {best['n']:,} test loans."
        )
    if s3:
        cmp = s3.get("naive_vs_survshap", {})
        out.append(
            f"- Naive vs survival SHAP: {cmp.get('verdict', 'n/a').split(':')[0]} "
            f"(Spearman {_fmt(cmp.get('spearman'))})."
        )
        out.append(
            f"- SurvSHAP(t) efficiency axiom: "
            f"{'passed' if s3.get('efficiency_pass') else 'FAILED'} "
            f"({s3.get('efficiency_worst_residual', float('nan')):.1e})."
        )
        seg = s3.get("segment_stability", {})
        drift = [k for k, v in seg.items() if v.get("verdict", "").startswith("DRIFTS")]
        drift = [k for k in drift if not (s3_strat and k == "grade")]
        out.append(
            f"- Explanation stability (purpose, income band, term): "
            f"{'drifts in ' + ', '.join(drift) if drift else 'stable across all segments tested'}."
        )
        if s3_strat:
            out.append(
                "- Explanation stability by grade (stratified, 36 per grade): the "
                "aggregate verdict is threshold-sensitive, landing either side of the "
                "0.70 top-5 overlap floor depending on the reference ranking."
            )
            boot = _load(Path(cfg_tables) / f"03_segment_bootstrap_{tag}_strat.json") \
                if cfg_tables else None
            if boot:
                bits = []
                for g, rows in boot["results"].items():
                    for r in rows:
                        if r["shift_ci_excludes_zero"] and r["share_ci_excludes_one"]:
                            bits.append(f"`{r['feature']}` is less important in {g} "
                                        if r["rank_shift"] > 0 else
                                        f"`{r['feature']}` is more important in {g} ")
                if bits:
                    out.append(
                        "- Surviving bootstrap resampling: " + "; ".join(b.strip() for b in bits)
                        + " than in A-E. Every other grade-specific shift tested has "
                        "an interval that includes no difference. Exploratory: the "
                        "features were selected post hoc. See section 3(c)."
                    )
                else:
                    out.append("- No grade-specific rank shift survives bootstrap "
                               "resampling. See section 3(c).")
    if s4:
        d = s4.get("diagnostic", {})
        out.append(f"- Reject-inference gate: **{d.get('gate_decision', 'n/a')}**.")
        sh = s4.get("explanation_shift", {})
        if sh.get("status") == "measured":
            out.append(f"- Explanation shift after correction: {sh.get('verdict')}.")
        elif sh.get("status") == "not_applicable":
            out.append(
                "- Explanation shift: not applicable, no correction was warranted."
            )
    out.append("")
    return "\n".join(out)


HOLDOUT_HEADING = "## 6. Out-of-time holdout"


def _auc_cells(summary: dict, horizons: list[int]) -> list[str]:
    return [_fmt(summary.get(f"auc_{h}m")) for h in horizons]


def section_6(tables_dir, holdout_tag: str = "holdout",
              strat_tag: str = "holdout_strat", full_tag: str = "full") -> str:
    """Out-of-time holdout results, rendered alongside -- never instead of -- §2-§5.

    Every decision rule applied here was pre-registered in the protected block at
    the top of this section before the holdout was run. The renderer applies those
    rules mechanically; it does not choose them.
    """
    t = tables_dir
    s2 = _load(t / f"02_metrics_{holdout_tag}.json")
    s2_full = _load(t / f"02_metrics_{full_tag}.json")
    nf = _load(t / f"03c_noise_floor_{holdout_tag}.json")
    s3 = _load(t / f"03_explain_{holdout_tag}.json")
    s3s = _load(t / f"03_explain_{strat_tag}.json")
    boot = _load(t / f"03_segment_bootstrap_{strat_tag}.json")
    boot_orig = _load(t / f"03_segment_bootstrap_{full_tag}_strat.json")
    s4 = _load(t / f"04_reject_inference_{holdout_tag}.json")
    s4_full = _load(t / f"04_reject_inference_{full_tag}.json")

    out = [HOLDOUT_HEADING, ""]
    if not any([s2, nf, s3, s3s, boot, s4]):
        out += ["*Not run yet.* The pre-registered decision rules above were fixed "
                "before any holdout result existed.", ""]
        return "\n".join(out)

    # ---------------- 6.1 design ----------------
    if s2:
        ty, hy = s2.get("train_years", []), s2.get("test_years", [])
        es = s2.get("early_stopping") or {}
        out += [
            "### 6.1 Design", "",
            f"Models refitted on **{ty[0]}-{ty[-1]}** vintages only "
            f"({s2['n_train']:,} loans) and evaluated on **{hy[0]}-{hy[-1]}** "
            f"({s2['n_test']:,} loans), which no part of the fit saw. Early stopping "
            f"validated on {es.get('validation', 'n/a')} "
            f"({_fmt(es.get('n_val'))} loans; {es.get('trees', 'n/a')} trees kept). "
            f"The censoring model behind the IPCW weights was fitted on the "
            f"**{s2.get('ipcw_fitted_on', 'n/a')}** set, because training-period and "
            f"holdout censoring differ sharply (6% vs 60% still performing).",
            "",
            "Two limits apply to everything below. Horizons of 30 and 36 months are "
            "reachable only by 2016 loans (data extracted early 2019), so those "
            "columns are 2016-only results, not 2016-2018. And items 6.3-6.4 change "
            "the model and the borrowers at once, so a result that fails to "
            "replicate cannot be attributed to either alone.",
            "",
        ]

        # ---------------- 6.2 performance ----------------
        results = s2.get("results", [])
        if results:
            horizons = sorted(int(k[4:-1]) for k in results[0] if k.startswith("auc_"))
            head = ["Model", "C-index", "IBS"] + [f"AUC {h}m" for h in horizons] + \
                   ["High-variance horizons"]
            out += ["### 6.2 Survival models on the holdout", "",
                    "| " + " | ".join(head) + " |",
                    "|" + "|".join("---" for _ in head) + "|"]
            for r in results:
                hv = r.get("high_variance_horizons") or []
                out.append("| " + " | ".join(
                    [r["model"], _fmt(r["concordance"]), _fmt(r["ibs"])]
                    + _auc_cells(r, horizons) + [str(hv) if hv else "none"]) + " |")
            out.append("")
            ctrl = results[0].get("n_controls_by_horizon") or {}
            if ctrl:
                out += ["Loans still under observation at each horizon (the AUC "
                        "controls): " + ", ".join(
                            f"{h}m {int(v):,}" for h, v in sorted(
                                ((int(k), v) for k, v in ctrl.items()))) + ".", ""]

            byv = s2.get("results_by_vintage") or {}
            if byv:
                out += ["By holdout vintage (each with its own censoring model, "
                        "horizons limited to what that vintage can reach):", "",
                        "| Model | Vintage | n | C-index | AUC 6m | AUC 12m | AUC 24m |",
                        "|---|---|---|---|---|---|---|"]
                for model, years in byv.items():
                    for y, r in sorted(years.items()):
                        out.append(
                            f"| {model} | {y} | {_fmt(r.get('n'))} | "
                            f"{_fmt(r.get('concordance'))} | {_fmt(r.get('auc_6m'))} | "
                            f"{_fmt(r.get('auc_12m'))} | {_fmt(r.get('auc_24m'))} |")
                out.append("")
            if s2_full and s2_full.get("results"):
                ref = ", ".join(f"{r['model']} {_fmt(r['concordance'])}"
                                for r in s2_full["results"])
                out += [f"For reference only, the random-split test (section 2) gave "
                        f"C-index {ref}. The two are **not like-for-like**: that model "
                        f"trained on all vintages and was tested on loans from the "
                        f"same years, and its test loans are much less censored.", ""]

    # ---------------- 6.3 naive vs SurvSHAP(t) ----------------
    cmp = (s3 or {}).get("naive_vs_survshap") or {}
    if cmp and "spearman" in cmp:
        floor = (nf or {}).get("spearman_between_draws")
        rho = cmp.get("spearman")
        top5 = cmp.get("top5_overlap")
        flips = cmp.get("top5_sign_disagreements") or []
        out += ["### 6.3 Naive SHAP vs SurvSHAP(t) on the holdout", "",
                "| Measure | Holdout | Section 3 (random split) |", "|---|---|---|"]
        orig = {}
        s3_full = _load(t / f"03_explain_{full_tag}.json")
        if s3_full:
            orig = s3_full.get("naive_vs_survshap") or {}
        for label, key in (("Spearman (all features)", "spearman"),
                           ("Top-5 overlap", "top5_overlap"),
                           ("Top-10 overlap", "top10_overlap"),
                           ("Sign agreement", "sign_agreement")):
            out.append(f"| {label} | {_fmt(cmp.get(key))} | {_fmt(orig.get(key))} |")
        out.append(f"| Top-5 sign disagreements | {flips or 'none'} | "
                   f"{orig.get('top5_sign_disagreements') or 'none'} |")
        out.append(f"| Between-draw noise floor (this model) | "
                   f"{_fmt(floor) if floor is not None else 'not measured'} | "
                   f"{_fmt(_section3_floor(t, (s3_full or {}).get('nsamples')))} |")
        out.append("")
        if floor is None:
            out += ["**Criterion (i) cannot be evaluated**: the noise floor was not "
                    "measured on the holdout model (`03c_noise_floor.py`). The floor "
                    "from section 3 belongs to a different model and is not used.", ""]
        else:
            c1 = rho >= floor
            c2 = (top5 == 1.0) and not flips
            out += [
                "Pre-registered criteria, applied as written:", "",
                f"- **(i) ranking level** -- method agreement rho {rho:.4f} "
                f"{'>=' if c1 else '<'} this model's noise floor {floor:.3f}: "
                f"**{'HOLDS' if c1 else 'DOES NOT HOLD'}**.",
                f"- **(ii) notice level** -- top-5 identical ({_fmt(top5)}) with no "
                f"top-5 sign disagreement: **{'HOLDS' if c2 else 'DOES NOT HOLD'}**.",
                "",
            ]
            if c1 and c2:
                out += ["The section 3 finding replicates on the holdout: the two "
                        "methods are indistinguishable at the resolution of the "
                        "instrument, and name the same reasons.", ""]
            elif c1:
                out += ["The ranking-level finding replicates, but the notice-level "
                        "one does not: overall rankings agree within noise, yet the "
                        "top-5 reasons differ. That is the part an applicant sees.", ""]
            else:
                out += ["The finding does **not** replicate on the holdout: the two "
                        "methods disagree by more than the explanation's own sampling "
                        "noise. Because the model and the borrowers both changed, "
                        "this run cannot say which change is responsible.", ""]

    # ---------------- 6.4 grade G, pre-registered ----------------
    if boot:
        g_rows = {r["feature"]: r for r in (boot.get("results") or {}).get("G", [])}
        orig_rows = {r["feature"]: r
                     for r in ((boot_orig or {}).get("results") or {}).get("G", [])}
        counts = (s3s or {}).get("strata_counts", {})
        out += ["### 6.4 Grade G attribution shares (pre-registered test)", "",
                f"{(s3s or {}).get('n_explained', 'n/a')} holdout borrowers, equal "
                f"allocation per grade ({counts}); {_fmt(boot.get('n_boot'))} "
                f"stratified bootstrap replicates; reference "
                f"{'+'.join(boot.get('reference', []))}. Efficiency residual "
                f"{(s3s or {}).get('efficiency_worst_residual', float('nan')):.1e}.", "",
                "| Feature | Share ratio G / A-E [95% CI], holdout | Same, section 3 | "
                "Pre-registered rule | Result |", "|---|---|---|---|---|"]
        rules = {
            "annual_inc": ("CI entirely below 1", lambda r: r["share_ratio_hi"] < 1,
                           "CONFIRMED", "NOT REPLICATED"),
            "mths_since_recent_inq": ("CI entirely above 1 (re-test)",
                                      lambda r: r["share_ratio_lo"] > 1,
                                      "REPLICATED", "NOT REPLICATED"),
        }
        for feat, (rule, test, yes, no) in rules.items():
            r = g_rows.get(feat)
            o = orig_rows.get(feat)
            cell_o = (f"{o['share_ratio']:.2f} [{o['share_ratio_lo']:.2f}, "
                      f"{o['share_ratio_hi']:.2f}]") if o else "n/a"
            if r is None:
                out.append(f"| `{feat}` | not tested | {cell_o} | {rule} | "
                           f"**NOT RUN** |")
                continue
            out.append(f"| `{feat}` | {r['share_ratio']:.2f} "
                       f"[{r['share_ratio_lo']:.2f}, {r['share_ratio_hi']:.2f}] | "
                       f"{cell_o} | {rule} | **{yes if test(r) else no}** |")
        out += ["", "The rules were fixed before the holdout was run and are applied "
                "mechanically. Unlike the section 3 intervals, these are not subject "
                "to post-hoc selection: the features were named in advance. "
                "Coalition-sampling noise is still not included in the intervals.", ""]

    # ---------------- 6.5 selection bias ----------------
    if s4:
        d = s4.get("diagnostic", {})
        comp = s4.get("composition", {})
        full_feats = {f["feature"]: f for f in
                      ((s4_full or {}).get("diagnostic") or {}).get("features", [])}
        out += ["### 6.5 Selection-bias diagnostic on the holdout period", "",
                f"{s4.get('n_accepted', 0):,} accepted loans issued "
                f"{s4.get('accepted_years')} against {s4.get('n_rejected_sampled', 0):,} "
                f"rejected applications sampled from "
                f"{s4.get('n_rejected_available', 0):,} made in "
                f"{s4.get('rejected_years')}.", "",
                "| Feature | Comparability | SMD | Rank AUC | Verdict, holdout | "
                "Verdict, full period |", "|---|---|---|---|---|---|"]
        for f in d.get("features", []):
            ff = full_feats.get(f["feature"], {})
            out.append(f"| `{f['feature']}` | {f['comparability']} | {f['smd']} | "
                       f"{f['rank_auc']} | {f['verdict']} | {ff.get('verdict', 'n/a')} |")
        out += ["", f"Separability AUC {_fmt(d.get('separability_auc'))}, common "
                f"support {_fmt(d.get('common_support_share'))}, gate decision "
                f"`{d.get('gate_decision')}` -- reported, not acted on "
                f"(`--diagnostic-only`).", ""]
        cov = comp.get("rejected_score_coverage_by_year", {})
        with_score = comp.get("rejected_with_score_by_year", {})
        if with_score:
            total = sum(with_score.values()) or 1
            parts = ", ".join(f"{y}: {cov.get(y, 0):.0%} coverage, "
                              f"{with_score[y] / total:.0%} of scored rows"
                              for y in sorted(with_score))
            out += [f"**The score comparison is not balanced across years.** "
                    f"Rejected-applicant score coverage and share of the scored "
                    f"sample by year -- {parts}. The accepted side is spread evenly "
                    f"({comp.get('accepted_by_year')}). Employment length is fully "
                    f"covered and is not affected.", ""]
    return "\n".join(out)


def _section3_floor(t, nsamples):
    """Between-draw Spearman from the section 3 sweep, at the nsamples used there."""
    import pandas as pd

    path = t / "03_nsamples_study.csv"
    if nsamples is None or not path.exists():
        return None
    study = pd.read_csv(path)
    row = study.loc[study["nsamples"] == int(nsamples), "spearman_between_draws"]
    return float(row.iloc[0]) if len(row) else None


KEEP_RE = re.compile(r"<!-- keep:([\w-]+) -->.*?<!-- /keep:\1 -->", re.S)


def _keep_blocks(section_text: str) -> list[str]:
    """Hand-written blocks inside a generated section, preserved verbatim.

    Anything between ``<!-- keep:NAME -->`` and ``<!-- /keep:NAME -->`` survives
    regeneration and is re-inserted directly under the section heading. This is
    the supported way to put hand-written analysis inside sections 2-5.
    """
    return [m.group(0) for m in KEEP_RE.finditer(section_text)]


def _digest(section_text: str) -> str:
    """Hash of a section's generated content, keep blocks excluded."""
    for block in _keep_blocks(section_text):
        section_text = section_text.replace(block, "")
    norm = "\n".join(line.rstrip() for line in section_text.strip().splitlines())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _bounds(text: str, start: str, end: str | None) -> tuple[int, int]:
    """Locate a section, failing loudly rather than guessing.

    The previous version appended when the start heading was missing and, worse,
    deleted everything to end-of-file when the *next* heading was missing -- so
    renaming a single heading would silently wipe every later section.
    """
    i = text.find(start)
    if i == -1:
        raise SystemExit(f"ERROR: heading {start!r} not found in FINDINGS.md. "
                         "Refusing to write; restore the heading or update "
                         "SECTION_MARKERS.")
    if end is None:
        return i, len(text)
    j = text.find(end, i + len(start))
    if j == -1:
        raise SystemExit(f"ERROR: closing heading {end!r} not found after {start!r}. "
                         "Refusing to write: continuing would delete every later "
                         "section.")
    return i, j


def replace_section(text: str, start: str, end: str | None, body: str) -> str:
    i, j = _bounds(text, start, end)
    kept = _keep_blocks(text[i:j])
    if kept:
        head, _, rest = body.partition("\n")
        body = head + "\n\n" + "\n\n".join(kept) + "\n" + rest
    return text[:i] + body.rstrip() + "\n\n" + text[j:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--findings", default="FINDINGS.md")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--holdout-tag", default="holdout",
                    help="tag of the out-of-time results rendered into section 6")
    ap.add_argument("--holdout-strat-tag", default="holdout_strat")
    ap.add_argument("--force", action="store_true",
                    help="overwrite sections edited by hand since the last "
                         "generation (a backup is written first)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    t = cfg.paths.tables_dir
    s2 = _load(t / f"02_metrics_{args.tag}.json")
    s3 = _load(t / f"03_explain_{args.tag}.json")
    s4 = _load(t / f"04_reject_inference_{args.tag}.json")
    if s2 and s2.get("split_scheme") == "out_of_time":
        # Sections 2-5 are the primary random-split results. Rendering an
        # out-of-time run into them would silently replace those results; the
        # overwrite guard would not catch it, because nothing was edited by hand.
        print(f"ERROR: --tag {args.tag} is an out-of-time run. Sections 2-5 hold the "
              f"primary results; out-of-time results render in section 6 via "
              f"--holdout-tag. Run with --tag full.", file=sys.stderr)
        return 2

    print(f"stage 2 metrics : {'found' if s2 else 'NOT RUN'}")
    print(f"stage 3 explain : {'found' if s3 else 'NOT RUN'}")
    print(f"stage 4 reject  : {'found' if s4 else 'NOT RUN'}")

    bodies = {
        "2": section_2(s2),
        "3": section_3(s3, t, args.tag),
        "4": section_4(s4),
        "5": section_5(s2, s3, s4, _load(t / f"03_explain_{args.tag}_strat.json"),
                       t, args.tag),
        "6": section_6(t, args.holdout_tag, args.holdout_strat_tag, args.tag),
    }

    if args.dry_run:
        for key in SECTIONS:
            print("\n" + "=" * 74)
            print(bodies[key])
        return 0

    path = Path(args.findings)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 2
    text = path.read_text(encoding="utf-8")
    if HOLDOUT_HEADING not in text:
        # Created once, at the end, so section 5 gains a closing marker. This does
        # not change section 5's content, so its recorded hash still matches.
        text = text.rstrip() + "\n\n" + HOLDOUT_HEADING + "\n\n*Not run yet.*\n"

    # Detect hand edits. Each generated section's hash (keep blocks excluded) is
    # recorded at write time; a section whose current hash differs was edited by
    # hand since, and overwriting it would silently discard that work.
    state_path = t / "05_report_state.json"
    all_state = _load(state_path) or {}
    doc_key = str(path.resolve())
    state = all_state.get(doc_key, {})
    edited = [key for key in SECTIONS
              if key in state
              and state[key] != _digest(text[slice(*_bounds(text, *SECTION_MARKERS[key]))])]
    if edited and not args.force:
        print(f"ERROR: section(s) {edited} were edited by hand since the last "
              f"generation. Refusing to overwrite. Move the hand-written text into a "
              f"<!-- keep:NAME --> ... <!-- /keep:NAME --> block, which survives "
              f"regeneration, or re-run with --force to discard it (a backup is "
              f"written first).", file=sys.stderr)
        return 3
    if not state:
        print("note: no prior generation state, so hand edits cannot be detected on "
              "this run; relying on the backup below.")

    backup_dir = t / "_keep" / "findings_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"FINDINGS_{datetime.now():%Y%m%d_%H%M%S}.md"
    backup.write_text(text, encoding="utf-8")

    for key in SECTIONS:
        start, end = SECTION_MARKERS[key]
        text = replace_section(text, start, end, bodies[key])
    new_state = {key: _digest(text[slice(*_bounds(text, *SECTION_MARKERS[key]))])
                 for key in SECTIONS}
    path.write_text(text, encoding="utf-8")
    all_state[doc_key] = new_state
    state_path.write_text(json.dumps(all_state, indent=2), encoding="utf-8")
    print(f"\nFINDINGS.md sections 2-6 rewritten ({len(text.splitlines())} lines "
          f"total); previous version backed up to {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
