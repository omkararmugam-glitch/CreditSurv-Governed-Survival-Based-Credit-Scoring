# Findings

An honest research log. Decisions are recorded when made, with rationale, and
negative or inconclusive results are reported as plainly as positive ones.
Sections for unfinished stages state what will be measured and what the
pre-registered thresholds are, so that a disappointing result cannot be quietly
reframed after the fact.

Last updated: 2026-09-21 (Stage 1 complete, run on the full dataset).

---

## 0. Environment constraint

**`scikit-survival` is not usable on this machine, and the project uses a
pure-Python survival stack instead.** This was discovered by probing rather than
assumed, and it is recorded because it changed the Stage 2 and Stage 3 designs.

What was found:

1. `scikit-survival` 0.28.0 publishes a `cp314` wheel, but hard-requires `ecos`,
   which has **no wheel for Python 3.13 or 3.14** and falls back to a source
   build that fails (`Microsoft Visual C++ 14.0 or greater is required`). Python
   3.11.9 is the only interpreter on this machine where `ecos` resolves.
2. **Smart App Control is enabled in enforcement mode**
   (`VerifiedAndReputablePolicyState = 1`). A clean Python 3.11 venv failed at
   import with `DLL load failed while importing _quad_tree: An Application
   Control policy has blocked this file` — a freshly-downloaded scikit-learn
   binary. Packages already installed are unaffected; the behaviour on new
   downloads was inconsistent across retries. Smart App Control cannot be
   re-enabled once disabled without reinstalling Windows.
3. The official **`survshap` 0.4.2** package (MI2DataLab, the authors of the
   SurvSHAP(t) paper) is pure Python and unpinned, but it hard-imports
   `sksurv.ensemble`, so it inherits the blocker above.

Resolution: a `--system-site-packages` venv reuses the compiled packages already
present and working on this machine (`scikit-learn` 1.9.0, `lightgbm` 4.7.0,
`shap` 0.52.0, `pandas` 2.3.3, `pyarrow` 22.0.0) and adds only pure-Python
packages, which Smart App Control has no reason to block. Verified end to end: a
Cox fit on synthetic data returned the expected coefficients and a c-index of
0.724.

| Component | Chosen | Instead of |
|---|---|---|
| Cox proportional hazards | `lifelines.CoxPHFitter` | `sksurv.linear_model.CoxPHSurvivalAnalysis` |
| Higher-capacity model | LightGBM discrete-time hazard | `sksurv` RandomSurvivalForest |
| Time-dependent attribution | SurvSHAP(t) implemented on `shap.KernelExplainer` | `survshap` package |

**This is not purely a workaround.** A RandomSurvivalForest on 2.26M × ~40 is not
feasible on a laptop, so that route would have meant subsampling heavily and
describing it as a full-data result. The discrete-time hazard model handles the
real dataset. The cost is that SurvSHAP(t) is a reimplementation rather than the
reference package; it will be validated against the **efficiency axiom**
(attributions at each `t` must sum to `S(t|x) − E[S(t)]`), which is a sharp
falsifiable check, and that test is a required part of Stage 3.

---

## 1. Survival labelling

### 1.1 The framing decision

Time origin is `issue_d`; time is measured in **whole months**. Lending Club
dates are month-granular strings (`"Dec-2018"`), so day-level resolution would be
invented precision. Integer months also feed the discrete-time hazard model
without further bucketing.

### 1.2 The event / censoring decision table

This is the single most consequential choice in the project — it sets the event
rate, which moves every downstream metric. It lives in
`src/creditsurv/labeling/status_rules.py` as pure functions with exhaustive tests.

| `loan_status` | Treatment | Observation ends at |
|---|---|---|
| Charged Off | **event** | `last_pymnt_d` |
| Default | **event** | `last_pymnt_d` |
| Fully Paid | censored | `last_pymnt_d` (payoff) |
| Current | censored | data cutoff |
| Issued | censored | data cutoff |
| In Grace Period | censored | `last_pymnt_d` |
| Late (16–30 days) | censored | `last_pymnt_d` |
| Late (31–120 days) | censored | `last_pymnt_d` |
| `Does not meet the credit policy. Status:*` | **excluded** | — |

An unrecognised status raises `UnknownStatusError` rather than defaulting to
censored, because a silent default here would bias everything downstream.

### 1.3 Four judgment calls, stated plainly

**Event time is `last_pymnt_d`, with no charge-off lag added.** Lending Club
charges off at roughly 150 days delinquent, so `last_pymnt_d` precedes the
recorded charge-off by about five months. The target is therefore *time to
cessation of payment* — directly observed, rather than inferred from an assumed
servicing policy. Sensitivity variant: `--event-lag 5`.

**Fully Paid is treated as censored at payoff, which is a known compromise.** It
is formally a *competing risk*: a loan repaid at month 14 can never default
afterwards, and it left the risk set for a reason correlated with being low-risk.
The censoring is therefore mildly informative and will, if anything, bias the
model toward optimism. This follows standard practice in the credit survival
literature and is recorded rather than hidden.

**Delinquent-but-not-charged-off loans are censored, which under-counts events.**
A loan 120 days late is largely destined to charge off. Censoring it is
conservative in the sense that it understates the event rate. Sensitivity
variant: `--late-as-event`. On synthetic data this switch moved the event rate
from 0.247 to 0.291, so the real-data effect is expected to be material and will
be reported.

**`Does not meet the credit policy` loans are excluded.** These are 2007–2008
originations under a different underwriting regime; mixing regimes contaminates
the explainability story. Variant: `--include-policy-exceptions`.

### 1.4 Other rules

* Durations are floored at 1 month — a loan cannot default inside one payment
  cycle, and zero-length rows break the Cox partial likelihood.
* A charged-off loan with no payment on record (`never_paid`) is kept as an event
  at month 1 rather than dropped.
* Rows whose duration exceeds `term + 12` months are treated as date errors and
  dropped, not winsorised.
* The data cutoff is **inferred** from the latest observed payment month rather
  than hardcoded.
* Every exclusion is counted by reason, and the row budget is asserted to
  reconcile: `n_retained + sum(drop_counts) == n_input`.

### 1.5 Leakage control

~40 of the 151 accepted-file columns are recorded after origination, and
`recoveries`, `total_rec_prncp` and `last_fico_range_low` are near-perfect outcome
proxies — a model using them scores near-perfectly and is worthless. The defence
is an **allowlist** in `io/schema.py`, so unknown or renamed columns fail closed,
plus `assert_no_leakage` on every feature matrix. `last_pymnt_d` is checked
explicitly, since it is a legitimate *label* input and therefore the most likely
column to leak in by accident. Verified on a synthetic file that deliberately
contained four leakage columns: none survived ingest.

### 1.6 Lending Club's own grade is excluded from the primary model

`grade`, `sub_grade` and `int_rate` are the output of *another* risk model.
Including them means partly predicting Lending Club's underwriter rather than
default itself, and it makes adverse-action reasoning circular — "your grade was
low" is not a permissible ECOA/Reg B reason, since Reg B requires the specific
underlying factors. They are ingested and retained for a `with_lc_grade`
benchmark variant, so the value of that signal can still be quantified.

### 1.7 Size strategy

| File | Raw | Approach |
|---|---|---|
| accepted | ~1.6 GB, ~2.26M × 151 | chunked at 250k rows, allowlisted to ~43 cols, `float32`/string, appended to Parquet under one fixed Arrow schema |
| rejected | ~1.6 GB, ~27.6M × 9 | chunked at 500k rows, renamed to accepted vocabulary |

**Measured, against the estimate.** Accepted Parquet came out at **174.9 MB**,
better than the 250-400 MB projected. Memory was *worse* than projected: the frame
expands to **2.30 GB** in pandas, not the ~1.5 GB estimated, because Parquet stores
strings compactly but pandas materialises them as Python objects. Casting the 13
low-cardinality string columns to `category` (lossless here) brings it to
**0.70 GB**. That mattered: this machine has 15.2 GB total with ~4 GB free, and the
original implementation copied the frame, which would have peaked near 10 GB and
swapped. `build_survival_target` was rewritten to materialise only retained rows,
and to classify by *unique* status value (9 calls) rather than per row (2.26M
calls). The full label build now runs in 1m45s.

Development runs on a **200k stratified sample** (by `issue_year` x `term_months`,
seed 20260921) with a full run at the end.

Every column is read as text and coerced numerically afterwards. This is slower
than pandas inference, but inference differs between chunks on a file this messy
and produces a schema mismatch part-way through a long ingest. Thousands
separators, `%` and `$` are stripped; a prospectus preamble line is located by
scanning; footer totals rows are dropped by requiring a usable `id`.

**One thing that does not fit, stated in advance:** the person-period expansion
for the discrete-time hazard model is ~45M rows at monthly granularity on the
full dataset. The plan is monthly bins on the dev sample and quarterly bins
(`time_bin_months: 3`, ~16M rows) for the full run, reporting both so the cost of
coarsening is visible rather than assumed away.

### 1.8 Accepted ↔ rejected alignment

Only six features are comparable across the two files, and two of those only
partially. Recorded in `schema.COMMON_FEATURE_MAP` with a comparability grade.

| Concept | Accepted | Rejected | Grade | Caveat |
|---|---|---|---|---|
| Loan amount | `loan_amnt` | Amount Requested | good | funded vs requested |
| Employment length | `emp_length` | Employment Length | good | higher missing rate on rejected |
| State | `addr_state` | State | good | — |
| ZIP | `zip_code` | Zip Code | good | both 3-digit |
| Score | `fico_range_low` | Risk_Score | **partial** | different instruments; comparable in rank, not scale; high missing rate in later vintages |
| DTI | `dti` | Debt-To-Income Ratio | **partial** | different definitions; rejected is self-reported text with implausible extremes (>1000%) |

**There is no income, no `revol_util`, no `home_ownership` and no credit-history
depth on the rejected side.** That is a hard ceiling on Stage 4, not a detail:
the selection-bias diagnostic can only be run on these six features, and any
reject-inference correction can only be justified on them. `Application Date` is
deliberately excluded (an application date is not an issue date; used for vintage
alignment only) and `Policy Code` is degenerate on both sides.

### 1.9 Results on real data

Run on the full Kaggle extract: `accepted_2007_to_2018Q4.csv` (1,597 MB) and
`rejected_2007_to_2018Q4.csv` (1,700 MB). No prospectus preamble in this build --
the header is row 0 on both files -- and all 73 allowlisted columns were present.

**Row budget.** 2,260,668 in, **2,257,790 retained (99.87%)**. Losses are
negligible and every one is accounted for:

| Reason | Rows |
|---|---|
| `excluded_status` (does not meet credit policy) | 2,749 |
| `missing_end_date` | 102 |
| `term_overrun` (beyond term + 12 months) | 27 |

**Status distribution (pre-exclusion).**

| `loan_status` | Rows | Share |
|---|---|---|
| Fully Paid | 1,076,751 | 47.63% |
| Current | 878,317 | 38.85% |
| Charged Off | 268,559 | 11.88% |
| Late (31-120 days) | 21,467 | 0.95% |
| In Grace Period | 8,436 | 0.37% |
| Late (16-30 days) | 4,349 | 0.19% |
| Does not meet credit policy: Fully Paid | 1,988 | 0.09% |
| Does not meet credit policy: Charged Off | 761 | 0.03% |
| Default | 40 | 0.00% |

**`Default` is effectively a non-category: 40 rows in 2.26M.** Events are therefore
almost entirely `Charged Off`. Including `Default` is still correct, but it changes
nothing, and any writeup implying two meaningful event sources would mislead.

**Target.** Event rate **0.1190** (268,593 events). Data cutoff **inferred as
2019-03-01** -- three months later than the "2018Q4" filename implies, because
`last_pymnt_d` runs past the last issue month. Hardcoding 2018-12 would have
truncated every still-current loan by three months.

Duration quantiles (months): p0=1, p10=5, p25=9, **p50=17**, p75=29, p90=36,
p99=54, p100=70.

| Outcome | n | Median duration |
|---|---|---|
| event | 268,593 | 14 |
| censored_payoff | 1,076,731 | 22 |
| censored_admin | 878,316 | 15 |
| censored_delinquent | 34,150 | 17 |

Defaults happen *earlier* than payoffs (median 14 vs 22 months) -- precisely the
information a binary target discards.

**Sanity checks, both passed.** Event rate rises monotonically across Lending
Club's own grade, which is strong evidence the labelling is correct. Grade is not a
model feature, so this is an independent check:

| Grade | n | Event rate |
|---|---|---|
| A | 432,923 | 0.0328 |
| B | 663,174 | 0.0793 |
| C | 649,384 | 0.1319 |
| D | 323,707 | 0.1886 |
| E | 135,078 | 0.2668 |
| F | 41,553 | 0.3488 |
| G | 11,971 | 0.3809 |

60-month loans default more than 36-month (0.1619 vs 0.1016), as expected.

**Kaplan-Meier.** S(6)=0.9777, S(12)=0.9388, S(18)=0.8970, S(24)=0.8573,
S(30)=0.8206, S(36)=0.7923, S(48)=0.7292, S(60)=0.6909.

**The survival framing is now justified empirically, not just in principle.**
Administrative censoring by vintage:

| Issue year | n | Still current | Median observed |
|---|---|---|---|
| 2007-2013 | 228,941 | 0.00% | 28-34 months |
| 2014 | 235,629 | 5.06% | 27 months |
| 2015 | 421,094 | 10.28% | 27 months |
| 2016 | 434,407 | 30.86% | 25 months |
| 2017 | 443,579 | 59.03% | 17 months |
| 2018 | 495,140 | **86.27%** | **8 months** |

A binary "defaults within 36 months" model would have to discard or mislabel **86%
of the 2018 book and 59% of 2017** -- 22% and 20% of the dataset respectively.
Survival analysis uses them correctly as censored. This is the strongest argument
for the framing, and it is a property of the data rather than a modelling
preference.

**Vintage-coverage claim confirmed.** The `num_*` and `mo_sin_*` families are
**100% missing for 2007-2011**, ~50% missing in 2012, and ~0% from 2013 onward.
That is 42,535 loans (1.9% of the data) with no extended features at all, which
vindicates keeping them in `EXTENDED_NUMERIC` rather than `CORE_NUMERIC`. Highest
overall missingness is `mths_since_last_record` (84.11%),
`mths_since_recent_bc_dlq` (77.01%) and `mths_since_last_major_derog` (74.31%) --
all structurally missing (no such record exists) rather than data faults, so they
need missing-as-signal handling, not imputation.

**Leakage guard verified on real data:** zero leakage columns in the ingested
frame, from a source file that contains all of them.

### 1.10 Rejected file: Stage 4 is more constrained than expected

27,648,741 rejected applications against 2,257,790 accepted -- a **12.2:1 ratio**,
implying an acceptance rate near 7.6%.

**`risk_score` is 66.90% missing (18,497,630 nulls), and the gaps fall exactly
where the volume is.** Coverage by application year:

| Year | n | `risk_score` coverage |
|---|---|---|
| 2007-2013 | 1,510,437 | 89-99% |
| 2014 | 1,933,700 | 86.84% |
| 2015 | 2,859,379 | **17.85%** |
| 2016 | 4,769,874 | **21.35%** |
| 2017 | 7,072,573 | 54.10% |
| 2018 | 9,496,782 | **6.83%** |

The four highest-volume years -- 79% of all rejected applications -- have the worst
coverage. This is worse than anticipated in 1.8 and it materially shrinks what
Stage 4 can do: the score dimension, the single most decision-relevant feature in
the file, is unusable for most of the rejected population.

**`dti_raw` parses 100% numeric but is not trustworthy as recorded.** Quantiles:
p1=**-1.0**, p25=8.1, p50=20.0, p75=36.6, p95=100.0, p99=455.2,
p100=**50,000,031.5**. 2.92% exceed 100% and 0.61% exceed 1000%. Negative and
50-million-percent values are self-reported junk; this needs winsorising before any
comparison, and the choice of cut will be stated explicitly in Stage 4.

**Preliminary SMDs (indicative only, not the gate):**

| Feature | Accepted mean | Rejected mean | SMD |
|---|---|---|---|
| Loan amount | 15,046.9 | 13,133.2 | 0.154 |
| Score | 698.6 | 628.2 | 1.039 |
| DTI (clipped at 100) | 18.8 | 27.5 | -0.399 |

**The 1.039 on score must not be read as selection bias.** It compares a FICO band
against Lending Club's `Risk_Score` -- different instruments on different scales,
graded "partial" in 1.8. Some of that gap is measurement, not selection. Reporting
it as evidence of distortion would be wrong, and the Stage 4 diagnostic will need a
rank-based comparison for this pair rather than a difference in means.

Loan amount (0.154) and DTI (-0.399) are like-for-like enough to be meaningful, and
both exceed the pre-registered 0.10 "notable" threshold, with DTI above the 0.25
"substantial" line. So there is real signal here -- but the gate also requires
common-support analysis, which has not been run, and the concern in section 4 about
near-deterministic cutoffs remains open.

---

## 2. Survival models

*Code complete and smoke-tested on a 5,000-loan slice; not yet run on the full
dataset.* Run `scripts/02_train_models.py --full`, then `scripts/05_report.py`, to
populate this section with real numbers. Will report concordance index and
time-dependent AUC at horizons
6/12/18/24/30/36 months for both the Cox baseline and the LightGBM discrete-time
hazard model, plus integrated Brier score and calibration. Risk will be reported
as explicitly age-dependent: how predicted hazard for a fixed borrower profile
changes as the loan seasons, not a single static score.

## 3. Explainability

*Code complete and smoke-tested; not yet run at full scale.* The SurvSHAP(t)
reimplementation passes the efficiency axiom exactly (residual 0.0 on the slice
run), which is the correctness precondition for anything reported here. Three
questions, with the answer format fixed in advance:

**(a) Does naive SHAP disagree with SurvSHAP(t) on *this* dataset?** The
literature says it should, but that will be **measured, not assumed** — rank
correlation and top-k overlap between the two attributions on the same model. A
finding of "they broadly agree here" is a legitimate result and will be reported
as such.

**(b) Adverse-action report.** SurvSHAP(t) output converted to an itemized ECOA /
Reg B style statement of specific principal reasons for a single applicant.

**(c) Segment stability.** Aggregated attributions across loan grade, loan purpose
and income band, reporting whether importance is stable or drifts.

## 4. Reject inference (diagnostic-gated)

*Code complete and smoke-tested; not yet run at full scale.*
**Thresholds are pre-registered here, before any result is seen:**

| Metric | Notable | Substantial |
|---|---|---|
| Standardized mean difference | \|SMD\| > 0.10 | > 0.25 |
| KS statistic | — | > 0.20 |
| Accept-vs-reject discriminator AUC | > 0.60 | > 0.75 |
| Common support | — | < 0.05 blocks correction |

At n in the millions **every KS p-value will be approximately zero**, so a
p-value-based gate would be meaningless here. The gate is effect-size based;
p-values will be reported but never used to decide.

Stated in advance: Lending Club used near-deterministic FICO/DTI cutoffs, so
there is a real possibility that common support is close to empty. If so, reject
inference is extrapolation rather than correction, and **that will be the reported
finding** — no correction will be forced in order to have something to show.

Whether or not a correction is applied, Stage 3's explanations will be re-run
before and after, and any shift in feature importance reported. This is the
project's own question; none of the source literature tests it.

## 5. Summary

*Pending completion of Stages 2–4.*
