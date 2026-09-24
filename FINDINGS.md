# Findings

An honest research log. Decisions are recorded when made, with rationale, and
negative or inconclusive results are reported as plainly as positive ones.
Sections for unfinished stages state what will be measured and what the
pre-registered thresholds are, so that a disappointing result cannot be quietly
reframed after the fact.

Last updated: 2026-09-22 (all stages run on the full dataset; grade-stratified segment analysis added).

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

Fitted on `outputs\data\accepted_labeled.parquet` (2,257,790 loans), random split, 1,806,232 train / 451,558 test. Lending Club grade excluded (primary spec).

Features: 55 numeric + 6 categorical.

| Model | C-index | IBS | AUC 6m | AUC 12m | AUC 18m | AUC 24m | AUC 30m | AUC 36m |
|---|---|---|---|---|---|---|---|---|
| cox | 0.6700 | 0.1008 | 0.6705 | 0.6752 | 0.6710 | 0.6690 | 0.6672 | 0.6429 |
| discrete_hazard | 0.6973 | 0.0998 | 0.7248 | 0.7118 | 0.6998 | 0.6920 | 0.6860 | 0.6388 |

Risk is reported as age-dependent in `outputs/tables/02_age_dependent_risk_*.csv`: the conditional probability of default over the next 12 months for a *fixed* borrower profile, at several loan ages. A single static score cannot express this.

## 3. Explainability

SurvSHAP(t) computed for 250 borrowers on the `discrete_hazard` model, `nsamples=600`, background 100 rows, at horizons [6, 12, 18, 24, 30, 36]. Cost: 2.75s per borrower (686s total).

### Implementation correctness

Efficiency-axiom worst residual: **0.00e+00** -> **PASS**. At every time point the attributions plus the base value reconstruct the model's own prediction, which is the falsifiable check that this reimplementation behaves like Shapley values rather than merely producing plausible numbers.

The first full-scale attempt failed this check outright (residual **5.65e+07**). KernelSHAP had been routed through LARS feature selection, which is unstable on this design matrix's near-constant columns -- rare-event counts and 13 missing-value indicators -- and returned degenerate active sets. Plain weighted least squares (`l1_reg=0`) makes the axiom exact. A test on well-conditioned random data did not reproduce the failure; it took real data to surface it.

### Global importance (time-integrated)

| Feature | Importance |
|---|---|
| `installment` | 0.03706 |
| `acc_open_past_24mths` | 0.02430 |
| `annual_inc` | 0.02064 |
| `tot_hi_cred_lim` | 0.01588 |
| `percent_bc_gt_75` | 0.01307 |
| `dti` | 0.01275 |
| `loan_amnt` | 0.01135 |
| `mths_since_last_record` | 0.01082 |
| `mo_sin_old_rev_tl_op` | 0.01055 |
| `addr_state` | 0.00987 |

### (a) Naive SHAP vs SurvSHAP(t) -- measured, not assumed

Both explainers use the same model, background set and `nsamples`; the only difference is that naive SHAP explains the scalar `1 - S(36.0 months)` while SurvSHAP(t) explains the whole curve. Any disagreement can therefore only come from the time dimension.

| Measure | Value |
|---|---|
| Spearman rank correlation | 0.9888 |
| Pearson correlation | 0.9976 |
| Top-5 overlap (Jaccard) | 1.0000 |
| Top-10 overlap (Jaccard) | 1.0000 |
| Sign agreement | 0.7945 |
| Mean time-varying share | 0.4640 |

Sign disagreements: ['total_acc', 'chargeoff_within_12_mths', 'tot_cur_bal', 'mths_since_recent_inq', 'num_actv_rev_tl', 'num_rev_tl_bal_gt_0', 'mths_since_last_delinq_missing', 'mths_since_last_record_missing', 'mths_since_recent_bc_dlq_missing', 'mths_since_recent_revol_delinq_missing', 'mths_since_recent_bc_missing', 'revol_util_missing', 'bc_util_missing', 'percent_bc_gt_75_missing', 'pct_tl_nvr_dlq_missing']. Within the top 5: none.

Most time-varying features: ['bc_open_to_buy', 'num_actv_bc_tl', 'total_bc_limit', 'num_bc_sats', 'num_op_rev_tl'].

**BROAD AGREEMENT: Spearman 0.989, top-5 overlap 1.000, sign agreement 0.795. On this dataset the scalar collapse does not materially change which features are named, so the literature's concern does not reproduce here at the level of reported reasons.**

#### Evidence: method difference vs. the measurement's own noise floor

A sweep held borrowers and background fixed and varied only the KernelSHAP coalition draw, measuring how much SurvSHAP(t) disagrees *with itself* from Monte Carlo sampling alone (`outputs/tables/03_nsamples_study.csv`):

| `nsamples` | Between-draw Spearman | sec/borrower |
|---|---|---|
| 146 | 0.883 | 0.85 |
| 210 | 0.883 | 0.82 |
| 350 | 0.890 | 1.18 |
| 600 **(used)** | 0.960 | 1.85 |
| 1200 | 0.973 | 4.50 |
| 2194 | 0.969 | 8.38 |

At nsamples=600, two draws of the *same* method agree at **rho = 0.960**. The two *different* methods agree at **rho = 0.9888**.

**This is a settled negative result.** The difference between naive SHAP and SurvSHAP(t) is smaller than the Monte Carlo noise within either one, so the two are statistically indistinguishable on this dataset. Top-5 and top-10 membership are identical and no top-5 feature disagrees on direction, so every reason that could appear in an adverse-action notice is the same under both. A larger borrower sample cannot overturn this: the effect is below the resolution of the instrument, not merely small.

What it does *not* say: attributions do vary with time (mean time-varying share 0.4640), and sign agreement across all features is only 0.7945. What fails to reproduce is the claim that collapsing that time structure to a scalar changes *which features are named*. Generalisation is limited to this dataset, model family and horizon grid; this book is also unusually heavily censored (39% still performing).

### (b) Adverse-action notice (ECOA / Regulation B)

Generated for the highest-risk explained applicant (`APP-000225`, predicted default probability 0.5567 by 36 months), with 4 principal reasons. Full text in `outputs/tables/03_adverse_action_notice_*.txt`.

| Rank | Feature | Disclosed reason |
|---|---|---|
| 1 | `acc_open_past_24mths` | Number of accounts opened recently |
| 2 | `dti` | Excessive obligations in relation to income |
| 3 | `annual_inc` | Income insufficient for amount of credit requested |
| 4 | `verification_status` | Unable to verify income |

Three Regulation B constraints are enforced in code, not left to the caller: reasons are capped at four per the Official Staff Commentary to 12 CFR 1002.9(b)(2); Lending Club grade and interest rate are refused as reasons because a failure-to-score disclosure does not satisfy the requirement; and only features whose attribution is *adverse* are eligible, since a feature that helped the applicant is not a reason for denial.

Features that *helped* this applicant, correctly excluded from the notice: ['purpose', 'pct_tl_nvr_dlq', 'addr_state', 'total_il_high_credit_limit', 'mths_since_last_record'].

### (c) Explanation stability across segments

Purpose, income band and term are measured on the unstratified 250-borrower sample, which is representative of the book. **Grade is measured on a separate grade-stratified sample** (below), because random sampling mirrors the population and left grades F and G too thin to measure at all.

| Segment | Levels | Min rank corr. | Mean top-5 overlap | Verdict |
|---|---|---|---|---|
| purpose | 4 | 0.9446 | 0.8333 | STABLE |
| income_band | 6 | 0.9476 | 0.7778 | STABLE |
| term | 2 | 0.9881 | 0.8333 | STABLE |

**purpose.** STABLE: minimum rank correlation 0.945 (floor 0.85), mean top-5 overlap 0.833 (floor 0.7), 2 feature(s) above CV 0.5. Importance is consistent across purpose, so a global explanation is a fair summary.

**income_band.** STABLE: minimum rank correlation 0.948 (floor 0.85), mean top-5 overlap 0.778 (floor 0.7), 4 feature(s) above CV 0.5. Importance is consistent across income_band, so a global explanation is a fair summary.

**term.** STABLE: minimum rank correlation 0.988 (floor 0.85), mean top-5 overlap 0.833 (floor 0.7), 0 feature(s) above CV 0.5. Importance is consistent across term, so a global explanation is a fair summary.

#### Grade, on a grade-stratified sample

252 borrowers, equal allocation per grade ({'A': 36, 'B': 36, 'C': 36, 'D': 36, 'E': 36, 'F': 36, 'G': 36}), same model and SurvSHAP(t) settings. Efficiency residual 5.55e-17.

| Grade | n | rho vs population | top-5 overlap vs population | rho vs pooled | top-5 overlap vs pooled |
|---|---|---|---|---|---|
| A | 36 | 0.928 | 0.667 | 0.943 | 1.000 |
| B | 36 | 0.964 | 0.667 | 0.971 | 0.667 |
| C | 36 | 0.972 | 0.667 | 0.985 | 1.000 |
| D | 36 | 0.975 | 0.667 | 0.981 | 0.667 |
| E | 36 | 0.972 | 0.667 | 0.980 | 0.667 |
| **F** | 36 | 0.966 | 0.667 | 0.985 | 1.000 |
| **G** | 36 | 0.965 | 0.429 | 0.977 | 0.667 |

Population top 5 (unstratified run): ['installment', 'acc_open_past_24mths', 'annual_inc', 'tot_hi_cred_lim', 'percent_bc_gt_75'].

**Pre-registered aggregate rule** (min rank correlation >= 0.85, mean top-5 overlap >= 0.7):

| Reference | Min rank corr. | Mean top-5 overlap | Verdict |
|---|---|---|---|
| population | 0.928 | 0.633 | DRIFTS |
| pooled | 0.943 | 0.810 | STABLE |

**The aggregate verdict depends on the reference ranking and sits at the threshold, so it is not robust in either direction.** Rank correlation is comfortably high for every grade; the whole question turns on top-5 membership, and there the mean lands just either side of 0.70.

Features whose rank in F or G falls **outside the entire range seen in grades A-E** and crosses the top-5 boundary:

| Feature | Grade | Rank there | Rank range in A-E |
|---|---|---|---|
| `annual_inc` | G | 9 | 2-3 |
| `mths_since_recent_inq` | G | 4 | 8-16 |
| `percent_bc_gt_75` | F | 10 | 5-7 |

Shared by the top 5 of every grade A-G: ['acc_open_past_24mths', 'installment'].

#### Borrower-sampling uncertainty (5,000 stratified bootstrap replicates, 95% intervals)

Borrowers were resampled with replacement *within* each grade, preserving 36 per grade, and every grade's ranking recomputed. `Shift` is the rank in the target grade minus the rank in pooled A-E (positive = less important in the target grade). `Share ratio` compares the feature's share of total attribution, target over A-E; unlike ranks it is continuous, so it checks that a rank move is not just a near-tie flipping.

| Grade | Feature | Rank there | Rank in A-E | Shift [CI] | Share ratio [CI] | P(outside A-E range) | Reading |
|---|---|---|---|---|---|---|---|
| G | `annual_inc` | 9 [4, 17] | 3 | +6 [+1, +14] | 0.59 [0.42, 0.79] | 0.92 | real difference in direction; top-5 exit/entry not established |
| G | `mths_since_recent_inq` | 4 [3, 11] | 11 | -7 [-9, +0] | 1.27 [0.99, 1.56] | 0.77 | **cannot exclude no difference** |
| G | `percent_bc_gt_75` | 8 [5, 13] | 6 | +2 [-2, +8] | 0.85 [0.70, 1.01] | 0.38 | **cannot exclude no difference** |
| G | `dti` | 5 [3, 13] | 5 | +0 [-3, +8] | 0.90 [0.68, 1.17] | 0.24 | **cannot exclude no difference** |
| F | `percent_bc_gt_75` | 10 [7, 14] | 6 | +4 [+0, +9] | 0.81 [0.66, 0.98] | 0.59 | **cannot exclude no difference** |
| F | `annual_inc` | 4 [3, 9] | 3 | +1 [+0, +6] | 0.84 [0.63, 1.09] | 0.40 | **cannot exclude no difference** |
| F | `dti` | 5 [3, 10] | 5 | +0 [-3, +4] | 1.05 [0.78, 1.37] | 0.19 | **cannot exclude no difference** |

Two reasons these intervals are, if anything, too narrow:

- **The features were chosen after looking at the data.** They are the ones whose point-estimate rank fell outside the A-E range, so they were selected *because* they looked extreme. Intervals on post-hoc-selected features are optimistic (winner's curse). This is an exploratory check, not a confirmatory test.
- **Coalition-sampling noise is not in them.** The bootstrap resamples borrowers but reuses each borrower's attribution. Re-running the identical sample with a different SHAP draw moved these ranks by 1-2 places (per-grade Spearman between the two runs 0.93-0.98).

**Not every point estimate survives resampling.** For `mths_since_recent_inq` in G, `percent_bc_gt_75` in G, `dti` in G, `percent_bc_gt_75` in F, `annual_inc` in F, `dti` in F the interval on the rank shift includes zero, so these could be noise from which 36 borrowers happened to be drawn and should not be reported as grade-specific behaviour.

## 4. Reject inference (diagnostic-gated)

<!-- keep:preregistration -->
### 4.0 Pre-registration (restored verbatim)

> **Provenance.** The text below was written on 2026-09-21, before Stage 4 was run
> on real data: the backup that preserves it is timestamped 21:21, and the first
> Stage 4 diagnostic output is timestamped 23:28. It was later overwritten when
> `05_report.py` regenerated this section, and is restored here verbatim from
> `outputs/tables/_keep/FINDINGS_pre_stage4_2026-09-21.md`. It now sits in a
> protected block that regeneration preserves.
>
> **One gap in it, stated plainly.** The two features graded "partial" (score and
> DTI) are judged on rank AUC, with notable/substantial thresholds of 0.55/0.65.
> Those values were fixed in code (`run_selection_diagnostic` defaults) before the
> real-data run, but they were **not** written into the table below. Only the
> thresholds in the table were pre-registered in this document.

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
<!-- /keep:preregistration -->

Compared 2,257,790 accepted loans against 1,000,000 sampled rejected applications on the features common to both files.

### Diagnostic result

| Feature | Comparability | Accepted mean | Rejected mean | SMD | Rank AUC | KS | Verdict |
|---|---|---|---|---|---|---|---|
| `loan_amnt` | good | 15054.274 | 13147.01 | 0.1537 | 0.6081 | 0.2287 | notable |
| `dti` | partial | 18.668 | 27.573 | -0.4343 | 0.4562 | 0.2331 | negligible |
| `score` | partial | 698.605 | 628.506 | 1.0355 | 0.8222 | 0.6543 | substantial |
| `emp_length_years` | good | 5.979 | 1.146 | 1.6768 | 0.8993 | 0.7712 | substantial |

- Separability AUC: **0.8938** (strong threshold 0.75)
- Top selection drivers: ['emp_length_years', 'score', 'dti']
- Common support: **0.6706** (floor 0.05)
- Bias detected: yes
- Support adequate: yes

### Gate decision: `APPLY_CORRECTION`

Distortion present (separability AUC 0.894; substantial on ['score', 'emp_length_years']) AND common support is 67.1%, above the 5% floor. A reweighting correction is warranted and will be applied.

Notes recorded during the diagnostic:

- dti: graded 'partial' -- judged on rank AUC (0.456), not SMD (-0.434), because the accepted and rejected columns are different instruments and an SMD between them conflates scale with selection.
- score: graded 'partial' -- judged on rank AUC (0.822), not SMD (1.036), because the accepted and rejected columns are different instruments and an SMD between them conflates scale with selection.

Every KS p-value in this comparison is effectively zero because of sample size, which is exactly why the gate is built on effect sizes. p-values are reported in the CSV for completeness and were not used to decide.

### Explanation shift before/after correction

This is the project's own question: does correcting for selection bias change *which features the model says matter*, not merely how well it scores? None of the reviewed literature tests it.

- Rank correlation before/after: **0.9709**
- Top-5 overlap: **1.0000**
- Top-5 before: ['acc_open_past_24mths', 'annual_inc', 'dti', 'installment', 'loan_amnt']
- Top-5 after: ['acc_open_past_24mths', 'annual_inc', 'dti', 'installment', 'loan_amnt']

**IMPORTANCE IS STABLE under correction**

Weighting diagnostics: propensity AUC 0.8938, effective sample size 1589230.5000 of 2,257,790 (ratio 0.7039), 45,156 weights clipped.

*Corrects selection only on the features listed in features_used. Selection operating through annual_inc, revol_util, home_ownership, credit history depth is not addressed and is invisible to this correction.*

## 5. Summary

Stages complete: 1, 2, 3, 4.

- Best concordance: **0.6973** (`discrete_hazard`), on 451,558 test loans.
- Naive vs survival SHAP: BROAD AGREEMENT (Spearman 0.9888).
- SurvSHAP(t) efficiency axiom: passed (0.0e+00).
- Explanation stability (purpose, income band, term): stable across all segments tested.
- Explanation stability by grade (stratified, 36 per grade): the aggregate verdict is threshold-sensitive, landing either side of the 0.70 top-5 overlap floor depending on the reference ranking.
- Surviving bootstrap resampling: `annual_inc` is less important in G than in A-E. Every other grade-specific shift tested has an interval that includes no difference. Exploratory: the features were selected post hoc. See section 3(c).
- Reject-inference gate: **apply_correction**.
- Explanation shift after correction: IMPORTANCE IS STABLE under correction.

## 6. Out-of-time holdout

<!-- keep:preregistration-holdout -->
### 6.0 Pre-registration (written before any holdout result existed)

Recorded on 2026-09-22, in its own commit, after the analysis code (commit
`74f01d2`) and before the holdout was run. At the time of writing no model had been
fitted on the 2007-2015 vintages and no holdout metric, explanation or bootstrap
existed. The rules below are applied mechanically by `scripts/05_report.py`.

**Design, fixed in advance.** Models are refitted on loans issued 2007-2015 only and
evaluated on loans issued 2016-2018 (`--split out_of_time --oot-cutoff 2016`), with the
same settings as the full run: quarterly time bins, negative subsampling 0.4, Cox on a
300,000-loan subsample. Early stopping validates on a random 10% of training loans. The
IPCW censoring model is fitted on the holdout. SurvSHAP(t) uses `nsamples=600`, a
100-row background, horizons 6/12/18/24/30/36 months and seed 20260921. Horizons of 30
and 36 months are reachable only by 2016 loans and are reported as such.

**H1 -- grade G income share (confirmatory).** Explain 36 holdout borrowers per grade
A-G with the holdout model. Bootstrap (5,000 stratified replicates, resampling
borrowers within grade) the ratio of `annual_inc`'s share of total attribution in
grade G to its share in pooled grades A-E.
- **CONFIRMED** if the 95% interval for that ratio lies entirely below 1.
- **NOT REPLICATED** otherwise, including when the interval merely includes 1.

In section 3 this feature was selected for testing after looking at the data, so its
interval there was optimistic. Here it is named in advance, so that caveat does not
apply to this test.

**H2 -- grade G inquiry recency (re-test of an inconclusive result).** Same procedure
for `mths_since_recent_inq`, whose section 3 interval included "no difference".
- **REPLICATED** (more important in G) if the 95% interval lies entirely above 1.
- **NOT REPLICATED** otherwise.

**H3 -- naive SHAP vs SurvSHAP(t) (same standard as section 3).** Explain 250 random
holdout borrowers with both methods under identical settings, and separately measure
SurvSHAP(t)'s between-draw noise floor on the holdout model at `nsamples=600`
(40 borrowers, two independent coalition draws). The floor from section 3 belongs to a
different model and is not reused.
- **(i) ranking level holds** if the method-vs-method Spearman is at least that floor.
- **(ii) notice level holds** if the top-5 features are identical and none of them
  differs in sign between the methods.

The section 3 finding counts as replicated only if both hold.

**Not pre-registered.** The Stage 2 metrics (item 2) and the selection-bias diagnostic
on 2016-2018 rejected applicants (item 5) are descriptive and have no pass/fail rule.
<!-- /keep:preregistration-holdout -->

*Not run yet.* The pre-registered decision rules above were fixed before any holdout result existed.

## 7. Bulk explanation: can TreeSHAP stand in for SurvSHAP(t)?

<!-- keep:preregistration-explainer -->
### 7.0 Pre-registration (written before the comparison was run)

**The problem.** SurvSHAP(t) costs about 2.7 seconds per applicant on the full
model. A 200 MB upload is roughly 1.9 million applicants, of which some 17% are
declined at the 0.30 threshold, so Regulation B reasons for one such file would
take about nine months of compute. It is also sampled, so an applicant's stated
reasons can change with which other applicants happen to be explained alongside
them. Neither is acceptable for a production decision file.

**The candidate.** TreeSHAP reads the fitted trees directly: exact, deterministic
and measured at 4 ms per applicant here, about 700 times faster. It explains the
per-period hazard margin rather than survival, so
`creditsurv/explain/tree_shap.py` combines the per-period Shapley values with the
applicant's own hazards. Because Shapley values are linear in the value function,
the result is an *exact* Shapley value of `sum_t h_t * f_t`, a first-order
expansion of cumulative hazard around the applicant's own point -- not an exact
Shapley value of the default probability itself. That is precisely why it is
validated here instead of assumed equivalent.

**What is being tested.** Whether TreeSHAP picks the same principal reasons a
notice would have stated under SurvSHAP(t), for applicants who are actually
declined. Both explainers are passed through the same
`build_adverse_action_notice`, so the Regulation B filtering, the geography
exclusion and the wording are identical and only the attributions differ.

**Sample.** At least 1,000 applicants drawn from the full model's test split,
restricted to those the configured threshold declines.

**Acceptance bar, fixed before running:**

| # | Criterion | Bar |
|---|---|---|
| E1 | Applicants whose **top stated reason** is identical under both explainers | >= 90% |
| E2 | Mean overlap of the **top-4 stated reason sets** (shared / 4) | >= 0.75 |

Both must hold. If either fails, TreeSHAP is not used for notices, and the
fallback is to make SurvSHAP(t) deterministic per applicant by seeding each call
from the applicant's row id, so that reasons stop depending on batch membership --
at unchanged cost.

**Reference measurement, for interpretation only.** SurvSHAP(t) is not perfectly
reproducible against itself: an earlier measurement found the fourth stated reason
moving with the background draw. So the same comparison is also run
SurvSHAP-against-SurvSHAP with two different seeds on a 300-applicant subset, to
show what agreement rate a *perfect* stand-in could achieve. This number is
reported alongside the result and explicitly **cannot rescue a failed bar**: if
TreeSHAP misses E1 or E2, it fails, whatever the ceiling turns out to be.

**Scope if it passes.** TreeSHAP becomes the reason generator for bulk scoring
runs only. SurvSHAP(t) remains the explainer for the research stages (3, 3b, 3c)
and for single-applicant work, and every scored row records which explainer
produced its reasons.

### 7.0b Addendum, written before any result was read

Two things fixed in advance so that neither can be chosen in hindsight.

**A secondary comparison, and what it cannot do.** The primary comparison holds
TreeSHAP against a *single* SurvSHAP(t) draw, and that draw carries the sampling
noise already recorded in this project: an applicant's fourth stated reason can
move with the background draw. So a second comparison is registered here, against
**SurvSHAP(t) averaged over 3 independent draws** on a 300-applicant subset, using
the mean attribution per feature before the reasons are selected. Averaging lowers
the noise in the target, so agreement with it is the better estimate of whether
TreeSHAP finds the same drivers as SurvSHAP *in expectation*.

It is **secondary and reported for interpretation only**. The verdict on E1 and E2
is decided by the primary comparison against a single draw, exactly as specified in
7.0. Neither the secondary numbers nor the SurvSHAP-against-itself ceiling can move
the verdict, and neither can change the `decision.bulk_explainer` setting. If the
primary comparison fails and the secondary one looks better, the recorded outcome
is still a failure, and the reason for the gap is noted as a finding rather than
used as grounds to proceed. Cost of the secondary run is about 40 minutes, which is
why it is worth doing at all.

**What bulk runs do if TreeSHAP fails.** Seeding each SurvSHAP(t) call from the
applicant's row id fixes reproducibility -- a notice stops depending on batch
membership -- but it does nothing about cost, and a declined applicant with no
stated reasons is a compliance gap whatever the reason for it. So in that case:

* Bulk runs explain **every** rejected applicant, not a capped subset. The cap
  stops being a normal operating mode.
* The run happens in the background with a **visible estimate of time remaining**,
  computed from the measured per-applicant cost and the number of rejected rows,
  shown before it starts and updated as it goes. At 2.7 s per applicant a file with
  100,000 declined applicants is about 76 hours, and the page must say so plainly
  rather than start silently.
* The `reasons not generated` marker then means one thing only: **the run was
  stopped before it finished**. Rows that were never reached keep the marker and are
  counted, so a partial file is never mistaken for a complete one.

**If the failure branch is taken, the long run must survive interruption.** A
76-hour explanation pass that loses everything to a reboot is not usable, so two
further requirements are recorded here, to be built **only if the primary verdict
is a fail**:

* **Resumable.** Explained rows are appended to disk as each one finishes, keyed by
  `row_id` and `applicant_id`, rather than held until the end. On restart the run
  reads that ledger and skips rows already explained. What makes this safe rather
  than merely convenient is the seeding: with each call seeded from the row id, a
  resumed run produces *identical* reasons to an uninterrupted one, which a test
  must assert by explaining a file, killing it part-way, resuming, and comparing
  the result with a single uninterrupted pass row by row.
* **Sleep.** The run asks Windows to keep the machine awake for its duration
  (`SetThreadExecutionState` with `ES_CONTINUOUS | ES_SYSTEM_REQUIRED` through
  ctypes, which needs no elevation), releases that request when it ends including
  on failure, and records in the log whether the request was granted. It does not
  prevent a deliberate sleep or a closed lid, so the page says plainly that sleep
  pauses a run and that a paused run resumes where it stopped.

This is the recorded plan for the failure branch, chosen before the result was
known.
<!-- /keep:preregistration-explainer -->

### 7.1 Result: TreeSHAP FAILS the pre-registered bar

Run on the full model's test split, 1,000 applicants declined at
a 0.30 36-month threshold, both explainers passed through the
same notice builder.

| # | Criterion | Bar | Observed | Result |
|---|---|---|---|---|
| E1 | Same top stated reason | >= 90% | **80.2%** | **FAIL** |
| E2 | Mean top-4 reason-set overlap | >= 0.75 | 0.84 | PASS |

**Verdict: FAIL.** Both criteria were required. TreeSHAP is **not** used to write
adverse-action reasons. `decision.bulk_explainer` stays `survshap`.

**The failure is real, not an artifact of SurvSHAP's own noise.** This is what the
reference ceiling was registered for: SurvSHAP(t) against itself under two seeds
agrees on the top reason for 95.3% of the same applicants, with
mean top-4 overlap 0.94. So sampling noise costs about
5 points of top-1 agreement, while TreeSHAP costs
about 20. Roughly three quarters of the gap is
genuine disagreement about which factor matters most, not noise.

What the two explainers agree on is the *set* of drivers: top-4 overlap of
0.84 passed comfortably, and 41.2% of
applicants had an identical top-4 set. They disagree on the ordering within that
set, which is exactly what a notice discloses first. That is consistent with what
section 3(a) already records about time-agnostic attributions: a hazard-weighted
margin is not the same quantity as S(t), and where two drivers are close the
ordering does not survive the substitution.

**Cost, for the record.** TreeSHAP took 6.7 ms per
applicant against SurvSHAP(t)'s 2.56 s, a factor of
about 381.
For 100,000 declined applicants that is 11 minutes versus
71 hours. The speed was never in doubt; the agreement was,
and it is the agreement that failed.

**TreeSHAP is kept, but not for notices.** The module stays in the codebase as a
fast screening tool -- ranking which applicants to review, checking a batch for
drivers that look wrong -- and every row that carries reasons records which
explainer produced them, so a file can never be ambiguous about it. Nothing that
states a reason to an applicant uses it.

**Consequence: the failure branch recorded in 7.0b is what bulk runs do.**
SurvSHAP(t) is seeded per applicant from the row id, so reasons no longer depend on
batch membership; bulk runs explain every rejected applicant rather than a capped
subset; the run is resumable and reports time remaining; and the
`reasons not generated` marker means only that a run was stopped early.

## 7b. What Lending Club's own pricing adds

Both models trained on the full labelled dataset with the out-of-time split
(train 884,664 loans from 2007-2015,
test 1,373,126 from 2016-2018), quarterly bins,
identical in every respect except the feature set. `holdout_lcgrade` adds `grade`,
`sub_grade` and `int_rate`; `holdout_nograde` is the primary specification.

| model | with grade / rate | primary (no grade) | gap |
|---|---|---|---|
| discrete hazard, concordance | **0.7158** | 0.6936 | 0.0222 |
| Cox, concordance | **0.7006** | 0.6602 | 0.0404 |
| discrete hazard, 12-month AUC | 0.7264 | 0.7025 | 0.0239 |
| discrete hazard, 36-month AUC | 0.6403 | 0.6417 | -0.0014 |
| discrete hazard, IBS (lower better) | 0.0978 | 0.0992 | -0.0014 |

**Lending Club's grade and rate are worth about 0.022 concordance to the tree model
and 0.040 to Cox.** That is the largest single feature effect measured anywhere in
this project -- bigger than the whole account-counts block, and roughly two thirds
of the gap between the primary model and a coin flip's distance from it. The
exclusion decided in section 0 therefore costs real accuracy, and saying so is the
point of measuring it.

Three things the table says beyond the headline:

* **Cox gains twice as much as the tree model** (0.040 against 0.022). `grade` is an
  ordinal summary of exactly the interactions a linear model cannot express and a
  GBM partly recovers on its own, so the benchmark flatters the weaker learner.
* **The gain is concentrated early.** At 12 months the tree model gains
  0.0239 AUC; at 36 months it gains
  -0.0014 -- nothing, or very
  slightly negative. Lending Club's pricing sorts who defaults *soon*; over a full
  term the primary features catch up. A single-number concordance hides that, which
  is the argument for the time-dependent view this project is built on.
* **The exclusion is still right, for reasons accuracy cannot settle.** `grade` is
  another model's output, so including it means partly predicting Lending Club's
  underwriter rather than default, and a notice whose principal reason is "your
  grade" is exactly the disclosure Regulation B's commentary forbids
  (section 3(b)). The cost is now measured rather than assumed, and it is the price
  of a model whose reasons can be disclosed.

## 7c. The credit score is missing from every model, and that is a defect

`fico_range_low` and `fico_range_high` are present on every accepted loan, are
pre-decision applicant facts, and are lawful to disclose -- FCRA 615(a) expects the
score and its key factors in an adverse-action notice, and this project's notice
template has a slot for them. They are in **no** trained model.

The cause is not a decision. `default_spec` drops the two raw FICO bounds as
superseded by `fico_midpoint`, and it is called on the *raw* columns, before
`add_derived_features` computes `fico_midpoint`. The intersection with "columns
actually present" then removes it, silently. The same bug removes
`emp_length_years`, `installment_to_income`, `loan_to_income` and `log_annual_inc`.
So five features -- including the single most standard credit-risk variable there
is -- were dropped by an ordering mistake, and the results in sections 2, 3, 6 and
7b were all produced without them.

Consequences worth stating plainly:

* Section 7b's measured cost of excluding `grade` is an **upper bound on what the
  exclusion costs a competent model**: some of what `grade` contributes is FICO,
  which the primary model never had either. A fair benchmark would give both sides
  the score.
* Every notice's FCRA score block reports a FICO the model never used.
* The ablation in section 7d cannot speak for a feature that was not in the model.

Stage 2 now takes `--with-derived`, which builds the spec after the derived features
exist. It is **off by default**: turning it on changes the model's inputs, so it
belongs to a new tag and a retrain, not to a silent correction of existing results.
`--drop-features` was added alongside it for the unpriced variants in 7e.

## 7d. What an absent column actually costs

Measured by scoring the holdout with one feature, or one group of features, set to
**missing** -- which is exactly what a scoring run does when a column is absent from
an uploaded file. Not an importance measure: a feature can matter and still cost
little when absent, because the trees route around it through correlated columns.

**An earlier version of this table is superseded.** The first ablation was run
against `02_models_holdout.pkl`, which turned out to hold a 200k dev-sample model
with monthly bins rather than the pre-registered full-data holdout (see the tag
incident below). Those files are kept, renamed to `..._holdout_devsample`, and their
JSON carries a `note` saying they are superseded. The numbers below replace them and
come from **`--model-tag full`**, the model the dashboard actually scores with:
50,000 test rows, baseline concordance **0.6927**, 12-month AUC **0.7045**.

### Groups, worst first

| group | features | concordance drop | 12m AUC drop |
|---|---|---|---|
| loan_structure | 2 | 0.0282 | 0.0324 |
| income_and_burden | 2 | 0.0232 | 0.0253 |
| account_counts | 14 | 0.0231 | 0.0229 |
| application_descriptors | 5 | 0.0172 | 0.0187 |
| utilisation | 7 | 0.0168 | 0.0136 |
| recency_months | 11 | 0.0156 | 0.0205 |
| inquiries | 2 | 0.0057 | 0.0087 |
| geography | 1 | 0.0039 | 0.0050 |
| balances_and_limits | 6 | 0.0025 | 0.0025 |
| delinquency_and_public_record | 12 | 0.0011 | 0.0015 |

### The fifteen most expensive single features

| feature | concordance drop | 12m AUC drop |
|---|---|---|
| installment | 0.0187 | 0.0229 |
| loan_amnt | 0.0169 | 0.0178 |
| annual_inc | 0.0167 | 0.0164 |
| application_type | 0.0094 | 0.0110 |
| acc_open_past_24mths | 0.0072 | 0.0095 |
| purpose | 0.0063 | 0.0095 |
| mo_sin_old_rev_tl_op | 0.0054 | 0.0060 |
| dti | 0.0047 | 0.0056 |
| bc_util | 0.0044 | 0.0050 |
| addr_state | 0.0039 | 0.0050 |
| tot_hi_cred_lim | 0.0035 | 0.0038 |
| revol_bal | 0.0033 | 0.0036 |
| mths_since_recent_inq | 0.0030 | 0.0053 |
| total_bc_limit | 0.0023 | 0.0002 |
| revol_util | 0.0023 | 0.0026 |

**Lender pricing, reported separately as asked.** The pricing group is
`grade`, `sub_grade`, `int_rate`, `installment`, and **three of those four are not
model features at all** -- excluded by the section 0 decision -- so the group reduces
to `installment` alone, at **0.0187**
concordance. On the full model that makes it the single most expensive column to
lose, ahead of `loan_amnt` and `annual_inc`. `installment` is computed by the lender
from the rate it assigned, so the model's most valuable input is partly a record of
Lending Club's own pricing decision. For a genuinely unpriced applicant it can be
derived from `loan_amnt`, `int_rate` and `term` by the standard amortisation formula
-- but only once a rate exists, which is the circularity section 7b describes.

**Geography.** `addr_state` costs
0.0039 concordance and
0.0050 AUC. It is the one feature
that may never be disclosed as a reason (section 3(b)), so a variant without it is
cheap to run and is being measured separately.

### Required and optional, with the thresholds fixed here

The thresholds are stated before use, and they are deliberately not "large drop =
required", because measurement does not support that framing: no single feature's
absence takes the model below usable. The worst case above leaves concordance at
0.6740.

| tier | rule | features | what a run does |
|---|---|---|---|
| **Required** | concordance drop >= 0.010, **or** structurally necessary | installment, loan_amnt, annual_inc | refuse the file, unless the column can be derived exactly (7e) |
| **Optional, costed** | drop 0.002 to 0.010 | 12 features | score, and report the measured cost per missing feature |
| **Optional, free** | drop < 0.002 | 46 features | score, and note the absence without a cost claim |

`installment` sits in the required tier but is **required-or-derivable**: a file with
`loan_amnt`, `int_rate` and `term` satisfies it by derivation, which is what lets a
raw-applicant file through.

**Cumulative budget.** Drops are not additive, so a file missing several optional
columns is reported with the sum of their individual drops as a **conservative upper
bound**, labelled as such: amber above 0.010 concordance, red above 0.030 (4.3% of
baseline). Where a whole group is absent, the group figure is used instead, because
that was measured jointly.

## 7e. The 2016 loan-amount change is real drift, not a data error

The drift check flags `loan_amnt` on out-of-time data. It is right to.

| period | loans | max | 99.9th percentile | mean | share above 35,000 |
|---|---|---|---|---|---|
| 2007-2015 | 884,664 | 35,000 | 35,000 | 14,773 | 0.000% |
| 2016-2018 | 1,373,126 | 40,000 | 40,000 | 15,236 | 3.260% |

Pre-2016 the maximum is **exactly** 35,000 with not one loan above it; from 2016 the
maximum is **exactly** 40,000 and 3.26% of loans exceed the old cap. That is a
product change -- Lending Club raised its maximum personal-loan size -- not a parsing
fault, a unit change or a corrupted column. Two consequences:

* A drift alert on `loan_amnt` for a 2016+ file is **correct and expected**, and the
  dashboard should not be read as reporting a data problem. Scoring 2016+ loans with
  a model trained to 2015 means extrapolating above the largest loan the model ever
  saw, for about one applicant in thirty.
* Any range check learned from training data will flag those loans as out of range.
  That is also correct: they are outside the training range. The run scores them and
  flags them, and does not clip them, which is the behaviour cleaning already has
  (`clip_numeric` off).

## 7f. Three pre-decision facts the model ignores, and one horizon that does not fit

### Why credit score, employment length and term are absent

All three are known about an applicant before any decision, and none reaches a
model. The reasons differ, and only one of them is a decision:

| fact | in the data as | why it is absent |
|---|---|---|
| credit score | `fico_range_low`, `fico_range_high` | **a defect.** `default_spec` drops the raw bounds as superseded by `fico_midpoint`, but runs before `add_derived_features` computes it (section 7c) |
| employment length | `emp_length` | **the same defect.** Superseded by `emp_length_years`, which the spec never sees |
| loan term | `term`, parsed to `term_months` | **an omission.** `term_months` is built in Stage 1 and used for stratified sampling and for the term-overrun rule, but it is in no schema feature list, so nothing ever put it in a model |

None of the three is excluded on fair-lending or leakage grounds. Score and
employment length are lawful to disclose and are standard credit-risk inputs; term
is chosen by the applicant and is the single strongest determinant of how long the
loan is exposed. The model has been predicting 36-month default probability without
knowing whether the loan runs for 36 months or 60.

Stage 2 now takes `--with-derived` (score, employment length and the three ratios)
and `--add-features term_months`, both off by default so no existing result moves,
and both leakage-checked. Section 7g reports what they are worth once the variants
are trained.

### The decision rule uses a 36-month horizon for 60-month loans

Measured on 60,000 loans from the full model's test split:

| | 36-month loans | 60-month loans |
|---|---|---|
| share of the portfolio | 71.4% | 28.6% |
| mean predicted default probability by 36 months | 0.1835 | 0.2181 |
| mean predicted default probability by 60 months | 0.2570 | 0.2997 |
| observed default rate | 0.1006 | 0.1619 |
| rejected by the current rule (PD36 >= 0.30) | 14.5% | 23.7% |

**Using a 36-month probability for a 60-month loan is not appropriate as a measure
of that loan's risk.** It stops counting two years before the loan does, and the
model puts about 8 percentage points of default probability in that window
(0.2181 -> 0.2997). The current rule is not blind to term -- 60-month loans are
already rejected at 23.7% against 14.5% -- but that happens through correlated
features, not because the horizon matches the exposure.

**What it would cost to fix naively.** Applying the same 0.30 cutoff to the
probability at each loan's own term end rejects **44.0%** of 60-month loans instead
of 23.7%. That flips 20.3% of 60-month decisions and 5.8% of all decisions. Such a
change is a credit-policy decision dressed as a technical correction, so it is not
made here.

**Two term-aware rules worth considering, neither applied:**

1. **Probability over the actual term, with term-specific cutoffs.** Reject on
   PD-at-term-end, using 0.30 for 36-month loans and **0.409** for 60-month loans --
   the cutoff measured to leave today's 60-month rejection rate unchanged. The
   quantity being compared then matches the exposure, and the change in who is
   rejected is deliberate rather than incidental.
2. **An annualised rule, one number for every term.** Reject when the default
   probability per year of exposure exceeds a single threshold. Today's 0.30 over
   three years is 0.10 per year, which for a 60-month loan means a PD60 cutoff of
   0.50. This is the cleaner rule -- term-neutral by construction, and it states the
   policy in a unit a credit committee can argue about -- but its effect on the
   60-month rejection rate has not been measured yet, and it must be before it could
   be adopted.

The published threshold is unchanged until that decision is taken.

## 7g. The schema layer: what an uploaded file is allowed to look like

Until now a file was refused unless its columns were named exactly as in training.
That is the wrong failure: the model does not care what a column is called, only what
is in it. `creditsurv/schema_match.py` and `creditsurv/derive.py` sit in front of the
model and change nothing about it.

**Recognition** uses three kinds of evidence, in descending order of trust: a written
synonym table, normalised-name similarity (case, punctuation and filler words
removed, camel case split), and the column's own content. Content can raise
confidence and, more usefully, **veto** a name: a column called `annual_income`
holding values between 0 and 1 is not an income, and is refused rather than renamed.
Two columns are never mapped onto one feature -- that choice belongs to a person, so
neither is pre-selected and the conflict is stated. Nothing is applied until it is
confirmed; the upload page shows the proposal with a confidence and a reason per
column, editable, and the command line takes `--map column=feature`.

**Derivation** computes what follows exactly from other columns: the monthly
instalment from amount, rate and term by the standard amortisation formula, the FICO
midpoint from the reported range, the band top from the band bottom, the two income
ratios, and `term_months` from a term written as "36 months". These are arithmetic,
not imputation, and the distinction is the point: a derived value is the number the
lender's own system would produce, while an imputed one is a guess that makes the
output look more certain than it is. Every derived column is reported as derived, in
the dashboard and in `run_summary.csv`.

**Required and optional** come from the measured ablation in 7d, with the thresholds
recorded there: required at or above 0.010 concordance, optional-with-a-cost between
0.002 and 0.010, optional-and-free below that. For the full model that makes
`installment`, `loan_amnt` and `annual_inc` required -- and `installment` is
required-or-derivable, which is what lets a raw-applicant file through. A file
missing an optional feature is scored, and the dashboard reports the measured cost of
its absence rather than an opinion about it.

**Explanations never cite a column the file did not contain.** A feature absent from
the upload is dropped from the attributions before reasons are selected, because
"your revolving balance" is not a reason anyone can act on when no revolving balance
was supplied.

**Messages** say what happened and what to do: what was recognised and as what, what
was derived and how, what is missing, and for each missing column either "not in
file" or which inputs a derivation would have needed -- for example
`Recognised: open_credit_lines -> open_acc` and
`Missing: revol_bal (not in file); installment (needs int_rate)`.

## 7a. Cheaper SurvSHAP(t) settings for bulk runs

<!-- keep:preregistration-settings -->
### 7a.0 Pre-registration (written before the comparison was run)

**Why.** TreeSHAP failed 7.1, so bulk runs keep SurvSHAP(t) at 2.56 s per applicant
-- 71 hours for 100,000 declined applicants. Parallelism divides that by the number
of cores; a cheaper setting would divide it again. The current settings are
`nsamples=600`, `n_background=100`, chosen in section 3 for a different purpose: the
headline number there was a **rank correlation over all 73 features**, which needs a
settled full ranking. A notice needs only the top four reasons, which is a weaker
requirement, so the settings may be over-specified for this use -- or may not be.
This tests it rather than assuming either way.

**What is compared.** The same 1,000 declined applicants from 7.1, the same notice
builder, full settings (`nsamples=600`, `n_background=100`) as the reference, against
these candidates:

| candidate | nsamples | n_background | relative cost |
|---|---|---|---|
| C1 | 300 | 100 | about 1/2 |
| C2 | 600 | 50 | about 1/2 |
| C3 | 300 | 50 | about 1/4 |
| C4 | 146 (the 2p floor) | 25 | about 1/16 |

**Acceptance bar, fixed before running:**

| # | Criterion | Bar |
|---|---|---|
| S1 | Applicants whose top stated reason matches the full-settings run | >= 90% |
| S2 | Mean top-4 stated reason-set overlap with the full-settings run | >= 0.85 |

Both must hold. The cheapest candidate that passes both is adopted **for bulk runs
only**; the research stages (3, 3b, 3c) and single-applicant work keep the full
settings, because their headline numbers are the full-ranking ones that justified
those settings. If no candidate passes, bulk runs keep the full settings and the
result is recorded as a negative one.

**How the numbers are to be read.** Section 7.1 measured SurvSHAP(t) against itself
under two seeds at the full settings: **95.3%** top-1 agreement and **0.94** mean
top-4 overlap. That is the ceiling any cheaper setting is competing with, and it is
also why S1 is set at 90% and S2 at 0.85 rather than higher: a candidate cannot be
asked to beat the method's own reproducibility. A candidate scoring near 95%/0.94 is
indistinguishable from the full settings; one scoring at the bar is measurably worse
but within the tolerance fixed here in advance.

**One thing this comparison cannot tell us.** Both sides of it are sampled, so a
candidate that agrees with the full settings agrees with *one draw* of them. The
7.1 ceiling bounds how much of any gap is noise; it does not remove the noise from
this measurement.
<!-- /keep:preregistration-settings -->

*Not run yet.* The bar above was fixed before the comparison existed.
