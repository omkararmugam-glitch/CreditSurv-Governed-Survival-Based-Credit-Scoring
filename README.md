# creditsurv — survival-based credit scoring with regulation-ready explainability

A time-to-default survival model for Lending Club loans, paired with
time-dependent feature attribution (SurvSHAP(t)-style) and a **diagnostic-gated**
reject-inference stage that only applies a correction when a pre-registered test
says selection bias is actually present.

No frontend, no API. A Python package plus terminal scripts; all results are
written to `outputs/`.

## Why survival rather than a binary classifier

A binary "defaults within N months" target throws away *when* a loan fails — a
borrower who stops paying in month 2 and one who stops in month 23 are the same
label — and it forces every still-performing loan into an arbitrary bucket. Since
Lending Club's 2018 vintages have only a few months of observation, a fixed
window either discards them or mislabels them as good. A survival model uses them
correctly as censored observations, and produces a full curve per borrower
instead of one number.

## Status

All five stages are built and have been run on the full dataset (2,257,790
loans). 300 tests pass with no real data required. Results are in
[FINDINGS.md](FINDINGS.md).

| Stage | Full-data run |
|---|---|
| 1 — Data prep & survival labelling | done — 2,257,790 loans retained |
| 2 — Cox + discrete-time hazard | done — C-index 0.670 / 0.697 on 451,558 test loans |
| 3 — SurvSHAP(t), naive comparison, adverse action, segments | done — 250 random + 252 grade-stratified borrowers |
| 4 — Selection-bias diagnostic & gated reject inference | done |
| 5 — FINDINGS.md assembly | done — regenerate with `05_report.py --tag full` |

## Setup

The project runs on the **pure-Python survival stack** (`lifelines` + LightGBM),
not `scikit-survival`. That is a deliberate choice forced by this machine, and
the reasoning is recorded in [FINDINGS.md](FINDINGS.md#0-environment-constraint).
The short version: `scikit-survival` requires `ecos`, which has no wheel for
Python 3.13+, and Smart App Control blocks freshly-downloaded compiled binaries
here. A `--system-site-packages` venv reuses the compiled packages already
installed and working, and adds only pure-Python ones.

```bash
python -m venv --system-site-packages .venv
./.venv/Scripts/python.exe -m pip install lifelines pyyaml
```

## Running it

Set the two CSV paths in [config/config.yaml](config/config.yaml) — that is a YAML
file to **edit**, not something to paste into a shell. Alternatively pass them on
the command line, which overrides the config:

```bash
./.venv/Scripts/python.exe scripts/00_ingest.py     --accepted-csv path/to/accepted_2007_to_2018Q4.csv     --rejected-csv path/to/rejected_2007_to_2018Q4.csv
```

The dataset is [wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
on Kaggle (~1.1 GB zipped, ~3.3 GB as two CSVs).

```bash
# Stage 1a — stream both CSVs to Parquet (run once; ~1.6 GB each)
./.venv/Scripts/python.exe scripts/00_ingest.py --config config/config.yaml

# Smoke-test the ingest on a slice first if you prefer
./.venv/Scripts/python.exe scripts/00_ingest.py --row-limit 50000

# Stage 1b — build the survival target + dev sample
./.venv/Scripts/python.exe scripts/01_build_labels.py

# Sensitivity variants on the labelling rule
./.venv/Scripts/python.exe scripts/01_build_labels.py --late-as-event --tag late_event
./.venv/Scripts/python.exe scripts/01_build_labels.py --event-lag 5 --tag lag5

# Stage 2 — fit Cox + discrete-time hazard, evaluate, write figures
./.venv/Scripts/python.exe scripts/02_train_models.py              # dev sample
./.venv/Scripts/python.exe scripts/02_train_models.py --full       # full dataset
./.venv/Scripts/python.exe scripts/02_train_models.py --with-lc-grade   # benchmark

# Stage 3 — SurvSHAP(t), naive-SHAP comparison, adverse action, segments
./.venv/Scripts/python.exe scripts/03_explain.py

# Stage 4 — selection-bias diagnostic, then correction only if warranted
./.venv/Scripts/python.exe scripts/04_reject_inference.py

# Stage 5 — write results into FINDINGS.md sections 2-5
./.venv/Scripts/python.exe scripts/05_report.py

# Inspect the data at any point (read-only)
./.venv/Scripts/python.exe scripts/inspect_data.py --what all

# Tests (no real data needed)
./.venv/Scripts/python.exe -m pytest -q
```

## Out-of-time holdout (one command)

`run_holdout.ps1` runs the whole holdout sequence (train 2007–2015, test
2016–2018: Stages 2, 3, 3c, stratified 3, 3b, 4, a Stage 5 report and a provenance
check), stopping at the first failure with no retry and no cleanup:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_holdout.ps1 -Size Small      # default
powershell -ExecutionPolicy Bypass -File .\run_holdout.ps1 -Size Small -PlanOnly  # print commands only
```

`-Size Small` and `-Size Medium` write under their own tags and only *preview*
FINDINGS section 6 (`05_report --dry-run`). Only `-Size Full` writes it. The command
list lives in `src/creditsurv/plan.py`, shared with the UI below, and
`tests/test_plan.py` pins it to exactly what the wrapper ran before that module existed.

## Running the UI

A local Streamlit app wraps the same scripts. It is optional; everything it does
is also available from the command line above.

```bash
./.venv/Scripts/python.exe -m streamlit run app/app.py  # from the project root
```

Then open http://localhost:8501. Streamlit and psutil come from the system
site-packages in this venv; elsewhere, install them with `pip install -e ".[ui]"`. The app binds to `localhost` only and sends no
usage statistics (`.streamlit/config.toml`).

| Page | What it does |
|---|---|
| **Score applicants** (home) | Upload a CSV; it is checked, cleaned, profiled, scored and explained, then written out as CSVs with Regulation B reasons, notices, a cleaning report and a drift check. Described below. |
| **Overview** | Every result tag and stage, with its provenance status: 🟢 verified (stamp re-hashes cleanly), 🟠 unverified (produced before stamping), 🔴 changed (an input or model it recorded has since been replaced), ⚪ not run. |
| **Run pipeline** | Runs the holdout sequence from `creditsurv.plan` (identical to `run_holdout.ps1`). Defaults to **Small** with **overwrite off**. Before starting, it lists any existing outputs that would make a stage refuse (the same check the scripts do); ticking Overwrite shows exactly which files are replaced and asks for confirmation. Tags `full` / `dev` are refused. |
| **Results viewer** | Tables, figures, the adverse-action notice and raw JSON for one tag, with its provenance status shown first. |
| **FINDINGS** | FINDINGS.md by section, protected keep-blocks marked, and `git diff` against the last commit, warning if anything above section 6 changed. |

### Score applicants (the home page)

Drop a CSV of applicants on the home page and the app runs the whole flow by
itself: checking the file, cleaning it exactly as the training pipeline does,
scoring every row with the trained survival model, explaining the rejected ones
with SurvSHAP(t), and writing finished files.

**Decision rule.** The model returns a probability, not a decision. The policy
that turns one into the other lives in `decision:` in
[config/config.yaml](config/config.yaml), is shown on the dashboard, and is
written into every `run_summary.csv`:

```yaml
decision:
  reject_at_or_above: 0.30   # 36-month default probability
```

The default was chosen on the full model's 451,558-loan test split. Rejecting at
0.30 declines 17.0% of applicants and takes the approved population's observed
default rate from 11.8% to 8.9%, while the rejected group defaults at 25.9%. It
is just above the 80th percentile of predicted risk. Every borrower in that
population had already been approved by Lending Club, so a real applicant pool is
riskier — see the Stage 4 selection-bias result.

**What a file needs.** Ten columns are required: `loan_amnt`, `installment`,
`annual_inc`, `dti`, `open_acc`, `revol_bal`, `delinq_2yrs`, `inq_last_6mths`,
`purpose`, `home_ownership`. The model's other 51 features are optional; where
they are absent the run is scored anyway and flagged as degraded, with the
coverage reported on screen and in every output file. A fixed alias table
(`creditsurv.batch.ALIASES`) renames common spellings such as `loan_purpose` and
`state`, and says on screen which ones it mapped. Columns that are not model
features are ignored and listed — including `credit_score` and `emp_length_years`,
which no trained model uses (see FINDINGS on the Stage 2 feature spec).
[tests/fixtures/sample_applicants.csv](tests/fixtures/sample_applicants.csv) is a
25-row example to try it with.

**What comes out**, in `outputs/runs/<timestamp>_<file>/` (never overwritten, not
in git) and as download buttons:

| file | contents |
|---|---|
| `scored_applicants.csv` | every applicant: `pd_12m`, `pd_36m`, decision, threshold, top three reasons, data-quality flags |
| `approved_applicants.csv` | approved rows only |
| `rejected_applicants.csv` | rejected rows with up to four Regulation B reasons, their attributions, the internal fair-lending flag and the notice file name |
| `adverse_action_notices.zip` | one formatted ECOA/Reg B notice per explained rejected applicant |
| `run_summary.csv` | one row: file, row count, model tag and SHA-256, threshold, approval rate, feature coverage, timings |
| `provenance.json` | the same provenance stamp the pipeline stages write |

**Large files.** An upload of `decision.background_above_mb` or more (default 25 MB)
runs in a detached background process via [creditsurv.runner](src/creditsurv/runner.py)
— the same machinery the pipeline page uses — so the browser can be closed and the
run keeps going; the page tails its log and picks the finished result up from disk.
Smaller files stay inline, because a background run reloads the model and its SHAP
background from scratch (20–40 s) that the page already has cached. "Always run in
the background" in the sidebar forces it either way. The same script works on its
own:

```bash
./.venv/Scripts/python.exe scripts/06_score_upload.py --file applicants.csv
./.venv/Scripts/python.exe scripts/06_score_upload.py --file applicants.csv \
    --model-tag holdout --threshold 0.25 --max-explained 500
./.venv/Scripts/python.exe scripts/06_score_upload.py --file applicants.csv \
    --chunk-rows 10000 --explainer auto        # less memory; faster reasons
```

Exit codes: 0 finished, 2 bad arguments or missing input, 3 the file or the model was
refused (the message says which and how to fix it). The upload cap is
`server.maxUploadSize` in `.streamlit/config.toml`, currently 500 MB; note the file is
held in memory as bytes and again as a DataFrame, so a file that size needs several GB
of RAM.

**Speed and memory.** Scoring is effectively free (50,000 applicants in about 2
seconds). The file is read, cleaned, scored and explained `decision.chunk_rows`
rows at a time (default 50,000) and the outputs are appended as it goes, so peak
memory follows the block size rather than the file size: measured on this machine,
a 20.3 MB / 190,000-row file peaked at **791 MB before** the change and **514 MB
after**, and doubling the file to 40.6 MB moved the peak by 4 MB (518 MB). About
350 MB of that is the model and its SHAP background, which is fixed. A smaller
block trades speed for memory: at `chunk_rows: 10000` the same file peaked at
357 MB and took 26 s instead of 21 s.

Scores, decisions and cleaning flags do not depend on the block size. The stated
*reasons* can: SurvSHAP(t) draws coalitions per call, so applicants explained in
one block together and in another separately may swap two near-tied reasons — the
per-applicant instability FINDINGS already records. The block size is written into
`run_summary.csv` and the provenance stamp for that reason.

**Explanations are capped, and the shortfall is stated.** SurvSHAP(t) costs about
2.7 s per applicant on the full model, so only *rejected* applicants are explained
— approved rows never consume the budget — in file order, up to
`decision.max_explained` (default 100). A rejected applicant past the cap is marked
`reasons not generated: explanation cap reached (decision.max_explained=N)` in
every output file, `run_summary.csv` carries `n_rejected_without_reasons`, and the
dashboard shows that count in red with the time it would take to clear it. A
declined applicant with no stated reasons is a compliance gap, so it is never a
blank cell. A 25-row fixture with five notices finishes in about 9 seconds.

What it does not do: it has no modelling code of its own, never passes
`--overwrite` unless you tick it, and never writes FINDINGS.md except through
`05_report.py` on the pre-registered Full run. Each stage still writes its own
provenance stamp.

Runs are separate background processes, so leaving or refreshing the page does not
stop them. Each run logs to `outputs/logs/runs/<time>_<tag>/` (git-ignored), and a
per-tag lock refuses a second run on the same tag while one is live. If the machine
sleeps or the process is ended, the run shows as **interrupted**; nothing is cleaned
up, so check the log and re-run.

## Cleaning and EDA

### Cleaning lives in one module

[src/creditsurv/cleaning.py](src/creditsurv/cleaning.py) is the single source of
truth for what happens to a *value* between the raw file and the design matrix.
Both training (Stage 2) and the dashboard's scoring path call it, so the two can no
longer disagree — before it existed, `$85,000` parsed to 85000.0 in training and to
missing when scoring.

The division of labour is deliberately narrow, so nothing moved twice:

| owns | what |
|---|---|
| `cleaning.py` | numeric text coercion, category alignment, optional clipping and rare-level pooling, duplicate rows, and the report |
| `build_design_matrix` | encoding: one-hot vs native categorical, the Cox median fill and standardisation, missing-indicator columns |
| labelling stage | row exclusions that need the outcome (unusable status, missing dates, term overrun); their counts are read into the report |

Learned values — category levels, percentile ranges, medians, clip bounds — are
fitted on the **training split only**, saved in the model bundle by Stage 2, and
reapplied unchanged when scoring. `clean()` has no code path that can refit, and
[tests/test_cleaning.py](tests/test_cleaning.py) pins that an upload of extreme
values leaves them untouched. A bundle trained before this module existed carries
none, so they are re-fitted from that bundle's own training split and the run says
so.

Every rule is a config switch with its reason in
[config/config.yaml](config/config.yaml) under `cleaning:`. The default is
**`v1-parity`**: the rule set the current trained models were produced under, which
reads, counts and flags but changes no value. The four switches that would alter a
model input (`clip_numeric`, `unseen_category_to_other`, `rare_category_min_count`,
`drop_duplicate_ids`) are off; turning one on invalidates the fitted models and
requires a retrain plus a note in FINDINGS saying which results used which cleaning
version. Stage 2 prints a warning when the policy in force would alter inputs, and
a test asserts the shipped config equals the parity policy.

Stage 2 writes `outputs/tables/00_cleaning_report_<tag>.json` per run: rows in and
out, rows dropped per rule, values imputed and clipped per column, unseen
categories, and the label-stage drop counts.

### EDA stage

[scripts/01b_eda.py](scripts/01b_eda.py) profiles the labelled training data and
writes `outputs/eda/<tag>/`: an `index.html` report plus every underlying CSV and
PNG. It is read-only by construction — it loads Parquet, writes tables and charts,
and never writes back to a data file. Overwrite-gated and provenance-stamped like
every other stage.

```bash
./.venv/Scripts/python.exe scripts/01b_eda.py --tag dev        # 200k dev sample
./.venv/Scripts/python.exe scripts/01b_eda.py --tag full       # labelled dataset
./.venv/Scripts/python.exe scripts/01b_eda.py --tag dev --sample 50000
```

It covers dataset overview, missing values, numeric distributions and summary
stats, categorical frequencies, outlier counts by two conventional rules, a
correlation heatmap with the highly correlated pairs listed, default rates by
grade, term, purpose, income band, home ownership and vintage, Kaplan-Meier curves
by grade and term, and a short list of observations — phrased as observations, not
conclusions. Accepted-vs-rejected numbers are *read* from
`04_reject_inference_<tag>.json` rather than recomputed, so this report and FINDINGS
section 4 cannot quote different selection-bias figures.

The dashboard's Data Profile tab calls the same [eda.py](src/creditsurv/eda.py)
functions on an uploaded file, so a figure here and a figure there always mean the
same thing.

### Drift check

[src/creditsurv/drift.py](src/creditsurv/drift.py) compares an upload with the
model's training split, feature by feature: population stability index for numeric
features, total variation distance for categorical ones, on the conventional bands
(below 0.10 stable, 0.10–0.25 moderate, above 0.25 large).

Below `decision.drift_min_rows` (default **500**) drift is not assessed at all and
the status is grey, "too few rows to assess drift". The reason is arithmetic: with
`k` bins and no real shift, PSI's expected value under the null is about
`(k - 1) / n`. This module uses 10 quantile bins plus a missing bin, so unshifted
data scores a median PSI of about 0.08 at 100 rows — within a whisker of the
"moderate" band, which small files would therefore trip constantly — and about 0.02
at 500 rows, a fifth of the band. Both figures are pinned by a test. A grey status
says nothing is wrong with the file; there is simply not enough of it to tell. The dashboard shows
green, amber, red or grey with the shifted features listed, and every run downloads
`data_drift.csv` alongside `cleaning_report.csv`.

The profile and the drift check run on a **uniform random sample of up to 50,000
rows drawn across the whole file** (priority sampling, one pass, bounded memory),
never on its opening rows. Loan files usually arrive sorted by date, so a
first-block sample would describe one vintage and report drift the file does not
have — a test builds exactly that file and checks the verdict follows the file
rather than its start. The count used is reported as `profiled_rows`. A feature that cannot be compared
— absent from the upload — is reported as unknown and holds the overall status at
amber rather than counting as evidence of stability.

## Re-running stages safely

Stages 1–4 (and `03b`) **refuse to overwrite existing results**. If any output
for the chosen `--tag` already exists, the script lists it and exits with code 4
before loading any data. To replace results deliberately, pass `--overwrite`;
to keep both, use a different `--tag`. This matters most for files that are not
in git — the fitted models in `outputs/models/` and the data in `outputs/data/` —
which cannot be recovered once replaced. Note in particular that a Stage 1
sensitivity flag without `--tag` would otherwise replace the primary dataset.

Every results JSON carries a `provenance` block: the git commit (and whether the
code was modified at run time), SHA-256 hashes of every input file and of any
model written, the config, and library versions. To check that no result's
inputs have been replaced since it was produced:

```bash
./.venv/Scripts/python.exe scripts/check_provenance.py     # exit 1 on any change
```

Results produced before stamping was added are covered by
`outputs/provenance_baseline.json`, a recorded hash of every untracked model and
data file.

`scripts/05_report.py` regenerates FINDINGS.md §2–§5 but refuses to overwrite a
section that was edited by hand since it last wrote it. Hand-written text inside
those sections belongs in a `<!-- keep:NAME --> … <!-- /keep:NAME -->` block,
which survives regeneration.

## Layout

```
src/creditsurv/
  io/         loaders.py   chunked CSV -> Parquet ingest
              schema.py    column allowlist + leakage guard
  labeling/   status_rules.py     loan_status -> event/censor decision table
              survival_target.py  (duration_months, event) construction
  features/   build.py     design matrices (Cox one-hot vs LightGBM native)
              encoders.py  ordinal / derived feature transforms
  models/     cox.py              lifelines Cox PH baseline
              discrete_hazard.py  LightGBM person-period hazard model
              evaluate.py         c-index, IPCW time-dependent AUC, Brier, calibration
  explain/    survshap.py      SurvSHAP(t) + efficiency-axiom check
              naive_shap.py    time-agnostic control condition
              compare.py       Stage 3(a) disagreement measures
              adverse_action.py  ECOA / Reg B notice generation
              segments.py      Stage 3(c) stability across segments
  reject_inference/  diagnostics.py  pre-registered selection-bias gate
                     correction.py   inverse-propensity reweighting
  reporting/  tables.py  figures.py
  cleaning.py the one cleaning module, used by training AND scoring
  eda.py      read-only profiling (EDA stage + dashboard profile tab)
  drift.py    PSI / category-share drift of an upload vs training
  plan.py     the holdout stage commands (shared by run_holdout.ps1 and the UI)
  runner.py   background run + log + per-tag lock (UI)
  status.py   read-only stage status and overwrite pre-flight (UI)
scripts/      00..05 one per stage, plus 01b EDA, 03b bootstrap, 06 batch scoring,
              inspect_data.py, check_provenance.py
app/          Streamlit UI (app.py + views/), a thin layer over the scripts
tests/        459 tests, synthetic fixtures only — no real data required
outputs/      data/ models/ (gitignored)   figures/ tables/ (in git: the evidence)
```

## Two things worth knowing before reading results

**Leakage.** ~40 of the accepted file's 151 columns are recorded *after*
origination, and several (`recoveries`, `total_rec_prncp`, `last_fico_range_low`)
are near-perfect outcome proxies. `io/schema.py` uses an allowlist, so unknown
columns fail closed, and `assert_no_leakage` runs on every feature matrix.

**Lending Club's own grade is excluded from the primary model.** `grade`,
`sub_grade` and `int_rate` are the output of *another* model. Including them means
partly predicting Lending Club's underwriter instead of default, and it makes
adverse-action reasons circular — "your grade was low" is not a permissible
ECOA/Reg B reason. They are retained for a `with_lc_grade` benchmark variant.
