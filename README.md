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
loans). 265 tests pass with no real data required. Results are in
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
scripts/      00..05 one per stage, plus inspect_data.py
tests/        264 tests, synthetic fixtures only — no real data required
outputs/      data/ (gitignored)  models/  figures/  tables/
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
