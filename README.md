# CreditSurv — Governed Survival-Based Credit Scoring

A credit scoring system that predicts *when* a loan is likely to default,
not just *whether* it will — paired with explanations a lender is legally
required to give, and a governance layer that stops an unapproved or
flawed model from being used for real decisions.

---

## Why survival, not a binary classifier

Most credit scoring predicts a single outcome: default or no default,
within a fixed window. This throws away *when* a default happens — a
borrower who stops paying in month 2 and one who stops in month 23 get
the same label — and it mishandles loans that are still being paid off
when the data was collected, forcing them into an arbitrary bucket.

This system instead predicts a full survival curve per borrower: the
probability they have *not yet* defaulted, at every point over the
loan's term. Loans still performing when observed are handled correctly
as censored data, not discarded or mislabeled.

## What it does

1. **Predicts time-to-default** using a discrete-time hazard model
   (gradient-boosted) and a Cox proportional-hazards baseline.
2. **Explains every decision** using a time-dependent attribution method
   built for survival models, converted into the specific, itemized
   reasons a lender must give an applicant under US lending law (ECOA /
   Regulation B).
3. **Checks for discriminatory bias before correcting for it** — a
   measurable diagnostic decides whether selection bias between accepted
   and rejected applicants is actually present before any correction is
   applied, rather than applying one by default.
4. **Refuses to let a bad model make real decisions.** A model registry
   enforces seven automatically-checked approval rules; every scoring
   run passes nine validation checks before its results are considered
   finished; and applicant-facing notices are structurally separated
   from internal fair-lending review records, so one can never leak into
   the other.

## Results

| Metric | Result |
|---|---|
| Concordance (discrimination), approved model | 0.696 on 451,558 held-out loans |
| Real, unseen 2016–2018 applicants | 86.2% approval, 0% flagged for fair-lending review |
| Cost of removing applicant geography entirely | 0.0011 concordance — negligible |
| Explanation method agreement (two independent runs of the same method) | 0.96 Spearman |
| Explanation method vs. a standard alternative | 0.99 Spearman — indistinguishable at this resolution |
| Fast explanation method tested for bulk use | Failed a pre-registered accuracy bar — kept only as an internal screening tool, never used for applicant-facing reasons |

Every number above is reproduced and explained in detail in `FINDINGS.md`.

## Research grounding

This project is built on current research in survival-based credit
scoring, explainable AI, and reject inference — and goes beyond it in
several specific, tested ways rather than simply applying it.

**Adopted directly.** The core explanation method is a time-dependent
technique designed specifically for survival models, used instead of
standard feature-attribution methods that are known to misrepresent
skewed, time-to-event outcomes. The reject-inference approach follows a
diagnose-before-correcting design: a measurable test decides whether
selection bias is actually present before any correction is applied.

**Tested, not assumed.** A documented concern that standard attribution
methods distort survival outcomes was tested directly on this project's
own data: the two methods agreed more closely with each other than
either agreed with itself across repeated runs, so the concern did not
reproduce here. A separate, documented warning that reject-inference
corrections don't reliably improve accuracy was also tested directly:
bias was detected, a correction was applied, and accuracy did not
improve — exactly as the warning predicts. Both results are reported
regardless of outcome.

**Extended beyond the published work.** Existing reject-inference
research asks whether a correction changes a model's accuracy. This
project asks a further question: does it change *why* the model decides
what it decides? Re-running the explanation step before and after
correction showed the overall ranking of important factors stayed
stable, but two specific features shifted substantially in weight while
remaining outside the top ranks — a finding a pure accuracy comparison
would never surface.

**A real, measured fair-lending result.** Applicant geography ranked
among the strongest features by one importance measure, yet removing it
entirely cost a negligible, measured amount of accuracy. That trade was
tested, not assumed, and the model used by this system excludes
geography as a result.

**A rigorous negative result.** A much faster alternative explanation
method was tested against a pre-registered accuracy bar before being
considered for production use. It failed on one required criterion. It
is kept only as an internal screening tool and is never used for
anything disclosed to an applicant.

**A defect found and fixed.** An earlier version of the model was
missing two legitimate, pre-decision applicant features due to a
specification error. Fixing it measurably improved accuracy and enabled
a model that uses no information from a lender's own pricing decision
and no geography at all, while matching the accuracy of the model that
used both.

**Governance the published research doesn't need, but a deployed system
does.** None of the techniques above address what happens when a model
is incomplete or wrong in production. This project adds a model registry
with automatically-checked approval rules, mandatory post-run validation
on every scoring run, and a strict separation between what an applicant
is legally told and what stays in an internal review record — closing a
real gap this project found in its own early output.

## Architecture

```
  Upload (CSV)
       │
       ▼
  REST API (FastAPI) ── one shared pipeline, two entry points
       │                 (the API/dashboard, and the command line)
       ▼
  Phase 1 — minutes, any file size
  check columns → clean → score → decide → check drift
       │
       ▼
  Phase 2 — explanations, with visible progress
  time-dependent attribution → Regulation B reasons → notices
       │
       ▼
  9 automatic checks must pass before a run is "finished"
       │
       ▼
  Results: decisions, reasons, notices, drift report — all downloadable
```

The Streamlit dashboard is a client of the API, not a direct caller of
the pipeline — it only talks to the API over HTTP, the same way the
command-line tool does. One pipeline, two ways in.

## What it can't do yet, stated plainly

- The proposed no-pricing, no-geography model's approval evidence is
  recorded in `FINDINGS.md`; check there for its current registry status
  before relying on it for anything beyond testing.
- A known limitation in the drift check: data reported before 2013 is
  structurally sparser than later data, which can make the drift check
  read "large" even on a genuinely unaffected file. The mechanism is
  understood and documented, not hidden; a fix is proposed but
  deliberately not applied yet, since changing drift thresholds changes
  what every past result means.
- The fast-explanation shortcut (TreeSHAP) is not used for anything an
  applicant sees, by design, after failing its accuracy bar.

## Setup

LightGBM and SHAP are blocked by Windows Smart App Control on some
machines. This project runs under WSL2 (Ubuntu) as a result.

```bash
wsl --install          # one-time, if not already installed
```

Inside WSL:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[ui]"
sudo apt install -y libgomp1
```

## Running it

From PowerShell, with WSL set up:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_linux.ps1
```

This starts the API (port 8000) and the dashboard (port 8501) together.
Open `http://localhost:8501`. The header of every page shows **API on
Linux (WSL)** in green when correctly connected.

<details>
<summary>Advanced: run the API and CLI separately</summary>

```bash
python -m uvicorn creditsurv.api.app:app --host 0.0.0.0 --port 8000
python -m streamlit run app/app.py --server.headless true
python scripts/06_score_upload.py --file path/to/applicants.csv
```
</details>

## Project structure

```
src/creditsurv/      core pipeline, API, registry, governance
app/                 Streamlit dashboard (a client of the API)
scripts/             command-line entry points
config/              model registry, thresholds
tests/               850+ tests, Windows and WSL
FINDINGS.md          full research log: every experiment, every result
```

## How this was built

Every design choice — architectural and literature-derived — was tested
against real data before being kept. Negative results are documented
with the same rigor as positive ones, and several bugs were found and
fixed through direct verification rather than assumption: a model
specification defect that silently excluded two legitimate features, an
inconsistency between what the model scored and what the drift check
measured, and a case where a feature's absence degraded predictions far
more than its measured importance would suggest. Each is documented in
`FINDINGS.md` as a finding, not hidden as a footnote.

---

**Status:** Core pipeline, governance layer, and dashboard are complete
and tested. Check `FINDINGS.md` and the model registry for the current
approval status of any specific model before using its output for a real
decision.
