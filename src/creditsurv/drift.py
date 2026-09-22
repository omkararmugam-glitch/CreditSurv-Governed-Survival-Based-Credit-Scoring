"""Does an uploaded file look like the data the model was trained on?

A model is only valid on a population resembling its training data, and a scoring
dashboard will happily return confident numbers for a file that does not. This
module compares the two, feature by feature:

* **numeric**: population stability index (PSI) over bins cut at the *training*
  quantiles, the standard credit-risk measure. Bands are the conventional ones --
  below 0.10 stable, 0.10-0.25 moderate shift, above 0.25 large shift.
* **categorical**: total variation distance (half the sum of absolute share
  differences), read on the same bands, plus the share of rows on levels the
  training data never contained.

Nothing here changes data, and it never refits anything: the reference
distribution comes from the training split the model was fitted on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["STABLE", "MODERATE", "LARGE", "MIN_ROWS", "DriftResult",
           "population_stability_index", "category_distance", "compare"]

STABLE, MODERATE, LARGE = 0.10, 0.25, 1.0
_EPS = 1e-6

MIN_ROWS = 500
"""Below this many rows, drift is not assessed at all.

PSI is a noisy statistic on small samples, and the noise is quantifiable: with
``k`` bins and no real shift, its expected value under the null is about
``(k - 1) / n``. This module uses 10 quantile bins plus a missing bin, so k - 1 =
10. Measured on unshifted normal data, the median PSI is about 0.08 at 100 rows --
within a whisker of the 0.10 "moderate" band, so individual features cross it
routinely -- and about 0.02 at 500 rows, roughly a fifth of the band. That is what
makes a reading above 0.10 mean something at 500 rows and nothing at 100. Below
the minimum the honest answer is that the file is too small to tell, reported as
its own grey status rather than as red. ``tests/test_drift.py`` pins both figures."""


def _band(score: float) -> str:
    if not np.isfinite(score):
        return "unknown"
    return "stable" if score < STABLE else ("moderate" if score < MODERATE else "large")


def population_stability_index(reference: pd.Series, current: pd.Series,
                               *, bins: int = 10) -> tuple[float, int]:
    """PSI of ``current`` against ``reference``, with bin edges from the reference.

    Missing values form their own bin: a file where a feature is absent or empty
    is exactly the case this check exists to catch.
    """
    ref = pd.to_numeric(reference, errors="coerce")
    cur = pd.to_numeric(current, errors="coerce")
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.nanquantile(ref.dropna(), qs)) if ref.notna().any() else np.array([])
    if edges.size < 2:
        return float("nan"), 0
    edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
    ref_counts = np.histogram(ref.dropna(), bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur.dropna(), bins=edges)[0].astype(float)
    ref_counts = np.append(ref_counts, ref.isna().sum())
    cur_counts = np.append(cur_counts, cur.isna().sum())
    p = np.where(ref_counts.sum() > 0, ref_counts / max(ref_counts.sum(), 1), 0.0)
    q = np.where(cur_counts.sum() > 0, cur_counts / max(cur_counts.sum(), 1), 0.0)
    p, q = np.clip(p, _EPS, None), np.clip(q, _EPS, None)
    return float(np.sum((q - p) * np.log(q / p))), int(len(edges) - 1)


def category_distance(reference: pd.Series, current: pd.Series) -> tuple[float, float, dict]:
    """Total variation distance, unseen-level share, and the biggest share moves."""
    ref = reference.astype("string").str.strip().fillna("(missing)")
    cur = current.astype("string").str.strip().fillna("(missing)")
    p = ref.value_counts(normalize=True)
    q = cur.value_counts(normalize=True)
    levels = p.index.union(q.index)
    p, q = p.reindex(levels, fill_value=0.0), q.reindex(levels, fill_value=0.0)
    tvd = float((p - q).abs().sum() / 2)
    unseen = float(q[~q.index.isin(ref.unique())].sum())
    moves = (q - p).sort_values(key=lambda s: s.abs(), ascending=False).head(3)
    return tvd, unseen, {str(k): round(float(v), 4) for k, v in moves.items()}


@dataclass
class DriftResult:
    table: pd.DataFrame
    status: str = "stable"
    n_moderate: int = 0
    n_large: int = 0
    n_unknown: int = 0
    notes: list[str] = field(default_factory=list)

    n_rows: int = 0

    @property
    def colour(self) -> str:
        return {"stable": "green", "moderate": "amber", "large": "red",
                "unknown": "amber", "insufficient": "grey"}[self.status]

    def headline(self) -> str:
        if self.status == "insufficient":
            return (f"Too few rows to assess drift: {self.n_rows:,} row(s), and at "
                    f"least {MIN_ROWS:,} are needed before the index means anything "
                    f"(a file this small scores as 'shifted' on features that have "
                    f"not moved). The scores themselves are unaffected.")
        if self.status == "stable":
            return ("This file looks like the training data: no feature shifted "
                    "materially (PSI below 0.10 throughout).")
        if self.status == "moderate":
            parts = []
            if self.n_moderate:
                parts.append(f"{self.n_moderate} feature(s) have shifted "
                             f"(PSI 0.10-0.25)")
            if self.n_unknown:
                parts.append(f"{self.n_unknown} feature(s) could not be compared "
                             f"(absent from the upload, or constant in training)")
            return " and ".join(parts) + ". Scores are less reliable for those features."
        return (f"{self.n_large} feature(s) shifted heavily (PSI above 0.25). The "
                f"model may not be valid for this population; treat the decisions "
                f"as indicative only.")


def compare(reference: pd.DataFrame, current: pd.DataFrame, spec, *,
            min_rows: int = MIN_ROWS) -> DriftResult:
    """Per-feature drift of ``current`` (the upload) against ``reference`` (training).

    Below ``min_rows`` the comparison is not made: see :data:`MIN_ROWS` for why a
    small file would otherwise be reported as shifted when it is not.
    """
    if len(current) < min_rows:
        table = pd.DataFrame([
            {"feature": col, "kind": "numeric" if col in spec.numeric else "categorical",
             "measure": "PSI" if col in spec.numeric else "TVD",
             "score": float("nan"), "status": "insufficient",
             "detail": f"only {len(current):,} row(s); {min_rows:,} needed",
             "reference_mean": "", "upload_mean": ""}
            for col in list(spec.numeric) + list(spec.categorical)
            if col in reference.columns])
        return DriftResult(table=table, status="insufficient", n_rows=len(current))

    rows = []
    for col in list(spec.numeric) + list(spec.categorical):
        if col not in reference.columns:
            continue
        numeric = col in spec.numeric
        if col not in current.columns:
            rows.append({"feature": col, "kind": "numeric" if numeric else "categorical",
                         "score": float("nan"), "measure": "PSI" if numeric else "TVD",
                         "status": "unknown", "detail": "absent from the upload",
                         "reference_mean": "", "upload_mean": ""})
            continue
        if numeric:
            score, nbins = population_stability_index(reference[col], current[col])
            ref_m = pd.to_numeric(reference[col], errors="coerce").mean()
            cur_m = pd.to_numeric(current[col], errors="coerce").mean()
            rows.append({"feature": col, "kind": "numeric", "measure": "PSI",
                         "score": round(score, 4) if np.isfinite(score) else score,
                         "status": _band(score),
                         "detail": f"{nbins} bins" if np.isfinite(score)
                                   else "not comparable",
                         "reference_mean": round(float(ref_m), 4) if pd.notna(ref_m) else "",
                         "upload_mean": round(float(cur_m), 4) if pd.notna(cur_m) else ""})
        else:
            tvd, unseen, moves = category_distance(reference[col], current[col])
            detail = "; ".join(f"{k} {v:+.2f}" for k, v in moves.items())
            if unseen:
                detail += f"; {unseen:.0%} of rows on unseen levels"
            rows.append({"feature": col, "kind": "categorical", "measure": "TVD",
                         "score": round(tvd, 4), "status": _band(tvd), "detail": detail,
                         "reference_mean": "", "upload_mean": ""})

    table = pd.DataFrame(rows).sort_values(
        ["status", "score"], ascending=[True, False],
        key=lambda s: s.map({"large": 0, "moderate": 1, "unknown": 2, "stable": 3})
        if s.name == "status" else s).reset_index(drop=True)
    n_large = int((table["status"] == "large").sum())
    n_moderate = int((table["status"] == "moderate").sum())
    n_unknown = int((table["status"] == "unknown").sum())
    # A feature that cannot be compared is not evidence of stability: a file
    # missing a feature entirely would otherwise report green.
    status = ("large" if n_large else
              "moderate" if (n_moderate or n_unknown) else "stable")
    notes = [f"{r.feature}: {r.measure} {r.score} ({r.status})"
             for r in table.itertuples()
             if r.status in ("large", "moderate", "unknown")][:10]
    return DriftResult(table=table, status=status, n_moderate=n_moderate,
                       n_large=n_large, n_unknown=n_unknown, notes=notes,
                       n_rows=len(current))
