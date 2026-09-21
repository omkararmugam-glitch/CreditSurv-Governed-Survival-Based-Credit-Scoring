"""Result serialisation: JSON for machines, Markdown for FINDINGS.md."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "write_json",
    "write_table",
    "to_markdown",
    "model_comparison_table",
    "explanation_shift_table",
]


def _jsonable(obj: Any) -> Any:
    """Coerce numpy / pandas objects into something ``json.dump`` accepts."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return [{str(k): _jsonable(v) for k, v in row.items()}
                for row in obj.to_dict(orient="records")]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(payload: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return path


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def to_markdown(df: pd.DataFrame, *, floatfmt: str = "{:.4f}", max_rows: int | None = None) -> str:
    """Render a DataFrame as a GitHub Markdown table, without tabulate."""
    if df.empty:
        return "_(no rows)_"
    view = df.head(max_rows) if max_rows else df

    def cell(v: Any) -> str:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(float(v))
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        return str(v)

    header = "| " + " | ".join(str(c) for c in view.columns) + " |"
    rule = "|" + "|".join("---" for _ in view.columns) + "|"
    rows = [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *rows])


def model_comparison_table(results: list) -> pd.DataFrame:
    """One row per (model, split) from a list of ``EvaluationResult``."""
    return pd.DataFrame([r.summary() for r in results])


def explanation_shift_table(
    before: pd.Series, after: pd.Series, *, k: int = 10
) -> pd.DataFrame:
    """Per-feature importance before vs after a reject-inference correction.

    This is the project's own question -- whether correcting for selection bias
    changes *which features the model says matter*, not merely how well it scores.
    None of the reviewed literature tests it.
    """
    features = list(dict.fromkeys(list(before.index) + list(after.index)))
    b = before.reindex(features)
    a = after.reindex(features)
    out = pd.DataFrame(
        {
            "feature": features,
            "importance_before": b.to_numpy(),
            "importance_after": a.to_numpy(),
            "rank_before": b.rank(ascending=False).to_numpy(),
            "rank_after": a.rank(ascending=False).to_numpy(),
        }
    )
    out["abs_change"] = (out["importance_after"] - out["importance_before"]).abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["pct_change"] = np.where(
            out["importance_before"].abs() > 1e-12,
            (out["importance_after"] - out["importance_before"])
            / out["importance_before"].abs()
            * 100.0,
            np.nan,
        )
    out["rank_change"] = out["rank_after"] - out["rank_before"]
    return out.sort_values("importance_before", ascending=False).head(k).reset_index(drop=True)
