"""Chunked ingest of the Lending Club CSVs into Parquet.

Size rationale
--------------
The accepted file is ~1.6 GB / ~2.26M rows x 151 columns; the rejected file is
of similar size but ~27.6M rows x 9 columns. Neither is loaded whole. Instead
each is streamed in chunks, restricted to an allowlist of columns, downcast, and
appended to a Parquet file with a single fixed Arrow schema. That takes accepted
to roughly 250-400 MB on disk and ~1.5 GB in memory, which is workable, and it
means the expensive parse happens exactly once.

Messiness handled explicitly
----------------------------
* **Prospectus preamble.** Some Lending Club exports begin with a free-text
  note before the real header row, so the header is located by scanning.
* **Footer totals.** Exports often end with rows like
  ``"Total amount funded in policy code 1: ..."``. These are dropped by
  requiring a usable ``id``.
* **Mixed types within a column.** Every column is read as text and then coerced
  numerically. This is slower than letting pandas infer, but inference differs
  between chunks on a file this messy, which produces a schema mismatch halfway
  through a 20-minute ingest.
* **Columns absent in some releases.** The allowlist is intersected with the
  real header, and whatever is missing is reported rather than raising.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import schema as sch

__all__ = [
    "find_header_row",
    "read_header",
    "ingest_accepted",
    "ingest_rejected",
    "stratified_sample",
]


def find_header_row(csv_path: str | Path, expected: set[str], max_scan: int = 10) -> int:
    """Return the 0-based index of the real header row.

    Scans the first ``max_scan`` lines for the row that overlaps ``expected``
    most strongly, which tolerates a prospectus note before the header.
    """
    best_row, best_hits = 0, -1
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        for i, row in enumerate(reader):
            if i >= max_scan:
                break
            hits = len({c.strip() for c in row} & expected)
            if hits > best_hits:
                best_row, best_hits = i, hits
    if best_hits <= 0:
        raise ValueError(
            f"No header row in the first {max_scan} lines of {csv_path} matched any "
            f"expected column. Is this the right file?"
        )
    return best_row


def read_header(csv_path: str | Path, header_row: int) -> list[str]:
    """Column names as they appear in the file."""
    df = pd.read_csv(csv_path, nrows=0, skiprows=header_row, low_memory=False)
    return [str(c).strip() for c in df.columns]


def _arrow_schema(columns: list[str], numeric: set[str]) -> pa.Schema:
    """One fixed schema for every chunk, so appends can never mismatch."""
    return pa.schema(
        [
            pa.field(c, pa.float32() if c in numeric else pa.string())
            for c in columns
        ]
    )


def _coerce(chunk: pd.DataFrame, numeric: set[str]) -> pd.DataFrame:
    """Coerce a text chunk to the target dtypes."""
    out = {}
    for col in chunk.columns:
        s = chunk[col]
        if col in numeric:
            cleaned = (
                s.astype("string")
                .str.strip()
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace("$", "", regex=False)
            )
            out[col] = pd.to_numeric(cleaned, errors="coerce").astype("float32")
        else:
            out[col] = s.astype("string").str.strip()
    return pd.DataFrame(out, columns=chunk.columns)


def _ingest(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    wanted: list[str],
    numeric: set[str],
    id_col: str | None,
    chunksize: int,
    row_limit: int | None,
    rename: dict[str, str] | None = None,
) -> dict:
    csv_path, out_path = Path(csv_path), Path(out_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header_row = find_header_row(csv_path, set(wanted))
    header = read_header(csv_path, header_row)
    usecols = [c for c in wanted if c in header]
    missing = [c for c in wanted if c not in header]
    if not usecols:
        raise ValueError(f"None of the requested columns exist in {csv_path}")

    final_names = [(rename or {}).get(c, c) for c in usecols]
    numeric_final = {(rename or {}).get(c, c) for c in numeric}
    arrow_schema = _arrow_schema(final_names, numeric_final)

    reader = pd.read_csv(
        csv_path,
        skiprows=header_row,
        usecols=usecols,
        dtype=str,
        chunksize=chunksize,
        low_memory=False,
        na_values=["", "n/a", "N/A", "NA", "null", "NULL", "none", "None"],
        keep_default_na=True,
        on_bad_lines="warn",
        encoding="utf-8",
        encoding_errors="replace",
    )

    n_read = n_written = n_dropped_junk = 0
    writer: pq.ParquetWriter | None = None
    try:
        for chunk in reader:
            chunk.columns = [str(c).strip() for c in chunk.columns]
            chunk = chunk.rename(columns=rename or {})
            chunk = chunk.reindex(columns=final_names)
            n_read += len(chunk)

            if id_col and id_col in chunk.columns:
                # Footer/total rows have no usable id.
                usable = pd.to_numeric(chunk[id_col], errors="coerce").notna()
                n_dropped_junk += int((~usable).sum())
                chunk = chunk.loc[usable]
            else:
                blank = chunk.isna().all(axis=1)
                n_dropped_junk += int(blank.sum())
                chunk = chunk.loc[~blank]

            if chunk.empty:
                continue

            table = pa.Table.from_pandas(
                _coerce(chunk, numeric_final), schema=arrow_schema, preserve_index=False
            )
            if writer is None:
                writer = pq.ParquetWriter(out_path, arrow_schema, compression="snappy")
            writer.write_table(table)
            n_written += len(chunk)

            if row_limit is not None and n_read >= row_limit:
                break
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise ValueError(f"No usable rows found in {csv_path}")

    return {
        "source": str(csv_path),
        "output": str(out_path),
        "header_row": header_row,
        "n_rows_read": n_read,
        "n_rows_written": n_written,
        "n_junk_rows_dropped": n_dropped_junk,
        "columns_written": final_names,
        "columns_requested_but_absent": missing,
        "output_size_mb": round(out_path.stat().st_size / 1e6, 1),
    }


def ingest_accepted(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    extended: bool = True,
    chunksize: int = 250_000,
    row_limit: int | None = None,
) -> dict:
    """Stream the accepted-loans CSV to Parquet, allowlisted and downcast."""
    wanted = sch.ingest_columns(extended=extended, with_lc_grade=True)
    numeric = set(sch.CORE_NUMERIC) | set(sch.EXTENDED_NUMERIC) | {"int_rate"}
    return _ingest(
        csv_path,
        out_path,
        wanted=wanted,
        numeric=numeric,
        id_col="id",
        chunksize=chunksize,
        row_limit=row_limit,
    )


def ingest_rejected(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    chunksize: int = 500_000,
    row_limit: int | None = None,
) -> dict:
    """Stream the rejected-applications CSV to Parquet with aligned names.

    ``dti_raw`` stays text at this stage: the rejected file stores DTI as a
    percentage string with implausible extremes, and deciding how to winsorise
    it belongs in the Stage 4 diagnostic, not in ingest.
    """
    wanted = list(sch.REJECTED_COLUMNS)
    numeric = {"Amount Requested", "Risk_Score", "Policy Code"}
    return _ingest(
        csv_path,
        out_path,
        wanted=wanted,
        numeric=numeric,
        id_col=None,
        chunksize=chunksize,
        row_limit=row_limit,
        rename=dict(sch.REJECTED_RENAMES),
    )


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    *,
    by: tuple[str, ...],
    seed: int,
) -> pd.DataFrame:
    """Proportional stratified sample, used for the development subset.

    Strata are allocated proportionally with a floor of one row each, so rare
    vintage/term combinations are not silently dropped from the dev sample.
    """
    if n >= len(df):
        return df.copy()
    cols = [c for c in by if c in df.columns]
    if not cols:
        return df.sample(n=n, random_state=seed)

    keys = df[cols].astype("string").fillna("NA").agg("|".join, axis=1)
    rng = np.random.default_rng(seed)
    counts = keys.value_counts()
    quota = (counts / counts.sum() * n).apply(np.floor).astype(int).clip(lower=1)

    # Hand any rounding remainder to the largest strata.
    deficit = n - int(quota.sum())
    if deficit > 0:
        for k in counts.index[: min(deficit, len(counts))]:
            quota[k] += 1

    parts = []
    for key, group in df.groupby(keys, sort=False):
        take = int(min(quota.get(key, 1), len(group)))
        parts.append(group.sample(n=take, random_state=int(rng.integers(1 << 31))))
    return pd.concat(parts).sample(frac=1.0, random_state=seed)
