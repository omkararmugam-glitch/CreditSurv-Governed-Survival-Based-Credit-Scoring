"""Seeded, resumable, parallel SurvSHAP(t) for bulk scoring.

Three properties, in the order they matter:

**Seeded per applicant.** :func:`creditsurv.explain.survshap.explain_survshap` seeds
numpy once per call, so an applicant's attributions depend on who else is in the
call -- which meant a notice could change with the block size (FINDINGS 7.1). Here
each row is explained on its own with a seed derived from its identifier, so the same
applicant always gets the same reasons, whatever else is being scored, in whatever
order, on however many cores.

**Parallel.** With per-row seeding the work is embarrassingly parallel and the result
is unchanged, because no row's seed depends on any other row. Workers are processes,
not threads: the cost is in numpy and LightGBM inside the Python layer of KernelSHAP,
which the GIL would serialise. If a pool cannot be started, the run continues in one
process and says so -- the seeds make that a difference in speed only.

**Resumable.** Each finished row is appended to a ledger as it completes, so a run
that is interrupted continues from where it stopped instead of starting again. Since
the seeds come from row identifiers, a resumed run produces the same output as an
uninterrupted one -- the property that makes resuming sound rather than merely
convenient, and the one ``tests/test_parallel_explain.py`` asserts.

The background is summarised **once** and passed to every row, so the k-means step
that dominates a small call runs a single time rather than per applicant.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .survshap import SurvShapExplanation, explain_survshap, summarise_background

__all__ = ["row_seed", "explain_rows", "explain_rows_parallel", "Ledger",
           "keep_awake", "suggest_workers"]


def row_seed(base_seed: int, row_id: object) -> int:
    """A stable seed for one applicant.

    Derived from the identifier rather than the position, so inserting a row, using
    a different block size or resuming a run cannot change anyone's reasons.
    """
    text = f"{base_seed}:{row_id}".encode("utf-8")
    import hashlib

    return int.from_bytes(hashlib.sha256(text).digest()[:4], "big")


def suggest_workers(n_rows: int, *, reserve_gb: float = 2.0,
                    per_worker_gb: float = 0.8) -> int:
    """How many worker processes this machine can afford.

    Each worker holds its own copy of the model and the summarised background, so
    the limit is memory, not cores. One core is left for the interface.
    """
    cores = max(1, (os.cpu_count() or 2) - 1)
    try:
        import psutil

        free_gb = psutil.virtual_memory().available / 1e9
        by_memory = max(1, int((free_gb - reserve_gb) / per_worker_gb))
    except Exception:                                  # pragma: no cover
        by_memory = 2
    return max(1, min(cores, by_memory, max(1, n_rows)))


@dataclass
class Ledger:
    """Rows already explained, so an interrupted run does not redo them."""

    path: Path
    done: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Ledger":
        path = Path(path)
        done: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:       # a half-written final line: ignore it and
                    continue             # let that row be explained again
                done[str(record.get("row_id"))] = record
        return cls(path=path, done=done)

    def has(self, row_id: object) -> bool:
        return str(row_id) in self.done

    def append(self, record: dict) -> None:
        self.done[str(record["row_id"])] = record
        # Created on first write, so a run refused at the checking step leaves no
        # directory behind.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
            fh.flush()


def keep_awake(enable: bool = True):
    """Ask Windows not to sleep while a long run is in progress.

    Returns ``(granted, note)``. It cannot prevent a deliberate sleep or a closed
    lid, so the caller says so rather than implying the run is safe from either.
    """
    if os.name != "nt":
        return False, "not Windows; no sleep request made"
    try:
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        flags = (ES_CONTINUOUS | ES_SYSTEM_REQUIRED) if enable else ES_CONTINUOUS
        ok = bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))
        if not enable:
            return ok, "sleep request released"
        return ok, ("the machine will not sleep on its own while this runs; a "
                    "deliberate sleep or a closed lid still pauses it, and the run "
                    "resumes where it stopped")
    except Exception as exc:                          # pragma: no cover
        return False, f"sleep request unavailable: {exc!r}"


def _explain_one(model, row: pd.DataFrame, background: pd.DataFrame,
                 times: np.ndarray, *, nsamples: int, seed: int) -> np.ndarray:
    """One applicant, seeded. ``background`` is already summarised."""
    np.random.seed(seed)
    expl = explain_survshap(model, row, background, times, nsamples=nsamples,
                            n_background=len(background), seed=seed, silent=True)
    return expl.phi[0]


_WORKER: dict = {}


def _init_worker(payload: dict) -> None:                # pragma: no cover - subprocess
    _WORKER.update(payload)


def _work(task: tuple) -> tuple:                        # pragma: no cover - subprocess
    position, row_id, values = task
    w = _WORKER
    row = pd.DataFrame([values], columns=w["columns"]).astype(w["dtypes"])
    phi = _explain_one(w["model"], row, w["background"], w["times"],
                       nsamples=w["nsamples"], seed=row_seed(w["seed"], row_id))
    return position, row_id, phi


def explain_rows(model, X: pd.DataFrame, background: pd.DataFrame,
                 times: np.ndarray, *, row_ids=None, nsamples: int = 600,
                 n_background: int = 100, seed: int = 20260921,
                 ledger: Ledger | None = None, progress=None) -> SurvShapExplanation:
    """Explain every row of ``X`` one at a time, each seeded from its identifier."""
    ids = list(row_ids if row_ids is not None else X.index)
    bg = summarise_background(background, n=n_background, seed=seed)
    phi = np.zeros((len(X), len(X.columns), len(np.atleast_1d(times))), dtype=float)
    for i, (row_id, (_, row)) in enumerate(zip(ids, X.iterrows())):
        if ledger is not None and ledger.has(row_id):
            record = ledger.done[str(row_id)]
            phi[i] = np.asarray(record["phi"], dtype=float)
            continue
        phi[i] = _explain_one(model, X.iloc[[i]], bg, times, nsamples=nsamples,
                              seed=row_seed(seed, row_id))
        if ledger is not None:
            ledger.append({"row_id": row_id, "phi": phi[i].tolist()})
        if progress:
            progress(i + 1, len(X))
    return _assemble(model, X, phi, times, bg, nsamples)


def explain_rows_parallel(model, X: pd.DataFrame, background: pd.DataFrame,
                          times: np.ndarray, *, row_ids=None, nsamples: int = 600,
                          n_background: int = 100, seed: int = 20260921,
                          workers: int | None = None, ledger: Ledger | None = None,
                          progress=None) -> SurvShapExplanation:
    """The same result as :func:`explain_rows`, computed on several cores.

    Identical output is not a hope here: each row's seed comes from its identifier,
    so the arithmetic in a worker is the arithmetic a single process would do.
    """
    ids = list(row_ids if row_ids is not None else X.index)
    bg = summarise_background(background, n=n_background, seed=seed)
    times = np.atleast_1d(np.asarray(times, dtype=float))
    phi = np.zeros((len(X), len(X.columns), len(times)), dtype=float)

    todo = []
    for i, row_id in enumerate(ids):
        if ledger is not None and ledger.has(row_id):
            phi[i] = np.asarray(ledger.done[str(row_id)]["phi"], dtype=float)
        else:
            todo.append((i, row_id, X.iloc[i].to_list()))
    done_already = len(X) - len(todo)
    if progress and done_already:
        progress(done_already, len(X))
    if not todo:
        return _assemble(model, X, phi, times, bg, nsamples)

    n_workers = workers or suggest_workers(len(todo))
    if n_workers <= 1:
        return explain_rows(model, X, background, times, row_ids=ids,
                            nsamples=nsamples, n_background=n_background, seed=seed,
                            ledger=ledger, progress=progress)

    from concurrent.futures import ProcessPoolExecutor

    payload = {"model": model, "background": bg, "times": times,
               "nsamples": nsamples, "seed": seed, "columns": list(X.columns),
               "dtypes": X.dtypes.to_dict()}
    completed = done_already
    try:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                                 initargs=(payload,)) as pool:
            for position, row_id, row_phi in pool.map(_work, todo, chunksize=1):
                phi[position] = row_phi
                if ledger is not None:
                    ledger.append({"row_id": row_id, "phi": row_phi.tolist()})
                completed += 1
                if progress:
                    progress(completed, len(X))
    except Exception as exc:
        # Workers are an optimisation, not a requirement. If the pool cannot start
        # or a worker dies -- a process-spawn environment that cannot re-import the
        # parent, a machine out of memory -- the work is done in this process
        # instead. The answer is the same either way, because each row is seeded
        # from its own id, so falling back costs time and nothing else.
        warnings.warn(f"parallel explanation unavailable ({exc!r}); continuing in one "
                      f"process", RuntimeWarning, stacklevel=2)
        return explain_rows(model, X, background, times, row_ids=ids,
                            nsamples=nsamples, n_background=n_background, seed=seed,
                            ledger=ledger, progress=progress)
    return _assemble(model, X, phi, times, bg, nsamples)


def _assemble(model, X: pd.DataFrame, phi: np.ndarray, times: np.ndarray,
              bg: pd.DataFrame, nsamples: int) -> SurvShapExplanation:
    times = np.atleast_1d(np.asarray(times, dtype=float))
    prediction = model.predict_survival(X, times)
    base = model.predict_survival(bg, times).mean(axis=0)
    return SurvShapExplanation(
        phi=phi, times=times, base=base, prediction=prediction,
        feature_names=tuple(X.columns), feature_values=X.copy(),
        nsamples=nsamples, n_background=len(bg))
