"""What the API process keeps between requests: loaded models, uploads, and the
scoring runs it is working on.

None of it is pipeline logic. The model cache holds what
:func:`creditsurv.batch.load_context` returns; the worker calls
:func:`creditsurv.batch.score_file`; the background path calls
:func:`creditsurv.runner.launch` with ``06_score_upload.py``, the command the
dashboard used for large files before the API existed.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..batch import BatchError, context_key, load_context, score_file
from ..stages import StageLog

__all__ = ["ContextCache", "UploadStore", "Upload", "Scorer", "RunRequest"]


class ContextCache:
    """Loaded models, shared by every request (D2 of the brief).

    Loading one unpickles the bundle and reads its training split for the SHAP
    background: seconds, and gigabytes at peak. The cache is keyed on
    :func:`creditsurv.batch.context_key` -- the model file, its cleaning values,
    its training data and the settings -- so a replaced file loads afresh and a
    stale model is never served. At most ``max_entries`` models stay in memory.
    """

    def __init__(self, cfg, *, loader=None, key_fn=None, max_entries: int = 2):
        self.cfg = cfg
        self._loader = loader or (lambda cfg, tag, model: load_context(cfg, tag, model))
        self._key_fn = key_fn or (lambda cfg, tag, model: context_key(cfg, tag, model))
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.max_entries = max_entries
        self.loads = 0
        self.hits = 0
        self.load_seconds: list[float] = []

    def resolve(self, tag: str | None, model: str | None) -> tuple[str, str]:
        d = self.cfg.decision
        return tag or d.model_tag, model or d.model

    def cached(self, tag: str | None, model: str | None) -> bool:
        tag, model = self.resolve(tag, model)
        return self._key_fn(self.cfg, tag, model) in self._items

    def get(self, tag: str | None = None, model: str | None = None):
        tag, model = self.resolve(tag, model)
        key = self._key_fn(self.cfg, tag, model)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                self.hits += 1
                return self._items[key]
            started = time.perf_counter()
            ctx = self._loader(self.cfg, tag, model)
            self.load_seconds.append(round(time.perf_counter() - started, 2))
            self.loads += 1
            self._items[key] = ctx
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)
            return ctx

    def describe(self) -> dict:
        return {"loaded": [list(k[:2]) for k in self._items], "loads": self.loads,
                "hits": self.hits, "load_seconds": self.load_seconds[-10:],
                "max_entries": self.max_entries}


_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class Upload:
    upload_id: str
    name: str
    path: Path
    size: int
    created: float = field(default_factory=time.time)


class UploadStore:
    """Uploaded files, on disk under ``<runs>/_uploads/<upload_id>/``.

    An upload outlives the page that sent it, which is what keeps a file from
    disappearing when the dashboard reruns: the page holds only the id.
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    def save(self, name: str, stream, chunk: int = 1 << 20) -> Upload:
        upload_id = uuid.uuid4().hex[:16]
        safe = _SAFE.sub("_", Path(name or "upload.csv").name)[:120] or "upload.csv"
        folder = self.root / upload_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / safe
        size = 0
        with open(path, "wb") as fh:
            while True:
                block = stream.read(chunk)
                if not block:
                    break
                fh.write(block)
                size += len(block)
        return Upload(upload_id, Path(name or safe).name, path, size)

    def get(self, upload_id: str) -> Upload | None:
        if not re.fullmatch(r"[0-9a-f]{16}", upload_id or ""):
            return None
        folder = self.root / upload_id
        files = [p for p in folder.iterdir() if p.is_file()] if folder.is_dir() else []
        if not files:
            return None
        p = files[0]
        return Upload(upload_id, p.name, p, p.stat().st_size, p.stat().st_mtime)

    def delete(self, upload_id: str) -> bool:
        up = self.get(upload_id)
        if up is None:
            return False
        shutil.rmtree(up.path.parent, ignore_errors=True)
        return True


@dataclass
class RunRequest:
    upload_id: str
    model_tag: str | None = None
    model: str | None = None
    threshold: float | None = None
    mapping: dict | None = None
    allow_unapproved_model: bool = False
    background: bool | None = None
    phase2: str = "auto"                  # "auto" | "defer"


def new_run_id(filename: str, runs_dir: Path) -> str:
    """``<timestamp>_<file stem>``, the name the dashboard always gave a run."""
    stem = _SAFE.sub("_", Path(filename).stem)[:40] or "upload"
    base = f"{datetime.now():%Y%m%d_%H%M%S}_{stem}"
    run_id, n = base, 1
    while (Path(runs_dir) / run_id).exists():
        n += 1
        run_id = f"{base}_{n}"
    return run_id


def place_input(upload: Upload, run_dir: Path) -> Path:
    """The upload as ``<run>/input_<name>``: hard-linked when the filesystem allows
    (instant, no second copy of a large file), copied otherwise. score_file sees the
    file already in place and does not copy it again."""
    run_dir.mkdir(parents=True, exist_ok=True)
    dest = run_dir / f"input_{upload.path.name}"
    try:
        os.link(upload.path, dest)
    except OSError:
        shutil.copyfile(upload.path, dest)
    return dest


class Scorer:
    """Runs Phase 1 in this process, one file at a time, with the shared model.

    One worker: two files scored at once would hold two copies of their blocks and
    two sets of outputs in memory with nothing gained. Requests queue behind it;
    the stage log says "queued" until the worker reaches them.
    """

    def __init__(self, cfg, contexts: ContextCache, *, phase2_launcher=None):
        self.cfg = cfg
        self.contexts = contexts
        self.phase2_launcher = phase2_launcher
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="phase1")
        self.active: set[str] = set()
        self._lock = threading.Lock()
        self.timings: dict[str, dict] = {}

    def busy(self) -> bool:
        return bool(self.active)

    def submit(self, run_id: str, run_dir: Path, input_path: Path, filename: str,
               req: RunRequest):
        # Registered as active before anything is written: a status read in between
        # would otherwise find a stage log with no live worker and call the run
        # interrupted.
        with self._lock:
            queued = bool(self.active)
            self.active.add(run_id)
        stages = StageLog(run_dir)
        stages("check", "running", "queued" if queued else "starting")
        return self._pool.submit(self._run, run_id, run_dir, input_path, filename,
                                 req, stages)

    def _run(self, run_id, run_dir, input_path, filename, req, stages) -> None:
        t0 = time.perf_counter()
        timing = self.timings.setdefault(run_id, {})
        try:
            cached = self.contexts.cached(req.model_tag, req.model)
            stages("check", "running", "model already loaded" if cached
                   else "loading the model (once; it stays loaded)")
            t = time.perf_counter()
            ctx = self.contexts.get(req.model_tag, req.model)
            timing["model_seconds"] = round(time.perf_counter() - t, 2)
            timing["model_cached"] = cached
            t = time.perf_counter()
            result = score_file(input_path, filename, self.cfg, ctx=ctx,
                                threshold=req.threshold, mapping=req.mapping,
                                progress=stages, run_dir=run_dir,
                                allow_unapproved_model=req.allow_unapproved_model)
            timing["score_file_seconds"] = round(time.perf_counter() - t, 2)
            stages.finish()
            if req.phase2 == "auto" and self.phase2_launcher is not None:
                self._auto_phase2(run_dir, result.summary)
        except BatchError as exc:
            if stages.state.get("finished") is None:
                stages.fail(exc.message, exc.detail, exc.fix)
        except Exception as exc:                        # never lose the reason
            stages.fail("The run stopped on an unexpected error.", repr(exc),
                        "Check the API log; nothing was cleaned up.")
        finally:
            timing["phase1_wall_seconds"] = round(time.perf_counter() - t0, 2)
            with self._lock:
                self.active.discard(run_id)

    def _auto_phase2(self, run_dir: Path, summary: dict) -> None:
        """Start Phase 2 by itself when the run is small enough not to ask: what the
        dashboard did on its own before, and what ``06_score_upload.py --phase2
        auto`` does. A larger run waits for the operator's choice."""
        if summary.get("phase2_state") == "not started":
            try:
                self.phase2_launcher(run_dir, "all", None)
            except Exception as exc:                    # LockHeld: already running
                (run_dir / "phase2_autostart_error.txt").write_text(
                    repr(exc), encoding="utf-8")
