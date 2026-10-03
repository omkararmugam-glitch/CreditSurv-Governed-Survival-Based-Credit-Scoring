"""Atomic file replacement that survives a reader on Windows.

Every progress and result file that a page or the API reads while it is being
written (Phase 2's status, results and notice zip, the runner's status, a run's
stage log) is written to a temporary file and moved into place, so a reader never
sees half a file. On Linux the move always succeeds. On Windows it fails with
"Access is denied" while any reader has the target open -- which, with the API
polling a run every second, happens -- and an unhandled failure there stopped
Phase 2 part-way. The move is retried for up to about four seconds instead.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

__all__ = ["replace"]


def replace(tmp: Path | str, path: Path | str, attempts: int = 20) -> None:
    """``os.replace(tmp, path)``, retried while Windows reports the target busy."""
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.02 * (attempt + 1))
