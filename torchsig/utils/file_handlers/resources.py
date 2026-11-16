"""Process-local synchronization utilities for file-backed resources."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["ProcessLocalResource"]


class ProcessLocalResource:
    """Track process ownership and serialize access to one resource group."""

    def __init__(self) -> None:
        self._pid = os.getpid()
        self._lock = threading.RLock()

    def process_changed(self) -> bool:
        """Adopt the current process and report whether ownership changed."""
        pid = os.getpid()
        changed = pid != self._pid
        self._pid = pid
        return changed

    @contextmanager
    def synchronized(self) -> Iterator[None]:
        """Serialize a seek/read or resource-transition operation."""
        with self._lock:
            yield

    def __getstate__(self) -> dict[str, int | None]:
        """Drop synchronization primitives and process ownership for pickle."""
        return {"_pid": None}

    def __setstate__(self, state: dict[str, int | None]) -> None:
        """Restore ownership and a fresh lock in the receiving process."""
        del state
        self._pid = os.getpid()
        self._lock = threading.RLock()
