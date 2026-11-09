"""Optional process-local metrics and validation policies for file handlers."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

__all__ = ["FileHandlerMetrics", "ValidationPolicy"]


class ValidationPolicy(str, Enum):
    """Control when expensive storage integrity validation runs."""

    STRICT = "strict"
    CACHED = "cached"
    LAZY = "lazy"

    @classmethod
    def parse(cls, value: str | ValidationPolicy) -> ValidationPolicy:
        """Return a validated policy value."""
        try:
            return cls(value)
        except ValueError as error:
            raise ValueError(f"validation_policy must be one of {[item.value for item in cls]}") from error


@dataclass
class FileHandlerMetrics:
    """Low-overhead counters that reset when accessed in another process."""

    enabled: bool = False
    bytes_read: int = 0
    bytes_written: int = 0
    records_read: int = 0
    records_written: int = 0
    read_seconds: float = 0.0
    write_seconds: float = 0.0
    validation_seconds: float = 0.0
    validation_cache_hits: int = 0
    validation_cache_misses: int = 0
    open_handles: int = 0
    _pid: int = 0

    def __post_init__(self) -> None:
        """Associate new counters with the current process."""
        self._pid = os.getpid()

    def _ensure_process(self) -> None:
        if self._pid == os.getpid():
            return
        fresh = type(self)(enabled=self.enabled)
        self.__dict__.update(fresh.__dict__)

    @staticmethod
    def payload_bytes(value: Any) -> int:
        """Estimate array payload bytes without traversing arbitrary objects."""
        data = getattr(value, "data", value)
        return int(data.nbytes) if isinstance(data, np.ndarray) else 0

    def record_read(self, value: Any, started: float) -> None:
        """Record one successful read operation."""
        if not self.enabled:
            return
        self._ensure_process()
        self.records_read += 1
        self.bytes_read += self.payload_bytes(value)
        self.read_seconds += time.perf_counter() - started

    def record_write(self, value: Any, started: float) -> None:
        """Record one successful write call."""
        if not self.enabled:
            return
        self._ensure_process()
        values = value if isinstance(value, (list, tuple)) else (value,)
        self.records_written += len(values)
        self.bytes_written += sum(self.payload_bytes(item) for item in values)
        self.write_seconds += time.perf_counter() - started

    def snapshot(self) -> dict[str, int | float | bool]:
        """Return public process-local counters as plain data."""
        self._ensure_process()
        return {name: value for name, value in self.__dict__.items() if name != "_pid"}
