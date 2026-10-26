"""Tests for process-safe bounded audio handle caching."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Self

import numpy as np
import pytest

from torchsig.utils.file_handlers.audio import AudioHandleCache


class FakeSoundFile:
    """Record seek/read/close operations without opening a real file."""

    def __init__(self, path: Path, mode: str) -> None:
        self.path = Path(path)
        self.mode = mode
        self.seeks: list[int] = []
        self.reads: list[tuple[int, str, bool]] = []
        self.closed = False

    def seek(self, frame: int) -> None:
        """Record a seek."""
        self.seeks.append(frame)

    def read(self, *, frames: int, dtype: str, always_2d: bool) -> np.ndarray:
        """Record and satisfy a bounded read."""
        self.reads.append((frames, dtype, always_2d))
        return np.zeros((frames, 2), dtype=np.float32)

    def close(self) -> None:
        """Record closure."""
        self.closed = True

    def __enter__(self) -> Self:
        """Return this fake context-managed handle."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Close this fake context-managed handle."""
        self.close()
        return False


@pytest.fixture
def fake_soundfile(monkeypatch):
    opened: list[FakeSoundFile] = []

    def factory(path: Path, mode: str) -> FakeSoundFile:
        handle = FakeSoundFile(path, mode)
        opened.append(handle)
        return handle

    monkeypatch.setattr("torchsig.utils.file_handlers.audio.sf.SoundFile", factory)
    return opened


def test_bounded_read_reuses_handle(fake_soundfile) -> None:
    cache = AudioHandleCache(2)
    path = Path("one.wav")

    cache.read(path, 3, 5)
    cache.read(path, 11, 7)

    assert len(fake_soundfile) == 1
    assert fake_soundfile[0].seeks == [3, 11]
    assert fake_soundfile[0].reads == [(5, "float32", True), (7, "float32", True)]


def test_lru_eviction_closes_oldest_handle(fake_soundfile) -> None:
    cache = AudioHandleCache(2)
    paths = [Path(f"{name}.wav") for name in ("one", "two", "three")]

    for path in paths:
        cache.read(path, 0, 1)

    assert fake_soundfile[0].closed
    assert not fake_soundfile[1].closed
    assert not fake_soundfile[2].closed


def test_zero_size_disables_caching(fake_soundfile) -> None:
    cache = AudioHandleCache(0)

    cache.read(Path("one.wav"), 2, 4)
    cache.read(Path("one.wav"), 2, 4)

    assert len(fake_soundfile) == 2
    assert all(handle.closed for handle in fake_soundfile)
    assert len(cache) == 0


def test_close_closes_every_handle(fake_soundfile) -> None:
    cache = AudioHandleCache(2)
    cache.read(Path("one.wav"), 0, 1)
    cache.read(Path("two.wav"), 0, 1)

    cache.close()

    assert all(handle.closed for handle in fake_soundfile)
    assert len(cache) == 0


def test_pickle_excludes_handles(fake_soundfile) -> None:
    cache = AudioHandleCache(2)
    cache.read(Path("one.wav"), 0, 1)

    restored = pickle.loads(pickle.dumps(cache))

    assert restored.max_size == 2
    assert len(restored) == 0


def test_pid_change_closes_and_reopens(fake_soundfile, monkeypatch) -> None:
    pid = 100
    monkeypatch.setattr("torchsig.utils.file_handlers.audio.os.getpid", lambda: pid)
    cache = AudioHandleCache(1)
    cache.read(Path("one.wav"), 0, 1)

    pid = 101
    cache.read(Path("one.wav"), 0, 1)

    assert len(fake_soundfile) == 2
    assert fake_soundfile[0].closed


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_cache_size(value) -> None:
    with pytest.raises((TypeError, ValueError), match="cache_size"):
        AudioHandleCache(value)
