"""Multiprocessing tests for the homogeneous HDF5 reader."""

from __future__ import annotations

import multiprocessing
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from torch.utils.data import DataLoader, Dataset

from torchsig.signals.signal_types import Signal
from torchsig.utils.file_handlers.homogeneous_hdf5 import (
    HomogeneousHDF5Reader,
    HomogeneousHDF5Writer,
)

if TYPE_CHECKING:
    from pathlib import Path

_WORKLOAD_SHAPES = {
    "iq": (64,),
    "wideband": (4_096,),
    "spectrogram": (16, 32),
}


def _identity_collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


class _HomogeneousReaderDataset(Dataset):
    def __init__(
        self,
        root: Path,
        length: int,
        open_in_parent: bool = False,
    ) -> None:
        self.root = root
        self.length = length
        self.reader = HomogeneousHDF5Reader(root)
        if open_in_parent:
            len(self.reader)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        signal = self.reader.read(idx)
        return {
            "index": idx,
            "shape": signal.data.shape,
            "dtype": signal.data.dtype.str,
            "first": float(signal.data.reshape(-1)[0].real),
            "sample_index": signal["sample_index"],
            "component_shapes": tuple(component.data.shape for component in signal.component_signals),
            "component_dtypes": tuple(component.data.dtype.str for component in signal.component_signals),
        }

    def __del__(self) -> None:
        self.reader.teardown()


def _signals(workload: str, count: int = 16) -> list[Signal]:
    shape = _WORKLOAD_SHAPES[workload]
    result = []
    for idx in range(count):
        dtype = np.float32 if workload == "spectrogram" else np.complex64
        data = np.full(shape, idx, dtype=dtype)
        components = [
            Signal(
                data=np.full(
                    (component_idx + 1, component_idx + 2),
                    idx,
                    dtype=np.float32 if component_idx % 2 == 0 else np.int16,
                )
            )
            for component_idx in range(idx % 4)
        ]
        result.append(
            Signal(
                data=data,
                component_signals=components,
                sample_index=idx,
            )
        )
    return result


def _write(root: Path, workload: str) -> list[Signal]:
    signals = _signals(workload)
    with HomogeneousHDF5Writer(root) as writer:
        writer.write(0, signals[:8])
        writer.write(1, signals[8:])
    return signals


def _collect(loader: DataLoader) -> list[dict[str, Any]]:
    return [item for batch in loader for item in batch]


def _assert_results(
    actual: list[dict[str, Any]],
    expected: list[Signal],
) -> None:
    assert [item["index"] for item in actual] == list(range(len(expected)))
    for item, signal in zip(actual, expected, strict=True):
        assert item["shape"] == signal.data.shape
        assert item["dtype"] == signal.data.dtype.str
        assert item["first"] == float(signal.data.reshape(-1)[0].real)
        assert item["sample_index"] == item["index"]
        assert item["component_shapes"] == tuple(component.data.shape for component in signal.component_signals)
        assert item["component_dtypes"] == tuple(component.data.dtype.str for component in signal.component_signals)


@pytest.mark.parametrize("workload", _WORKLOAD_SHAPES)
@pytest.mark.parametrize("num_workers", [0, 2])
def test_homogeneous_reader_dataloader_order_and_content(
    tmp_path,
    workload,
    num_workers,
) -> None:
    expected = _write(tmp_path, workload)
    context = "spawn" if num_workers else None
    dataset = _HomogeneousReaderDataset(
        tmp_path,
        len(expected),
        open_in_parent=num_workers > 0,
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=num_workers,
        multiprocessing_context=context,
        collate_fn=_identity_collate,
    )

    _assert_results(_collect(loader), expected)


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="fork multiprocessing context is unavailable",
)
def test_homogeneous_reader_reopens_parent_handle_after_fork(
    tmp_path,
) -> None:
    expected = _write(tmp_path, "iq")
    dataset = _HomogeneousReaderDataset(
        tmp_path,
        len(expected),
        open_in_parent=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=4,
        multiprocessing_context="fork",
        collate_fn=_identity_collate,
    )

    _assert_results(_collect(loader), expected)
    assert dataset.reader.read(0)["sample_index"] == 0


def test_homogeneous_reader_repeated_persistent_worker_epochs(
    tmp_path,
) -> None:
    expected = _write(tmp_path, "spectrogram")
    dataset = _HomogeneousReaderDataset(tmp_path, len(expected))
    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
        multiprocessing_context="spawn",
        persistent_workers=True,
        collate_fn=_identity_collate,
    )

    _assert_results(_collect(loader), expected)
    _assert_results(_collect(loader), expected)
