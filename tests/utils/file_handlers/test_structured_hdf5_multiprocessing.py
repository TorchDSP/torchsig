"""Multiprocessing tests for the structured HDF5 reader."""

from __future__ import annotations

import multiprocessing
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from torchsig.datasets import StructuredHDF5Dataset
from torchsig.utils.file_handlers import StructuredHDF5Writer

if TYPE_CHECKING:
    from pathlib import Path


def _identity_collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


class _StructuredReaderDataset(Dataset):
    def __init__(self, root: Path, length: int, open_in_parent: bool = False) -> None:
        self.dataset = StructuredHDF5Dataset(root)
        self.length = length
        if open_in_parent:
            len(self.dataset)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        return {
            "index": int(sample["index"]),
            "worker_pid": os.getpid(),
            "reader_pid": self.dataset.reader._pid,
            "data": sample["data"],
        }

    def __del__(self) -> None:
        self.dataset.close()


def _write(root: Path, count: int = 24) -> list[dict[str, Any]]:
    samples = [
        {
            "index": np.int64(index),
            "data": np.full((3, 4), index, dtype=np.float32),
        }
        for index in range(count)
    ]
    with StructuredHDF5Writer(root) as writer:
        writer.write(0, samples[:11])
        writer.write(1, samples[11:])
    return samples


def _collect(loader: DataLoader) -> list[dict[str, Any]]:
    return [item for batch in loader for item in batch]


def _assert_results(actual: list[dict[str, Any]], expected: list[dict[str, Any]], shuffled: bool = False) -> None:
    if shuffled:
        actual.sort(key=lambda item: item["index"])
    assert [item["index"] for item in actual] == list(range(len(expected)))
    for item, sample in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(item["data"], sample["data"])
        assert item["reader_pid"] == item["worker_pid"]


@pytest.mark.parametrize("num_workers", [0, 2])
def test_structured_reader_dataloader_order_content_and_process_handles(tmp_path, num_workers) -> None:
    expected = _write(tmp_path)
    dataset = _StructuredReaderDataset(tmp_path, len(expected), open_in_parent=num_workers > 0)
    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers else None,
        collate_fn=_identity_collate,
    )

    actual = _collect(loader)
    _assert_results(actual, expected)
    worker_pids = {item["worker_pid"] for item in actual}
    assert len(worker_pids) == (num_workers or 1)


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="fork multiprocessing context is unavailable")
def test_structured_reader_reopens_parent_handle_after_fork_with_four_workers(tmp_path) -> None:
    expected = _write(tmp_path)
    dataset = _StructuredReaderDataset(tmp_path, len(expected), open_in_parent=True)
    parent_pid = dataset.dataset.reader._pid
    loader = DataLoader(
        dataset,
        batch_size=3,
        num_workers=4,
        multiprocessing_context="fork",
        collate_fn=_identity_collate,
        shuffle=True,
        generator=torch.Generator().manual_seed(5),
    )

    actual = _collect(loader)
    _assert_results(actual, expected, shuffled=True)
    assert parent_pid == os.getpid()
    assert all(item["worker_pid"] != parent_pid for item in actual)
    assert len({item["worker_pid"] for item in actual}) == 4
    assert dataset.dataset[0]["index"] == 0


def test_structured_reader_spawn_persistent_workers_across_epochs(tmp_path) -> None:
    expected = _write(tmp_path)
    dataset = _StructuredReaderDataset(tmp_path, len(expected), open_in_parent=True)
    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
        multiprocessing_context="spawn",
        persistent_workers=True,
        collate_fn=_identity_collate,
        shuffle=True,
        generator=torch.Generator().manual_seed(7),
    )

    first = _collect(loader)
    second = _collect(loader)
    _assert_results(first, expected, shuffled=True)
    _assert_results(second, expected, shuffled=True)
    assert {item["worker_pid"] for item in first} == {item["worker_pid"] for item in second}
