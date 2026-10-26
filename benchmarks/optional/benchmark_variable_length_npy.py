#!/usr/bin/env python3
"""Compare shuffled reads from packed NPY, per-record NPY, and packed HDF5."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset

from torchsig.signals.signal_types import Signal
from torchsig.utils.file_handlers import PackedHDF5Reader, PackedHDF5Writer, PackedNPYReader, PackedNPYWriter


class _LayoutDataset(Dataset):
    """Process-local reader adapter for DataLoader measurements."""

    def __init__(self, layout: str, root: Path, length: int) -> None:
        self.layout = layout
        self.root = root
        self.length = length
        self.reader = None
        self.pid = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[float, int]:
        pid = os.getpid()
        if self.pid != pid:
            self.reader = None
            self.pid = pid
        if self.layout == "per_record_npy":
            data = np.load(self.root / f"{index:08d}.npy", mmap_mode="r")
        else:
            if self.reader is None:
                reader_type = PackedNPYReader if self.layout == "packed_npy" else PackedHDF5Reader
                self.reader = reader_type(self.root)
            data = self.reader.read(index).data
        array = np.asarray(data)
        return float(np.sum(array.real, dtype=np.float64)), array.size


def _identity_collate(batch):
    return batch


def _directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _fd_count() -> int | None:
    fd_root = Path("/proc/self/fd")
    return len(list(fd_root.iterdir())) if fd_root.exists() else None


def _measure(read, indices: list[int]) -> dict[str, float | int | None]:
    before = _fd_count()
    start = time.perf_counter()
    checksum = 0.0

    for index in indices:
        value = read(index)
        array = np.asarray(value)
        checksum += float(np.sum(array.real, dtype=np.float64))

    elapsed = time.perf_counter() - start
    after = _fd_count()

    return {
        "seconds": elapsed,
        "records_per_second": len(indices) / elapsed,
        "fd_delta": None if before is None or after is None else after - before,
        "checksum": checksum,
    }


def _measure_dataloader(layout: str, root: Path, indices: list[int], workers: int) -> dict[str, float | int]:
    dataset = Subset(_LayoutDataset(layout, root, len(indices)), indices)
    loader = DataLoader(dataset, batch_size=32, num_workers=workers, collate_fn=_identity_collate)
    start = time.perf_counter()
    observed = 0
    for batch in loader:
        observed += len(batch)
    elapsed = time.perf_counter() - start
    return {"workers": workers, "seconds": elapsed, "records_per_second": observed / elapsed}


def run(root: Path, record_count: int, min_length: int, max_length: int, seed: int, workers: list[int]) -> dict:
    """Create all layouts and return deterministic shuffled-read measurements."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(min_length, max_length + 1, size=record_count)
    signals = [Signal(data=np.full(int(length), index, dtype=np.complex64), record_index=index) for index, length in enumerate(lengths)]
    order = rng.permutation(record_count).tolist()

    packed_root = root / "packed-npy"
    with PackedNPYWriter(packed_root) as writer:
        writer.write(0, signals)
    packed_reader = PackedNPYReader(packed_root)

    files_root = root / "per-record-npy"
    files_root.mkdir(parents=True, exist_ok=True)
    for index, signal in enumerate(signals):
        np.save(files_root / f"{index:08d}.npy", signal.data, allow_pickle=False)

    hdf5_root = root / "packed-hdf5"
    with PackedHDF5Writer(hdf5_root, compression=None) as writer:
        writer.write(0, signals)
    hdf5_reader = PackedHDF5Reader(hdf5_root)

    roots = {"packed_npy": packed_root, "per_record_npy": files_root, "packed_hdf5": hdf5_root}

    results = {
        "parameters": {"records": record_count, "min_length": min_length, "max_length": max_length, "seed": seed},
        "layouts": {
            "packed_npy": {
                "storage_bytes": _directory_size(packed_root),
                **_measure(lambda index: packed_reader.read(index).data, order),
            },
            "per_record_npy": {
                "storage_bytes": _directory_size(files_root),
                **_measure(lambda index: np.load(files_root / f"{index:08d}.npy", mmap_mode="r"), order),
            },
            "packed_hdf5": {
                "storage_bytes": _directory_size(hdf5_root),
                **_measure(lambda index: hdf5_reader.read(index).data, order),
            },
        },
    }
    for layout, layout_root in roots.items():
        results["layouts"][layout]["dataloader"] = [_measure_dataloader(layout, layout_root, order, worker_count) for worker_count in workers]
    hdf5_reader.teardown()
    packed_reader.teardown()
    return results


def main() -> None:
    """Run the benchmark from command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument("--min-length", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--workers", default="0,2", help="Comma-separated DataLoader worker counts")
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    if args.records < 1 or args.min_length < 0 or args.max_length < args.min_length:
        parser.error("records must be positive and lengths must satisfy 0 <= min <= max")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        workers = [int(value) for value in args.workers.split(",")]
    except ValueError:
        parser.error("workers must be a comma-separated list of non-negative integers")
    if not workers or any(value < 0 for value in workers):
        parser.error("workers must contain non-negative integers")
    results = run(args.output_dir, args.records, args.min_length, args.max_length, args.seed, workers)
    payload = json.dumps(results, indent=2)
    if args.results:
        args.results.write_text(payload + os.linesep, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
