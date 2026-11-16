"""Compare online deterministic transforms with structured HDF5 training reads.

Run a representative matrix with:
    python benchmarks/optional/benchmark_structured_hdf5.py \
        --samples 1024 --workers 0 2 4 --access sequential shuffled

The report includes materialization time, cold first-batch latency, steady
epoch throughput, output size, and break-even epoch count. It does not assert
that materialization is faster for every workload or storage system.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from torchsig.datasets import StructuredHDF5Dataset, materialize_structured_dataset

Access = Literal["sequential", "shuffled"]


class DeterministicTransformDataset(Dataset):
    """Generate deterministic complex IQ and an FFT-derived model sample."""

    def __init__(self, length: int, iq_length: int, seed: int) -> None:
        self.length = length
        self.iq_length = iq_length
        self.seed = seed

    def __len__(self) -> int:
        """Return the configured number of deterministic samples."""
        return self.length

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, np.ndarray | np.int64]]:
        """Generate and transform one reproducible IQ sample."""
        generator = np.random.default_rng(self.seed + index)
        iq = (generator.standard_normal(self.iq_length) + 1j * generator.standard_normal(self.iq_length)).astype(np.complex64)
        spectrum = np.fft.fftshift(np.fft.fft(iq)).astype(np.complex64)
        model_input = np.stack((spectrum.real, spectrum.imag)).astype(np.float32)
        peak = int(np.argmax(np.abs(spectrum)))
        target = {
            "class": np.int64(index % 16),
            "detection": np.array([peak / self.iq_length, float(np.abs(spectrum[peak]))], dtype=np.float32),
        }
        return model_input, target


@dataclass(frozen=True)
class Result:
    """One online-versus-materialized benchmark result."""

    workers: int
    access: Access
    compression: str
    chunk_samples: int
    materialization_seconds: float
    output_mib: float
    online_first_batch_seconds: float
    materialized_first_batch_seconds: float
    online_samples_per_second: float
    materialized_samples_per_second: float
    break_even_epochs: float

    def as_row(self) -> dict[str, str | int]:
        """Return stable, presentation-ready report fields."""
        break_even = "never" if math.isinf(self.break_even_epochs) else f"{self.break_even_epochs:.2f}"
        return {
            "workers": self.workers,
            "access": self.access,
            "compression": self.compression,
            "chunk_samples": self.chunk_samples,
            "materialization_s": f"{self.materialization_seconds:.4f}",
            "output_mib": f"{self.output_mib:.2f}",
            "online_first_batch_s": f"{self.online_first_batch_seconds:.4f}",
            "materialized_first_batch_s": f"{self.materialized_first_batch_seconds:.4f}",
            "online_samples_s": f"{self.online_samples_per_second:.1f}",
            "materialized_samples_s": f"{self.materialized_samples_per_second:.1f}",
            "break_even_epochs": break_even,
        }


def _consume_batch(batch: Any) -> float:
    """Touch collated numeric payloads so timed work cannot be skipped."""
    if isinstance(batch, dict):
        return sum(_consume_batch(value) for value in batch.values())
    if isinstance(batch, (tuple, list)):
        return sum(_consume_batch(value) for value in batch)
    if isinstance(batch, torch.Tensor):
        return float(batch.reshape(-1)[0]) if batch.numel() else 0.0
    array = np.asarray(batch)
    return float(np.real(array.reshape(-1)[0])) if array.size else 0.0


def _loader(dataset: Dataset, batch_size: int, workers: int, access: Access, seed: int) -> DataLoader:
    """Create a reproducible loader for one benchmark case."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=access == "shuffled",
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers,
        persistent_workers=workers > 0,
    )


def _measure_loader(loader: DataLoader, sample_count: int, warmup_epochs: int) -> tuple[float, float]:
    """Return cold first-batch latency and median steady epoch duration."""
    start = time.perf_counter()
    first = next(iter(loader))
    first_batch_seconds = time.perf_counter() - start
    _consume_batch(first)

    for _ in range(warmup_epochs):
        for batch in loader:
            _consume_batch(batch)

    durations = []
    for _ in range(3):
        start = time.perf_counter()
        count = 0
        for batch in loader:
            _consume_batch(batch)
            count += len(batch[0])
        duration = time.perf_counter() - start
        if count != sample_count:
            raise RuntimeError(f"Loader produced {count} samples; expected {sample_count}")
        durations.append(duration)
    return first_batch_seconds, float(np.median(durations))


def run_benchmark(
    root: Path,
    *,
    samples: int,
    iq_length: int,
    batch_size: int,
    workers: tuple[int, ...],
    access_modes: tuple[Access, ...],
    compression: str | None,
    chunk_samples: int,
    seed: int,
    warmup_epochs: int,
) -> list[Result]:
    """Materialize one workload and measure all requested DataLoader cases."""
    source = DeterministicTransformDataset(samples, iq_length, seed)
    materialized_root = root / "structured"
    start = time.perf_counter()
    materialized = materialize_structured_dataset(
        source,
        materialized_root,
        batch_size=batch_size,
        progress=False,
        overwrite=True,
        validation="sampled",
        validation_samples=min(32, samples),
        validation_seed=seed,
        writer_kwargs={
            "compression": compression,
            "shuffle": compression is not None,
            "fletcher32": False,
            "chunk_samples": chunk_samples,
        },
    )
    materialization_seconds = time.perf_counter() - start
    materialized.close()
    output_mib = (materialized_root / "data.h5").stat().st_size / (1024**2)

    results = []
    for worker_count in workers:
        for access in access_modes:
            online_loader = _loader(source, batch_size, worker_count, access, seed)
            structured = StructuredHDF5Dataset(materialized_root)
            try:
                materialized_loader = _loader(structured, batch_size, worker_count, access, seed)
                online_first, online_epoch = _measure_loader(online_loader, samples, warmup_epochs)
                materialized_first, materialized_epoch = _measure_loader(materialized_loader, samples, warmup_epochs)
                del online_loader, materialized_loader
            finally:
                structured.close()
            savings = online_epoch - materialized_epoch
            break_even = materialization_seconds / savings if savings > 0 else math.inf
            results.append(
                Result(
                    workers=worker_count,
                    access=access,
                    compression=compression or "none",
                    chunk_samples=chunk_samples,
                    materialization_seconds=materialization_seconds,
                    output_mib=output_mib,
                    online_first_batch_seconds=online_first,
                    materialized_first_batch_seconds=materialized_first,
                    online_samples_per_second=samples / online_epoch,
                    materialized_samples_per_second=samples / materialized_epoch,
                    break_even_epochs=break_even,
                )
            )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--iq-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--access", choices=("sequential", "shuffled"), nargs="+", default=["sequential", "shuffled"])
    parser.add_argument("--compression", choices=("none", "lzf"), default="none")
    parser.add_argument("--chunk-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Optional CSV output path")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("samples", "iq_length", "batch_size", "chunk_samples"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if any(worker < 0 for worker in args.workers):
        raise ValueError("--workers values must be non-negative")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be non-negative")


def _write_report(results: list[Result], output: Path | None) -> None:
    fieldnames = list(results[0].as_row())
    rows = [result.as_row() for result in results]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            file_writer = csv.DictWriter(handle, fieldnames=fieldnames)
            file_writer.writeheader()
            file_writer.writerows(rows)


def main() -> None:
    """Run the benchmark and emit a CSV report."""
    args = _parse_args()
    _validate_args(args)
    benchmark_root = Path(tempfile.mkdtemp(prefix="torchsig-structured-benchmark-"))
    try:
        results = run_benchmark(
            benchmark_root,
            samples=args.samples,
            iq_length=args.iq_length,
            batch_size=args.batch_size,
            workers=tuple(args.workers),
            access_modes=tuple(args.access),
            compression=None if args.compression == "none" else args.compression,
            chunk_samples=args.chunk_samples,
            seed=args.seed,
            warmup_epochs=args.warmup_epochs,
        )
        _write_report(results, args.output)
    finally:
        shutil.rmtree(benchmark_root)


if __name__ == "__main__":
    main()
