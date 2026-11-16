"""Compare TorchSig HDF5 readers with equivalent fixed-shape sample arrays.

This is a warm-cache raw-I/O benchmark, not a model-training benchmark. It
creates equivalent legacy, packed, homogeneous, and structured datasets, checks
their values and dtypes, and emits JSON/CSV measurements. Run ``--help`` for
the configurable matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import h5py
import numpy as np
import torch

import torchsig
from torchsig.signals.signal_types import Signal
from torchsig.utils.file_handlers.hdf5 import HDF5Reader, HDF5Writer
from torchsig.utils.file_handlers.homogeneous_hdf5 import HomogeneousHDF5Reader, HomogeneousHDF5Writer
from torchsig.utils.file_handlers.packed_hdf5 import PackedHDF5Reader, PackedHDF5Writer
from torchsig.utils.file_handlers.structured_hdf5 import StructuredHDF5Reader, StructuredHDF5Writer

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

Format = Literal["standard", "packed", "homogeneous", "structured"]
Operation = Literal["sequential_individual", "random_individual", "contiguous_batch", "shuffled_batch"]
Workload = Literal["iq", "spectrogram"]

FORMATS: tuple[Format, ...] = ("standard", "packed", "homogeneous", "structured")
OPERATIONS: tuple[Operation, ...] = ("sequential_individual", "random_individual", "contiguous_batch", "shuffled_batch")
READER_CLASSES = {
    "standard": HDF5Reader,
    "packed": PackedHDF5Reader,
    "homogeneous": HomogeneousHDF5Reader,
    "structured": StructuredHDF5Reader,
}


@dataclass(frozen=True)
class DatasetConfiguration:
    """One generated reader dataset in the benchmark matrix."""

    format: Format
    workload: Workload
    compression: str
    chunk_samples: int | None
    root: str
    file_size_bytes: int


def _csv_values(value: str, allowed: set[str], label: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = set(values) - allowed
    if not values or invalid:
        raise ValueError(f"{label} must contain only {sorted(allowed)}; got {sorted(invalid)}")
    return values


def generate_workload(workload: Workload, samples: int, iq_length: int, image_size: int, seed: int) -> np.ndarray:
    """Generate deterministic representative IQ or image-like arrays."""
    rng = np.random.default_rng(seed)
    if workload == "iq":
        real = rng.standard_normal((samples, iq_length), dtype=np.float32)
        imag = rng.standard_normal((samples, iq_length), dtype=np.float32)
        return (real + 1j * imag).astype(np.complex64)
    return rng.standard_normal((samples, image_size, image_size), dtype=np.float32)


def _signals(data: np.ndarray) -> list[Signal]:
    return [Signal(data=array, class_index=index % 8) for index, array in enumerate(data)]


def _writer_options(compression: str) -> dict[str, Any]:
    enabled = compression != "none"
    return {"compression": None if not enabled else compression, "shuffle": enabled, "fletcher32": False}


def write_dataset(format_name: Format, root: Path, data: np.ndarray, compression: str, chunk_samples: int | None, batch_size: int) -> int:
    """Write one equivalent dataset and return its HDF5 file size."""
    options = _writer_options(compression)
    if format_name == "standard":
        writer: Any = HDF5Writer(root, max_batches_in_memory=4, **options)
        values: Sequence[Any] = _signals(data)
    elif format_name == "packed":
        writer = PackedHDF5Writer(root, max_batches_in_memory=4, **options)
        values = _signals(data)
    elif format_name == "homogeneous":
        writer = HomogeneousHDF5Writer(root, chunk_samples=chunk_samples, **options)
        values = _signals(data)
    else:
        writer = StructuredHDF5Writer(root, chunk_samples=chunk_samples, **options)
        values = list(data)
    with writer:
        for batch_index, start in enumerate(range(0, len(values), batch_size)):
            writer.write(batch_index, values[start : start + batch_size])
    return (root / "data.h5").stat().st_size


def _array(sample: Any) -> np.ndarray:
    return np.asarray(sample if isinstance(sample, np.ndarray) else sample.data)


def _close(reader: Any) -> None:
    close = getattr(reader, "close", None) or getattr(reader, "teardown", None)
    if close is not None:
        close()


def validate_dataset(configuration: DatasetConfiguration, expected: np.ndarray) -> None:
    """Check representative values, shapes, and dtypes before timing."""
    reader = READER_CLASSES[configuration.format](configuration.root)
    try:
        if len(reader) != len(expected):
            raise AssertionError(f"{configuration.format} length {len(reader)} != {len(expected)}")
        for index in sorted({0, len(expected) // 2, len(expected) - 1}):
            actual = _array(reader.read(index))
            if actual.dtype != expected[index].dtype or actual.shape != expected[index].shape or not np.array_equal(actual, expected[index]):
                raise AssertionError(f"{configuration.format} differs at sample {index}")
    finally:
        _close(reader)


def _native_contiguous(reader: Any, format_name: Format, start: int, stop: int) -> Iterable[np.ndarray]:
    if format_name == "homogeneous":
        return reader.read_batch(start, stop)
    if format_name == "structured":
        return reader.read_batch(start, stop)
    return (_array(reader.read(index)) for index in range(start, stop))


def _checksum(values: Iterable[Any]) -> float:
    result = 0.0
    for value in values:
        array = _array(value)
        if array.size:
            result += float(np.real(array.flat[0]))
    return result


def read_operation(reader: Any, format_name: Format, operation: Operation, indices: Sequence[int], batch_size: int) -> float:
    """Execute one complete read operation and return an anti-optimization checksum."""
    if operation in {"sequential_individual", "random_individual"}:
        return _checksum(reader.read(index) for index in indices)
    checksum = 0.0
    if operation == "contiguous_batch":
        for start in range(0, len(indices), batch_size):
            stop = min(start + batch_size, len(indices))
            checksum += _checksum(_native_contiguous(reader, format_name, start, stop))
        return checksum
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        values = reader.read_indices(batch_indices) if format_name == "structured" else (reader.read(index) for index in batch_indices)
        checksum += _checksum(values)
    return checksum


def _indices(operation: Operation, samples: int, reads: int, seed: int) -> tuple[int, ...]:
    count = min(samples, reads)
    if operation in {"sequential_individual", "contiguous_batch"}:
        return tuple(range(count))
    values = np.arange(samples)
    np.random.default_rng(seed).shuffle(values)
    return tuple(int(value) for value in values[:count])


def measure(configuration: DatasetConfiguration, operation: Operation, samples: int, reads: int, batch_size: int, warmups: int, repetitions: int, seed: int) -> dict[str, Any]:
    """Measure first-read latency and warm-cache throughput for one case."""
    reader_class = READER_CLASSES[configuration.format]
    start = time.perf_counter()
    first_reader = reader_class(configuration.root)
    try:
        first = _array(first_reader.read(0))
        first_checksum = float(np.real(first.flat[0])) if first.size else 0.0
    finally:
        _close(first_reader)
    first_read_seconds = time.perf_counter() - start

    indices = _indices(operation, samples, reads, seed)
    reader = reader_class(configuration.root)
    try:
        for _ in range(warmups):
            read_operation(reader, configuration.format, operation, indices, batch_size)
        elapsed = []
        checksums = []
        for _ in range(repetitions):
            start = time.perf_counter()
            checksums.append(read_operation(reader, configuration.format, operation, indices, batch_size))
            elapsed.append(time.perf_counter() - start)
    finally:
        _close(reader)
    if not all(np.isfinite([first_checksum, *checksums])):
        raise RuntimeError("Non-finite benchmark checksum")
    rates = [len(indices) / duration for duration in elapsed]
    return {
        **asdict(configuration),
        "operation": operation,
        "samples_read": len(indices),
        "batch_size": batch_size,
        "warmups": warmups,
        "repetitions": repetitions,
        "first_read_seconds": first_read_seconds,
        "median_seconds": statistics.median(elapsed),
        "median_samples_per_second": statistics.median(rates),
        "stdev_samples_per_second": statistics.stdev(rates) if len(rates) > 1 else None,
        "native_contiguous_batch": configuration.format in {"homogeneous", "structured"},
        "native_shuffled_batch": configuration.format == "structured",
        "warm_filesystem_cache": True,
    }


def configurations(
    formats: Sequence[Format], workloads: Sequence[Workload], compressions: Sequence[str], chunk_sizes: Sequence[int], output_dir: Path, arrays: dict[Workload, np.ndarray], batch_size: int
) -> list[DatasetConfiguration]:
    """Generate the supported format/configuration matrix."""
    result = []
    for workload in workloads:
        for format_name in formats:
            supported_chunks: Sequence[int | None] = chunk_sizes if format_name in {"homogeneous", "structured"} else (None,)
            for compression in compressions:
                for chunk_samples in supported_chunks:
                    suffix = "na" if chunk_samples is None else str(chunk_samples)
                    root = output_dir / "datasets" / f"{workload}-{format_name}-{compression}-chunk{suffix}"
                    size = write_dataset(format_name, root, arrays[workload], compression, chunk_samples, batch_size)
                    configuration = DatasetConfiguration(format_name, workload, compression, chunk_samples, str(root), size)
                    validate_dataset(configuration, arrays[workload])
                    result.append(configuration)
    return result


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torchsig": torchsig.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "h5py": h5py.__version__,
    }


def write_results(path: Path, rows: list[dict[str, Any]], arguments: dict[str, Any]) -> None:
    """Write machine-readable JSON and CSV benchmark results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": _environment(),
        "arguments": arguments,
        "format_controls": {
            "standard": {"compression": True, "chunk_samples": False},
            "packed": {"compression": True, "chunk_samples": False},
            "homogeneous": {"compression": True, "chunk_samples": True},
            "structured": {"compression": True, "chunk_samples": True},
        },
        "measurements": rows,
    }
    path.write_text(json.dumps(payload, indent=2))
    fields = sorted({key for row in rows for key in row})
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated benchmark datasets")
    parser.add_argument("--results", type=Path, required=True, help="JSON output path; a sibling CSV is also written")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--reads", type=int, default=128)
    parser.add_argument("--iq-length", type=int, default=4096)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--formats", default=",".join(FORMATS))
    parser.add_argument("--workloads", default="iq,spectrogram")
    parser.add_argument("--compressions", default="none,lzf")
    parser.add_argument("--chunk-sizes", default="1,8,32")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing generated datasets directory")
    return parser.parse_args()


def main() -> None:
    """Generate equivalent files, validate them, benchmark reads, and report results."""
    args = _parse_args()
    if min(args.samples, args.reads, args.iq_length, args.image_size, args.batch_size, args.repetitions) < 1 or args.warmups < 0:
        raise ValueError("counts, dimensions, batch size, and repetitions must be positive; warmups must be non-negative")
    formats = _csv_values(args.formats, set(FORMATS), "formats")
    workloads = _csv_values(args.workloads, {"iq", "spectrogram"}, "workloads")
    compressions = _csv_values(args.compressions, {"none", "lzf"}, "compressions")
    try:
        chunk_sizes = tuple(int(value) for value in args.chunk_sizes.split(",") if value.strip())
    except ValueError as error:
        raise ValueError("chunk sizes must be comma-separated positive integers") from error
    if not chunk_sizes or any(value < 1 for value in chunk_sizes):
        raise ValueError("chunk sizes must be positive")
    dataset_dir = args.output_dir / "datasets"
    if dataset_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Generated datasets already exist: {dataset_dir}; pass --overwrite to replace them")
        shutil.rmtree(dataset_dir)
    arrays = {workload: generate_workload(workload, args.samples, args.iq_length, args.image_size, args.seed) for workload in workloads}
    configs = configurations(formats, workloads, compressions, chunk_sizes, args.output_dir, arrays, args.batch_size)
    print(f"Validated {len(configs)} equivalent datasets.")
    rows = []
    total = len(configs) * len(OPERATIONS)
    for number, configuration in enumerate(configs, start=1):
        for operation_index, operation in enumerate(OPERATIONS):
            case_number = (number - 1) * len(OPERATIONS) + operation_index + 1
            print(f"[{case_number}/{total}] {configuration.workload} {configuration.format} {configuration.compression} chunk={configuration.chunk_samples} {operation}")
            rows.append(measure(configuration, operation, args.samples, args.reads, args.batch_size, args.warmups, args.repetitions, args.seed))
    arguments = vars(args).copy()
    arguments.update(formats=formats, workloads=workloads, compressions=compressions, chunk_sizes=chunk_sizes)
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in arguments.items()}
    write_results(args.results, rows, arguments)
    print(f"JSON: {args.results}")
    print(f"CSV: {args.results.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
