"""Compare standard, packed, and homogeneous HDF5 across signal workloads.

The matrix covers narrowband IQ, wideband IQ, and spectrogram observations,
each with variable component counts and with compression enabled and disabled.
It also measures end-to-end ``DatasetCreator`` writes and sequential or
shuffled DataLoader epochs with 0, 2, and 8 workers.

Run the complete matrix with:
    pytest benchmarks/optional/benchmark_homogeneous_hdf5.py --benchmark-only

Run one workload with, for example:
    pytest benchmarks/optional/benchmark_homogeneous_hdf5.py \
        --benchmark-only -k wideband
"""

import multiprocessing
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import torchsig.utils.file_handlers.homogeneous_hdf5 as homogeneous_module
from torchsig.datasets.datasets import StaticTorchSigDataset
from torchsig.signals.signal_types import Signal
from torchsig.utils.file_handlers.hdf5 import HDF5Reader, HDF5Writer
from torchsig.utils.file_handlers.homogeneous_hdf5 import (
    HomogeneousHDF5Reader,
    HomogeneousHDF5Writer,
)
from torchsig.utils.file_handlers.packed_hdf5 import (
    PackedHDF5Reader,
    PackedHDF5Writer,
)
from torchsig.utils.writer import DatasetCreator, identity_collate_fn


@dataclass(frozen=True)
class Workload:
    """Configuration for one representative signal workload."""

    name: str
    signal_count: int
    shape: tuple[int, ...]
    dtype: np.dtype
    max_components: int
    batch_size: int
    read_count: int


WORKLOADS = (
    Workload(
        name="narrowband-iq",
        signal_count=512,
        shape=(2_048,),
        dtype=np.dtype(np.complex64),
        max_components=3,
        batch_size=32,
        read_count=64,
    ),
    Workload(
        name="wideband-iq",
        signal_count=96,
        shape=(65_536,),
        dtype=np.dtype(np.complex64),
        max_components=12,
        batch_size=8,
        read_count=16,
    ),
    Workload(
        name="spectrogram",
        signal_count=128,
        shape=(128, 256),
        dtype=np.dtype(np.float32),
        max_components=8,
        batch_size=16,
        read_count=16,
    ),
)

FORMATS: dict[str, tuple[Callable, Callable]] = {
    "standard": (HDF5Writer, HDF5Reader),
    "packed": (PackedHDF5Writer, PackedHDF5Reader),
    "homogeneous": (HomogeneousHDF5Writer, HomogeneousHDF5Reader),
}
COMPRESSIONS = {
    "none": None,
    "lzf": "lzf",
}
CASES = tuple(
    (workload, compression_name, compression, format_name, *classes) for workload in WORKLOADS for compression_name, compression in COMPRESSIONS.items() for format_name, classes in FORMATS.items()
)
WORKER_CASES = tuple(
    (workload, format_name, *classes, num_workers, shuffled)
    for workload in WORKLOADS
    for format_name, classes in FORMATS.items()
    for num_workers in (0, 2, 8)
    for shuffled in (False, True)
)


def _case_id(case: tuple) -> str:
    workload, compression_name, _, format_name, *_ = case
    return f"{workload.name}-{compression_name}-{format_name}"


def _worker_case_id(case: tuple) -> str:
    workload, format_name, _, _, num_workers, shuffled = case
    access = "shuffled" if shuffled else "sequential"
    return f"{workload.name}-{format_name}-{access}-{num_workers}-workers"


def _identity_worker_collate(batch: list[np.ndarray]) -> list[np.ndarray]:
    return batch


def _random_array(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    if np.issubdtype(dtype, np.complexfloating):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    return rng.standard_normal(shape).astype(dtype)


def _iq_components(data: np.ndarray, count: int) -> list[Signal]:
    components = []
    for component_idx in range(count):
        length = 256 + 32 * component_idx
        start = component_idx * 1_024
        components.append(
            Signal(
                data=data[start : start + length],
                component_index=component_idx,
            )
        )
    return components


def _spectrogram_components(data: np.ndarray, count: int) -> list[Signal]:
    components = []
    for component_idx in range(count):
        height = 8 + component_idx
        width = 16 + 2 * component_idx
        row = (component_idx * 11) % (data.shape[0] - height + 1)
        column = (component_idx * 17) % (data.shape[1] - width + 1)
        components.append(
            Signal(
                data=data[row : row + height, column : column + width],
                component_index=component_idx,
            )
        )
    return components


def _make_signals(workload: Workload) -> list[Signal]:
    rng = np.random.default_rng(0)
    signals = []
    for idx in range(workload.signal_count):
        data = _random_array(rng, workload.shape, workload.dtype)
        component_count = idx % (workload.max_components + 1)
        components = _spectrogram_components(data, component_count) if workload.name == "spectrogram" else _iq_components(data, component_count)
        signals.append(
            Signal(
                data=data,
                component_signals=components,
                sample_index=idx,
            )
        )
    return signals


def _write(
    writer_class: Callable,
    root: Path,
    signals: list[Signal],
    workload: Workload,
    compression: str | None,
) -> int:
    writer = writer_class(
        root,
        compression=compression,
        shuffle=compression is not None,
        fletcher32=False,
    )
    writer.setup()
    try:
        for batch_idx, start in enumerate(range(0, len(signals), workload.batch_size)):
            writer.write(
                batch_idx,
                signals[start : start + workload.batch_size],
            )
    finally:
        writer.teardown()
    return (root / "data.h5").stat().st_size


def _dataset_creator_write(
    writer_class: Callable,
    root: Path,
    signals: list[Signal],
    workload: Workload,
    compression: str | None,
) -> int:
    loader = DataLoader(
        signals,
        batch_size=workload.batch_size,
        collate_fn=identity_collate_fn,
    )
    DatasetCreator(
        dataloader=loader,
        root=root,
        overwrite=True,
        file_handler=writer_class,
        multithreading=False,
        compression=compression,
        shuffle=compression is not None,
        fletcher32=False,
        tqdm_desc="Benchmark creation",
    ).create()
    return (root / "data.h5").stat().st_size


def _profile_dataset_creator_write(
    writer_class: Callable,
    root: Path,
    signals: list[Signal],
    workload: Workload,
    compression: str | None,
) -> tuple[int, int]:
    tracemalloc.start()
    try:
        file_size = _dataset_creator_write(
            writer_class,
            root,
            signals,
            workload,
            compression,
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return file_size, peak


def _random_read(reader, indices: tuple[int, ...]) -> float:
    return sum(float(reader.read(idx).data.reshape(-1)[0].real) for idx in indices)


def _packed_contiguous_read(
    reader: PackedHDF5Reader,
    start: int,
    stop: int,
) -> np.ndarray:
    return np.stack([reader.read(idx).data for idx in range(start, stop)])


def _contiguous_signal_read(reader, start: int, stop: int) -> int:
    signals = (
        reader.read_signals_batch(start, stop)
        if isinstance(reader, HomogeneousHDF5Reader)
        else [reader.read(idx) for idx in range(start, stop)]
    )
    return sum(1 + len(signal.component_signals) for signal in signals)


def _open_reader(reader_class: Callable, root: Path) -> int:
    reader = reader_class(root)
    try:
        return len(reader)
    finally:
        reader.teardown()


def _cold_reader_open_peak(
    reader_class: Callable,
    root: Path,
) -> int:
    if reader_class is HomogeneousHDF5Reader:
        homogeneous_module._VALIDATED_FILES.pop(str(root / "data.h5"), None)  # noqa: SLF001
    tracemalloc.start()
    try:
        _open_reader(reader_class, root)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def _worker_loader(
    root: Path,
    workload: Workload,
    reader_class: Callable,
    num_workers: int,
    shuffled: bool,
) -> DataLoader:
    context = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
    generator = torch.Generator().manual_seed(0)
    return DataLoader(
        StaticTorchSigDataset(
            root=root,
            file_handler_class=reader_class,
            target_labels=[],
        ),
        batch_size=workload.read_count,
        num_workers=num_workers,
        multiprocessing_context=context if num_workers else None,
        collate_fn=_identity_worker_collate,
        shuffle=shuffled,
        generator=generator,
    )


def _first_worker_batch(
    root: Path,
    workload: Workload,
    reader_class: Callable,
    num_workers: int,
    shuffled: bool,
) -> tuple[int, ...]:
    iterator = iter(
        _worker_loader(
            root,
            workload,
            reader_class,
            num_workers,
            shuffled,
        )
    )
    try:
        batch = next(iterator)
        return (len(batch), *batch[0].shape)
    finally:
        if num_workers:
            iterator._shutdown_workers()  # noqa: SLF001


def _worker_epoch(
    root: Path,
    workload: Workload,
    reader_class: Callable,
    num_workers: int,
    shuffled: bool,
) -> int:
    count = 0
    for batch in _worker_loader(
        root,
        workload,
        reader_class,
        num_workers,
        shuffled,
    ):
        count += len(batch)
    return count


def _set_extra_info(
    benchmark,
    workload: Workload,
    compression_name: str,
    file_size: int,
) -> None:
    benchmark.extra_info["workload"] = workload.name
    benchmark.extra_info["compression"] = compression_name
    benchmark.extra_info["signals"] = workload.signal_count
    benchmark.extra_info["shape"] = workload.shape
    benchmark.extra_info["max_components"] = workload.max_components
    benchmark.extra_info["file_size_mib"] = file_size / (1024**2)


@pytest.mark.benchmark
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_benchmark_homogeneous_reader_open(
    benchmark,
    tmp_path,
    case,
) -> None:
    """Measure warm reader opening and record cold-open peak allocation."""
    (
        workload,
        compression_name,
        compression,
        format_name,
        writer_class,
        reader_class,
    ) = case
    signals = _make_signals(workload)
    root = tmp_path / f"dataset-{format_name}"
    file_size = _write(
        writer_class,
        root,
        signals,
        workload,
        compression,
    )
    peak_bytes = _cold_reader_open_peak(reader_class, root)
    assert benchmark(_open_reader, reader_class, root) == len(signals)
    _set_extra_info(benchmark, workload, compression_name, file_size)
    benchmark.extra_info["cold_open_peak_mib"] = peak_bytes / (1024**2)


@pytest.mark.benchmark
@pytest.mark.parametrize("case", WORKER_CASES, ids=_worker_case_id)
def test_benchmark_homogeneous_worker_startup(
    benchmark,
    tmp_path,
    case,
) -> None:
    """Measure DataLoader construction and first-batch latency."""
    (
        workload,
        format_name,
        writer_class,
        reader_class,
        num_workers,
        shuffled,
    ) = case
    signals = _make_signals(workload)
    root = tmp_path / f"dataset-{format_name}"
    _write(
        writer_class,
        root,
        signals,
        workload,
        compression=None,
    )
    shape = benchmark(
        _first_worker_batch,
        root,
        workload,
        reader_class,
        num_workers,
        shuffled,
    )
    assert shape == (workload.read_count, *workload.shape)
    benchmark.extra_info["workload"] = workload.name
    benchmark.extra_info["format"] = format_name
    benchmark.extra_info["workers"] = num_workers
    benchmark.extra_info["access"] = "shuffled" if shuffled else "sequential"
    benchmark.extra_info["batch_size"] = workload.read_count


@pytest.mark.benchmark
@pytest.mark.parametrize("case", WORKER_CASES, ids=_worker_case_id)
def test_benchmark_homogeneous_worker_epoch(
    benchmark,
    tmp_path,
    case,
) -> None:
    """Measure full-epoch DataLoader throughput."""
    (
        workload,
        format_name,
        writer_class,
        reader_class,
        num_workers,
        shuffled,
    ) = case
    signals = _make_signals(workload)
    root = tmp_path / f"dataset-{format_name}"
    _write(
        writer_class,
        root,
        signals,
        workload,
        compression=None,
    )
    count = benchmark(
        _worker_epoch,
        root,
        workload,
        reader_class,
        num_workers,
        shuffled,
    )
    assert count == workload.signal_count
    benchmark.extra_info["workload"] = workload.name
    benchmark.extra_info["format"] = format_name
    benchmark.extra_info["workers"] = num_workers
    benchmark.extra_info["access"] = "shuffled" if shuffled else "sequential"
    benchmark.extra_info["signals"] = workload.signal_count


@pytest.mark.benchmark
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_benchmark_dataset_creator_write(
    benchmark,
    tmp_path,
    case,
) -> None:
    """Measure end-to-end DatasetCreator time, size, and peak allocation."""
    (
        workload,
        compression_name,
        compression,
        format_name,
        writer_class,
        _,
    ) = case
    signals = _make_signals(workload)
    profile_root = tmp_path / f"profile-{format_name}"
    file_size, peak_bytes = _profile_dataset_creator_write(
        writer_class,
        profile_root,
        signals,
        workload,
        compression,
    )
    benchmark_root = tmp_path / f"benchmark-{format_name}"
    measured_size = benchmark.pedantic(
        _dataset_creator_write,
        args=(
            writer_class,
            benchmark_root,
            signals,
            workload,
            compression,
        ),
        iterations=1,
        rounds=1,
    )
    size_tolerance = max(4_096, file_size // 100)
    assert abs(measured_size - file_size) <= size_tolerance
    _set_extra_info(benchmark, workload, compression_name, measured_size)
    benchmark.extra_info["format"] = format_name
    benchmark.extra_info["peak_python_mib"] = peak_bytes / (1024**2)
    benchmark.extra_info["profile_file_size_mib"] = file_size / (1024**2)


@pytest.mark.benchmark
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_benchmark_homogeneous_format_write(
    benchmark,
    tmp_path,
    case,
) -> None:
    """Measure full writes and record the resulting file size."""
    (
        workload,
        compression_name,
        compression,
        _,
        writer_class,
        _,
    ) = case
    signals = _make_signals(workload)
    root = tmp_path / "dataset"
    file_size = benchmark(
        _write,
        writer_class,
        root,
        signals,
        workload,
        compression,
    )
    _set_extra_info(benchmark, workload, compression_name, file_size)


@pytest.mark.benchmark
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_benchmark_homogeneous_format_random_read(
    benchmark,
    tmp_path,
    case,
) -> None:
    """Measure warm random reads including component reconstruction."""
    (
        workload,
        compression_name,
        compression,
        format_name,
        writer_class,
        reader_class,
    ) = case
    signals = _make_signals(workload)
    root = tmp_path / f"dataset-{format_name}"
    file_size = _write(
        writer_class,
        root,
        signals,
        workload,
        compression,
    )
    indices = tuple(
        int(value)
        for value in np.random.default_rng(1).integers(
            0,
            len(signals),
            size=workload.read_count,
        )
    )
    reader = reader_class(root)
    try:
        assert np.isfinite(_random_read(reader, indices))
        assert np.isfinite(benchmark(_random_read, reader, indices))
    finally:
        reader.teardown()
    _set_extra_info(benchmark, workload, compression_name, file_size)


@pytest.mark.benchmark
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_benchmark_homogeneous_format_contiguous_batch_read(
    benchmark,
    tmp_path,
    case,
) -> None:
    """Measure top-level-only contiguous batch reads."""
    (
        workload,
        compression_name,
        compression,
        format_name,
        writer_class,
        reader_class,
    ) = case
    signals = _make_signals(workload)
    root = tmp_path / f"dataset-{format_name}"
    file_size = _write(
        writer_class,
        root,
        signals,
        workload,
        compression,
    )
    start = (workload.signal_count - workload.read_count) // 2
    stop = start + workload.read_count
    reader = reader_class(root)
    try:
        actual = (
            benchmark(reader.read_batch, start, stop)
            if isinstance(reader, HomogeneousHDF5Reader)
            else benchmark(
                _packed_contiguous_read,
                reader,
                start,
                stop,
            )
        )
        assert actual.shape == (workload.read_count, *workload.shape)
    finally:
        reader.teardown()
    _set_extra_info(benchmark, workload, compression_name, file_size)


@pytest.mark.benchmark
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_benchmark_homogeneous_contiguous_signal_batch_read(
    benchmark,
    tmp_path,
    case,
) -> None:
    """Measure contiguous reads preserving metadata and variable components."""
    (
        workload,
        compression_name,
        compression,
        format_name,
        writer_class,
        reader_class,
    ) = case
    signals = _make_signals(workload)
    root = tmp_path / f"dataset-{format_name}"
    file_size = _write(
        writer_class,
        root,
        signals,
        workload,
        compression,
    )
    start = (workload.signal_count - workload.read_count) // 2
    stop = start + workload.read_count
    expected_count = sum(
        1 + len(signal.component_signals)
        for signal in signals[start:stop]
    )
    reader = reader_class(root)
    try:
        assert (
            benchmark(_contiguous_signal_read, reader, start, stop)
            == expected_count
        )
    finally:
        reader.teardown()
    _set_extra_info(benchmark, workload, compression_name, file_size)
