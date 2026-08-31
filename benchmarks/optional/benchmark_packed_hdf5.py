"""Optional comparison of object-per-record and packed HDF5 formats.

Run with:
    pytest benchmarks/optional/benchmark_packed_hdf5.py --benchmark-only
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.file_handlers.hdf5 import HDF5Reader, HDF5Writer
from torchsig.utils.file_handlers.packed_hdf5 import (
    PackedHDF5Reader,
    PackedHDF5Writer,
)

NUM_SIGNALS = 256
NUM_SAMPLES = 2_048
BATCH_SIZE = 32
READS_PER_ROUND = 64
WRITE_BATCH_SIZES = (1, 8, 32, NUM_SIGNALS)
VALIDATION_SIGNAL_COUNTS = (128, 1_024, 4_096)
VALIDATION_NUM_SAMPLES = 16
PACKED_WRITE_CONFIGS = {
    "uncompressed": {
        "compression": None,
        "shuffle": False,
        "fletcher32": False,
    },
    "lzf": {
        "compression": "lzf",
        "shuffle": False,
        "fletcher32": False,
    },
    "lzf-shuffle-checksum": {
        "compression": "lzf",
        "shuffle": True,
        "fletcher32": True,
    },
}

WRITERS: dict[str, Callable] = {
    "current": HDF5Writer,
    "packed": PackedHDF5Writer,
}

READ_FORMATS: dict[str, tuple[Callable, Callable]] = {
    "current": (HDF5Writer, HDF5Reader),
    "packed": (PackedHDF5Writer, PackedHDF5Reader),
}


@pytest.fixture(scope="module")
def signals() -> list[Signal]:
    """Create representative top-level and component signals."""
    rng = np.random.default_rng(0)
    parent = HierarchicalMetadataObject(metadata={"sample_rate": 1_000_000.0, "dataset_name": "packed-benchmark"})
    result = []
    for idx in range(NUM_SIGNALS):
        data = (rng.standard_normal(NUM_SAMPLES) + 1j * rng.standard_normal(NUM_SAMPLES)).astype(np.complex64)
        component = Signal(
            data=data[:256],
            class_name="component",
            center_freq=float(idx),
        )
        component.add_parent(parent, register=False)
        signal = Signal(
            data=data,
            component_signals=[component],
            class_name="benchmark",
            sample_index=idx,
        )
        signal.add_parent(parent, register=False)
        result.append(signal)
    return result


def _write(
    writer_class: Callable,
    root: Path,
    signals: list[Signal],
    *,
    batch_size: int = BATCH_SIZE,
    **writer_kwargs,
) -> int:
    options = {
        "shuffle": False,
        "fletcher32": False,
        "max_batches_in_memory": 4,
    }
    options.update(writer_kwargs)
    writer = writer_class(root, **options)
    writer.setup()
    try:
        for batch_idx, start in enumerate(range(0, len(signals), batch_size)):
            writer.write(batch_idx, signals[start : start + batch_size])
    finally:
        writer.teardown()
    return (root / "data.h5").stat().st_size


def _read(reader, indices: tuple[int, ...]) -> float:
    return sum(float(reader.read(idx).data[0].real) for idx in indices)


def _open_and_validate(root: Path) -> int:
    """Open a packed reader, trigger validation, and close it."""
    reader = PackedHDF5Reader(root)
    try:
        return len(reader)
    finally:
        reader.teardown()


def _open_and_read_first(root: Path) -> tuple[int, ...]:
    """Open and validate a packed reader, read one signal, and close it."""
    reader = PackedHDF5Reader(root)
    try:
        return reader.read(0).data.shape
    finally:
        reader.teardown()


def _validation_signals(signal_count: int) -> list[Signal]:
    """Create small signals which isolate validation cost from IQ reads."""
    data = np.arange(VALIDATION_NUM_SAMPLES, dtype=np.complex64)
    return [Signal(data=data, class_name="validation", sample_index=idx) for idx in range(signal_count)]


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("format_name", "writer_class"),
    WRITERS.items(),
    ids=WRITERS,
)
def test_benchmark_hdf5_format_write(
    benchmark,
    tmp_path,
    signals: list[Signal],
    format_name: str,
    writer_class: Callable,
) -> None:
    """Measure complete file creation, writing, flushing, and closing."""
    del format_name
    root = tmp_path / "dataset"
    file_size = benchmark(_write, writer_class, root, signals)
    benchmark.extra_info["file_size_mib"] = file_size / (1024**2)
    reader_class = PackedHDF5Reader if writer_class is PackedHDF5Writer else HDF5Reader
    reader = reader_class(root)
    try:
        assert len(reader) == NUM_SIGNALS
        assert reader.read(NUM_SIGNALS - 1).data.shape == (NUM_SAMPLES,)
    finally:
        reader.teardown()


@pytest.mark.benchmark
@pytest.mark.parametrize("batch_size", WRITE_BATCH_SIZES)
def test_benchmark_packed_hdf5_write_batch_size(
    benchmark,
    tmp_path,
    signals: list[Signal],
    batch_size: int,
) -> None:
    """Measure packed write performance as the generated batch size changes."""
    root = tmp_path / "dataset"
    file_size = benchmark(
        _write,
        PackedHDF5Writer,
        root,
        signals,
        batch_size=batch_size,
        compression=None,
    )
    benchmark.extra_info["batch_size"] = batch_size
    benchmark.extra_info["signals"] = len(signals)
    benchmark.extra_info["file_size_mib"] = file_size / (1024**2)

    reader = PackedHDF5Reader(root)
    try:
        assert len(reader) == len(signals)
    finally:
        reader.teardown()


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("configuration", "writer_kwargs"),
    PACKED_WRITE_CONFIGS.items(),
    ids=PACKED_WRITE_CONFIGS,
)
def test_benchmark_packed_hdf5_write_filters(
    benchmark,
    tmp_path,
    signals: list[Signal],
    configuration: str,
    writer_kwargs: dict,
) -> None:
    """Measure packed write time and file size for common HDF5 filters."""
    root = tmp_path / "dataset"
    file_size = benchmark(
        _write,
        PackedHDF5Writer,
        root,
        signals,
        **writer_kwargs,
    )
    benchmark.extra_info["configuration"] = configuration
    benchmark.extra_info["signals"] = len(signals)
    benchmark.extra_info["file_size_mib"] = file_size / (1024**2)

    reader = PackedHDF5Reader(root)
    try:
        assert len(reader) == len(signals)
    finally:
        reader.teardown()


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("format_name", "writer_class", "reader_class"),
    [(name, *classes) for name, classes in READ_FORMATS.items()],
    ids=READ_FORMATS,
)
def test_benchmark_hdf5_format_warm_random_read(
    benchmark,
    tmp_path_factory,
    signals: list[Signal],
    format_name: str,
    writer_class: Callable,
    reader_class: Callable,
) -> None:
    """Measure repeated random access after warming the selected records."""
    root = tmp_path_factory.mktemp(f"packed-reader-{format_name}")
    _write(writer_class, root, signals)
    indices = tuple(int(idx) for idx in np.random.default_rng(1).integers(0, NUM_SIGNALS, size=READS_PER_ROUND))
    reader = reader_class(root)
    try:
        assert np.isfinite(_read(reader, indices))
        result = benchmark(_read, reader, indices)
        assert np.isfinite(result)
    finally:
        reader.teardown()


@pytest.mark.benchmark
@pytest.mark.parametrize("signal_count", VALIDATION_SIGNAL_COUNTS)
def test_benchmark_packed_hdf5_cold_open_validation_scaling(
    benchmark,
    tmp_path,
    signal_count: int,
) -> None:
    """Measure reader creation, full integrity validation, and close."""
    root = tmp_path / "dataset"
    signals = _validation_signals(signal_count)
    file_size = _write(
        PackedHDF5Writer,
        root,
        signals,
        batch_size=min(BATCH_SIZE, signal_count),
        compression=None,
    )

    actual_count = benchmark(_open_and_validate, root)

    assert actual_count == signal_count
    benchmark.extra_info["signals"] = signal_count
    benchmark.extra_info["records"] = signal_count
    benchmark.extra_info["file_size_mib"] = file_size / (1024**2)


@pytest.mark.benchmark
@pytest.mark.parametrize("signal_count", VALIDATION_SIGNAL_COUNTS)
def test_benchmark_packed_hdf5_cold_first_read_scaling(
    benchmark,
    tmp_path,
    signal_count: int,
) -> None:
    """Measure cold open, full validation, first signal read, and close."""
    root = tmp_path / "dataset"
    signals = _validation_signals(signal_count)
    file_size = _write(
        PackedHDF5Writer,
        root,
        signals,
        batch_size=min(BATCH_SIZE, signal_count),
        compression=None,
    )

    shape = benchmark(_open_and_read_first, root)

    assert shape == (VALIDATION_NUM_SAMPLES,)
    benchmark.extra_info["signals"] = signal_count
    benchmark.extra_info["records"] = signal_count
    benchmark.extra_info["file_size_mib"] = file_size / (1024**2)
