"""Optional end-to-end benchmark for the current HDF5 writer.

Run with:
    pytest benchmarks/optional/benchmark_hdf5_writers.py --benchmark-only
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.file_handlers.hdf5 import HDF5Reader, HDF5Writer

NUM_SIGNALS = 64
NUM_SAMPLES = 2_048
BATCH_SIZE = 16

WRITER_IMPLEMENTATIONS: dict[str, Callable] = {
    "current": HDF5Writer,
}


@pytest.fixture(scope="module")
def signals() -> list[Signal]:
    """Build representative complex signals with shared and component metadata."""
    rng = np.random.default_rng(0)
    parent = HierarchicalMetadataObject(
        metadata={"sample_rate": 1_000_000.0, "dataset_name": "writer-benchmark"}
    )
    result = []
    for idx in range(NUM_SIGNALS):
        data = (
            rng.standard_normal(NUM_SAMPLES)
            + 1j * rng.standard_normal(NUM_SAMPLES)
        ).astype(np.complex64)
        component = Signal(
            data=data[:256],
            parent=parent,
            class_name="component",
            center_freq=float(idx),
        )
        result.append(
            Signal(
                data=data,
                component_signals=[component],
                parent=parent,
                class_name="benchmark",
                sample_index=idx,
            )
        )
    return result


def _write_dataset(
    writer_class: Callable,
    root: Path,
    signals: list[Signal],
) -> int:
    writer = writer_class(
        root,
        shuffle=False,
        fletcher32=False,
        max_batches_in_memory=4,
    )
    writer.setup()
    try:
        for batch_idx, start in enumerate(range(0, len(signals), BATCH_SIZE)):
            writer.write(batch_idx, signals[start : start + BATCH_SIZE])
    finally:
        writer.teardown()
    return (root / "data.h5").stat().st_size


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("implementation_name", "writer_class"),
    WRITER_IMPLEMENTATIONS.items(),
    ids=WRITER_IMPLEMENTATIONS,
)
def test_benchmark_hdf5_writer(
    benchmark,
    tmp_path,
    signals: list[Signal],
    implementation_name: str,
    writer_class: Callable,
) -> None:
    """Measure setup, all writes, final flush, and close."""
    del implementation_name
    root = tmp_path / "dataset"
    file_size = benchmark(_write_dataset, writer_class, root, signals)
    assert file_size > 0

    reader = HDF5Reader(root)
    try:
        assert len(reader) == NUM_SIGNALS
        assert reader.read(NUM_SIGNALS - 1).data.shape == (NUM_SAMPLES,)
    finally:
        reader.teardown()
