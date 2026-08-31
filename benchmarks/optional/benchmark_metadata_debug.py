"""Optional benchmarks for metadata resolution and debug logging."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import (
    HierarchicalMetadataObject,
    MetadataAttributeError,
)
from torchsig.utils.metadata_logging import metadata_logging_context

if TYPE_CHECKING:
    from collections.abc import Iterator

LOGGER_NAME = "torchsig.metadata"


class MetadataContextBenchmarkDataset(TorchSigIterableDataset):
    """Minimal dataset isolating automatic sample-context overhead."""

    def __generate_new_signal__(self) -> Signal:
        """Return an empty signal without running RF generation."""
        return Signal(num_signals_max=1)


@pytest.fixture
def metadata_objects() -> tuple[
    HierarchicalMetadataObject,
    HierarchicalMetadataObject,
]:
    """Create objects representing local and inherited metadata lookups."""
    parent = HierarchicalMetadataObject(metadata={"field": 1})
    local = HierarchicalMetadataObject(metadata={"field": 1})
    inherited = HierarchicalMetadataObject(parent=parent)
    return local, inherited


@pytest.fixture
def filtered_debug_logger() -> Iterator[None]:
    """Configure the metadata logger to filter DEBUG records."""
    logger = logging.getLogger(LOGGER_NAME)
    previous_level = logger.level
    previous_propagate = logger.propagate
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    try:
        yield
    finally:
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


@pytest.fixture
def emitted_debug_logger() -> Iterator[None]:
    """Configure DEBUG records to be handled without formatting or I/O."""
    logger = logging.getLogger(LOGGER_NAME)
    handler = logging.NullHandler()
    previous_level = logger.level
    previous_propagate = logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


def lookup_missing(obj: HierarchicalMetadataObject) -> None:
    """Perform a missing lookup while retaining its exception cost."""
    try:
        obj["missing"]
    except MetadataAttributeError:
        return
    raise AssertionError("missing metadata lookup unexpectedly succeeded")


@pytest.mark.benchmark(group="metadata-lookup")
def test_benchmark_local_lookup_debug_disabled(benchmark, metadata_objects):
    """Benchmark the normal local metadata lookup path."""
    local, _ = metadata_objects

    result = benchmark(local.__getitem__, "field")

    assert result == 1


@pytest.mark.benchmark(group="metadata-lookup")
def test_benchmark_inherited_lookup_debug_disabled(benchmark, metadata_objects):
    """Benchmark the normal inherited metadata lookup path."""
    _, inherited = metadata_objects

    result = benchmark(inherited.__getitem__, "field")

    assert result == 1


@pytest.mark.benchmark(group="metadata-lookup")
def test_benchmark_missing_lookup_debug_disabled(benchmark, metadata_objects):
    """Benchmark a normal missing metadata lookup and exception."""
    local, _ = metadata_objects

    benchmark(lookup_missing, local)


@pytest.mark.benchmark(group="metadata-resolution")
def test_benchmark_explain_local_metadata(benchmark, metadata_objects):
    """Benchmark structured resolution of local metadata."""
    local, _ = metadata_objects

    result = benchmark(local.explain_metadata, "field")

    assert result.source == "local"


@pytest.mark.benchmark(group="metadata-resolution")
def test_benchmark_explain_inherited_metadata(benchmark, metadata_objects):
    """Benchmark structured resolution of inherited metadata."""
    _, inherited = metadata_objects

    result = benchmark(inherited.explain_metadata, "field")

    assert result.source == "inherited"


@pytest.mark.benchmark(group="metadata-debug-filtered")
def test_benchmark_local_lookup_debug_filtered(
    benchmark,
    metadata_objects,
    filtered_debug_logger,
):
    """Benchmark enabled debug mode when DEBUG records are filtered."""
    local, _ = metadata_objects
    local.enable_metadata_debug()

    result = benchmark(local.__getitem__, "field")

    assert result == 1


@pytest.mark.benchmark(group="metadata-debug-filtered")
def test_benchmark_inherited_lookup_debug_filtered(
    benchmark,
    metadata_objects,
    filtered_debug_logger,
):
    """Benchmark inherited lookup when DEBUG records are filtered."""
    _, inherited = metadata_objects
    inherited.enable_metadata_debug()

    result = benchmark(inherited.__getitem__, "field")

    assert result == 1


@pytest.mark.benchmark(group="metadata-debug-emitted")
def test_benchmark_local_lookup_debug_emitted(
    benchmark,
    metadata_objects,
    emitted_debug_logger,
):
    """Benchmark local lookup when a no-op handler receives debug records."""
    local, _ = metadata_objects
    local.enable_metadata_debug()

    result = benchmark(local.__getitem__, "field")

    assert result == 1


@pytest.mark.benchmark(group="metadata-debug-emitted")
def test_benchmark_inherited_lookup_debug_emitted(
    benchmark,
    metadata_objects,
    emitted_debug_logger,
):
    """Benchmark inherited lookup when a no-op handler receives records."""
    _, inherited = metadata_objects
    inherited.enable_metadata_debug()

    result = benchmark(inherited.__getitem__, "field")

    assert result == 1


@pytest.mark.benchmark(group="metadata-debug-suppressed")
def test_benchmark_lookup_rejected_by_key_filter(
    benchmark,
    metadata_objects,
    emitted_debug_logger,
):
    """Benchmark an enabled lookup rejected before record construction."""
    local, _ = metadata_objects
    local.enable_metadata_debug(keys={"different_field"})

    result = benchmark(local.__getitem__, "field")

    assert result == 1


@pytest.mark.benchmark(group="metadata-debug-suppressed")
def test_benchmark_lookup_rejected_by_rate_limit(
    benchmark,
    metadata_objects,
    emitted_debug_logger,
):
    """Benchmark an enabled lookup after its event limit is exhausted."""
    local, _ = metadata_objects
    local.enable_metadata_debug(max_events=0)

    result = benchmark(local.__getitem__, "field")

    assert result == 1


@pytest.mark.benchmark(group="metadata-debug-emitted")
def test_benchmark_local_lookup_debug_with_value(
    benchmark,
    metadata_objects,
    emitted_debug_logger,
):
    """Benchmark an emitted record with a bounded value representation."""
    local, _ = metadata_objects
    local.enable_metadata_debug(include_values=True)

    result = benchmark(local.__getitem__, "field")

    assert result == 1


@pytest.mark.benchmark(group="metadata-debug-emitted")
def test_benchmark_local_lookup_with_correlation_context(
    benchmark,
    metadata_objects,
    emitted_debug_logger,
):
    """Benchmark an emitted record with active correlation fields."""
    local, _ = metadata_objects
    local.enable_metadata_debug()

    with metadata_logging_context(
        session_id="benchmark-session",
        dataset_id="benchmark-dataset",
        sample_index=42,
        worker_id=1,
        fields={"split": "train"},
    ):
        result = benchmark(local.__getitem__, "field")

    assert result == 1


@pytest.mark.benchmark(group="metadata-debug-snapshot")
def test_benchmark_completed_metadata_snapshot(
    benchmark,
    emitted_debug_logger,
):
    """Benchmark a completed sample snapshot with component metadata."""
    dataset = HierarchicalMetadataObject(metadata={"sample_rate": 10_000_000})
    component = Signal(parent=dataset, class_name="bpsk", snr_db=20.0)
    sample = Signal(
        parent=dataset,
        component_signals=[component],
        center_freq=0.0,
    )
    dataset.enable_metadata_debug(
        events={"snapshot"},
        include_values=True,
    )

    benchmark(
        dataset.log_metadata_snapshot,
        sample,
        include_components=True,
    )

    assert dataset.metadata_debug_statistics.emitted_events > 0


@pytest.mark.benchmark(group="dataset-metadata-context")
def test_benchmark_dataset_next_debug_disabled(benchmark):
    """Benchmark dataset iteration when correlation context is inactive."""
    dataset = MetadataContextBenchmarkDataset(
        signal_generators=[],
        target_labels=None,
        validate_init=False,
    )

    result = benchmark(next, dataset)

    assert isinstance(result, Signal)


@pytest.mark.benchmark(group="dataset-metadata-context")
def test_benchmark_dataset_next_debug_enabled(benchmark):
    """Benchmark dataset iteration with automatic correlation enabled."""
    dataset = MetadataContextBenchmarkDataset(
        signal_generators=[],
        target_labels=None,
        validate_init=False,
    )
    dataset.enable_metadata_debug(max_events=0)

    result = benchmark(next, dataset)

    assert isinstance(result, Signal)
