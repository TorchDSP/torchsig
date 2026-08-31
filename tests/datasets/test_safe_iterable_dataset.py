"""Unit tests for ``SafeTorchSigIterableDataset``."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

# ----------------------------------------------------------------------
# Imports from the real package
# ----------------------------------------------------------------------
from torchsig.datasets.datasets import (
    SafeTorchSigIterableDataset,
)
from torchsig.signals.signal_types import Signal
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.metadata_logging import (
    MetadataLoggingContext,
    get_metadata_logging_context,
    metadata_logging_context,
)


# ----------------------------------------------------------------------
# Dummy signal generator
# ----------------------------------------------------------------------
class DummySignalGenerator(dict):
    """A minimal signal generator for testing purposes.

    This generator creates a deterministic Signal object with a fixed set of
    IQ samples and the minimal metadata required by TorchSig's signal processing
    pipeline. It's designed for unit testing and doesn't represent a real RF signal.

    The generator produces a Signal with:
    - Complex IQ data: [1+2j, 3+4j]
    - Metadata: class_index, SNR bounds, center frequency, bandwidth, and duration

    Returns:
        Signal: A Signal object with the predefined data and metadata.

    Note:
        This generator inherits from dict to support metadata assignment via
        dictionary-style access, which is required by TorchSig's dataset machinery.
    """

    def __call__(self) -> Signal:
        """Generate a test Signal with predefined IQ data and metadata.

        Returns:
            Signal: A Signal object containing:
                - data: Array of complex values [1+2j, 3+4j]
                - class_index: 0
                - snr_db_min: 0.0
                - snr_db_max: 0.0
                - center_freq: 1500.0
                - bandwidth: 500.0
                - duration_in_samples: 2
        """
        sig = Signal(data=np.array([1.0 + 2j, 3.0 + 4j], dtype=np.complex64))
        sig["class_index"] = 0
        sig["snr_db_min"] = 0.0
        sig["snr_db_max"] = 0.0
        sig["center_freq"] = 1500.0
        sig["bandwidth"] = 500.0
        sig["duration_in_samples"] = sig.data.size
        return sig


# ----------------------------------------------------------------------
# Minimal metadata
# ----------------------------------------------------------------------
def _minimal_metadata() -> dict:
    md = TorchSigDefaults().default_dataset_metadata.copy()
    md.update(
        {
            "num_iq_samples_dataset": 256,
            "fft_size": 64,
            "fft_stride": 64,
            "sample_rate": 10_000,
            "frequency_min": 1_000,
            "frequency_max": 2_000,
            "signal_center_freq_min": 1_000,
            "signal_center_freq_max": 2_000,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "signal_duration_in_samples_min": 2000,
            "signal_duration_in_samples_max": 8000,
            "bandwidth_min": 1_000,
            "bandwidth_max": 2_000,
            "noise_level": 0.0,
            "cochannel_overlap_probability": 0.0,
        }
    )
    return md


# ----------------------------------------------------------------------
# Helper: transform that can be forced to fail a few times
# ----------------------------------------------------------------------
def _counter_transform(counter: list[int], failures_before_success: int = 0, multiply: float | None = None) -> Callable[[Signal], Signal]:
    def _transform(sig: Signal) -> Signal:
        if counter[0] < failures_before_success:
            counter[0] += 1
            raise RuntimeError(f"forced failure #{counter[0]}")
        if multiply is not None:
            sig.data = sig.data * multiply
        return sig

    return _transform


# ----------------------------------------------------------------------
# Helper to create the dataset
# ----------------------------------------------------------------------
def _make_safe_dataset(
    transforms: list[Callable[[Signal], Signal]] | None = None,
    target_labels: list | None = None,
) -> SafeTorchSigIterableDataset:
    md = _minimal_metadata()
    return SafeTorchSigIterableDataset(
        signal_generators=[DummySignalGenerator()],
        transforms=transforms or [],
        component_transforms=[],
        target_labels=target_labels,
        validate_init=False,
        **md,
    )


# ----------------------------------------------------------------------
# Fixture that captures logs
# ----------------------------------------------------------------------
@pytest.fixture
def default_logging(caplog):
    caplog.set_level(logging.WARNING)
    return caplog


# ----------------------------------------------------------------------
# Helpers for log asserts
# ----------------------------------------------------------------------
def _has_exact_fallback_warning(records):
    """Return True iff records contain the sentence 'Retries exhausted'."""
    return any("Retries exhausted" in r.message for r in records)


def _count_retry_warnings(records):
    """Return the number of lines containing 'retry NUMBER/MINUS failed'."""
    return len([r for r in records if re.search(r"retry \d+/\d+ failed", r.message)])


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_safe_dataset_normal_operation(default_logging):
    ds = _make_safe_dataset(
        transforms=[lambda s: s],
        target_labels=["class_index"],
    )
    val = next(ds)
    assert isinstance(val, (tuple, list))
    data, lbl = val
    assert isinstance(data, np.ndarray)
    assert data.shape[0] == 256
    assert data.dtype == np.complex64
    assert isinstance(lbl, int)


def test_safe_dataset_fallback_original(default_logging):
    counter = [0]
    ds = _make_safe_dataset(
        transforms=[_counter_transform(counter, failures_before_success=1)],
        target_labels=[],
    )
    ds.pipeline_fallback = "original"
    val = next(ds)
    assert isinstance(val, np.ndarray)
    assert val.shape[0] == 256
    assert val.dtype == np.complex64
    assert _count_retry_warnings(default_logging.records) == 0


def test_safe_dataset_fallback_zero(default_logging):
    counter = [0]
    ds = _make_safe_dataset(
        transforms=[_counter_transform(counter, failures_before_success=1)],
        target_labels=[],
    )
    ds.pipeline_fallback = "zero"
    val = next(ds)
    assert isinstance(val, np.ndarray)
    assert val.shape[0] == 256
    assert val.dtype == np.complex64
    assert np.allclose(val, np.zeros_like(val))
    assert _count_retry_warnings(default_logging.records) == 0


def test_safe_dataset_fallback_retry_success(default_logging):
    counter = [0]
    mul = 7.0
    ds = _make_safe_dataset(
        transforms=[_counter_transform(counter, failures_before_success=2, multiply=mul)],
        target_labels=[],
    )
    ds.pipeline_fallback = "retry"
    ds.pipeline_max_retries = 3
    val = next(ds)
    assert isinstance(val, np.ndarray)
    assert val.shape[0] == 256
    assert val.dtype == np.complex64
    # Exactly two retry logs
    assert _count_retry_warnings(default_logging.records) == 2
    # No explicit 'Retries exhausted' line
    assert not _has_exact_fallback_warning(default_logging.records)


def test_safe_dataset_fallback_retry_exhausted(default_logging):
    counter = [0]
    ds = _make_safe_dataset(
        transforms=[_counter_transform(counter, failures_before_success=100)],
        target_labels=[],
    )
    ds.pipeline_fallback = "retry"
    ds.pipeline_max_retries = 2
    val = next(ds)
    assert isinstance(val, np.ndarray)
    assert val.shape[0] == 256
    assert val.dtype == np.complex64
    # Two retry logs
    assert _count_retry_warnings(default_logging.records) == 2
    # Exactly one explicit 'Retries exhausted' line
    assert _has_exact_fallback_warning(default_logging.records)


def test_safe_dataset_fallback_original(default_logging):
    counter = [0]
    ds = _make_safe_dataset(
        transforms=[_counter_transform(counter, failures_before_success=1)],
        target_labels=[],
    )
    ds.pipeline_fallback = "original"

    val = next(ds)

    assert isinstance(val, np.ndarray)
    assert val.shape[0] == 256
    assert val.dtype == np.complex64
    assert _count_retry_warnings(default_logging.records) == 1
    assert _has_exact_fallback_warning(default_logging.records)


def test_safe_dataset_fallback_zero(default_logging):
    counter = [0]
    ds = _make_safe_dataset(
        transforms=[_counter_transform(counter, failures_before_success=1)],
        target_labels=[],
    )
    ds.pipeline_fallback = "zero"

    val = next(ds)

    assert isinstance(val, np.ndarray)
    assert val.shape[0] == 256
    assert val.dtype == np.complex64
    assert np.allclose(val, np.zeros_like(val))
    assert _count_retry_warnings(default_logging.records) == 1
    assert _has_exact_fallback_warning(default_logging.records)


def test_safe_dataset_generation_failure_reraises_without_raw_signal(default_logging, monkeypatch):
    ds = _make_safe_dataset(
        transforms=[],
        target_labels=[],
    )

    ds.pipeline_fallback = "retry"
    ds.pipeline_max_retries = 2

    def bad_generate(self):
        raise RuntimeError("raw generation failed")

    monkeypatch.setattr(
        SafeTorchSigIterableDataset,
        "__generate_new_signal__",
        bad_generate,
    )

    with pytest.raises(RuntimeError, match="raw generation failed"):
        next(ds)

    assert _count_retry_warnings(default_logging.records) == 2
    assert _has_exact_fallback_warning(default_logging.records)


def test_safe_dataset_retries_keep_same_sample_logging_context(default_logging):
    attempts = []

    def retrying_transform(signal):
        attempts.append(get_metadata_logging_context())
        if len(attempts) < 3:
            raise RuntimeError("retry transform")
        return signal

    dataset = _make_safe_dataset(
        transforms=[retrying_transform],
        target_labels=[],
    )
    dataset["dataset_id"] = "safe-debug-dataset"
    dataset.pipeline_fallback = "retry"
    dataset.pipeline_max_retries = 3

    with metadata_logging_context(session_id="safe-session"):
        next(dataset)

    assert len(attempts) == 3
    assert all(context.session_id == "safe-session" for context in attempts)
    assert all(context.dataset_id == "safe-debug-dataset" for context in attempts)
    assert all(context.sample_index == 0 for context in attempts)
    assert all(context.worker_id == 0 for context in attempts)
    assert all(dict(context.fields)["stage"] == "transform" for context in attempts)
    assert get_metadata_logging_context() == MetadataLoggingContext()


def test_safe_dataset_logs_completed_metadata_snapshot(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")

    def mark_complete(signal):
        signal["complete"] = True
        return signal

    dataset = _make_safe_dataset(
        transforms=[mark_complete],
        target_labels=None,
    )
    dataset["dataset_id"] = "safe-snapshot-dataset"
    dataset.enable_metadata_debug(
        keys={"complete", "class_index"},
        events={"snapshot"},
        include_values=True,
    )

    next(dataset)

    snapshots = [record for record in caplog.records if record.metadata_event == "snapshot"]
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.metadata_snapshot == {"complete": "True"}
    assert snapshot.metadata_component_snapshots == ({"class_index": "0"},)
    assert snapshot.metadata_dataset_id == "safe-snapshot-dataset"
    assert snapshot.metadata_sample_index == 0
    assert snapshot.metadata_correlation_fields["stage"] == "transform"
