"""Unit tests for the PipelineFailOverEnabled mix-in and the
SafeTorchSigIterableDataset that uses it.

The tests cover:

1.  “original” fallback -- return the data untouched when a transform
    raises.
2.  “zero” fallback -- return a zero-array of the same shape.
3.  “retry” fallback -- attempt `pipeline_max_retries` times before
    falling back to the chosen “original” or “zero” behaviour.
4.  Normal success - when no exception is raised the transformer
    runs once and its result is returned.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import pytest

from torchsig.datasets.pipeline_failover import PipelineFailOverEnabled


# --------------------------------------------------------------------------
#  Dummy signal object used by the tests
# --------------------------------------------------------------------------
class DummySignal(dict):
    """Very small stand-in for ``torchsig.signals.signal_types.Signal``.
    The real class behaves like a dict but also supports ``.data``.
    """

    def __init__(self, data: np.ndarray):
        super().__init__()
        self["data"] = data
        self.data = data  # convenience attribute


# --------------------------------------------------------------------------
#  Minimal TorchSigIterableDataset that uses the mix-in
# --------------------------------------------------------------------------
class SafeTorchSigIterableDataset(PipelineFailOverEnabled, Iterator):
    """Minimal Dataset used only for the tests.

    This is a lightweight implementation that does not use the heavy TorchSig machinery.
    The ``__next__`` method mimics the real class behavior by calling
    `_run_with_fallback` around a user supplied ``apply_func``.
    """

    def __init__(self, base_signal: DummySignal, apply_func, target_labels=None):
        super().__init__()
        self.base_signal = base_signal
        self.apply_func = apply_func
        self.target_labels = target_labels or []

    def __iter__(self):
        """Return the iterator object.

        Returns:
            self: The dataset instance itself as it is its own iterator.
        """
        return self

    def __next__(self):
        """Return the next sample from the dataset.

        Returns:
            The result of applying the function to the base signal with fail-over handling.
        """
        return self._run_with_fallback(
            self.apply_func,
            self.base_signal,
            self.target_labels,
            fallback_raw_signal=self.base_signal,
        )


# --------------------------------------------------------------------------
#  Helper functions used in the tests
# --------------------------------------------------------------------------
def dummy_apply_func(signal: DummySignal, target_labels=None):
    """Dummy implementation that simply republishes ``signal.data``.
    All real transform logic is replaced in the tests by patching
    this function -- see the individual test functions below.
    """
    return signal.data


# --------------------------------------------------------------------------
#  Tests ---------------------------------------------------------------
# --------------------------------------------------------------------------
@pytest.fixture
def base_signal():
    """Return a deterministic 1-D complex signal."""
    return DummySignal(data=np.array([1.0 + 2j, 3.0 + 4j], dtype=np.complex64))


def test_fallback_original(base_signal, caplog):
    """When an exception is raised, the original data is returned."""
    caplog.set_level(logging.WARNING)

    # Simulate a transform that always fails
    def bad_apply(signal, target_labels=None):
        raise RuntimeError("transform error")

    dataset = SafeTorchSigIterableDataset(base_signal, apply_func=bad_apply, target_labels=[])
    dataset.pipeline_fallback = "original"

    sample = next(dataset)
    # should get the *original* data back
    assert isinstance(sample, np.ndarray)
    assert np.allclose(sample, base_signal.data)
    # a warning should have been emitted
    assert any("transform error" in rec.message for rec in caplog.records)


def test_fallback_zero(base_signal, caplog):
    """When an exception is raised, a zero array of matching shape is returned."""
    caplog.set_level(logging.WARNING)

    def bad_apply(signal, target_labels=None):
        raise RuntimeError("transform error")

    dataset = SafeTorchSigIterableDataset(base_signal, apply_func=bad_apply, target_labels=[])
    dataset.pipeline_fallback = "zero"

    sample = next(dataset)
    assert isinstance(sample, np.ndarray)
    assert np.allclose(sample, np.zeros_like(base_signal.data))
    assert any("transform error" in rec.message for rec in caplog.records)


def test_fallback_retry_successful(base_signal, caplog):
    """Failed attempts should be retried up to ``pipeline_max_retries``.
    When the transform finally succeeds, the correct result is returned.
    """
    caplog.set_level(logging.WARNING)
    attempts = {"count": 0}

    def flaky_apply(signal, target_labels=None):
        # fails the first two calls, succeeds on the third
        if attempts["count"] < 2:
            attempts["count"] += 1
            raise RuntimeError(f"attempt {attempts['count']}")
        return signal.data * 10.0  # success path

    dataset = SafeTorchSigIterableDataset(base_signal, apply_func=flaky_apply, target_labels=[])
    dataset.pipeline_fallback = "retry"
    dataset.pipeline_max_retries = 3

    # The third attempt should succeed -- result multiplied by 10
    result = next(dataset)
    assert np.allclose(result, base_signal.data * 10.0)
    # Two warning logs for the failed attempts
    failures = [rec for rec in caplog.records if "attempt" in rec.message]
    assert len(failures) == 2
    # No final error warning (because success)
    assert not any("Pipeline failed" in rec.message for rec in caplog.records)


def test_fallback_retry_exhausted(base_signal, caplog):
    """If the transform fails for all allowed retries the mix-in falls back
    to the configured fallback (here “original”).
    """
    caplog.set_level(logging.WARNING)

    def always_fail(signal, target_labels=None):
        raise RuntimeError("always bad")

    dataset = SafeTorchSigIterableDataset(base_signal, apply_func=always_fail, target_labels=[])
    dataset.pipeline_fallback = "retry"
    dataset.pipeline_max_retries = 2  # only two retries

    result = next(dataset)
    # After two retries, fallback to original data
    assert np.allclose(result, base_signal.data)
    # Two retry warnings + one final warning that the fallback was used.
    retry_warnings = [rec for rec in caplog.records if "retry" in rec.message]
    assert len(retry_warnings) == 2
    fallback_warnings = [rec for rec in caplog.records if "fallback" in rec.message]
    print("fallback_warnings")
    print(fallback_warnings)
    assert any("pipeline" in rec.message.lower() and "failed" in rec.message.lower() for rec in fallback_warnings)


def test_success_no_fallback_needed(base_signal):
    """When the transform succeeds the result is returned unchanged."""

    def good_apply(signal, target_labels=None):
        return signal.data * 3.0

    dataset = SafeTorchSigIterableDataset(base_signal, apply_func=good_apply, target_labels=[])
    dataset.pipeline_fallback = "original"  # any value is irrelevant here

    result = next(dataset)
    assert np.allclose(result, base_signal.data * 3.0)


def test_target_labels_passthrough(base_signal):
    """The helper should pass through target labels when requested."""

    # Define a dummy transform that merely copies the data
    def copy_apply(signal, target_labels=None):
        # pretend we also add a metadata entry
        signal["new_meta"] = 42
        return signal.data

    dataset = SafeTorchSigIterableDataset(base_signal, apply_func=copy_apply, target_labels=["new_meta"])

    # monkeypatch the mixin's run method to use our dummy function that returns (data, label)
    def dummy_func(signal, target_labels):
        return signal.data, 42

    dataset.apply_func = dummy_func
    dataset.target_labels = ["new_meta"]

    # should return tuple (data, label)
    data, lbl = next(dataset)
    assert np.allclose(data, base_signal.data)
    assert lbl == 42


def test_run_with_fallback_reraises_when_no_fallback_signal(base_signal, caplog):
    """If a stage fails before a raw signal exists, the final exception is re-raised."""
    caplog.set_level(logging.WARNING)

    def generation_fails():
        raise RuntimeError("generation failed")

    dataset = SafeTorchSigIterableDataset(base_signal, apply_func=dummy_apply_func, target_labels=[])
    dataset.pipeline_fallback = "retry"
    dataset.pipeline_max_retries = 2

    with pytest.raises(RuntimeError, match="generation failed"):
        dataset._run_with_fallback(
            generation_fails,
            fallback_raw_signal=None,
        )

    retry_warnings = [rec for rec in caplog.records if "Pipeline retry" in rec.message and "generation failed" in rec.message]
    assert len(retry_warnings) == 2
    assert any("Retries exhausted" in rec.message for rec in caplog.records)
