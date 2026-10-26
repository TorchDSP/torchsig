"""Optional benchmarks for the pure-Python clock-jitter fallback.

Run this module explicitly and save its results when comparing branches::

    python -m pytest benchmarks/optional/benchmark_clock_jitter_python.py \
        --benchmark-only --benchmark-save=clock-jitter-python

The functional transform normally selects the Numba implementation when it is
available. These benchmarks replace that internal selection with the Python
implementation so they measure the fallback consistently on every system.
"""

from __future__ import annotations

import numpy as np
import pytest

from torchsig.transforms import functional
from torchsig.utils.dsp import TorchSigComplexDataType, sampling_clock_impairments

NUM_SAMPLES = 4_096
SEED = 42


def _complex_noise() -> np.ndarray:
    """Return deterministic complex64 input for the benchmark cases."""
    rng = np.random.default_rng(SEED)
    return (rng.standard_normal(NUM_SAMPLES) + 1j * rng.standard_normal(NUM_SAMPLES)).astype(
        TorchSigComplexDataType
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "jitter_ppm",
    [0.0, 10.0],
    ids=["integer-positions", "interpolated-positions"],
)
def test_benchmark_clock_jitter_python_fallback(benchmark, monkeypatch, jitter_ppm: float) -> None:
    """Measure ``clock_jitter`` while forcing its pure-Python implementation."""
    monkeypatch.setattr(functional, "_sampling_clock_impairments", sampling_clock_impairments)
    data = _complex_noise()

    def run() -> np.ndarray:
        return functional.clock_jitter(
            data=data,
            jitter_ppm=jitter_ppm,
            rng=np.random.default_rng(SEED),
        )

    result = benchmark(run)

    assert result.shape == data.shape
    assert result.dtype == TorchSigComplexDataType
    assert np.all(np.isfinite(result))
