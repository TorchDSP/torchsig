"""Optional benchmarks for the wideband Doppler transform.

Run this module explicitly so its relatively expensive resampling cases do not
become part of the routine benchmark suite::

    python -m pytest benchmarks/optional/benchmark_doppler.py --benchmark-only
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from torchsig.transforms.functional import doppler
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.dsp import TorchSigComplexDataType

if TYPE_CHECKING:
    from collections.abc import Callable


SEED = 42
PROPAGATION_SPEED = 2.9979e8

# Keeping implementations in a lookup makes it straightforward to add an
# optimized candidate while retaining the current implementation as a baseline.
DOPPLER_IMPLEMENTATIONS: dict[str, Callable[..., np.ndarray]] = {
    "current": doppler,
}


def _complex_noise(num_samples: int, seed: int = SEED) -> np.ndarray:
    """Return deterministic complex64 input for a benchmark case."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples)).astype(TorchSigComplexDataType)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "num_samples",
    [256, 1_024, 8_192, 65_536],
    ids=lambda value: f"n={value}",
)
@pytest.mark.parametrize(
    "velocity",
    [-1e6, 10.0, 1e6],
    ids=["receding", "near-unity", "approaching"],
)
@pytest.mark.parametrize(
    ("implementation_name", "implementation"),
    DOPPLER_IMPLEMENTATIONS.items(),
    ids=DOPPLER_IMPLEMENTATIONS,
)
def test_benchmark_doppler_functional(
    benchmark,
    implementation_name: str,
    implementation: Callable[..., np.ndarray],
    velocity: float,
    num_samples: int,
) -> None:
    """Measure steady-state functional Doppler performance."""
    del implementation_name
    data = _complex_noise(num_samples)
    kwargs = {
        "data": data,
        "velocity": velocity,
        "propagation_speed": PROPAGATION_SPEED,
    }

    # Keep one-time setup, including possible future JIT compilation, outside
    # the measured region. The benchmark still measures normal per-call filter
    # lookup/loading performed by the implementation.
    warm_result = implementation(**kwargs)
    assert warm_result.shape == data.shape
    assert warm_result.dtype == TorchSigComplexDataType

    result = benchmark(implementation, **kwargs)

    assert result.shape == data.shape
    assert result.dtype == TorchSigComplexDataType
    assert np.all(np.isfinite(result))


def _dataset_metadata(num_samples: int) -> dict:
    """Create compact metadata suitable for repeatable generation benchmarks."""
    metadata = TorchSigDefaults().default_dataset_metadata
    fft_size = min(256, num_samples)
    metadata.update(
        {
            "num_iq_samples_dataset": num_samples,
            "fft_size": fft_size,
            "fft_stride": fft_size,
            "sample_rate": float(num_samples),
            "frequency_min": -num_samples / 2,
            "frequency_max": num_samples / 2,
            "signal_center_freq_min": 0.0,
            "signal_center_freq_max": 0.0,
            "signal_duration_in_samples_min": num_samples,
            "signal_duration_in_samples_max": num_samples,
            "bandwidth_min": 1.0,
            "bandwidth_max": 1.0,
            "num_signals_min": 1,
            "num_signals_max": 1,
        }
    )
    return metadata
