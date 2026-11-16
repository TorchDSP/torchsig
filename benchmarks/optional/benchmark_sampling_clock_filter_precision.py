"""Optional sampling-clock coefficient-precision benchmarks.

Run explicitly with::

    python -m pytest benchmarks/optional/benchmark_sampling_clock_filter_precision.py --benchmark-only

The frequency-response cases compare the designed float64 coefficients with
the float32 coefficients used by public transforms. Runtime cases pass each
representation through both resampler implementations; those implementations
retain the public complex64 output contract and internally use float32 filter
banks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy import signal as sp

from torchsig.utils.dsp import prototype_polyphase_filter, sampling_clock_impairments
from torchsig.utils.dsp_numba import sampling_clock_impairments_numba_wrapper

if TYPE_CHECKING:
    from collections.abc import Callable

UPRATE = 5000
NUM_SAMPLES = 4096
RESPONSE_FREQUENCIES = np.array([0.0, 0.35, 0.45, 0.55, 0.65, 0.70, 0.75, 1.0])
IMPLEMENTATIONS: dict[str, Callable[..., np.ndarray]] = {
    "numpy": sampling_clock_impairments,
    "numba": sampling_clock_impairments_numba_wrapper,
}


def _prototype(dtype: np.dtype) -> np.ndarray:
    return prototype_polyphase_filter(UPRATE).astype(dtype)


@pytest.mark.benchmark
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_benchmark_sampling_clock_filter_response(benchmark, dtype) -> None:
    """Measure response calculation and report coefficient storage and levels."""
    taps = _prototype(dtype)

    def calculate_response() -> np.ndarray:
        _, response = sp.freqz(taps, worN=2 * np.pi * RESPONSE_FREQUENCIES / UPRATE)
        return 20 * np.log10(np.abs(response) / np.abs(response[0]))

    response_db = benchmark(calculate_response)
    benchmark.extra_info["coefficient_bytes"] = taps.nbytes
    benchmark.extra_info["response_db"] = response_db.tolist()
    assert abs(response_db[1]) < 0.05
    assert response_db[5] < -80.0


@pytest.mark.benchmark
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("implementation_name", IMPLEMENTATIONS, ids=IMPLEMENTATIONS)
def test_benchmark_sampling_clock_resampler(benchmark, dtype, implementation_name: str) -> None:
    """Measure resampling cost when supplied each coefficient representation."""
    implementation = IMPLEMENTATIONS[implementation_name]
    taps = _prototype(dtype)
    rng = np.random.default_rng(21)
    data = (rng.standard_normal(NUM_SAMPLES) + 1j * rng.standard_normal(NUM_SAMPLES)).astype(np.complex64)
    kwargs = {
        "h": taps,
        "x": data,
        "uprate": UPRATE,
        "drate": UPRATE,
        "jitter_ppm": 10.0,
        "drift_ppm": 0.0,
        "initial_phase": 0.0,
    }

    # Exclude one-time Numba compilation and filter loading from steady-state timing.
    warm_result = implementation(rng=np.random.default_rng(9), **kwargs)
    result = benchmark(implementation, rng=np.random.default_rng(9), **kwargs)

    benchmark.extra_info["coefficient_bytes"] = taps.nbytes
    assert warm_result.shape == result.shape
    assert result.dtype == np.complex64
    assert np.all(np.isfinite(result))
