import numpy as np
import pytest

from torchsig.transforms import functional
from torchsig.transforms.functional import passband_ripple

# Constants
_EPS = np.finfo(np.float32).eps
TLL = 10 * np.log10(_EPS)  # "Ten Log Lynx" - a safe lower bound for 20*log10(x)


def _safe_log10(x):
    """A log10 that clamps its input to [EPS, inf) to avoid -inf."""
    x = np.maximum(x, _EPS)
    return np.log10(x)


# Benchmark test cases
@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("num_taps", "max_ripple_db", "signal_length"),
    [
        (3, 2.0, 1_000),
        (10, 1.0, 10_000),
        (32, 0.5, 100_000),
        (50, 0.1, 50_000),
    ],
)
def test_passband_ripple_benchmark(benchmark, num_taps, max_ripple_db, signal_length):
    """Benchmark the passband_ripple function with various configurations."""
    # Generate random test data
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, signal_length) + 1j * rng.normal(0, 1, signal_length)

    # Run the benchmark
    result = benchmark(
        passband_ripple,
        data,
        num_taps=num_taps,
        max_ripple_db=max_ripple_db,
        rng=rng,
    )

    # Basic sanity checks on the result
    assert result.shape == data.shape
    assert np.iscomplexobj(result)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


# Additional benchmark for the worst-case scenario (when filter can't be found)
@pytest.mark.benchmark
def test_passband_ripple_worst_case_benchmark(benchmark, monkeypatch):
    """Benchmark fallback when filter construction fails."""

    def fail_filter(*_args, **_kwargs):
        raise RuntimeError("injected filter construction failure")

    monkeypatch.setattr(functional, "_fft_filter", fail_filter)

    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 1000) + 1j * rng.normal(0, 1, 1000)

    result = benchmark(
        passband_ripple,
        data,
        num_taps=50,
        max_ripple_db=0.01,
        fallback="original",
        rng=rng,
    )

    assert result is data
