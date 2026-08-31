"""Tests for the numba-accelerated sampling clock impairments.

Verifies the numba implementation matches the pure-NumPy reference (to float32
precision) and that the clock_drift / clock_jitter transforms remain correct
and reproducible.
"""

import numpy as np
import pytest

pytest.importorskip("numba")

from torchsig.transforms import functional as F
from torchsig.utils.dsp import (
    TorchSigRealDataType,
    prototype_polyphase_filter,
    sampling_clock_impairments,
)
from torchsig.utils.dsp_numba import (
    digital_agc_numba,
    partition_polyphase_numba,
    sampling_clock_impairments_numba,
    sampling_clock_impairments_numba_wrapper,
)

UPRATE = 5000


def _filter_and_data(seed=123, n=4096):
    h = prototype_polyphase_filter(num_branches=UPRATE).astype(TorchSigRealDataType)
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    return h, x


@pytest.mark.slow
@pytest.mark.parametrize(
    "jitter_ppm,drift_ppm",
    [(0.0, 10.0), (10.0, 0.0), (0.0, 0.0)],
)
def test_numba_matches_reference(jitter_ppm, drift_ppm):
    """Numba output matches the NumPy reference to float32 precision."""
    h, x = _filter_and_data()
    kw = dict(h=h, x=x, uprate=UPRATE, drate=UPRATE, jitter_ppm=jitter_ppm, drift_ppm=drift_ppm)

    ref = sampling_clock_impairments(rng=np.random.default_rng(42), **kw)
    out = sampling_clock_impairments_numba_wrapper(rng=np.random.default_rng(42), **kw)

    assert len(out) == len(ref)
    assert out.dtype == np.complex64
    np.testing.assert_allclose(out, ref, atol=1e-4)


def test_numba_reproducible():
    """Same seed yields identical numba output."""
    h, x = _filter_and_data()
    kw = dict(h=h, x=x, uprate=UPRATE, drate=UPRATE, jitter_ppm=0.0, drift_ppm=10.0)
    a = sampling_clock_impairments_numba_wrapper(rng=np.random.default_rng(7), **kw)
    b = sampling_clock_impairments_numba_wrapper(rng=np.random.default_rng(7), **kw)
    np.testing.assert_array_equal(a, b)


def test_functional_uses_numba():
    """functional.py wired the accelerated implementation in."""
    assert F._sampling_clock_impairments is sampling_clock_impairments_numba_wrapper


@pytest.mark.parametrize("transform", [F.clock_drift, F.clock_jitter])
def test_clock_transforms_preserve_length(transform):
    """clock_drift / clock_jitter return the same number of samples as the input."""
    _, x = _filter_and_data(n=4096)
    out = transform(x, 10.0, np.random.default_rng(5))
    assert len(out) == len(x)
    assert out.dtype == np.complex64
    assert np.all(np.isfinite(out))


# --------------------------------------------------------------------------- #
# digital_agc
# --------------------------------------------------------------------------- #


_AGC_ARGS = (0.0, 1e-4, 1e-3, 0.1, 1e-3, 0.0, 1.0, -80.0, 10.0)


def _agc_data(seed=11, n=4096):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    x[5] = 0  # exercise the zero-sample branch
    return x


def test_digital_agc_matches_reference():
    """Numba digital_agc matches the pure-NumPy reference to float precision."""
    x = _agc_data()
    ref = F._digital_agc_python(x, *_AGC_ARGS)
    out = digital_agc_numba(x, *_AGC_ARGS)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_digital_agc_functional_uses_numba_and_preserves_length():
    """functional.digital_agc uses numba and preserves length/dtype."""
    assert F._digital_agc_numba is digital_agc_numba
    x = _agc_data()
    out = F.digital_agc(x)
    assert len(out) == len(x)
    assert out.dtype == np.complex64
    assert np.all(np.isfinite(out))


def test_partition_polyphase_numba_scales_and_zero_pads_taps():
    h = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    out = partition_polyphase_numba(h, up_rate=3, taps_per_phase=2)

    expected = np.array(
        [
            [3.0, 12.0],
            [6.0, 15.0],
            [9.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(out, expected)


def test_sampling_clock_numba_wrapper_does_not_consume_rng_when_no_jitter_or_drift():
    class ExplodingRng:
        def normal(self, *args, **kwargs):
            raise AssertionError("rng.normal should not be called")

    h = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    x = np.array([1 + 1j, 2 - 1j, -1 + 0.5j], dtype=np.complex64)

    out = sampling_clock_impairments_numba_wrapper(
        h=h,
        x=x,
        uprate=2,
        drate=2,
        jitter_ppm=0.0,
        drift_ppm=0.0,
        rng=ExplodingRng(),
    )

    assert out.dtype == np.complex64
    assert np.all(np.isfinite(out))


def test_sampling_clock_numba_wrapper_handles_real_valued_complex_input():
    h = np.array([0.5, 0.25, -0.125, 0.0], dtype=np.float32)
    x = np.array([1, 2, 3, 4], dtype=np.complex64)

    out = sampling_clock_impairments_numba_wrapper(
        h=h,
        x=x,
        uprate=2,
        drate=2,
        jitter_ppm=0.0,
        drift_ppm=0.0,
        rng=np.random.default_rng(123),
    )

    assert out.dtype == np.complex64
    np.testing.assert_array_equal(out.imag, np.zeros_like(out.imag))


@pytest.mark.parametrize(
    "x",
    [
        np.zeros(16, dtype=np.complex64),
        np.ones(16, dtype=np.complex64),
        np.linspace(-1, 1, 16).astype(np.complex64),
    ],
)
def test_digital_agc_handles_special_input_patterns(x):
    out = digital_agc_numba(x, *_AGC_ARGS)

    assert out.shape == x.shape
    assert out.dtype == np.complex64
    assert np.all(np.isfinite(out))


def test_digital_agc_empty_input_returns_empty_complex64_array():
    x = np.array([], dtype=np.complex64)

    out = digital_agc_numba(x, *_AGC_ARGS)

    assert out.shape == (0,)
    assert out.dtype == np.complex64


@pytest.mark.parametrize(
    "sample,expected_finite",
    [
        (0 + 0j, True),
        (1 + 0j, True),
        (1 + 1j, True),
        (-1 - 1j, True),
    ],
)
def test_digital_agc_single_sample_inputs(sample, expected_finite):
    x = np.array([sample], dtype=np.complex64)

    out = digital_agc_numba(x, *_AGC_ARGS)

    assert out.shape == (1,)
    assert out.dtype == np.complex64
    assert np.isfinite(out[0])


def test_partition_polyphase_numba_py_func_covers_kernel_body():
    h = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    out = partition_polyphase_numba.py_func(h, 3, 2)

    expected = np.array(
        [
            [3.0, 12.0],
            [6.0, 15.0],
            [9.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(out, expected)


@pytest.mark.slow
def test_sampling_clock_impairments_numba_py_func_covers_kernel_body():
    h, x = _filter_and_data(n=128)

    uprate = 8
    drate = 8
    taps_per_phase = int(np.ceil(len(h) / uprate))
    h_pfb = partition_polyphase_numba.py_func(h, uprate, taps_per_phase)
    h_pfb_reversed = np.ascontiguousarray(np.flip(h_pfb, axis=1))

    padded_len = len(x) + 2 * taps_per_phase - 1
    max_start = padded_len - taps_per_phase
    num_output_samples = int(np.ceil(padded_len * uprate / drate)) + 1

    rng = np.random.default_rng(123)
    jitter_drift_pool = rng.normal(0.0, 1.0, num_output_samples * 2).astype(np.float32) * 1e-6

    out = sampling_clock_impairments_numba.py_func(
        h,
        x.real.astype(np.float32),
        x.imag.astype(np.float32),
        uprate,
        drate,
        10.0,
        10.0,
        jitter_drift_pool,
        h_pfb_reversed,
        taps_per_phase,
        padded_len,
        max_start,
        num_output_samples,
    )

    assert out.dtype == np.complex64
    assert len(out) > 0
    assert np.all(np.isfinite(out))


def test_sampling_clock_impairments_numba_py_func_handles_empty_output():
    h = np.array([1.0], dtype=np.float32)
    x = np.array([], dtype=np.complex64)

    out = sampling_clock_impairments_numba.py_func(
        h,
        x.real.astype(np.float32),
        x.imag.astype(np.float32),
        1,
        1,
        0.0,
        0.0,
        np.zeros(2, dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
        1,
        0,
        -1,
        1,
    )

    assert out.shape == (0,)
    assert out.dtype == np.complex64


def test_digital_agc_numba_py_func_covers_kernel_body():
    x = _agc_data(n=64)

    out = digital_agc_numba.py_func(x, *_AGC_ARGS)

    ref = F._digital_agc_python(x, *_AGC_ARGS)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_digital_agc_numba_py_func_empty_input():
    x = np.array([], dtype=np.complex64)

    out = digital_agc_numba.py_func(x, *_AGC_ARGS)

    assert out.shape == (0,)
    assert out.dtype == np.complex64
