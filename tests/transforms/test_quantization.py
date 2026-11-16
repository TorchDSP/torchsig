"""Regression tests for the shared scalar and batched quantization kernel."""

from __future__ import annotations

import numpy as np
import pytest

from torchsig.transforms.functional import _quantize_iq, quantize
from torchsig.utils.dsp import TorchSigComplexDataType, TorchSigRealDataType


def _legacy_quantize(data: np.ndarray, num_bits: int, adjustment_db: float, mode: str) -> np.ndarray:
    """Reproduce the pre-464 scalar implementation for exact comparison."""
    levels = np.arange(-(2**num_bits) // 2, (2**num_bits) // 2) / ((2**num_bits) // 2)
    distance = levels[1] - levels[0]
    if mode == "floor":
        thresholds = levels + distance / 2
    elif mode == "ceiling":
        thresholds = levels - distance / 2
    else:
        raise ValueError

    maximum = max(np.max(np.abs(data.real)), np.max(np.abs(data.imag)))
    if maximum == 0:
        return np.zeros(data.shape, dtype=TorchSigComplexDataType)
    scale = 10 ** (adjustment_db / 20)
    scaled = data * scale / maximum
    real = np.zeros(len(data), dtype=TorchSigRealDataType)
    imag = np.zeros(len(data), dtype=TorchSigRealDataType)
    for component, output in ((scaled.real, real), (scaled.imag, imag)):
        negative = np.where(component <= thresholds[0])[0]
        positive = np.where(component >= thresholds[-1])[0]
        output[negative] = levels[0]
        output[positive] = levels[-1]
        remaining = np.setdiff1d(np.arange(len(data)), negative)
        remaining = np.setdiff1d(remaining, positive)
        output[remaining] = levels[np.digitize(component[remaining], thresholds)]
    return ((real + 1j * imag) * maximum / scale).astype(TorchSigComplexDataType)


@pytest.mark.parametrize("num_bits", [2, 4, 8, 12])
@pytest.mark.parametrize("rounding_mode", ["floor", "ceiling"])
@pytest.mark.parametrize("adjustment_db", [-6.0, 0.0, 3.0])
@pytest.mark.parametrize("length", [31, 1024])
def test_scalar_quantization_exactly_matches_legacy(
    num_bits: int,
    rounding_mode: str,
    adjustment_db: float,
    length: int,
) -> None:
    """The optimized numerical path must preserve established scalar output."""
    rng = np.random.default_rng(464 + length)
    data = (rng.normal(size=length) + 1j * rng.normal(size=length)).astype(np.complex64)

    actual = quantize(data, num_bits, adjustment_db, rounding_mode)
    expected = _legacy_quantize(data, num_bits, adjustment_db, rounding_mode)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == TorchSigComplexDataType
    assert actual.shape == data.shape


@pytest.mark.parametrize("rounding_mode", ["floor", "ceiling"])
def test_quantization_preserves_threshold_and_clipping_boundaries(rounding_mode: str) -> None:
    """Exact thresholds and values outside the representable range retain legacy semantics."""
    levels = np.arange(-2, 2) / 2
    distance = levels[1] - levels[0]
    thresholds = levels + distance / 2 if rounding_mode == "floor" else levels - distance / 2
    scale = 2.0
    values = np.array(
        [
            thresholds[0] / scale,
            np.nextafter(thresholds[0], np.inf) / scale,
            thresholds[-1] / scale,
            -1.0,
            1.0,
        ],
        dtype=np.float32,
    )
    data = (values + 1j * values[::-1]).astype(np.complex64)

    actual = quantize(data, 2, 20 * np.log10(scale), rounding_mode)
    expected = _legacy_quantize(data, 2, 20 * np.log10(scale), rounding_mode)

    np.testing.assert_array_equal(actual, expected)


def test_quantization_preserves_input_and_handles_zero_rows() -> None:
    """Scalar and row-wise calls do not mutate inputs and preserve zero signals."""
    data = np.array([[0j, 0j, 0j], [1 + 2j, -3 + 0.5j, 0j]], dtype=np.complex64)
    original = data.copy()

    actual = _quantize_iq(data, 4, -2.0, "floor", axis=1)

    np.testing.assert_array_equal(data, original)
    np.testing.assert_array_equal(actual[0], np.zeros(3, dtype=np.complex64))
    np.testing.assert_array_equal(actual[1], quantize(data[1], 4, -2.0, "floor"))
    assert actual.dtype == TorchSigComplexDataType
    assert actual.shape == data.shape


@pytest.mark.parametrize("invalid", [np.nan + 1j, np.inf + 1j, 1 + np.inf * 1j])
def test_quantize_rejects_nonfinite_input(invalid: complex) -> None:
    """The optimized scalar path retains finite-input validation."""
    with pytest.raises(ValueError, match=r"NaN|infinite"):
        quantize(np.array([invalid], dtype=np.complex64), 8)
