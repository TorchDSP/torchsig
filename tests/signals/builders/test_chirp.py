"""Unit tests for the chirp signal generator."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from torchsig.signals.builders.chirp import chirp
from torchsig.signals.builders.chirpss import ChirpSSSignalGenerator, chirpss_modulator, chirpss_modulator_baseband
from torchsig.signals.builders.lfm import LFMSignalGenerator, lfm_modulator, lfm_modulator_baseband


@pytest.mark.parametrize("seed", [0, 42, 2026])
@pytest.mark.parametrize(
    ("modulator", "args"),
    [
        (lfm_modulator_baseband, ("radar", 8192, 4)),
        (lfm_modulator_baseband, ("data", 8192, 4)),
        (chirpss_modulator_baseband, (8192, 4)),
    ],
)
def test_default_sweep_length_preserves_legacy_draw(modulator, args, seed):
    """Unbounded calls retain the legacy length draw, waveform, and RNG state."""
    rng = np.random.default_rng(seed)
    legacy_rng = np.random.default_rng(seed)
    legacy_length = legacy_rng.integers(low=128, high=4096)

    expected = modulator(*args, rng=legacy_rng, samples_per_symbol_min=legacy_length, samples_per_symbol_max=legacy_length)
    actual = modulator(*args, rng=rng)

    np.testing.assert_array_equal(actual, expected)
    assert rng.bit_generator.state == legacy_rng.bit_generator.state


@pytest.mark.parametrize(
    ("modulator", "args"),
    [
        (lfm_modulator_baseband, ("radar", 256, 4)),
        (chirpss_modulator_baseband, (256, 4)),
    ],
)
def test_pinned_sweep_length_does_not_draw_length(modulator, args):
    """Equal bounds set the actual sweep period without consuming a length draw."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.zeros((4,), dtype=int)
    rng.uniform.return_value = 1.0

    result = modulator(*args, rng=rng, samples_per_symbol_min=64, samples_per_symbol_max=64)

    rng.integers.assert_called_once_with(0, 1 if modulator is lfm_modulator_baseband else 128, 4)
    np.testing.assert_allclose(result[:64], result[64:128])


@pytest.mark.parametrize(
    ("modulator", "args"),
    [
        (lfm_modulator_baseband, ("radar", 256, 4)),
        (chirpss_modulator_baseband, (256, 4)),
    ],
)
def test_bounded_sweep_length_draws_inclusive_range(modulator, args):
    """The upper bound must be reachable via NumPy's exclusive high argument."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [65, np.zeros((4,), dtype=int)]
    rng.uniform.return_value = 1.0

    modulator(*args, rng=rng, samples_per_symbol_min=64, samples_per_symbol_max=65)

    assert rng.integers.call_args_list[0] == call(low=64, high=66)


@pytest.mark.parametrize("bounds", [(0, 64), (64, 257), (80, 64), (64, None), (None, 64), (1.5, 64), (True, 64)])
@pytest.mark.parametrize(
    ("modulator", "args"),
    [
        (lfm_modulator_baseband, ("radar", 256, 4)),
        (chirpss_modulator_baseband, (256, 4)),
    ],
)
def test_invalid_sweep_bounds_raise(modulator, args, bounds):
    """Configured bounds must describe a whole sweep within the burst."""
    with pytest.raises(ValueError, match="samples_per_symbol_min"):
        modulator(*args, rng=np.random.default_rng(0), samples_per_symbol_min=bounds[0], samples_per_symbol_max=bounds[1])


@pytest.mark.parametrize(
    ("modulator", "args"),
    [
        (lfm_modulator, ("radar", 1, 8, 1024)),
        (chirpss_modulator, (1, 8, 1024)),
    ],
)
def test_output_sweep_period_scales_with_resampling(modulator, args):
    """A 64-sample baseband sweep repeats after 128 output samples at 2x resampling."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = lambda _low, _high=None, size=None: np.zeros((size,), dtype=int)
    rng.uniform.return_value = 1.0

    signal = modulator(*args, rng=rng, samples_per_symbol_min=64, samples_per_symbol_max=64)
    instantaneous_frequency = np.angle(signal[1:] * np.conj(signal[:-1]))

    np.testing.assert_allclose(instantaneous_frequency[256:640], instantaneous_frequency[384:768], atol=1e-6)
    assert np.mean(np.abs(instantaneous_frequency[256:640] - instantaneous_frequency[320:704])) > 0.1


@pytest.mark.parametrize(
    ("generator_class", "modulator_name", "extra_metadata"),
    [
        (LFMSignalGenerator, "torchsig.signals.builders.lfm.lfm_modulator", {"lfm_type": "radar"}),
        (ChirpSSSignalGenerator, "torchsig.signals.builders.chirpss.chirpss_modulator", {}),
    ],
)
def test_signal_generator_passes_sweep_bounds(generator_class, modulator_name, extra_metadata):
    """Generator metadata must reach the authoritative modulator path."""
    generator = generator_class(
        sample_rate=8,
        bandwidth_min=1,
        bandwidth_max=1,
        signal_duration_in_samples_min=256,
        signal_duration_in_samples_max=256,
        samples_per_symbol_min=32,
        samples_per_symbol_max=64,
        **extra_metadata,
    )
    with patch(modulator_name, return_value=np.ones(256, dtype=np.complex64)) as modulator:
        generator.generate()

    assert modulator.call_args.kwargs == {"samples_per_symbol_min": 32, "samples_per_symbol_max": 64}


@pytest.mark.parametrize("samples_per_symbol", [0, -1, -100])
def test_chirp_rejects_nonpositive_samples_per_symbol(samples_per_symbol):
    """Nonpositive symbol lengths should raise a ValueError."""
    with pytest.raises(
        ValueError,
        match="samples_per_symbol must be positive",
    ):
        chirp(
            f0=0.1,
            f1=0.2,
            samples_per_symbol=samples_per_symbol,
        )


@pytest.mark.parametrize(
    ("f0", "f1"),
    [
        (np.nan, 0.2),
        (0.1, np.nan),
        (np.inf, 0.2),
        (0.1, np.inf),
        (-np.inf, 0.2),
        (0.1, -np.inf),
    ],
)
def test_chirp_rejects_nonfinite_frequencies(f0, f1):
    """NaN and infinite frequency values should be rejected."""
    with pytest.raises(
        ValueError,
        match="f0 and f1 must be finite numbers",
    ):
        chirp(
            f0=f0,
            f1=f1,
            samples_per_symbol=16,
        )


@pytest.mark.parametrize("samples_per_symbol", [1, 2, 17, 128])
def test_chirp_returns_expected_shape(samples_per_symbol):
    """The result should contain exactly the requested number of samples."""
    result = chirp(
        f0=0.1,
        f1=0.2,
        samples_per_symbol=samples_per_symbol,
    )

    assert result.shape == (samples_per_symbol,)


def test_chirp_returns_complex64():
    """The generated chirp should use complex64 storage."""
    result = chirp(
        f0=0.1,
        f1=0.2,
        samples_per_symbol=32,
    )

    assert result.dtype == np.complex64


@pytest.mark.parametrize(
    ("f0", "f1"),
    [
        (0.0, 0.0),
        (0.1, 0.2),
        (0.2, 0.1),
        (-0.2, 0.2),
    ],
)
def test_chirp_has_unit_magnitude(f0, f1):
    """Every complex exponential sample should have magnitude one."""
    result = chirp(
        f0=f0,
        f1=f1,
        samples_per_symbol=64,
    )

    np.testing.assert_allclose(
        np.abs(result),
        np.ones(64),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("phi", [0.0, 45.0, 90.0, -90.0, 360.0])
def test_chirp_applies_initial_phase(phi):
    """The first sample should equal the requested initial phase."""
    result = chirp(
        f0=0.1,
        f1=0.2,
        samples_per_symbol=32,
        phi=phi,
    )

    expected = np.exp(1j * np.deg2rad(phi))

    assert result[0] == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_chirp_single_sample_uses_initial_phase():
    """A one-sample chirp should avoid division by zero and apply phi."""
    result = chirp(
        f0=0.1,
        f1=0.9,
        samples_per_symbol=1,
        phi=30.0,
    )

    expected = np.array(
        [np.exp(1j * np.deg2rad(30.0))],
        dtype=np.complex64,
    )

    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_chirp_matches_linear_frequency_modulation_formula():
    """The generated samples should match the documented LFM equation."""
    f0 = 0.05
    f1 = 0.25
    samples_per_symbol = 11
    phi = 35.0

    result = chirp(
        f0=f0,
        f1=f1,
        samples_per_symbol=samples_per_symbol,
        phi=phi,
    )

    t = np.arange(samples_per_symbol, dtype=np.float64)
    chirp_rate = (f1 - f0) / (samples_per_symbol - 1)
    phase = 2 * np.pi * (f0 * t + 0.5 * chirp_rate * t**2)
    expected = np.exp(
        1j * (phase + np.deg2rad(phi)),
    ).astype(np.complex64)

    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_chirp_constant_frequency_matches_complex_tone():
    """Equal start and end frequencies should produce a constant tone."""
    frequency = 0.125
    samples_per_symbol = 16

    result = chirp(
        f0=frequency,
        f1=frequency,
        samples_per_symbol=samples_per_symbol,
    )

    t = np.arange(samples_per_symbol, dtype=np.float64)
    expected = np.exp(1j * 2 * np.pi * frequency * t).astype(np.complex64)

    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_chirp_zero_frequency_returns_constant_phase():
    """A zero-frequency chirp should remain at its initial phase."""
    phi = 60.0

    result = chirp(
        f0=0.0,
        f1=0.0,
        samples_per_symbol=20,
        phi=phi,
    )

    expected_sample = np.exp(1j * np.deg2rad(phi))
    expected = np.full(
        20,
        expected_sample,
        dtype=np.complex64,
    )

    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_chirp_default_phase_is_zero():
    """Omitting phi should be equivalent to explicitly passing zero."""
    result_default = chirp(
        f0=0.1,
        f1=0.3,
        samples_per_symbol=32,
    )
    result_explicit = chirp(
        f0=0.1,
        f1=0.3,
        samples_per_symbol=32,
        phi=0.0,
    )

    np.testing.assert_array_equal(result_default, result_explicit)


def test_chirp_is_deterministic():
    """Identical arguments should always produce identical samples."""
    first = chirp(
        f0=-0.15,
        f1=0.2,
        samples_per_symbol=50,
        phi=17.0,
    )
    second = chirp(
        f0=-0.15,
        f1=0.2,
        samples_per_symbol=50,
        phi=17.0,
    )

    np.testing.assert_array_equal(first, second)


def test_chirp_negative_sweep_matches_formula():
    """A descending chirp should correctly use a negative chirp rate."""
    f0 = 0.3
    f1 = -0.1
    samples_per_symbol = 9

    result = chirp(
        f0=f0,
        f1=f1,
        samples_per_symbol=samples_per_symbol,
    )

    t = np.arange(samples_per_symbol, dtype=np.float64)
    chirp_rate = (f1 - f0) / (samples_per_symbol - 1)
    expected = np.exp(1j * 2 * np.pi * (f0 * t + 0.5 * chirp_rate * t**2)).astype(np.complex64)

    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
