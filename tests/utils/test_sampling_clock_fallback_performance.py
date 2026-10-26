"""Performance guard for the vectorized Python sampling-clock fallback."""

import numpy as np

from torchsig.utils.dsp import partition_polyphase, polyphase_resample_at_positions


def test_vectorized_partition_matches_phase_slicing() -> None:
    taps = np.arange(1, 18, dtype=np.float32)
    uprate = 5
    taps_per_phase = 4

    output = partition_polyphase(taps, uprate, taps_per_phase)

    expected = np.zeros((uprate, taps_per_phase), dtype=np.float32)
    for phase in range(uprate):
        phase_taps = taps[phase::uprate] * uprate
        expected[phase, : len(phase_taps)] = phase_taps
    np.testing.assert_array_equal(output, expected)


def test_vectorized_position_resampler_handles_large_position_batch() -> None:
    rng = np.random.default_rng(7)
    data = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)).astype(np.complex64)
    taps = np.hanning(64).astype(np.float32)
    positions = np.linspace(0.0, 4095.0 * 8, 4096, endpoint=False)

    output = polyphase_resample_at_positions(taps, data, uprate=8, positions=positions)

    assert output.shape == positions.shape
    assert output.dtype == np.complex64
    assert np.all(np.isfinite(output))
