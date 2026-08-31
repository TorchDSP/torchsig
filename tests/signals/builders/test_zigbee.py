"""Unit tests for Zigbee signal builder."""

import numpy as np
import pytest

from torchsig.signals.builders.zigbee import (
    ZIGBEE_CHIP_SEQS,
    ZigBeeSignalGenerator,
    _oqpsk_half_sine,
    build_zigbee_chip_stream,
    zigbee_modulator,
    zigbee_modulator_baseband,
)
from torchsig.signals.signal_lists import CLASS_FAMILY_DICT, TorchSigSignalLists
from torchsig.utils.dsp import TorchSigComplexDataType
from torchsig.utils.signal_building import lookup_signal_generator_by_string

ZIGBEE_METADATA = {
    "sample_rate": 10_000_000,
    "bandwidth_min": 1_500_000,
    "bandwidth_max": 2_500_000,
    "signal_duration_in_samples_min": 4096,
    "signal_duration_in_samples_max": 4096,
}

# Samples per chip used by zigbee_modulator at baseband
NOMINAL_OSR = 4


def test_zigbee_chip_seqs_shape():
    """There are 16 chip sequences, each 32 chips."""
    assert ZIGBEE_CHIP_SEQS.shape == (16, 32)
    assert set(np.unique(ZIGBEE_CHIP_SEQS)).issubset({0, 1})


def test_zigbee_chip_stream_length():
    """Chip stream is exactly the requested length."""
    rng = np.random.default_rng(0)
    stream = build_zigbee_chip_stream(500, rng)
    assert len(stream) == 500


def test_zigbee_modulator_output():
    """The modulator returns finite complex IQ of the right length."""
    rng = np.random.default_rng(42)
    num_samples = 4096
    iq = zigbee_modulator(2_000_000, 10_000_000, num_samples, rng)
    assert iq.dtype == TorchSigComplexDataType
    assert len(iq) == num_samples
    assert np.all(np.isfinite(iq))


def test_zigbee_modulator_invalid_args():
    """Invalid bandwidth/sample-rate raise."""
    with pytest.raises(ValueError):
        zigbee_modulator(0, 10_000_000, 4096)
    with pytest.raises(ValueError):
        zigbee_modulator(6_000_000, 10_000_000, 4096)


def test_zigbee_generator_generate():
    """Generator produces a Signal with correct metadata."""
    signal = ZigBeeSignalGenerator(metadata=ZIGBEE_METADATA, seed=1)()
    assert signal.class_name == "zigbee"
    assert len(signal.data) == ZIGBEE_METADATA["signal_duration_in_samples_min"]


def test_zigbee_generator_reproducible():
    """Same seed yields identical IQ."""
    a = ZigBeeSignalGenerator(metadata=ZIGBEE_METADATA, seed=6).generate()
    b = ZigBeeSignalGenerator(metadata=ZIGBEE_METADATA, seed=6).generate()
    np.testing.assert_array_equal(a.data, b.data)


def test_zigbee_registered_and_in_signal_lists():
    """'zigbee' resolves through lookup and is its own family."""
    assert isinstance(lookup_signal_generator_by_string("zigbee"), ZigBeeSignalGenerator)
    assert CLASS_FAMILY_DICT["zigbee"] == "zigbee"
    lists = TorchSigSignalLists()
    assert "zigbee" in lists.zigbee_signals


def _burst_support(iq, live_threshold=0.1):
    """Locates the samples that carry signal energy.

    ZigBee O-QPSK has a nominally constant envelope, so the median magnitude is
    a robust reference level and samples far below it are dead air introduced
    by padding rather than by the modulation.

    Args:
        iq: Complex baseband samples.
        live_threshold: Fraction of the median magnitude below which a sample
            counts as dead.

    Returns:
        tuple: (span_fraction, dead_fraction, longest_dead_run)
    """
    mag = np.abs(iq)
    live = mag > live_threshold * np.median(mag)
    idx = np.flatnonzero(live)
    if idx.size == 0:
        return 0.0, 1.0, len(iq)

    span_fraction = (idx[-1] - idx[0] + 1) / len(iq)
    dead = ~live
    padded = np.concatenate(([False], dead, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    longest_dead_run = int(np.max(edges[1::2] - edges[0::2])) if edges.size else 0
    return span_fraction, float(dead.mean()), longest_dead_run


def _edge_to_centre_power_ratio(iq, edge_fraction=0.1):
    """Compares mean power in the outer edges of the window against its centre.

    Head/tail padding drives this to zero while a burst that fills the window
    keeps it near one, independent of any absolute threshold.
    """
    mag = np.abs(iq)
    n = len(mag)
    edge = max(1, int(n * edge_fraction))
    outer = np.concatenate([mag[:edge], mag[-edge:]])
    centre = mag[n // 4 : 3 * n // 4]
    return float(np.mean(outer**2) / np.mean(centre**2))


@pytest.mark.parametrize("osr", [2, 4, 8, 16])
def test_zigbee_shaper_emits_osr_samples_per_chip(osr):
    """The O-QPSK shaper emits exactly osr samples for every input chip.

    In 802.15.4 each rail's half-sine pulse spans two chip periods and the Q
    rail lags I by one full chip period. A shaper that instead uses a one-chip
    pulse with a half-chip offset produces osr/2 samples per chip, halving
    every burst while leaving the returned array length untouched.
    """
    rng = np.random.default_rng(0)
    num_chips = 200
    chips = rng.integers(0, 2, num_chips).astype(np.int8)
    iq = _oqpsk_half_sine(chips, osr)
    assert len(iq) == num_chips * osr, f"shaper produced {len(iq)} samples for {num_chips} chips at osr={osr}, i.e. {len(iq) / num_chips} samples/chip instead of {osr}"


@pytest.mark.parametrize("max_num_samples", [256, 1024, 3276, 8192])
def test_zigbee_baseband_is_not_zero_padded(max_num_samples):
    """Baseband output is filled with signal, not padded out to length.

    Isolated zeros at pulse boundaries are expected; a run of them is not, and
    means pad_head_tail_to_length made up the shortfall.
    """
    rng = np.random.default_rng(1)
    iq = zigbee_modulator_baseband(max_num_samples, NOMINAL_OSR, rng)
    assert len(iq) == max_num_samples

    span, dead_fraction, longest_dead_run = _burst_support(iq)
    assert span > 0.95, f"burst spans only {span:.1%} of the baseband window"
    assert dead_fraction < 0.05, f"{dead_fraction:.1%} of baseband samples are dead"
    assert longest_dead_run <= 2 * NOMINAL_OSR, f"found {longest_dead_run} consecutive dead samples, which indicates head/tail zero padding rather than pulse-boundary zeros"


@pytest.mark.parametrize(
    "bandwidth, sample_rate, num_samples",
    [
        (2_000_000, 10_000_000, 4096),
        (1_500_000, 10_000_000, 8192),
        (2_500_000, 10_000_000, 2048),
        (1_750_000, 15_360_000, 512),
        (2_250_000, 20_000_000, 12345),
    ],
)
def test_zigbee_modulator_burst_fills_requested_duration(bandwidth, sample_rate, num_samples):
    """The burst occupies the whole requested window, not just its centre.

    Thresholds are loose on purpose: a correct build measures above 0.99 span
    and above 0.9 edge-to-centre power.
    """
    rng = np.random.default_rng(42)
    iq = zigbee_modulator(bandwidth, sample_rate, num_samples, rng)
    assert len(iq) == num_samples
    assert np.all(np.isfinite(iq))

    span, dead_fraction, longest_dead_run = _burst_support(iq)
    assert span > 0.90, f"burst spans only {span:.1%} of the {num_samples}-sample window"
    assert dead_fraction < 0.10, f"{dead_fraction:.1%} of samples carry no energy"
    assert longest_dead_run <= max(16, num_samples // 50), f"found {longest_dead_run} consecutive dead samples in a {num_samples}-sample burst"

    edge_ratio = _edge_to_centre_power_ratio(iq)
    assert edge_ratio > 0.25, f"edge power is only {edge_ratio:.1%} of centre power, so the burst is narrower than the window it will be annotated with"


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_zigbee_generator_burst_fills_annotated_duration(seed):
    """Energy fills the duration the Signal metadata reports.

    The dataset derives duration_in_samples from len(signal.data), so dead air
    inside signal.data becomes a time bounding box that is too wide.
    """
    expected = ZIGBEE_METADATA["signal_duration_in_samples_min"]
    signal = ZigBeeSignalGenerator(metadata=ZIGBEE_METADATA, seed=seed)()
    assert len(signal.data) == expected

    span, dead_fraction, _ = _burst_support(signal.data)
    assert span > 0.90, f"metadata claims {expected} samples but energy spans only {span:.1%} of them"
    assert dead_fraction < 0.10
    assert _edge_to_centre_power_ratio(signal.data) > 0.25


@pytest.mark.parametrize(
    "bandwidth, sample_rate",
    [(1_500_000, 10_000_000), (2_000_000, 10_000_000), (2_500_000, 20_000_000)],
)
def test_zigbee_occupied_bandwidth_matches_annotation(bandwidth, sample_rate):
    """Occupied bandwidth tracks the annotated bandwidth.

    Compressing the waveform in time expands it in frequency, so the same
    samples-per-chip error that halves the burst also doubles the chip rate and
    invalidates the frequency bounding box. Half-sine O-QPSK holds about 99% of
    its power within roughly 1.25x the chip rate, so a correct build lands near
    1.25 and the regression pushes it past 2.5.
    """
    from scipy.signal import welch

    rng = np.random.default_rng(11)
    iq = zigbee_modulator(bandwidth, sample_rate, 8192, rng)

    freqs, psd = welch(iq, fs=sample_rate, nperseg=2048, return_onesided=False, detrend=False)
    order = np.argsort(freqs)
    freqs, psd = freqs[order], psd[order]
    cumulative = np.cumsum(psd) / np.sum(psd)
    low = freqs[np.searchsorted(cumulative, 0.005)]
    high = freqs[np.searchsorted(cumulative, 0.995)]
    ratio = (high - low) / bandwidth

    assert 0.8 < ratio < 1.8, f"99% occupied bandwidth is {ratio:.2f}x the annotated bandwidth ({(high - low) / 1e6:.2f} MHz vs {bandwidth / 1e6:.2f} MHz)"
