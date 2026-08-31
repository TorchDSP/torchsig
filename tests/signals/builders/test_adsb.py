"""Unit tests for ADS-B signal builder."""

import numpy as np
import pytest

from torchsig.signals.builders.adsb import (
    ADSB_PREAMBLE_CHIPS,
    AdsBSignalGenerator,
    adsb_modulator,
    build_adsb_chip_stream,
)
from torchsig.signals.signal_lists import CLASS_FAMILY_DICT, TorchSigSignalLists
from torchsig.utils.dsp import TorchSigComplexDataType
from torchsig.utils.signal_building import lookup_signal_generator_by_string

ADSB_METADATA = {
    "sample_rate": 10_000_000,
    "bandwidth_min": 1_000_000,
    "bandwidth_max": 2_000_000,
    "signal_duration_in_samples_min": 4096,
    "signal_duration_in_samples_max": 4096,
}


def test_adsb_preamble_chips():
    """Preamble has 16 chips with pulses at indices 0, 2, 7, 9."""
    assert len(ADSB_PREAMBLE_CHIPS) == 16
    assert ADSB_PREAMBLE_CHIPS[0] == 1
    assert ADSB_PREAMBLE_CHIPS[2] == 1
    assert ADSB_PREAMBLE_CHIPS[7] == 1
    assert ADSB_PREAMBLE_CHIPS[9] == 1
    assert ADSB_PREAMBLE_CHIPS[1] == 0


def test_adsb_chip_stream_starts_with_preamble():
    """Long-frame chip stream starts with the fixed preamble."""
    rng = np.random.default_rng(0)
    stream = build_adsb_chip_stream(500, "long", rng)
    np.testing.assert_array_equal(stream[:16], ADSB_PREAMBLE_CHIPS)


def test_adsb_chip_stream_frame_sizes():
    """Long and short streams have different frame periods."""
    rng = np.random.default_rng(1)
    long_chips = 16 + 112 * 2  # 240
    short_chips = 16 + 56 * 2  # 128
    l = build_adsb_chip_stream(long_chips, "long", rng)
    s = build_adsb_chip_stream(short_chips, "short", rng)
    assert len(l) == long_chips
    assert len(s) == short_chips


@pytest.mark.parametrize("frame_type", ["long", "short"])
def test_adsb_modulator_output(frame_type):
    """The modulator returns finite complex IQ of the requested length."""
    rng = np.random.default_rng(42)
    num_samples = 4096
    iq = adsb_modulator(frame_type, 1_500_000, 10_000_000, num_samples, rng)
    assert iq.dtype == TorchSigComplexDataType
    assert len(iq) == num_samples
    assert np.all(np.isfinite(iq))


def test_adsb_modulator_invalid_args():
    """Invalid bandwidth/frame type raise."""
    with pytest.raises(ValueError):
        adsb_modulator("long", 0, 10_000_000, 4096)
    with pytest.raises(ValueError):
        adsb_modulator("long", 6_000_000, 10_000_000, 4096)
    with pytest.raises(ValueError):
        adsb_modulator("bogus", 1_000_000, 10_000_000, 4096)


@pytest.mark.parametrize(
    "frame_type,expected_name",
    [("long", "adsb-long"), ("short", "adsb-short")],
)
def test_adsb_generator_class_name(frame_type, expected_name):
    """class_name reflects the frame type."""
    md = dict(ADSB_METADATA, frame_type=frame_type)
    signal = AdsBSignalGenerator(metadata=md, seed=1)()
    assert signal.class_name == expected_name


def test_adsb_generator_reproducible():
    """Same seed yields identical IQ."""
    a = AdsBSignalGenerator(metadata=ADSB_METADATA, seed=5).generate()
    b = AdsBSignalGenerator(metadata=ADSB_METADATA, seed=5).generate()
    np.testing.assert_array_equal(a.data, b.data)


def test_adsb_registered_and_in_signal_lists():
    """'adsb-long' and 'adsb-short' resolve and share the 'adsb' family."""
    assert isinstance(lookup_signal_generator_by_string("adsb-long"), AdsBSignalGenerator)
    assert isinstance(lookup_signal_generator_by_string("adsb-short"), AdsBSignalGenerator)
    assert CLASS_FAMILY_DICT["adsb-long"] == "adsb"
    assert CLASS_FAMILY_DICT["adsb-short"] == "adsb"
    lists = TorchSigSignalLists()
    assert "adsb-long" in lists.adsb_signals
    assert "adsb-short" in lists.adsb_signals
