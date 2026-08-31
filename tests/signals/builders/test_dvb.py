"""Unit tests for DVB-S2 type signal builder."""

import numpy as np
import pytest

from torchsig.signals.builders.constellation_maps import all_symbol_maps
from torchsig.signals.builders.dvb import (
    DVBS2_CONSTELLATIONS,
    DVBS2_PLHEADER_SYMBOLS,
    DVBS2SignalGenerator,
    build_dvbs2_symbol_stream,
    build_plheader,
    dvbs2_modulator,
    pi2_bpsk,
    xfecframe_symbols,
)
from torchsig.signals.signal_lists import CLASS_FAMILY_DICT, TorchSigSignalLists
from torchsig.utils.dsp import TorchSigComplexDataType
from torchsig.utils.signal_building import lookup_signal_generator_by_string

DVBS2_METADATA = {
    "sample_rate": 10_000_000,
    "bandwidth_min": 2_000_000,
    "bandwidth_max": 3_000_000,
    "signal_duration_in_samples_min": 8192,
    "signal_duration_in_samples_max": 8192,
}


def test_dvbs2_apsk_maps_present():
    """16/32APSK maps exist with the right cardinality and unit average power."""
    for name, size in [("16apsk", 16), ("32apsk", 32)]:
        const = np.asarray(all_symbol_maps[name])
        assert len(const) == size
        assert np.isclose(np.mean(np.abs(const) ** 2), 1.0, atol=1e-6)
    # 16APSK has two distinct rings (4 + 12)
    radii = np.round(np.abs(np.asarray(all_symbol_maps["16apsk"])), 4)
    assert len(np.unique(radii)) == 2


def test_pi2_bpsk_unit_modulus_and_rotation():
    """pi/2-BPSK symbols are unit modulus and adjacent symbols are pi/2 apart."""
    sym = pi2_bpsk(np.array([0, 0, 0, 0]))
    assert np.allclose(np.abs(sym), 1.0)
    # all-zero bits -> pure pi/2 rotation between consecutive symbols
    phase_step = np.angle(sym[1] / sym[0])
    assert np.isclose(abs(phase_step), np.pi / 2)


def test_build_plheader_length():
    """PLHEADER is 90 unit-modulus symbols."""
    plheader = build_plheader()
    assert len(plheader) == DVBS2_PLHEADER_SYMBOLS == 90
    assert np.allclose(np.abs(plheader), 1.0)


def test_xfecframe_symbols():
    """XFECFRAME symbol counts match FECFRAME_bits / bits_per_symbol."""
    assert xfecframe_symbols("normal", "qpsk") == 32400
    assert xfecframe_symbols("normal", "8psk") == 21600
    assert xfecframe_symbols("normal", "16apsk") == 16200
    assert xfecframe_symbols("short", "qpsk") == 8100
    with pytest.raises(ValueError):
        xfecframe_symbols("bogus", "qpsk")


def test_build_dvbs2_symbol_stream_starts_with_plheader():
    """The stream begins with the PLHEADER and has the requested length."""
    rng = np.random.default_rng(0)
    plheader = build_plheader()
    stream = build_dvbs2_symbol_stream(500, "qpsk", "normal", False, rng)
    assert len(stream) == 500
    np.testing.assert_array_equal(stream[:90], plheader)


@pytest.mark.parametrize("constellation", DVBS2_CONSTELLATIONS)
def test_dvbs2_modulator_output(constellation):
    """The modulator returns finite complex IQ of the requested length."""
    rng = np.random.default_rng(42)
    num_samples = 8192
    iq = dvbs2_modulator(
        constellation_name=constellation,
        frame_type="normal",
        pilots=True,
        alpha_rolloff=0.25,
        bandwidth=2_500_000,
        sample_rate=10_000_000,
        num_samples=num_samples,
        rng=rng,
    )
    assert iq.dtype == TorchSigComplexDataType
    assert len(iq) == num_samples
    assert np.all(np.isfinite(iq))


def test_dvbs2_modulator_invalid_args():
    """Invalid bandwidth and roll-off raise."""
    with pytest.raises(ValueError):
        dvbs2_modulator("qpsk", "normal", False, 0.25, 0, 10_000_000, 8192)
    with pytest.raises(ValueError):
        dvbs2_modulator("qpsk", "normal", False, 0.25, 6_000_000, 10_000_000, 8192)
    with pytest.raises(ValueError):
        dvbs2_modulator("qpsk", "normal", False, 1.5, 2_500_000, 10_000_000, 8192)


def test_dvbs2_generator_generate():
    """The generator produces a Signal with correct metadata and data."""
    signal = DVBS2SignalGenerator(metadata=DVBS2_METADATA, seed=123)()
    assert signal.class_name == "dvbs2"
    assert signal.center_freq == 0
    assert len(signal.data) == DVBS2_METADATA["signal_duration_in_samples_min"]


def test_dvbs2_generator_reproducible():
    """Same seed yields identical IQ."""
    a = DVBS2SignalGenerator(metadata=DVBS2_METADATA, seed=5).generate()
    b = DVBS2SignalGenerator(metadata=DVBS2_METADATA, seed=5).generate()
    np.testing.assert_array_equal(a.data, b.data)


def test_dvbs2_registered_and_in_signal_lists():
    """'dvbs2' resolves and is in the 'dvb' family."""
    assert isinstance(lookup_signal_generator_by_string("dvbs2"), DVBS2SignalGenerator)
    assert CLASS_FAMILY_DICT["dvbs2"] == "dvb"
    lists = TorchSigSignalLists()
    assert "dvbs2" in lists.dvb_signals
