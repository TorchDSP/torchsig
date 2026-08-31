"""Unit tests for WiFi type signal builders."""

import numpy as np
import pytest

from torchsig.signals.builders.wifi import (
    WIFI_DATA_SUBCARRIERS,
    WIFI_NUM_SUBCARRIERS,
    WIFI_OVERSAMPLING_NOMINAL,
    WIFI_PILOT_SUBCARRIERS,
    Wifi80211aSignalGenerator,
    build_ltf,
    build_stf,
    num_data_ofdm_symbols,
    wifi_80211a_modulator,
    wifi_80211a_modulator_baseband,
)
from torchsig.signals.signal_lists import CLASS_FAMILY_DICT, TorchSigSignalLists
from torchsig.utils.dsp import TorchSigComplexDataType
from torchsig.utils.signal_building import lookup_signal_generator_by_string

WIFI_METADATA = {
    "sample_rate": 20_000_000,
    "bandwidth_min": 7_000_000,
    "bandwidth_max": 9_000_000,
    "signal_duration_in_samples_min": 8192,
    "signal_duration_in_samples_max": 8192,
}


def test_wifi_subcarrier_layout():
    """48 data + 4 pilot subcarriers, none overlapping, DC excluded."""
    assert len(WIFI_DATA_SUBCARRIERS) == 48
    assert len(WIFI_PILOT_SUBCARRIERS) == 4
    assert set(WIFI_DATA_SUBCARRIERS).isdisjoint(WIFI_PILOT_SUBCARRIERS)
    assert 0 not in WIFI_DATA_SUBCARRIERS


def test_wifi_training_field_lengths():
    """STF and LTF have the standard 160-sample length (oversampled)."""
    ifft_size = WIFI_OVERSAMPLING_NOMINAL * WIFI_NUM_SUBCARRIERS
    assert len(build_stf(ifft_size)) == 160 * WIFI_OVERSAMPLING_NOMINAL
    assert len(build_ltf(ifft_size)) == 160 * WIFI_OVERSAMPLING_NOMINAL


def test_wifi_num_data_symbols_control_frames():
    """RTS/CTS/ACK map to the expected number of DATA OFDM symbols at 6 Mbps."""
    assert num_data_ofdm_symbols(20, 24) == 8  # RTS
    assert num_data_ofdm_symbols(14, 24) == 6  # CTS / ACK


def test_wifi_control_frame_symbol_counts_in_baseband():
    """Control-frame baseband length is fixed regardless of requested length."""
    osr = WIFI_OVERSAMPLING_NOMINAL
    ifft_size = osr * WIFI_NUM_SUBCARRIERS
    symbol_len = ifft_size + osr * 16
    preamble = 2 * (160 * osr) + symbol_len  # STF + LTF + SIGNAL
    rng = np.random.default_rng(0)
    rts = wifi_80211a_modulator_baseband("rts", "bpsk", 100, osr, rng)
    ack = wifi_80211a_modulator_baseband("ack", "bpsk", 100, osr, rng)
    assert len(rts) == preamble + 8 * symbol_len
    assert len(ack) == preamble + 6 * symbol_len
    # control frames are shorter than (typical) data frames
    assert len(ack) < len(rts)


@pytest.mark.parametrize("frame_type", ["data", "rts", "cts", "ack"])
def test_wifi_modulator_output(frame_type):
    """The modulator returns finite complex IQ of the requested length."""
    rng = np.random.default_rng(42)
    num_samples = 8192
    iq = wifi_80211a_modulator(
        frame_type=frame_type,
        constellation_name="qpsk",
        bandwidth=8_000_000,
        sample_rate=20_000_000,
        num_samples=num_samples,
        rng=rng,
    )
    assert iq.dtype == TorchSigComplexDataType
    assert len(iq) == num_samples
    assert np.all(np.isfinite(iq))


def test_wifi_modulator_invalid_args():
    """Invalid bandwidth and frame types raise."""
    with pytest.raises(ValueError):
        wifi_80211a_modulator("data", "qpsk", 0, 20_000_000, 8192)
    with pytest.raises(ValueError):
        wifi_80211a_modulator("data", "qpsk", 11_000_000, 20_000_000, 8192)
    with pytest.raises(ValueError):
        wifi_80211a_modulator_baseband("bogus", "qpsk", 8192, WIFI_OVERSAMPLING_NOMINAL)


@pytest.mark.parametrize(
    "frame_type,expected_name",
    [("data", "80211a"), ("rts", "80211a_rts"), ("cts", "80211a_cts"), ("ack", "80211a_ack")],
)
def test_wifi_generator_class_name(frame_type, expected_name):
    """class_name reflects the frame type."""
    md = dict(WIFI_METADATA, frame_type=frame_type)
    signal = Wifi80211aSignalGenerator(metadata=md, seed=1)()
    assert signal.class_name == expected_name
    assert signal.center_freq == 0
    assert len(signal.data) == WIFI_METADATA["signal_duration_in_samples_min"]


def test_wifi_generator_reproducible():
    """Same seed yields identical IQ."""
    a = Wifi80211aSignalGenerator(metadata=WIFI_METADATA, seed=9).generate()
    b = Wifi80211aSignalGenerator(metadata=WIFI_METADATA, seed=9).generate()
    np.testing.assert_array_equal(a.data, b.data)


def test_wifi_registered_in_lookup_table():
    """All frame generators resolve, but only data appears in 'all'."""
    control_frame_names = {"80211a_rts", "80211a_cts", "80211a_ack"}
    for name in ["80211a", "80211a_rts", "80211a_cts", "80211a_ack"]:
        assert isinstance(lookup_signal_generator_by_string(name), Wifi80211aSignalGenerator)

    all_gen = lookup_signal_generator_by_string("all")
    class_names = {generator.class_name for generator in all_gen.signal_generators if hasattr(generator, "class_name")}
    assert "80211a" in class_names
    assert control_frame_names.isdisjoint(class_names)


def test_wifi_in_signal_lists():
    """Only the 802.11a data signal is exposed in the WiFi class list."""
    control_frame_names = {"80211a_rts", "80211a_cts", "80211a_ack"}

    assert CLASS_FAMILY_DICT["80211a"] == "wifi"
    assert control_frame_names.isdisjoint(CLASS_FAMILY_DICT)
    assert "wifi" in TorchSigSignalLists.family_list
    lists = TorchSigSignalLists()
    assert lists.wifi_signals == ["80211a"]
