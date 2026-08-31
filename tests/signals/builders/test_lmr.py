"""Unit tests for Land Mobile Radio (LMR) type signal builders."""

import numpy as np
import pytest

from torchsig.signals.builders.lmr import (
    DMR_LEVEL_BY_DIBIT,
    DMR_SYMBOLS_PER_BURST,
    DMR_SYNC_PATTERNS,
    P25_MOD_INDEX,
    DMRSignalGenerator,
    P25SignalGenerator,
    build_dmr_symbol_stream,
    build_p25_symbol_stream,
    dmr_modulator,
    hex_sync_to_levels,
    p25_modulator,
)
from torchsig.signals.signal_lists import CLASS_FAMILY_DICT, TorchSigSignalLists
from torchsig.utils.dsp import TorchSigComplexDataType
from torchsig.utils.signal_building import lookup_signal_generator_by_string

# --------------------------------------------------------------------------- #
# Digital Mobile Radio (DMR) tests
# --------------------------------------------------------------------------- #

DMR_METADATA = {
    "sample_rate": 10_000_000,
    "bandwidth_min": 2_500_000,
    "bandwidth_max": 3_333_333,
    "signal_duration_in_samples_min": 3276,
    "signal_duration_in_samples_max": 4096,
}


def test_hex_sync_to_levels_known_pattern():
    """SYNC hex expands to the expected number of normalized 4FSK levels."""
    levels = hex_sync_to_levels(DMR_SYNC_PATTERNS["bs_sourced_voice"])
    # 48-bit pattern -> 24 dibits -> 24 symbols
    assert len(levels) == 24
    # every level must be one of the four valid deviation levels
    assert np.all(np.isin(levels, DMR_LEVEL_BY_DIBIT))


def test_build_dmr_symbol_stream_embeds_sync():
    """Each burst embeds the fixed SYNC field in the expected position."""
    rng = np.random.default_rng(0)
    sync_levels = hex_sync_to_levels(DMR_SYNC_PATTERNS["bs_sourced_voice"])
    stream = build_dmr_symbol_stream(DMR_SYMBOLS_PER_BURST, sync_levels, rng)

    assert len(stream) == DMR_SYMBOLS_PER_BURST
    payload = (DMR_SYMBOLS_PER_BURST - len(sync_levels)) // 2
    np.testing.assert_array_equal(stream[payload : payload + len(sync_levels)], sync_levels)


def test_build_dmr_symbol_stream_length():
    """The stream is truncated to exactly the requested number of symbols."""
    rng = np.random.default_rng(0)
    sync_levels = hex_sync_to_levels(DMR_SYNC_PATTERNS["bs_sourced_voice"])
    stream = build_dmr_symbol_stream(50, sync_levels, rng)
    assert len(stream) == 50


def test_dmr_modulator_output():
    """The modulator returns constant-modulus complex IQ of the right length."""
    rng = np.random.default_rng(42)
    sync_levels = hex_sync_to_levels(DMR_SYNC_PATTERNS["bs_sourced_voice"])
    num_samples = 4096
    iq = dmr_modulator(
        bandwidth=3_000_000,
        sample_rate=10_000_000,
        num_samples=num_samples,
        sync_levels=sync_levels,
        rng=rng,
    )
    assert iq.dtype == TorchSigComplexDataType
    assert len(iq) == num_samples
    assert np.all(np.isfinite(iq))
    # 4FSK is a constant-envelope modulation; the steady-state interior should
    # have (near) constant modulus (ignore padded head/tail).
    interior = np.abs(iq[num_samples // 4 : 3 * num_samples // 4])
    assert interior.std() / interior.mean() < 0.05


def test_dmr_modulator_invalid_args():
    """Invalid bandwidth/sample-rate combinations raise."""
    sync_levels = hex_sync_to_levels(DMR_SYNC_PATTERNS["bs_sourced_voice"])
    with pytest.raises(ValueError):
        dmr_modulator(0, 10_000_000, 4096, sync_levels)
    with pytest.raises(ValueError):
        dmr_modulator(6_000_000, 10_000_000, 4096, sync_levels)  # bw > fs/2


def test_dmr_generator_generate():
    """The generator produces a Signal with correct metadata and data."""
    gen = DMRSignalGenerator(metadata=DMR_METADATA, seed=123)
    # __call__ (not generate) applies the class_name and any transforms.
    signal = gen()
    assert signal.class_name == "dmr"
    assert signal.center_freq == 0
    assert DMR_METADATA["bandwidth_min"] <= signal.bandwidth <= DMR_METADATA["bandwidth_max"]
    assert DMR_METADATA["signal_duration_in_samples_min"] <= len(signal.data) <= DMR_METADATA["signal_duration_in_samples_max"]


def test_dmr_generator_reproducible():
    """Two generators with the same seed produce identical IQ."""
    sig_a = DMRSignalGenerator(metadata=DMR_METADATA, seed=7).generate()
    sig_b = DMRSignalGenerator(metadata=DMR_METADATA, seed=7).generate()
    np.testing.assert_array_equal(sig_a.data, sig_b.data)


def test_dmr_registered_in_lookup_table():
    """'dmr' resolves through the string lookup and via 'all'."""
    gen = lookup_signal_generator_by_string("dmr")
    assert isinstance(gen, DMRSignalGenerator)

    all_gen = lookup_signal_generator_by_string("all")
    class_names = [g.class_name for g in all_gen.signal_generators if hasattr(g, "class_name")]
    assert "dmr" in class_names


def test_dmr_in_signal_lists():
    """'dmr' is registered as in the lmr family in the signal lists."""
    assert CLASS_FAMILY_DICT["dmr"] == "lmr"
    assert "dmr" in TorchSigSignalLists.all_signals
    assert "lmr" in TorchSigSignalLists.family_list
    # the per-family buckets are populated by __post_init__ on instantiation.
    lists = TorchSigSignalLists()
    assert "dmr" in lists.lmr_signals


# --------------------------------------------------------------------------- #
# APCO P25 (25) tests
# --------------------------------------------------------------------------- #

P25_METADATA = {
    "sample_rate": 10_000_000,
    "bandwidth_min": 2_500_000,
    "bandwidth_max": 3_333_333,
    "signal_duration_in_samples_min": 4096,
    "signal_duration_in_samples_max": 4096,
}


def test_p25_mod_index():
    """P25 outer modulation index is 2 * 1800 / 4800 = 0.75."""
    assert np.isclose(P25_MOD_INDEX, 0.75)


def test_p25_symbol_stream_starts_with_sync():
    """The symbol stream begins with the 24-symbol frame sync."""
    from torchsig.signals.builders.lmr import _P25_SYNC_LEVELS, P25_SYNC_SYMBOLS

    rng = np.random.default_rng(0)
    stream = build_p25_symbol_stream(200, rng)
    assert len(stream) == 200
    np.testing.assert_array_equal(stream[:P25_SYNC_SYMBOLS], _P25_SYNC_LEVELS)


def test_p25_modulator_output():
    """The modulator returns finite constant-envelope IQ of the right length."""
    rng = np.random.default_rng(42)
    num_samples = 4096
    iq = p25_modulator(3_000_000, 10_000_000, num_samples, rng)
    assert iq.dtype == TorchSigComplexDataType
    assert len(iq) == num_samples
    assert np.all(np.isfinite(iq))
    interior = np.abs(iq[num_samples // 4 : 3 * num_samples // 4])
    assert interior.std() / interior.mean() < 0.05


def test_p25_modulator_invalid_args():
    """Invalid bandwidth/sample-rate raise."""
    with pytest.raises(ValueError):
        p25_modulator(0, 10_000_000, 4096)
    with pytest.raises(ValueError):
        p25_modulator(6_000_000, 10_000_000, 4096)


def test_p25_generator_generate():
    """Generator produces a Signal with correct metadata."""
    signal = P25SignalGenerator(metadata=P25_METADATA, seed=1)()
    assert signal.class_name == "p25"
    assert signal.center_freq == 0
    assert len(signal.data) == P25_METADATA["signal_duration_in_samples_min"]


def test_p25_generator_reproducible():
    """Same seed yields identical IQ."""
    a = P25SignalGenerator(metadata=P25_METADATA, seed=7).generate()
    b = P25SignalGenerator(metadata=P25_METADATA, seed=7).generate()
    np.testing.assert_array_equal(a.data, b.data)


def test_p25_registered_and_in_signal_lists():
    """'p25' resolves through lookup and is in the lmr family."""
    assert isinstance(lookup_signal_generator_by_string("p25"), P25SignalGenerator)
    assert CLASS_FAMILY_DICT["p25"] == "lmr"
    lists = TorchSigSignalLists()
    assert "p25" in lists.lmr_signals
