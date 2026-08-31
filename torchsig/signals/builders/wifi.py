"""WiFi Signal Builder and Modulator modules."""

from __future__ import annotations

import numpy as np

from torchsig.signals.builder import BaseSignalGenerator
from torchsig.signals.builders.constellation_maps import all_symbol_maps
from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import (
    TorchSigComplexDataType,
    multistage_polyphase_resampler,
    pad_head_tail_to_length,
    slice_head_tail_to_length,
)

# --------------------------------------------------------------------------- #
# 802.11a
# --------------------------------------------------------------------------- #
"""IEEE 802.11a (OFDM PHY) Signal Builder and Modulator Module.

A *structured* signal: the recognizable 802.11a frame is built explicitly as
    STF (short training) -> LTF (long training) -> SIGNAL (PHY header) -> DATA
rather than as a stream of random OFDM symbols.

This is a structural model: the training fields, subcarrier allocation (48 data +
4 pilots, DC/guard nulls), cyclic prefixes, and per-frame OFDM-symbol counts are
faithful, but the bit-processing chain (scrambler, convolutional code,
interleaver, real FCS) is intentionally skipped -- the coded subcarrier values
are drawn at random. This captures the spectral/temporal *structure* needed for
RF-ML while staying lightweight.

Frame types
-----------
    * "data" : a generic data frame; fills the requested duration.
    * "rts"  : Request To Send  control frame (PSDU 20 bytes -> 8 OFDM symbols).
    * "cts"  : Clear To Send    control frame (PSDU 14 bytes -> 6 OFDM symbols).
    * "ack"  : Acknowledgement  control frame (PSDU 14 bytes -> 6 OFDM symbols).
Control frames are short bursts sent at the 6 Mbps basic rate (BPSK, R=1/2).

The 802.11a channel is a 64-point FFT (52 used subcarriers, 312.5 kHz spacing,
16-sample cyclic prefix). As in the generic OFDM builder, the waveform is built
oversampled by 4 and resampled so the channel maps to the dataset-requested
bandwidth.
"""


# 802.11a OFDM PHY parameters (IEEE Std 802.11, Clause 17)
WIFI_NUM_SUBCARRIERS: int = 64  # FFT size of the channel
WIFI_CP_LEN: int = 16  # cyclic prefix (normal guard interval), samples
WIFI_LTF_GI_LEN: int = 32  # LTF double guard interval, samples
WIFI_OVERSAMPLING_NOMINAL: int = 4  # baseband oversampling, matches ofdm.py

# Used-subcarrier layout, indexed -26..26 (DC = 0 is null).
WIFI_PILOT_SUBCARRIERS: tuple[int, ...] = (-21, -7, 7, 21)
# Pilot base values (polarity is fixed to +1 in this toy; the real standard
# multiplies these by a 127-length polarity sequence per OFDM symbol).
WIFI_PILOT_VALUES: dict[int, complex] = {-21: 1.0, -7: 1.0, 7: 1.0, 21: -1.0}
# Data subcarriers: -26..26 excluding DC and the four pilots (48 total).
WIFI_DATA_SUBCARRIERS: tuple[int, ...] = tuple(k for k in range(-26, 27) if k != 0 and k not in WIFI_PILOT_SUBCARRIERS)

# Short Training Field frequency sequence S_{-26,26} (12 active subcarriers at
# multiples of 4), scaled to unit average power over the active tones.
_STF_NONZERO: dict[int, complex] = {
    -24: 1 + 1j,
    -20: -1 - 1j,
    -16: 1 + 1j,
    -12: -1 - 1j,
    -8: -1 - 1j,
    -4: 1 + 1j,
    4: -1 - 1j,
    8: -1 - 1j,
    12: 1 + 1j,
    16: 1 + 1j,
    20: 1 + 1j,
    24: 1 + 1j,
}
_STF_SCALE: float = np.sqrt(13.0 / 6.0)

# Long Training Field frequency sequence L_{-26,26} (53 values, DC = 0).
_LTF_SEQUENCE: tuple[int, ...] = (
    1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    1,
    1,
    1,
    1,
    1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    1,
    1,
    1,
    1,
    0,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    1,
    1,
    1,
)

# Data bits per OFDM symbol (N_DBPS) by data-subcarrier constellation. Only used
# to compute the number of DATA symbols for a given PSDU length; values use the
# mandatory mod/code rate for each constellation.
WIFI_N_DBPS: dict[str, int] = {"bpsk": 24, "qpsk": 48, "16qam": 96, "64qam": 192}

# PSDU length in bytes for the supported MAC control frames.
WIFI_CONTROL_FRAME_BYTES: dict[str, int] = {"rts": 20, "cts": 14, "ack": 14}

# SERVICE field (16 bits) + TAIL (6 bits) overhead added before the PSDU.
_WIFI_SERVICE_TAIL_BITS: int = 16 + 6


def _subcarrier_index(k: int) -> int:
    """Maps a subcarrier index in [-26, 26] to its position in a length-53 array."""
    return k + 26


def num_data_ofdm_symbols(length_bytes: int, n_dbps: int) -> int:
    """Number of DATA OFDM symbols for a PSDU of the given length.

    Implements N_SYM = ceil((16 + 8*LENGTH + 6) / N_DBPS) per IEEE 802.11.

    Args:
        length_bytes: PSDU length in octets.
        n_dbps: Data bits per OFDM symbol for the chosen rate.

    Returns:
        int: Number of DATA OFDM symbols (at least 1).
    """
    total_bits = _WIFI_SERVICE_TAIL_BITS + 8 * length_bytes
    return max(1, int(np.ceil(total_bits / n_dbps)))


def _spectrum_from_subcarriers(values_53: np.ndarray, ifft_size: int) -> np.ndarray:
    """Places a length-53 (-26..26) subcarrier vector into an IFFT-sized spectrum.

    Args:
        values_53: Complex subcarrier values indexed -26..26.
        ifft_size: Size of the (oversampled) IFFT.

    Returns:
        np.ndarray: Complex spectrum of length ifft_size ready for np.fft.ifft.
    """
    spectrum = np.zeros(ifft_size, dtype=TorchSigComplexDataType)
    bins = np.arange(-26, 27) % ifft_size
    spectrum[bins] = values_53
    return spectrum


def _ofdm_symbol_time(values_53: np.ndarray, ifft_size: int, cp_len: int) -> np.ndarray:
    """IFFTs a subcarrier vector and prepends a cyclic prefix.

    Args:
        values_53: Complex subcarrier values indexed -26..26.
        ifft_size: Oversampled IFFT size.
        cp_len: Oversampled cyclic-prefix length to prepend.

    Returns:
        np.ndarray: Time-domain OFDM symbol with cyclic prefix.
    """
    spectrum = _spectrum_from_subcarriers(values_53, ifft_size)
    symbol = np.fft.ifft(spectrum)
    return np.concatenate((symbol[ifft_size - cp_len :], symbol))


def build_stf(ifft_size: int) -> np.ndarray:
    """Builds the oversampled Short Training Field (10 short symbols, 160 samples)."""
    values = np.zeros(53, dtype=TorchSigComplexDataType)
    for k, val in _STF_NONZERO.items():
        values[_subcarrier_index(k)] = _STF_SCALE * val
    symbol = np.fft.ifft(_spectrum_from_subcarriers(values, ifft_size))
    # STF spans 160 samples (2.5 base symbols); tile the periodic symbol and cut.
    osr = ifft_size // WIFI_NUM_SUBCARRIERS
    stf_len = 160 * osr
    tiled = np.tile(symbol, int(np.ceil(stf_len / len(symbol))))
    return tiled[:stf_len]


def build_ltf(ifft_size: int) -> np.ndarray:
    """Builds the oversampled Long Training Field (GI2 + 2 long symbols, 160 samples)."""
    values = np.array(_LTF_SEQUENCE, dtype=TorchSigComplexDataType)
    symbol = np.fft.ifft(_spectrum_from_subcarriers(values, ifft_size))
    osr = ifft_size // WIFI_NUM_SUBCARRIERS
    gi2 = symbol[ifft_size - WIFI_LTF_GI_LEN * osr :]
    return np.concatenate((gi2, symbol, symbol))


def _make_payload_symbol(constellation_name: str, ifft_size: int, cp_len: int, rng: np.random.Generator) -> np.ndarray:
    """Builds one SIGNAL/DATA OFDM symbol: random data subcarriers + fixed pilots.

    Args:
        constellation_name: Subcarrier constellation (e.g. 'bpsk', 'qpsk').
        ifft_size: Oversampled IFFT size.
        cp_len: Oversampled cyclic-prefix length.
        rng: Random number generator.

    Returns:
        np.ndarray: Time-domain OFDM symbol with cyclic prefix.
    """
    symbol_map = all_symbol_maps[constellation_name]
    symbol_map = symbol_map / np.sqrt(np.mean(np.abs(symbol_map) ** 2))

    values = np.zeros(53, dtype=TorchSigComplexDataType)
    # random (uncoded) data subcarriers
    data_idx = rng.integers(0, len(symbol_map), len(WIFI_DATA_SUBCARRIERS))
    for k, sym in zip(WIFI_DATA_SUBCARRIERS, symbol_map[data_idx]):
        values[_subcarrier_index(k)] = sym
    # fixed pilots (toy: polarity = +1)
    for k in WIFI_PILOT_SUBCARRIERS:
        values[_subcarrier_index(k)] = WIFI_PILOT_VALUES[k]

    return _ofdm_symbol_time(values, ifft_size, cp_len)


def wifi_80211a_modulator_baseband(
    frame_type: str,
    constellation_name: str,
    max_num_samples: int,
    oversampling_rate_nominal: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Builds an 802.11a frame at complex baseband (oversampled).

    Args:
        frame_type: One of 'data', 'rts', 'cts', 'ack'.
        constellation_name: DATA-subcarrier constellation for 'data' frames.
            Control frames are forced to BPSK.
        max_num_samples: Target oversampled length; for 'data' frames the number
            of DATA symbols is chosen to fill this length.
        oversampling_rate_nominal: Baseband oversampling factor.
        rng: Random number generator for reproducibility.

    Returns:
        np.ndarray: Oversampled baseband 802.11a frame.

    Raises:
        ValueError: If frame_type or constellation_name is unsupported, or
            max_num_samples/oversampling_rate_nominal are not positive.
    """
    if max_num_samples <= 0:
        raise ValueError("max_num_samples must be positive")
    if oversampling_rate_nominal <= 0:
        raise ValueError("oversampling_rate_nominal must be positive")
    if frame_type not in ("data", "rts", "cts", "ack"):
        raise ValueError(f"unsupported 802.11a frame_type: {frame_type}")

    if rng is None:
        rng = np.random.default_rng()

    # Control frames are sent at the 6 Mbps basic rate (BPSK).
    if frame_type in WIFI_CONTROL_FRAME_BYTES:
        constellation_name = "bpsk"
    if constellation_name not in WIFI_N_DBPS:
        raise ValueError(f"unsupported 802.11a constellation: {constellation_name}")

    ifft_size = oversampling_rate_nominal * WIFI_NUM_SUBCARRIERS
    cp_len = oversampling_rate_nominal * WIFI_CP_LEN
    symbol_len = ifft_size + cp_len  # oversampled DATA/SIGNAL symbol length

    # Preamble + PHY header
    stf = build_stf(ifft_size)
    ltf = build_ltf(ifft_size)
    signal_field = _make_payload_symbol(constellation_name, ifft_size, cp_len, rng)
    preamble_len = len(stf) + len(ltf) + len(signal_field)

    # Number of DATA OFDM symbols
    if frame_type in WIFI_CONTROL_FRAME_BYTES:
        num_data_symbols = num_data_ofdm_symbols(WIFI_CONTROL_FRAME_BYTES[frame_type], WIFI_N_DBPS[constellation_name])
    else:
        # Generic data frame: fill the requested duration.
        num_data_symbols = max(1, int(np.floor((max_num_samples - preamble_len) / symbol_len)))

    data_symbols = [_make_payload_symbol(constellation_name, ifft_size, cp_len, rng) for _ in range(num_data_symbols)]

    return np.concatenate([stf, ltf, signal_field, *data_symbols]).astype(TorchSigComplexDataType)


def wifi_80211a_modulator(
    frame_type: str,
    constellation_name: str,
    bandwidth: float,
    sample_rate: float,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """802.11a modulator: builds a frame and resamples it to the target bandwidth.

    Args:
        frame_type: One of 'data', 'rts', 'cts', 'ack'.
        constellation_name: DATA-subcarrier constellation for 'data' frames.
        bandwidth: Desired channel bandwidth of the signal (Hz).
        sample_rate: Sampling rate for the IQ signal (Hz).
        num_samples: Number of IQ samples to produce.
        rng: Random number generator for reproducibility.

    Returns:
        np.ndarray: 802.11a frame at the appropriate bandwidth, length num_samples.
        Control frames are short bursts zero-padded within the window.

    Raises:
        ValueError: If bandwidth or sample_rate are not positive.
        ValueError: If bandwidth exceeds sample_rate/2.
        ValueError: If num_samples is not positive.
    """
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if bandwidth > sample_rate / 2:
        raise ValueError("bandwidth must be less than sample_rate/2")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    if rng is None:
        rng = np.random.default_rng()

    # Resampling parameters (mirrors the generic OFDM builder).
    oversampling_rate = sample_rate / bandwidth
    resample_rate_ideal = oversampling_rate / WIFI_OVERSAMPLING_NOMINAL
    num_samples_baseband = int(np.ceil(num_samples / resample_rate_ideal))

    frame_baseband = wifi_80211a_modulator_baseband(
        frame_type,
        constellation_name,
        num_samples_baseband,
        WIFI_OVERSAMPLING_NOMINAL,
        rng,
    )

    frame_correct_bw = multistage_polyphase_resampler(frame_baseband, resample_rate_ideal)

    # Fit to requested length: pad short bursts, slice if longer than the window.
    frame_correct_bw = slice_head_tail_to_length(frame_correct_bw, num_samples) if len(frame_correct_bw) > num_samples else pad_head_tail_to_length(frame_correct_bw, num_samples)

    return frame_correct_bw.astype(TorchSigComplexDataType)


class Wifi80211aSignalGenerator(BaseSignalGenerator):
    """IEEE 802.11a Signal Generator.

    Builds a structured 802.11a OFDM frame (STF, LTF, SIGNAL header, DATA),
    optionally as an RTS/CTS/ACK control frame.
    """

    def __init__(self, **kwargs: dict[str, str | float | int]) -> None:
        """Initializes the 802.11a Signal Generator.

        Args:
            **kwargs: Metadata parameters including:

                - sample_rate: Sampling rate (Hz)
                - bandwidth_min: Minimum bandwidth (Hz)
                - bandwidth_max: Maximum bandwidth (Hz)
                - signal_duration_in_samples_min: Minimum signal duration (samples)
                - signal_duration_in_samples_max: Maximum signal duration (samples)
                - frame_type: (optional) 'data', 'rts', 'cts', or 'ack'. Default 'data'.
                - constellation_name: (optional) DATA-subcarrier constellation for
                  'data' frames ('bpsk', 'qpsk', '16qam', '64qam'). Default 'qpsk'.

        Raises:
            ValueError: If required metadata fields are missing or invalid.
        """
        super().__init__(**kwargs)
        self.required_metadata_fields = [
            "sample_rate",
            "bandwidth_min",
            "bandwidth_max",
            "signal_duration_in_samples_min",
            "signal_duration_in_samples_max",
        ]
        try:
            frame_type = self["frame_type"]
        except AttributeError:
            frame_type = "data"
        class_suffix = "" if frame_type == "data" else f"_{frame_type}"
        self.set_default_class_name(f"80211a{class_suffix}")

    def generate(self) -> Signal:
        """Generates an 802.11a signal based on the configured parameters.

        Returns:
            Signal: Generated 802.11a signal with metadata.

        Raises:
            ValueError: If required metadata fields are missing or invalid.
        """
        sample_rate = self["sample_rate"]
        num_iq_samples_signal = self.random_generator.integers(
            low=self["signal_duration_in_samples_min"],
            high=self["signal_duration_in_samples_max"] + 1,
        )
        bandwidth = self.random_generator.integers(low=self["bandwidth_min"], high=self["bandwidth_max"] + 1)

        try:
            frame_type = self["frame_type"]
        except AttributeError:
            frame_type = "data"
        try:
            constellation_name = self["constellation_name"]
        except AttributeError:
            constellation_name = "qpsk"

        signal_data = wifi_80211a_modulator(
            frame_type,
            constellation_name,
            bandwidth,
            sample_rate,
            num_iq_samples_signal,
            self.random_generator,
        )

        return Signal(data=signal_data, center_freq=0, bandwidth=bandwidth)
