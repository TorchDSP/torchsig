"""DVB-S2 Signal Builder and Modulator Module.

Digital Video Broadcasting - Satellite - Second Generation (DVB-S2) is
a *structured* single-carrier satellite signal (ETSI EN 302 307). The
recognizable PLFRAME structure is built explicitly:

    PLFRAME = PLHEADER (90 symbols) + XFECFRAME (data, with optional pilots)

where the PLHEADER is a fixed 26-symbol SOF + 64-symbol PLSC, modulated with
pi/2-BPSK -- the marker a DVB-S2 receiver locks onto. Data symbols use QPSK,
8PSK, 16APSK, or 32APSK, root-raised-cosine pulse shaped.

This is a limited model of the DVB-S2 standard as a signal builder: the
PLHEADER modulation, frame layout, pilot-block insertion, RRC shaping, and
constellations are faithful, but the FEC chain (BBheader, BCH/LDPC, scrambling)
and the exact PLSC/MODCOD bit encoding are skipped -- payload symbols are random.
PLFRAMEs are laid back-to-back and the stream is resampled to the dataset's
requested bandwidth, like the other builders.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sp

from torchsig.signals.builder import BaseSignalGenerator
from torchsig.signals.builders.constellation_maps import all_symbol_maps
from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import (
    TorchSigComplexDataType,
    estimate_filter_length,
    multistage_polyphase_resampler,
    pad_head_tail_to_length,
    slice_head_tail_to_length,
    slice_tail_to_length,
    srrc_taps,
)

# DVB-S2 physical-layer framing constants (ETSI EN 302 307)
DVBS2_SOF_SYMBOLS: int = 26  # Start Of Frame
DVBS2_PLSC_SYMBOLS: int = 64  # Physical Layer Signalling Code
DVBS2_PLHEADER_SYMBOLS: int = DVBS2_SOF_SYMBOLS + DVBS2_PLSC_SYMBOLS  # 90
DVBS2_PILOT_BLOCK_SYMBOLS: int = 36  # pilot block length
DVBS2_PILOT_SLOT_SYMBOLS: int = 16 * 90  # data symbols between pilot blocks (1440)

# Fixed SOF bit pattern (26 bits, ETSI EN 302 307 constant 0x18D2E82). Stored
# zero-padded to 4 bytes; only the low 26 bits are used.
DVBS2_SOF_HEX: str = "018D2E82"

# Bits per symbol per supported constellation.
DVBS2_BITS_PER_SYMBOL: dict[str, int] = {
    "qpsk": 2,
    "8psk": 3,
    "16apsk": 4,
    "32apsk": 5,
}
DVBS2_CONSTELLATIONS: tuple[str, ...] = tuple(DVBS2_BITS_PER_SYMBOL)

# FECFRAME size in bits (normal vs short frames).
DVBS2_FECFRAME_BITS: dict[str, int] = {"normal": 64800, "short": 16200}

# Allowed RRC roll-off factors.
DVBS2_ROLLOFFS: tuple[float, ...] = (0.20, 0.25, 0.35)


def pi2_bpsk(bits: np.ndarray) -> np.ndarray:
    """Modulates a bit sequence with pi/2-BPSK (as used in the DVB-S2 PLHEADER).

    Even-index symbols map to +/-(1+j)/sqrt(2); odd-index symbols are rotated by
    +pi/2 (multiplied by j).

    Args:
        bits: Array of bits (0/1).

    Returns:
        np.ndarray: Complex pi/2-BPSK symbols, unit modulus.
    """
    bits = np.asarray(bits)
    base = (1 - 2 * bits) * (1 + 1j) / np.sqrt(2)
    rotation = 1j ** (np.arange(len(bits)) % 2)
    return (base * rotation).astype(TorchSigComplexDataType)


def build_plheader(rng: np.random.Generator | None = None) -> np.ndarray:
    """Builds the 90-symbol DVB-S2 PLHEADER (pi/2-BPSK).

    The 26-bit SOF is the fixed standard pattern; the 64-bit PLSC is a fixed
    placeholder (the exact MODCOD encoding is skipped in this toy).

    Args:
        rng: Unused; present for signature consistency.

    Returns:
        np.ndarray: 90 complex PLHEADER symbols.
    """
    sof_bits = np.unpackbits(np.frombuffer(bytes.fromhex(DVBS2_SOF_HEX), dtype=np.uint8))
    sof_bits = sof_bits[-DVBS2_SOF_SYMBOLS:]  # low 26 bits
    # Fixed PLSC placeholder pattern (alternating), toy stand-in for MODCOD code.
    plsc_bits = np.tile([0, 1], DVBS2_PLSC_SYMBOLS // 2)
    return pi2_bpsk(np.concatenate([sof_bits, plsc_bits]))


def xfecframe_symbols(frame_type: str, constellation_name: str) -> int:
    """Number of data symbols in one XFECFRAME for the given frame type/constellation.

    Args:
        frame_type: 'normal' or 'short'.
        constellation_name: One of the DVB-S2 constellations.

    Returns:
        int: XFECFRAME length in symbols.

    Raises:
        ValueError: If frame_type or constellation_name is unsupported.
    """
    if frame_type not in DVBS2_FECFRAME_BITS:
        raise ValueError(f"unsupported DVB-S2 frame_type: {frame_type}")
    if constellation_name not in DVBS2_BITS_PER_SYMBOL:
        raise ValueError(f"unsupported DVB-S2 constellation: {constellation_name}")
    return DVBS2_FECFRAME_BITS[frame_type] // DVBS2_BITS_PER_SYMBOL[constellation_name]


def build_dvbs2_symbol_stream(
    num_symbols: int,
    constellation_name: str,
    frame_type: str,
    pilots: bool,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Builds a structured DVB-S2 symbol stream of the requested length.

    The stream is composed of back-to-back PLFRAMEs (PLHEADER + XFECFRAME), with
    36-symbol pilot blocks inserted every 1440 data symbols when enabled, then
    truncated to num_symbols. Payload symbols are random.

    Args:
        num_symbols: Number of symbols to produce.
        constellation_name: Data-symbol constellation.
        frame_type: 'normal' or 'short' FECFRAME.
        pilots: Whether to insert pilot blocks.
        rng: Random number generator for reproducibility.

    Returns:
        np.ndarray: Complex symbol stream, length num_symbols.

    Raises:
        ValueError: If num_symbols is not positive.
    """
    if num_symbols <= 0:
        raise ValueError("num_symbols must be positive")

    if rng is None:
        rng = np.random.default_rng()

    symbol_map = all_symbol_maps[constellation_name]
    symbol_map = symbol_map / np.sqrt(np.mean(np.abs(symbol_map) ** 2))
    data_per_frame = xfecframe_symbols(frame_type, constellation_name)
    # pilot symbol: unmodulated pi/2-BPSK reference, (1+j)/sqrt(2)
    pilot_symbol = np.array([(1 + 1j) / np.sqrt(2)], dtype=TorchSigComplexDataType)

    chunks: list[np.ndarray] = []
    total = 0
    while total < num_symbols:
        plheader = build_plheader(rng)
        chunks.append(plheader)
        total += len(plheader)

        produced = 0
        while produced < data_per_frame and total < num_symbols:
            block = min(
                DVBS2_PILOT_SLOT_SYMBOLS if pilots else data_per_frame,
                data_per_frame - produced,
            )
            data = symbol_map[rng.integers(0, len(symbol_map), block)]
            chunks.append(data.astype(TorchSigComplexDataType))
            produced += block
            total += block
            if pilots and produced < data_per_frame:
                chunks.append(np.repeat(pilot_symbol, DVBS2_PILOT_BLOCK_SYMBOLS))
                total += DVBS2_PILOT_BLOCK_SYMBOLS

    return np.concatenate(chunks)[:num_symbols]


def dvbs2_modulator_baseband(
    constellation_name: str,
    frame_type: str,
    pilots: bool,
    alpha_rolloff: float,
    max_num_samples: int,
    oversampling_rate_nominal: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """DVB-S2 modulator at complex baseband.

    Builds a structured PLFRAME symbol stream and root-raised-cosine pulse shapes it.

    Args:
        constellation_name: Data-symbol constellation.
        frame_type: 'normal' or 'short' FECFRAME.
        pilots: Whether to insert pilot blocks.
        alpha_rolloff: RRC roll-off factor (0 < alpha < 1).
        max_num_samples: Maximum number of samples to produce.
        oversampling_rate_nominal: Baseband samples per symbol.
        rng: Random number generator for reproducibility.

    Returns:
        np.ndarray: DVB-S2 modulated signal at baseband.

    Raises:
        ValueError: If max_num_samples or oversampling_rate_nominal are not positive.
        ValueError: If alpha_rolloff is not in (0, 1).
    """
    if max_num_samples <= 0:
        raise ValueError("max_num_samples must be positive")
    if oversampling_rate_nominal <= 0:
        raise ValueError("oversampling_rate_nominal must be positive")
    if not 0 < alpha_rolloff < 1:
        raise ValueError("alpha_rolloff must be between 0 and 1")

    if rng is None:
        rng = np.random.default_rng()

    samples_per_symbol = oversampling_rate_nominal

    # Root-raised-cosine pulse shape.
    attenuation_db = 120
    pulse_shape_filter_length = estimate_filter_length(alpha_rolloff, attenuation_db, 1)
    pulse_shape_filter_span = int(np.ceil((pulse_shape_filter_length - 1) / (2 * samples_per_symbol)))
    pulse_shape = srrc_taps(samples_per_symbol, pulse_shape_filter_span, alpha_rolloff)

    # Number of symbols that fit within max_num_samples after pulse shaping.
    subtract_off_symbols = 2 * pulse_shape_filter_span
    num_symbols = max(1, int(np.floor(max_num_samples / samples_per_symbol)) - subtract_off_symbols)

    symbols = build_dvbs2_symbol_stream(num_symbols, constellation_name, frame_type, pilots, rng)
    modulated = sp.upfirdn(pulse_shape, symbols, up=samples_per_symbol, down=1)

    # Adjust signal length
    if len(modulated) > max_num_samples:
        modulated = slice_tail_to_length(modulated, max_num_samples)
    elif len(modulated) < max_num_samples:
        modulated = pad_head_tail_to_length(modulated, max_num_samples)

    return modulated.astype(TorchSigComplexDataType)


def dvbs2_modulator(
    constellation_name: str,
    frame_type: str,
    pilots: bool,
    alpha_rolloff: float,
    bandwidth: float,
    sample_rate: float,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """DVB-S2 modulator.

    Args:
        constellation_name: Data-symbol constellation.
        frame_type: 'normal' or 'short' FECFRAME.
        pilots: Whether to insert pilot blocks.
        alpha_rolloff: RRC roll-off factor (0 < alpha < 1).
        bandwidth: Desired symbol-rate bandwidth of the signal (Hz).
        sample_rate: Sampling rate for the IQ signal (Hz).
        num_samples: Number of IQ samples to produce.
        rng: Random number generator for reproducibility.

    Returns:
        np.ndarray: DVB-S2 modulated signal at the appropriate bandwidth.

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

    # Resampling parameters (mirrors the constellation builder).
    oversampling_rate = sample_rate / bandwidth
    oversampling_rate_baseband = 4
    resample_rate_ideal = oversampling_rate / oversampling_rate_baseband
    num_samples_baseband = max(oversampling_rate_baseband, int(np.floor(num_samples / resample_rate_ideal)))

    signal_baseband = dvbs2_modulator_baseband(
        constellation_name,
        frame_type,
        pilots,
        alpha_rolloff,
        num_samples_baseband,
        oversampling_rate_baseband,
        rng,
    )

    signal_correct_bw = multistage_polyphase_resampler(signal_baseband, resample_rate_ideal)

    # Adjust signal length
    signal_correct_bw = slice_head_tail_to_length(signal_correct_bw, num_samples) if len(signal_correct_bw) > num_samples else pad_head_tail_to_length(signal_correct_bw, num_samples)

    return signal_correct_bw.astype(TorchSigComplexDataType)


class DVBS2SignalGenerator(BaseSignalGenerator):
    """DVB-S2 Signal Generator.

    Builds a structured DVB-S2 single-carrier waveform: pi/2-BPSK PLHEADER
    followed by RRC-shaped QPSK/8PSK/16APSK/32APSK data, with optional pilots.
    """

    def __init__(self, **kwargs: dict[str, str | float | int]) -> None:
        """Initializes the DVB-S2 Signal Generator.

        Args:
            **kwargs: Metadata parameters including:

                - sample_rate: Sampling rate (Hz)
                - bandwidth_min: Minimum bandwidth (Hz)
                - bandwidth_max: Maximum bandwidth (Hz)
                - signal_duration_in_samples_min: Minimum signal duration (samples)
                - signal_duration_in_samples_max: Maximum signal duration (samples)
                - constellation_name: (optional) one of 'qpsk', '8psk', '16apsk',
                  '32apsk'. If absent, a random one is chosen per signal.
                - frame_type: (optional) 'normal' or 'short'. Default 'normal'.
                - pilots: (optional) bool, insert pilot blocks. Default random.
                - alpha_rolloff: (optional) RRC roll-off. If absent, a random
                  standard value (0.20/0.25/0.35) is chosen per signal.

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
        self.set_default_class_name("dvbs2")

    def generate(self) -> Signal:
        """Generates a DVB-S2 signal based on the configured parameters.

        Returns:
            Signal: Generated DVB-S2 signal with metadata.

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
            constellation_name = self["constellation_name"]
        except AttributeError:
            constellation_name = DVBS2_CONSTELLATIONS[self.random_generator.integers(0, len(DVBS2_CONSTELLATIONS))]
        try:
            frame_type = self["frame_type"]
        except AttributeError:
            frame_type = "normal"
        try:
            pilots = bool(self["pilots"])
        except AttributeError:
            pilots = bool(self.random_generator.integers(0, 2))
        try:
            alpha_rolloff = self["alpha_rolloff"]
        except AttributeError:
            alpha_rolloff = DVBS2_ROLLOFFS[self.random_generator.integers(0, len(DVBS2_ROLLOFFS))]

        signal_data = dvbs2_modulator(
            constellation_name,
            frame_type,
            pilots,
            alpha_rolloff,
            bandwidth,
            sample_rate,
            num_iq_samples_signal,
            self.random_generator,
        )

        return Signal(data=signal_data, center_freq=0, bandwidth=bandwidth)
