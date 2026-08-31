"""Land Mobile Radio (LMR) Signal Builders and Modulators module."""

from __future__ import annotations

import numpy as np
import scipy.signal as sp

from torchsig.signals.builder import BaseSignalGenerator
from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import (
    TorchSigComplexDataType,
    multistage_polyphase_resampler,
    pad_head_tail_to_length,
    slice_head_tail_to_length,
    slice_tail_to_length,
    srrc_taps,
)

# --------------------------------------------------------------------------- #
# Digital Modile Radio (DMR)
# --------------------------------------------------------------------------- #

"""DMR is a type of Land Mobile Radio (LMR) standard. It
has a defined burst/frame structure: fixed SYNC patterns
interleaved with random payload symbols. The framing is what makes the
waveform recognizable, so it is modeled explicitly here.

DMR (ETSI TS 102 361) is a 4-level FSK (4FSK / C4FM-style) modulation:
    * symbol rate      : 4800 symbols/s
    * deviations       : {+1944, +648, -648, -1944} Hz (outer/inner = +/-3, +/-1)
    * channel spacing  : 12.5 kHz
    * pulse shaping     : root-raised-cosine

Toy simplifications (relative to the full standard):
    * No TDMA two-slot timing, FEC, or embedded signalling is modeled.
    * A single recognizable 24-symbol SYNC field is placed in each 132-symbol
      burst; the remaining symbols are random payload.
    * The absolute deviation/symbol-rate ratios (i.e. the 4FSK *structure*) are
      preserved, while the occupied bandwidth is scaled to the dataset-requested
      bandwidth via resampling, exactly like the other TorchSig builders.
"""

# DMR physical-layer constants (ETSI TS 102 361-1)
DMR_SYMBOL_RATE: float = 4800.0  # symbols/s
DMR_OUTER_DEVIATION_HZ: float = 1944.0  # Hz, the +/-3 deviation level
DMR_RRC_ALPHA: float = 0.2  # root-raised-cosine roll-off

# Outer 4FSK modulation index h = 2 * deviation / symbol_rate.
# The inner levels (+/-1) sit at one third of this deviation.
DMR_MOD_INDEX: float = 2.0 * DMR_OUTER_DEVIATION_HZ / DMR_SYMBOL_RATE

# Mapping from an information dibit (2 bits, MSB first) to its normalized 4FSK
# deviation level, where the outer deviation is +/-1.0 (TS 102 361-1, Table).
#   dibit 01 -> +1944 Hz (+3)   -> +1.0
#   dibit 00 -> + 648 Hz (+1)   -> +1/3
#   dibit 10 -> - 648 Hz (-1)   -> -1/3
#   dibit 11 -> -1944 Hz (-3)   -> -1.0
# Indexed by the integer value of the dibit (00->0, 01->1, 10->2, 11->3).
DMR_LEVEL_BY_DIBIT: np.ndarray = np.array([1.0 / 3.0, 1.0, -1.0 / 3.0, -1.0])

# A burst is 264 bits = 132 symbols: 54 payload + 24 sync + 54 payload symbols.
DMR_SYMBOLS_PER_BURST: int = 132
DMR_SYNC_SYMBOLS: int = 24
DMR_PAYLOAD_SYMBOLS: int = (DMR_SYMBOLS_PER_BURST - DMR_SYNC_SYMBOLS) // 2

# A selection of the 48-bit (24-symbol) SYNC patterns defined by the standard,
# expressed as hex. Any one of these makes the burst structure recognizable.
DMR_SYNC_PATTERNS: dict[str, str] = {
    "bs_sourced_voice": "755FD7DF75F7",
    "bs_sourced_data": "DFF57D75DF5D",
    "ms_sourced_voice": "7F7D5DD57DFD",
    "ms_sourced_data": "D5D7F77FD757",
}
DEFAULT_DMR_SYNC_PATTERN: str = "bs_sourced_voice"


def hex_sync_to_levels(sync_hex: str) -> np.ndarray:
    """Converts a hex SYNC pattern into an array of normalized 4FSK levels.

    Each hex character expands to 4 bits; bits are grouped into dibits
    (MSB first) and mapped to their deviation level via DMR_LEVEL_BY_DIBIT.

    Args:
        sync_hex: SYNC pattern as a hex string (e.g. '755FD7DF75F7').

    Returns:
        np.ndarray: Normalized 4FSK deviation levels for the SYNC field.

    Raises:
        ValueError: If the pattern does not contain an even number of dibits.
    """
    num_bits = len(sync_hex) * 4
    if num_bits % 2 != 0:
        raise ValueError("SYNC pattern must contain an even number of bits")
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(sync_hex), dtype=np.uint8))
    dibits = bits.reshape(-1, 2)
    dibit_values = dibits[:, 0] * 2 + dibits[:, 1]
    return DMR_LEVEL_BY_DIBIT[dibit_values]


def build_dmr_symbol_stream(
    num_symbols: int,
    sync_levels: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Builds a structured DMR 4FSK level stream of the requested length.

    The stream is composed of repeated 132-symbol bursts, each laid out as
    [payload | SYNC | payload], then truncated to num_symbols. Payload symbols
    are random; the SYNC field is fixed, giving the burst its structure.

    Args:
        num_symbols: Number of 4FSK symbols to produce.
        sync_levels: Normalized 4FSK levels for the fixed SYNC field.
        rng: Random number generator for reproducibility. If None, a new
            default generator is created.

    Returns:
        np.ndarray: Normalized 4FSK deviation levels, length num_symbols.

    Raises:
        ValueError: If num_symbols is not positive.
    """
    if num_symbols <= 0:
        raise ValueError("num_symbols must be positive")

    if rng is None:
        rng = np.random.default_rng()

    num_bursts = int(np.ceil(num_symbols / DMR_SYMBOLS_PER_BURST))
    bursts = []
    for _ in range(num_bursts):
        payload_pre = DMR_LEVEL_BY_DIBIT[rng.integers(0, 4, DMR_PAYLOAD_SYMBOLS)]
        payload_post = DMR_LEVEL_BY_DIBIT[rng.integers(0, 4, DMR_PAYLOAD_SYMBOLS)]
        bursts.append(np.concatenate([payload_pre, sync_levels, payload_post]))

    stream = np.concatenate(bursts)
    return stream[:num_symbols]


def dmr_modulator_baseband(
    max_num_samples: int,
    oversampling_rate_nominal: int,
    sync_levels: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """DMR 4FSK modulator at complex baseband.

    Builds a structured 4FSK level stream, root-raised-cosine pulse shapes the
    frequency deviation, and integrates it into instantaneous phase (CPFSK).

    Args:
        max_num_samples: Maximum number of samples to produce.
        oversampling_rate_nominal: Baseband samples per symbol.
        sync_levels: Normalized 4FSK levels for the fixed SYNC field.
        rng: Random number generator for reproducibility. If None, a new
            default generator is created.

    Returns:
        np.ndarray: DMR modulated signal at baseband.

    Raises:
        ValueError: If max_num_samples or oversampling_rate_nominal are not positive.
    """
    if max_num_samples <= 0:
        raise ValueError("max_num_samples must be positive")
    if oversampling_rate_nominal <= 0:
        raise ValueError("oversampling_rate_nominal must be positive")

    if rng is None:
        rng = np.random.default_rng()

    samples_per_symbol = oversampling_rate_nominal

    # Root-raised-cosine frequency pulse, spanning a few symbols.
    pulse_shape_filter_span = 4
    pulse_shape = srrc_taps(samples_per_symbol, pulse_shape_filter_span, DMR_RRC_ALPHA)

    # Number of symbols that fit within max_num_samples after pulse shaping.
    max_num_samples_minus_pulse_shape = max_num_samples - len(pulse_shape) + 1
    num_symbols = max(1, int(np.floor(max_num_samples_minus_pulse_shape / samples_per_symbol)))

    # Build the structured level stream and pulse-shape the frequency deviation.
    levels = build_dmr_symbol_stream(num_symbols, sync_levels, rng)
    freq = sp.upfirdn(pulse_shape, levels, up=samples_per_symbol, down=1)

    # Integrate frequency into phase (CPFSK). The scaling is chosen so that a
    # single symbol of normalized level L contributes pi * h * L radians, i.e.
    # 2*pi*(deviation*L)/symbol_rate, independent of the pulse-shape gain.
    phase = np.cumsum(freq) * (np.pi * DMR_MOD_INDEX / np.sum(pulse_shape))
    modulated = np.exp(1j * phase)

    # Adjust signal length
    if len(modulated) > max_num_samples:
        modulated = slice_tail_to_length(modulated, max_num_samples)
    elif len(modulated) < max_num_samples:
        modulated = pad_head_tail_to_length(modulated, max_num_samples)

    return modulated


def dmr_modulator(
    bandwidth: float,
    sample_rate: float,
    num_samples: int,
    sync_levels: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """DMR 4FSK modulator.

    Args:
        bandwidth: Desired 3 dB bandwidth of the signal (Hz).
        sample_rate: Sampling rate for the IQ signal (Hz).
        num_samples: Number of IQ samples to produce.
        sync_levels: Normalized 4FSK levels for the fixed SYNC field.
        rng: Random number generator for reproducibility. If None, a new
            default generator is created.

    Returns:
        np.ndarray: DMR modulated signal at the appropriate bandwidth.

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

    # Baseband modulation parameters
    oversampling_rate_nominal = 4
    oversampling_rate = sample_rate / bandwidth
    resample_rate_ideal = oversampling_rate / oversampling_rate_nominal

    # Calculate baseband samples
    max_num_samples = max(oversampling_rate_nominal, int(np.floor(num_samples / resample_rate_ideal)))

    # Generate and resample signal
    baseband_signal = dmr_modulator_baseband(max_num_samples, oversampling_rate_nominal, sync_levels, rng)

    dmr_correct_bw = multistage_polyphase_resampler(baseband_signal, resample_rate_ideal)
    dmr_correct_bw *= 1 / resample_rate_ideal

    # Adjust signal length
    dmr_correct_bw = slice_head_tail_to_length(dmr_correct_bw, num_samples) if len(dmr_correct_bw) > num_samples else pad_head_tail_to_length(dmr_correct_bw, num_samples)

    return dmr_correct_bw.astype(TorchSigComplexDataType)


class DMRSignalGenerator(BaseSignalGenerator):
    """DMR Signal Generator.

    Implements a toy Digital Mobile Radio (4FSK) waveform with a structured
    burst layout: fixed SYNC fields interleaved with random payload symbols.
    """

    def __init__(self, **kwargs: dict[str, str | float | int]) -> None:
        """Initializes DMR Signal Generator.

        Args:
            **kwargs: Metadata parameters including:

                - sample_rate: Sampling rate (Hz)
                - bandwidth_min: Minimum bandwidth (Hz)
                - bandwidth_max: Maximum bandwidth (Hz)
                - signal_duration_in_samples_min: Minimum signal duration (samples)
                - signal_duration_in_samples_max: Maximum signal duration (samples)
                - sync_pattern: (optional) Name of the SYNC pattern to use, one of
                  DMR_SYNC_PATTERNS. Defaults to a burst-sourced voice sync.

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
        self.set_default_class_name("dmr")

    def generate(self) -> Signal:
        """Generates a DMR signal based on the configured parameters.

        Returns:
            Signal: Generated DMR signal with metadata.

        Raises:
            ValueError: If required metadata fields are missing or invalid.
            ValueError: If the configured sync_pattern is unknown.
        """
        # Get parameters from metadata
        sample_rate = self["sample_rate"]
        num_iq_samples_signal = self.random_generator.integers(
            low=self["signal_duration_in_samples_min"],
            high=self["signal_duration_in_samples_max"] + 1,
        )
        bandwidth = self.random_generator.integers(low=self["bandwidth_min"], high=self["bandwidth_max"] + 1)

        # Resolve the SYNC pattern (default if not specified in metadata).
        try:
            sync_pattern = self["sync_pattern"]
        except AttributeError:
            sync_pattern = DEFAULT_DMR_SYNC_PATTERN
        if sync_pattern not in DMR_SYNC_PATTERNS:
            raise ValueError(f"unknown DMR sync_pattern '{sync_pattern}'; expected one of {list(DMR_SYNC_PATTERNS)}")
        sync_levels = hex_sync_to_levels(DMR_SYNC_PATTERNS[sync_pattern])

        # Generate signal
        signal_data = dmr_modulator(
            bandwidth,
            sample_rate,
            num_iq_samples_signal,
            sync_levels,
            self.random_generator,
        )

        return Signal(data=signal_data, center_freq=0, bandwidth=bandwidth)


# --------------------------------------------------------------------------- #
# APCO Project 25 (P25)
# --------------------------------------------------------------------------- #
"""APCO Project 25 Phase I (C4FM) Signal Builder.

P25 Phase I is a land-mobile radio standard used by North American public-safety
agencies. Like DMR it uses 4-level FM at 4800 Bd over 12.5 kHz channels, but
with different deviation ratios, frame structure, and sync word.

Physical layer (TIA-102.BAAA):
    * Symbol rate      : 4800 Bd
    * Deviations       : ±1800 Hz (outer, symbol ±3) and ±600 Hz (inner, symbol ±1)
    * Modulation index : h = 2 × 1800 / 4800 = 0.75
    * Pulse shaping    : root-raised-cosine, α = 0.2
    * Channel spacing  : 12.5 kHz

Frame structure (one logical channel frame, 1080 symbols):
    * 24-symbol frame sync (fixed pattern)
    * 32-symbol NID (random for toy)
    * 1024-symbol payload (random)

Toy simplifications:
    * No IMBE voice codec, FEC, or signalling is modeled.
    * NID and payload are random 4-level symbols.
"""

P25_SYMBOL_RATE: float = 4800.0
P25_OUTER_DEVIATION_HZ: float = 1800.0
P25_RRC_ALPHA: float = 0.2
P25_MOD_INDEX: float = 2.0 * P25_OUTER_DEVIATION_HZ / P25_SYMBOL_RATE  # 0.75

# Dibit → normalized level mapping (same encoding as DMR, different deviations)
# dibit 01 → +1800 Hz → +1.0
# dibit 00 → + 600 Hz → +1/3
# dibit 10 → - 600 Hz → -1/3
# dibit 11 → -1800 Hz → -1.0
P25_LEVEL_BY_DIBIT: np.ndarray = np.array([1.0 / 3.0, 1.0, -1.0 / 3.0, -1.0])

# Frame sync: 48-bit / 24-symbol pattern (TIA-102.BAAA)
P25_FRAME_SYNC_HEX: str = "5575F5FF7FF5"

P25_SYNC_SYMBOLS: int = 24
P25_NID_SYMBOLS: int = 32
P25_PAYLOAD_SYMBOLS: int = 1024
P25_FRAME_SYMBOLS: int = P25_SYNC_SYMBOLS + P25_NID_SYMBOLS + P25_PAYLOAD_SYMBOLS  # 1080


def _hex_to_levels(sync_hex: str) -> np.ndarray:
    """Converts a hex sync pattern into normalized 4-level deviation values."""
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(sync_hex), dtype=np.uint8))
    dibits = bits.reshape(-1, 2)
    dibit_vals = dibits[:, 0] * 2 + dibits[:, 1]
    return P25_LEVEL_BY_DIBIT[dibit_vals]


_P25_SYNC_LEVELS: np.ndarray = _hex_to_levels(P25_FRAME_SYNC_HEX)


def build_p25_symbol_stream(
    num_symbols: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Builds a P25 C4FM level stream: repeated frames of [sync | NID | payload].

    Args:
        num_symbols: Number of 4-level symbols to produce.
        rng: Random number generator.

    Returns:
        np.ndarray: Normalized deviation levels, length num_symbols.

    Raises:
        ValueError: If num_symbols is not positive.
    """
    if num_symbols <= 0:
        raise ValueError("num_symbols must be positive")

    if rng is None:
        rng = np.random.default_rng()

    num_frames = int(np.ceil(num_symbols / P25_FRAME_SYMBOLS))
    frames = []
    for _ in range(num_frames):
        nid = P25_LEVEL_BY_DIBIT[rng.integers(0, 4, P25_NID_SYMBOLS)]
        payload = P25_LEVEL_BY_DIBIT[rng.integers(0, 4, P25_PAYLOAD_SYMBOLS)]
        frames.append(np.concatenate([_P25_SYNC_LEVELS, nid, payload]))

    return np.concatenate(frames)[:num_symbols]


def p25_modulator_baseband(
    max_num_samples: int,
    oversampling_rate_nominal: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """P25 C4FM modulator at complex baseband.

    Args:
        max_num_samples: Maximum output samples.
        oversampling_rate_nominal: Samples per symbol at baseband.
        rng: Random number generator.

    Returns:
        np.ndarray: Baseband P25 signal, exactly max_num_samples long.

    Raises:
        ValueError: If max_num_samples or oversampling_rate_nominal are not positive.
    """
    if max_num_samples <= 0:
        raise ValueError("max_num_samples must be positive")
    if oversampling_rate_nominal <= 0:
        raise ValueError("oversampling_rate_nominal must be positive")

    if rng is None:
        rng = np.random.default_rng()

    samples_per_symbol = oversampling_rate_nominal
    filter_span = 4
    pulse_shape = srrc_taps(samples_per_symbol, filter_span, P25_RRC_ALPHA)

    max_minus_filter = max_num_samples - len(pulse_shape) + 1
    num_symbols = max(1, int(np.floor(max_minus_filter / samples_per_symbol)))

    levels = build_p25_symbol_stream(num_symbols, rng)
    freq = sp.upfirdn(pulse_shape, levels, up=samples_per_symbol, down=1)
    phase = np.cumsum(freq) * (np.pi * P25_MOD_INDEX / np.sum(pulse_shape))
    modulated = np.exp(1j * phase)

    if len(modulated) > max_num_samples:
        modulated = slice_tail_to_length(modulated, max_num_samples)
    elif len(modulated) < max_num_samples:
        modulated = pad_head_tail_to_length(modulated, max_num_samples)

    return modulated


def p25_modulator(
    bandwidth: float,
    sample_rate: float,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """P25 C4FM modulator: builds level stream and resamples to target bandwidth.

    Args:
        bandwidth: Desired signal bandwidth (Hz).
        sample_rate: Capture sampling rate (Hz).
        num_samples: Number of IQ samples to produce.
        rng: Random number generator.

    Returns:
        np.ndarray: P25 IQ at the requested bandwidth, length num_samples.

    Raises:
        ValueError: If bandwidth or sample_rate are not positive, bandwidth > sample_rate/2,
            or num_samples is not positive.
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

    oversampling_rate_nominal = 4
    oversampling_rate = sample_rate / bandwidth
    resample_rate_ideal = oversampling_rate / oversampling_rate_nominal
    max_num_samples = max(oversampling_rate_nominal, int(np.floor(num_samples / resample_rate_ideal)))

    baseband = p25_modulator_baseband(max_num_samples, oversampling_rate_nominal, rng)
    correct_bw = multistage_polyphase_resampler(baseband, resample_rate_ideal)
    correct_bw *= 1 / resample_rate_ideal

    correct_bw = slice_head_tail_to_length(correct_bw, num_samples) if len(correct_bw) > num_samples else pad_head_tail_to_length(correct_bw, num_samples)
    return correct_bw.astype(TorchSigComplexDataType)


class P25SignalGenerator(BaseSignalGenerator):
    """APCO P25 Phase I Signal Generator.

    Implements a toy P25 C4FM waveform with a structured frame layout:
    fixed 24-symbol sync word, random 32-symbol NID, random 1024-symbol payload.
    """

    def __init__(self, **kwargs: dict[str, str | float | int]) -> None:
        """Initializes the P25 Signal Generator.

        Args:
            **kwargs: Metadata parameters including:

                - sample_rate: Sampling rate (Hz)
                - bandwidth_min: Minimum bandwidth (Hz)
                - bandwidth_max: Maximum bandwidth (Hz)
                - signal_duration_in_samples_min: Minimum signal duration (samples)
                - signal_duration_in_samples_max: Maximum signal duration (samples)
        """
        super().__init__(**kwargs)
        self.required_metadata_fields = [
            "sample_rate",
            "bandwidth_min",
            "bandwidth_max",
            "signal_duration_in_samples_min",
            "signal_duration_in_samples_max",
        ]
        self.set_default_class_name("p25")

    def generate(self) -> Signal:
        """Generates a P25 signal.

        Returns:
            Signal: Generated P25 signal with metadata.
        """
        sample_rate = self["sample_rate"]
        num_iq_samples_signal = self.random_generator.integers(
            low=self["signal_duration_in_samples_min"],
            high=self["signal_duration_in_samples_max"] + 1,
        )
        bandwidth = self.random_generator.integers(low=self["bandwidth_min"], high=self["bandwidth_max"] + 1)
        signal_data = p25_modulator(bandwidth, sample_rate, num_iq_samples_signal, self.random_generator)
        return Signal(data=signal_data, center_freq=0, bandwidth=bandwidth)
