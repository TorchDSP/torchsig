"""IEEE 802.15.4 / ZigBee Signal Builder.

802.15.4 at 2.4 GHz uses O-QPSK with DSSS: each 4-bit data nibble is mapped to
one of 16 standard 32-chip spreading sequences, then modulated as offset QPSK
with half-sine pulse shaping. The preamble (4x0x00) and SFD (0xA7) produce a
distinctive correlation signature at the start of every packet.

Physical layer (IEEE 802.15.4-2015, 2.4 GHz):
    * Chip rate  : 2 Mchips/s
    * Data rate  : 250 kbps (4 bits per 32-chip symbol)
    * Modulation : O-QPSK, half-sine pulse shape (MSK-equivalent envelope)
    * Spreading  : each 4-bit nibble → 32-chip sequence (Table 70)

Packet structure:
    4-byte preamble (0x00) | 1-byte SFD (0xA7) | 1-byte PHR (length) | PSDU

Toy simplifications:
    * No FCS, no MAC header encoding; PSDU bytes are random.
    * Packets are concatenated without inter-frame gaps.
"""

from __future__ import annotations

import numpy as np

from torchsig.signals.builder import BaseSignalGenerator
from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import (
    TorchSigComplexDataType,
    multistage_polyphase_resampler,
    pad_head_tail_to_length,
    slice_head_tail_to_length,
    slice_tail_to_length,
)

# 16 chip sequences from IEEE 802.15.4-2015 Table 70 (2.4 GHz, chip 0 = first)
ZIGBEE_CHIP_SEQS: np.ndarray = np.array(
    [
        [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
        [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
        [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1],
        [0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
    ],
    dtype=np.int8,
)

_ZIGBEE_SFD: int = 0xA7
_CHIPS_PER_NIBBLE: int = 32


def _bytes_to_chips(data: bytes) -> np.ndarray:
    """Converts bytes to 802.15.4 DSSS chips (LSB nibble first per byte)."""
    chunks = []
    for byte in data:
        lo = byte & 0x0F
        hi = (byte >> 4) & 0x0F
        chunks.append(ZIGBEE_CHIP_SEQS[lo])
        chunks.append(ZIGBEE_CHIP_SEQS[hi])
    return np.concatenate(chunks).astype(np.int8)


def build_zigbee_chip_stream(num_chips: int, rng: np.random.Generator) -> np.ndarray:
    """Builds a chip stream of repeated 802.15.4 packets.

    Args:
        num_chips: Total chips to produce.
        rng: Random number generator.

    Returns:
        np.ndarray: Chip stream of length num_chips, values in {0, 1}.
    """
    chips = []
    while sum(len(c) for c in chips) < num_chips:
        psdu_len = int(rng.integers(1, 128))
        psdu = rng.integers(0, 256, psdu_len, dtype=np.uint8).tobytes()
        phr = bytes([psdu_len & 0x7F])
        preamble = bytes(4)
        sfd = bytes([_ZIGBEE_SFD])
        packet_bytes = preamble + sfd + phr + psdu
        chips.append(_bytes_to_chips(packet_bytes))

    return np.concatenate(chips)[:num_chips].astype(np.int8)


def _oqpsk_half_sine(chips: np.ndarray, osr: int) -> np.ndarray:
    """Modulates a chip stream as O-QPSK with half-sine pulse shaping.

    Each rail's half-sine pulse spans two chip periods (2*osr samples) and the
    Q rail is delayed by one full chip period (osr samples), per IEEE 802.15.4.
    Output length is exactly len(chips) * osr.

    Args:
        chips: Binary chip stream, values in {0, 1}.
        osr: Samples per chip.

    Returns:
        np.ndarray: Complex baseband IQ.
    """
    bipolar = 1.0 - 2.0 * chips.astype(np.float32)  # {0,1} → {+1,-1}

    # Half-sine pulse
    samples_per_rail_symbol = 2 * osr
    t = np.arange(samples_per_rail_symbol)
    half_sine = np.sin(np.pi * t / samples_per_rail_symbol).astype(np.float32)

    i_chips = bipolar[0::2]
    q_chips = bipolar[1::2]
    n = len(i_chips)
    out_len = n * samples_per_rail_symbol  # ie, len(chips) * osr

    # Build upsampled (zero-stuffed) and convolve
    i_up = np.zeros(out_len, dtype=np.float32)
    i_up[::samples_per_rail_symbol] = i_chips
    q_up = np.zeros(out_len, dtype=np.float32)
    q_up[::samples_per_rail_symbol] = q_chips

    i_samples = np.convolve(i_up, half_sine)[:out_len]
    q_samples = np.convolve(q_up, half_sine)[:out_len]

    iq = np.zeros(out_len, dtype=TorchSigComplexDataType)
    iq.real[:] = i_samples
    iq.imag[osr:] = q_samples[: out_len - osr]  # Q lags I by one chip period
    return iq


def zigbee_modulator_baseband(
    max_num_samples: int,
    oversampling_rate_nominal: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """802.15.4 O-QPSK modulator at complex baseband.

    Args:
        max_num_samples: Maximum output samples.
        oversampling_rate_nominal: Samples per chip at baseband.
        rng: Random number generator.

    Returns:
        np.ndarray: Baseband ZigBee signal, exactly max_num_samples long.

    Raises:
        ValueError: If max_num_samples or oversampling_rate_nominal are not positive.
    """
    if max_num_samples <= 0:
        raise ValueError("max_num_samples must be positive")
    if oversampling_rate_nominal <= 0:
        raise ValueError("oversampling_rate_nominal must be positive")

    if rng is None:
        rng = np.random.default_rng()

    num_chips = max(2, int(np.ceil(max_num_samples / oversampling_rate_nominal)))
    # Need even number of chips so both I and Q rails are populated
    if num_chips % 2:
        num_chips += 1

    chips = build_zigbee_chip_stream(num_chips, rng)
    iq = _oqpsk_half_sine(chips, oversampling_rate_nominal)

    if len(iq) < max_num_samples - oversampling_rate_nominal:
        raise ValueError(f"modulator produced {len(iq)} samples, expected >= {max_num_samples}")

    if len(iq) > max_num_samples:
        iq = slice_tail_to_length(iq, max_num_samples)
    elif len(iq) < max_num_samples:
        iq = pad_head_tail_to_length(iq, max_num_samples)

    return iq.astype(TorchSigComplexDataType)


def zigbee_modulator(
    bandwidth: float,
    sample_rate: float,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """ZigBee modulator: builds chip stream and resamples to target bandwidth.

    Note: the chip rate maps to bandwidth here, but the occupied null-to-null
        bandwidth is about 1.5x that.

    Args:
        bandwidth: Desired signal bandwidth (Hz).
        sample_rate: Capture sampling rate (Hz).
        num_samples: Number of IQ samples to produce.
        rng: Random number generator.

    Returns:
        np.ndarray: ZigBee IQ at the requested bandwidth, length num_samples.

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

    baseband = zigbee_modulator_baseband(max_num_samples, oversampling_rate_nominal, rng)
    correct_bw = multistage_polyphase_resampler(baseband, resample_rate_ideal)
    correct_bw *= 1 / resample_rate_ideal

    correct_bw = slice_head_tail_to_length(correct_bw, num_samples) if len(correct_bw) > num_samples else pad_head_tail_to_length(correct_bw, num_samples)
    return correct_bw.astype(TorchSigComplexDataType)


class ZigBeeSignalGenerator(BaseSignalGenerator):
    """IEEE 802.15.4 / ZigBee Signal Generator.

    Builds structured 802.15.4 packets: 4-byte preamble, SFD 0xA7, PHR,
    random PSDU; DSSS-spread and O-QPSK modulated with half-sine pulse shaping.
    """

    def __init__(self, **kwargs: dict[str, str | float | int]) -> None:
        """Initializes the ZigBee Signal Generator.

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
        self.set_default_class_name("zigbee")

    def generate(self) -> Signal:
        """Generates a ZigBee signal.

        Returns:
            Signal: Generated ZigBee signal with metadata.
        """
        sample_rate = self["sample_rate"]
        num_iq_samples_signal = self.random_generator.integers(
            low=self["signal_duration_in_samples_min"],
            high=self["signal_duration_in_samples_max"] + 1,
        )
        bandwidth = self.random_generator.integers(low=self["bandwidth_min"], high=self["bandwidth_max"] + 1)
        signal_data = zigbee_modulator(bandwidth, sample_rate, num_iq_samples_signal, self.random_generator)
        return Signal(data=signal_data, center_freq=0, bandwidth=bandwidth)
