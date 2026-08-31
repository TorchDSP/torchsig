"""Bluetooth Low Energy (BLE) Signal Builder.

BLE uses GFSK (BT=0.5, h=0.5) at 1 Msymbol/s. The distinctive structural
feature is a fixed 4-byte Access Address — 0x8E89BED6 for advertising — that
follows a short preamble, making BLE packets immediately identifiable by their
autocorrelation with the known access address sequence.

Physical layer (Bluetooth Core Spec 5.x, Vol 6, Part B):
    * Modulation   : GFSK, BT = 0.5, modulation index h = 0.5
    * Symbol rate  : 1 Msymbol/s
    * Occupied BW  : ~1 MHz

Advertising packet structure:
    8-bit preamble (0xAA) | 32-bit access address | PDU (2–39 bytes) | 24-bit CRC

Toy simplifications:
    * Access address fixed to the advertising channel value (0x8E89BED6).
    * PDU type, payload, and CRC are random bits.
    * Packets are concatenated without inter-frame gaps.
"""

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
)

# Advertising channel access address (Bluetooth Core Spec 5.x, §2.1.2)
BTLE_ACCESS_ADDRESS: int = 0x8E89BED6

_PREAMBLE_BITS: np.ndarray = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64)  # 0xAA, MSB first

_AA_BITS: np.ndarray = np.unpackbits(np.array([0x8E, 0x89, 0xBE, 0xD6], dtype=np.uint8)).astype(np.float64)

_BLE_GFSK_BT: float = 0.5
_BLE_MOD_INDEX: float = 0.5


def _gaussian_pulse(samples_per_symbol: int, bt: float) -> np.ndarray:
    """Gaussian frequency pulse for GFSK."""
    m = 2
    n = np.arange(-m * samples_per_symbol, m * samples_per_symbol + 1)
    p = np.exp(-2 * np.pi**2 * bt**2 / np.log(2) * (n / samples_per_symbol) ** 2)
    return p / np.sum(p)


def build_btle_bit_stream(num_bits: int, rng: np.random.Generator) -> np.ndarray:
    """Builds a stream of BLE advertising packets as bipolar symbols.

    Args:
        num_bits: Total bipolar symbols to produce.
        rng: Random number generator.

    Returns:
        np.ndarray: Bipolar {-1, +1} stream, length num_bits.

    Raises:
        ValueError: If num_bits is not positive.
    """
    if num_bits <= 0:
        raise ValueError("num_bits must be positive")

    bits = []
    while sum(len(b) for b in bits) < num_bits:
        # PDU: 2-byte header + random 0–37 byte payload
        pdu_payload_bytes = int(rng.integers(0, 38))
        pdu_bytes = 2 + pdu_payload_bytes
        pdu_bits = np.unpackbits(rng.integers(0, 256, pdu_bytes, dtype=np.uint8)).astype(np.float64)
        crc_bits = np.unpackbits(rng.integers(0, 256, 3, dtype=np.uint8)).astype(np.float64)
        packet = np.concatenate([_PREAMBLE_BITS, _AA_BITS, pdu_bits, crc_bits])
        bits.append(packet)

    stream = np.concatenate(bits)[:num_bits]
    return 2.0 * stream - 1.0  # {0,1} → {-1,+1}


def btle_modulator_baseband(
    max_num_samples: int,
    oversampling_rate_nominal: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """BLE GFSK modulator at complex baseband.

    Args:
        max_num_samples: Maximum output samples.
        oversampling_rate_nominal: Samples per bit at baseband.
        rng: Random number generator.

    Returns:
        np.ndarray: Baseband BLE signal, exactly max_num_samples long.

    Raises:
        ValueError: If max_num_samples or oversampling_rate_nominal are not positive.
    """
    if max_num_samples <= 0:
        raise ValueError("max_num_samples must be positive")
    if oversampling_rate_nominal <= 0:
        raise ValueError("oversampling_rate_nominal must be positive")

    if rng is None:
        rng = np.random.default_rng()

    sps = oversampling_rate_nominal
    rect = np.ones(sps)
    gauss = _gaussian_pulse(sps, _BLE_GFSK_BT)
    pulse_shape = sp.convolve(gauss, rect)

    max_minus_filter = max_num_samples - len(pulse_shape) + 1
    num_bits = max(1, int(np.floor(max_minus_filter / sps)))

    symbols = build_btle_bit_stream(num_bits, rng)
    freq = sp.upfirdn(pulse_shape, symbols, up=sps, down=1)
    phase = np.cumsum(freq) * (np.pi * _BLE_MOD_INDEX / np.sum(rect))
    modulated = np.exp(1j * phase)

    if len(modulated) > max_num_samples:
        modulated = slice_tail_to_length(modulated, max_num_samples)
    elif len(modulated) < max_num_samples:
        modulated = pad_head_tail_to_length(modulated, max_num_samples)

    return modulated


def btle_modulator(
    bandwidth: float,
    sample_rate: float,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """BLE GFSK modulator: builds packet stream and resamples to target bandwidth.

    Args:
        bandwidth: Desired signal bandwidth (Hz).
        sample_rate: Capture sampling rate (Hz).
        num_samples: Number of IQ samples to produce.
        rng: Random number generator.

    Returns:
        np.ndarray: BLE IQ at the requested bandwidth, length num_samples.

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

    baseband = btle_modulator_baseband(max_num_samples, oversampling_rate_nominal, rng)
    correct_bw = multistage_polyphase_resampler(baseband, resample_rate_ideal)
    correct_bw *= 1 / resample_rate_ideal

    correct_bw = slice_head_tail_to_length(correct_bw, num_samples) if len(correct_bw) > num_samples else pad_head_tail_to_length(correct_bw, num_samples)
    return correct_bw.astype(TorchSigComplexDataType)


class BTLESignalGenerator(BaseSignalGenerator):
    """Bluetooth Low Energy Signal Generator.

    Builds structured BLE advertising packets: 0xAA preamble, 0x8E89BED6
    access address, random PDU, random CRC; GFSK modulated at BT=0.5, h=0.5.
    """

    def __init__(self, **kwargs: dict[str, str | float | int]) -> None:
        """Initializes the BLE Signal Generator.

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
        self.set_default_class_name("btle")

    def generate(self) -> Signal:
        """Generates a BLE signal.

        Returns:
            Signal: Generated BLE signal with metadata.
        """
        sample_rate = self["sample_rate"]
        num_iq_samples_signal = self.random_generator.integers(
            low=self["signal_duration_in_samples_min"],
            high=self["signal_duration_in_samples_max"] + 1,
        )
        bandwidth = self.random_generator.integers(low=self["bandwidth_min"], high=self["bandwidth_max"] + 1)
        signal_data = btle_modulator(bandwidth, sample_rate, num_iq_samples_signal, self.random_generator)
        return Signal(data=signal_data, center_freq=0, bandwidth=bandwidth)
