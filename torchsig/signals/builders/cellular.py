"""Cellular Signal Builders."""

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

# --------------------------------------------------------------------------- #
# GSM (Global System for Mobile)
# --------------------------------------------------------------------------- #
"""GSM is a 2G cellular standard using GMSK modulation with a TDMA frame
structure. The distinctive feature is a known 26-bit Training Sequence Code
(TSC) embedded in the middle of each burst, giving the waveform a periodic
autocorrelation peak that differentiates it from generic GMSK.

Physical layer (ETSI TS 45.001/45.002):
    * Modulation    : GMSK, BT = 0.3, modulation index h = 0.5
    * Symbol rate   : 270 833.333 Bd (13/48 MHz)
    * Channel BW    : 200 kHz
    * Burst length  : 156.25 bits (148 active + 8.25 guard)

Normal burst layout (148 active bits):
    3 tail | 57 data | 1 steal | 26 TSC | 1 steal | 57 data | 3 tail

Toy simplifications:
    * No encryption, FEC, or frequency-hopping; data bits are random.
    * Guard bits are modeled as zeros (off-air silence).
    * The burst is repeated continuously to fill the requested duration.
"""

# Eight standard Training Sequence Codes (ETSI TS 45.002, Table 5.2.3a)
GSM_TSC: tuple[tuple[int, ...], ...] = (
    (0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1),
    (0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1),
    (0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0),
    (0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0),
    (0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1),
    (0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0),
    (1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1),
    (1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0),
)

_GSM_TAIL: np.ndarray = np.zeros(3, dtype=np.float64)
_GSM_GUARD: np.ndarray = np.zeros(9, dtype=np.float64)  # 8.25 guard → 9 zeros


def build_gsm_burst(tsc_idx: int, rng: np.random.Generator) -> np.ndarray:
    """Builds one 157-bit GSM Normal Burst (148 active + 9 guard bits).

    Args:
        tsc_idx: Index 0–7 selecting the Training Sequence Code.
        rng: Random number generator.

    Returns:
        np.ndarray: Bipolar {-1, +1} bit stream, length 157.
    """
    data_pre = rng.integers(0, 2, 57).astype(np.float64)
    data_post = rng.integers(0, 2, 57).astype(np.float64)
    steal_pre = rng.integers(0, 2, 1).astype(np.float64)
    steal_post = rng.integers(0, 2, 1).astype(np.float64)
    tsc = np.array(GSM_TSC[tsc_idx], dtype=np.float64)

    bits = np.concatenate([_GSM_TAIL, data_pre, steal_pre, tsc, steal_post, data_post, _GSM_TAIL, _GSM_GUARD])
    return 2.0 * bits - 1.0  # {0,1} → {-1,+1}


def build_gsm_bit_stream(num_bits: int, rng: np.random.Generator) -> np.ndarray:
    """Builds a continuous stream of GSM Normal Bursts.

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

    burst_len = 157  # 148 active + 9 guard
    num_bursts = int(np.ceil(num_bits / burst_len))
    bursts = [build_gsm_burst(int(rng.integers(0, 8)), rng) for _ in range(num_bursts)]
    return np.concatenate(bursts)[:num_bits]


def _gaussian_pulse(samples_per_symbol: int, bt: float) -> np.ndarray:
    """Gaussian frequency pulse for GMSK (BT-product pulse shaping)."""
    m = 3
    n = np.arange(-m * samples_per_symbol, m * samples_per_symbol + 1)
    p = np.exp(-2 * np.pi**2 * bt**2 / np.log(2) * (n / samples_per_symbol) ** 2)
    return p / np.sum(p)


def gsm_modulator_baseband(
    max_num_samples: int,
    oversampling_rate_nominal: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """GSM GMSK modulator at complex baseband.

    Args:
        max_num_samples: Maximum output samples.
        oversampling_rate_nominal: Samples per bit at baseband.
        rng: Random number generator.

    Returns:
        np.ndarray: Baseband GSM signal, exactly max_num_samples long.

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
    gauss = _gaussian_pulse(sps, bt=0.3)
    pulse_shape = sp.convolve(gauss, rect)

    max_minus_filter = max_num_samples - len(pulse_shape) + 1
    num_bits = max(1, int(np.floor(max_minus_filter / sps)))

    symbols = build_gsm_bit_stream(num_bits, rng)
    freq = sp.upfirdn(pulse_shape, symbols, up=sps, down=1)
    # h=0.5 MSK: scale so one symbol-period of rectangular freq = pi/2 phase
    phase = np.cumsum(freq) * (np.pi * 0.5 / np.sum(rect))
    modulated = np.exp(1j * phase)

    if len(modulated) > max_num_samples:
        modulated = slice_tail_to_length(modulated, max_num_samples)
    elif len(modulated) < max_num_samples:
        modulated = pad_head_tail_to_length(modulated, max_num_samples)

    return modulated


def gsm_modulator(
    bandwidth: float,
    sample_rate: float,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """GSM GMSK modulator: builds burst stream and resamples to target bandwidth.

    Args:
        bandwidth: Desired signal bandwidth (Hz).
        sample_rate: Capture sampling rate (Hz).
        num_samples: Number of IQ samples to produce.
        rng: Random number generator.

    Returns:
        np.ndarray: GSM IQ at the requested bandwidth, length num_samples.

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

    baseband = gsm_modulator_baseband(max_num_samples, oversampling_rate_nominal, rng)
    correct_bw = multistage_polyphase_resampler(baseband, resample_rate_ideal)
    correct_bw *= 1 / resample_rate_ideal

    correct_bw = slice_head_tail_to_length(correct_bw, num_samples) if len(correct_bw) > num_samples else pad_head_tail_to_length(correct_bw, num_samples)
    return correct_bw.astype(TorchSigComplexDataType)


class GSMSignalGenerator(BaseSignalGenerator):
    """GSM Signal Generator.

    Builds a structured GSM GMSK waveform: Normal Bursts with standard
    Training Sequence Codes interleaved with random data bits.
    """

    def __init__(self, **kwargs: dict[str, str | float | int]) -> None:
        """Initializes the GSM Signal Generator.

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
        self.set_default_class_name("gsm")

    def generate(self) -> Signal:
        """Generates a GSM signal.

        Returns:
            Signal: Generated GSM signal with metadata.
        """
        sample_rate = self["sample_rate"]
        num_iq_samples_signal = self.random_generator.integers(
            low=self["signal_duration_in_samples_min"],
            high=self["signal_duration_in_samples_max"] + 1,
        )
        bandwidth = self.random_generator.integers(low=self["bandwidth_min"], high=self["bandwidth_max"] + 1)
        signal_data = gsm_modulator(bandwidth, sample_rate, num_iq_samples_signal, self.random_generator)
        return Signal(data=signal_data, center_freq=0, bandwidth=bandwidth)
