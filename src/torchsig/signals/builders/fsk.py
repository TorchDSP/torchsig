"""Frequency Shift Keying (FSK) and related Signal Builder and Modulator Module"""

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


def get_fsk_freq_map(n: int) -> np.ndarray:
    """Generates frequency symbol maps for FSK and MSK variants.

    Args:
        n: Number of frequency points in the constellation.

    Returns:
        np.ndarray: Array of frequency points for the constellation.
    """
    return np.linspace(-1 + (1 / n), 1 - (1 / n), n, endpoint=True)


def get_fsk_mod_index(fsk_type: str, rng: np.random.Generator | None = None) -> float:
    """Determines the modulation index for different FSK variants.

    The modulation index is derived from the symbol spacing in the frequency domain.
    Orthogonal FSK has a modulation index of 1.0, and MSK/GMSK have 0.5.

    Args:
        fsk_type: Type of FSK modulation ('fsk', 'gfsk', 'msk', 'gmsk').
        rng: Random number generator for reproducibility. If None, creates a new default generator.

    Returns:
        float: Modulation index.

    Raises:
        ValueError: If fsk_type is not one of the supported types.
    """
    if rng is None:
        rng = np.random.default_rng()

    if fsk_type == "gfsk":
        # Bluetooth GFSK
        return rng.uniform(0.1, 0.5)
    if fsk_type in ("msk", "gmsk"):
        # MSK and GMSK
        return 0.5
    if fsk_type == "fsk":
        # FSK - 50% chance for orthogonal (mod_idx=1) or non-orthogonal
        orthogonal_probability = 0.50
        return (
            1.0
            if rng.uniform(0, 1) < orthogonal_probability
            else rng.uniform(0.7, 1.01)
        )
    raise ValueError(f"Unexpected fsk_type: {fsk_type}")


def gaussian_taps(
    samples_per_symbol: int, bt: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Designs a Gaussian pulse shape for GMSK and GFSK.

    Args:
        samples_per_symbol: Number of samples per symbol.
        bt: Time-bandwidth product (0.0 to 1.0).
        rng: Random number generator for reproducibility. If None, creates a new default generator.

    Returns:
        np.ndarray: Filter weights for the Gaussian pulse shape.

    Raises:
        ValueError: If bt is not in the valid range (0.0 to 1.0).
    """
    if rng is None:
        rng = np.random.default_rng()

    if not 0.0 <= bt <= 1.0:
        raise ValueError("bt must be between 0.0 and 1.0")

    m = rng.integers(1, 5)  # Randomize filter span
    n = np.arange(-m * samples_per_symbol, m * samples_per_symbol + 1)
    p = np.exp(-2 * np.pi**2 * bt**2 / np.log(2) * (n / float(samples_per_symbol)) ** 2)
    return p / np.sum(p)


def fsk_modulator_baseband(
    constellation_size: int,
    fsk_type: str,
    max_num_samples: int,
    oversampling_rate_nominal: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """FSK modulator at baseband.

    Args:
        constellation_size: Number of points in the constellation.
        fsk_type: Type of FSK modulation ('fsk', 'gfsk', 'msk', 'gmsk').
        max_num_samples: Maximum number of samples to produce.
        oversampling_rate_nominal: Oversampling rate (sampling_rate/bandwidth).
        rng: Random number generator for reproducibility. If None, creates a new default generator.

    Returns:
        np.ndarray: FSK modulated signal at baseband.

    Raises:
        ValueError: If max_num_samples or oversampling_rate_nominal are not positive.
    """
    # Input validation
    if max_num_samples <= 0:
        raise ValueError("max_num_samples must be positive")
    if oversampling_rate_nominal <= 0:
        raise ValueError("oversampling_rate_nominal must be positive")

    if rng is None:
        rng = np.random.default_rng()

    # Determine modulation index
    mod_idx = get_fsk_mod_index(fsk_type, rng)

    # Get FSK frequency symbol map
    const = get_fsk_freq_map(constellation_size)
    const_oversampled = const / oversampling_rate_nominal

    # Calculate modulation order and samples per symbol
    mod_order = len(const)
    samples_per_symbol = int(mod_order * oversampling_rate_nominal)

    # Create pulse shape
    pulse_shape = np.ones(samples_per_symbol)

    if "g" in fsk_type:  # GMSK, GFSK
        bt = rng.uniform(0.1, 0.5)  # Randomize time-bandwidth product
        taps = gaussian_taps(samples_per_symbol, bt, rng)
        pulse_shape = sp.convolve(taps, pulse_shape)

    # Calculate number of symbols
    max_num_samples_minus_pulse_shape = max_num_samples - len(pulse_shape) + 1
    num_symbols = max(
        1, int(np.floor(max_num_samples_minus_pulse_shape / samples_per_symbol))
    )

    # Generate symbols
    symbol_nums = rng.integers(0, len(const_oversampled), num_symbols)
    symbols = const_oversampled[symbol_nums]

    # Apply pulse shaping and modulation
    filtered = sp.upfirdn(pulse_shape, symbols, up=samples_per_symbol, down=1)
    phase = np.cumsum(np.array(filtered) * 1j * mod_idx * np.pi)
    modulated = np.exp(phase)

    # Adjust signal length
    if len(modulated) > max_num_samples:
        modulated = slice_tail_to_length(modulated, max_num_samples)
    elif len(modulated) < max_num_samples:
        modulated = pad_head_tail_to_length(modulated, max_num_samples)

    return modulated


def fsk_modulator(
    constellation_size: int,
    fsk_type: str,
    bandwidth: float,
    sample_rate: float,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """FSK modulator.

    Args:
        constellation_size: Number of points in the constellation.
        fsk_type: Type of FSK modulation ('fsk', 'gfsk', 'msk', 'gmsk').
        bandwidth: Desired 3 dB bandwidth of the signal (Hz).
        sample_rate: Sampling rate for the IQ signal (Hz).
        num_samples: Number of IQ samples to produce.
        rng: Random number generator for reproducibility. If None, creates a new default generator.

    Returns:
        np.ndarray: FSK modulated signal at the appropriate bandwidth.

    Raises:
        ValueError: If bandwidth or sample_rate are not positive.
        ValueError: If bandwidth exceeds sample_rate/2.
        ValueError: If num_samples is not positive.
    """
    # Input validation
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
    max_num_samples = max(
        oversampling_rate_nominal, int(np.floor(num_samples / resample_rate_ideal))
    )

    # Generate and resample signal
    baseband_signal = fsk_modulator_baseband(
        constellation_size, fsk_type, max_num_samples, oversampling_rate_nominal, rng
    )

    fsk_correct_bw = multistage_polyphase_resampler(
        baseband_signal, resample_rate_ideal
    )
    fsk_correct_bw *= 1 / resample_rate_ideal

    # Adjust signal length
    fsk_correct_bw = (
        slice_head_tail_to_length(fsk_correct_bw, num_samples)
        if len(fsk_correct_bw) > num_samples
        else pad_head_tail_to_length(fsk_correct_bw, num_samples)
    )

    return fsk_correct_bw.astype(TorchSigComplexDataType)


class FSKSignalGenerator(BaseSignalGenerator):
    """FSK Signal Generator.

    Implements FSK waveforms with configurable parameters.
    """

    def __init__(self, **kwargs: dict[str, str | float | int]) -> None:
        """Initializes FSK Signal Generator.

        Args:
            **kwargs: Metadata parameters including:
                - sample_rate: Sampling rate (Hz)
                - bandwidth_min: Minimum bandwidth (Hz)
                - bandwidth_max: Maximum bandwidth (Hz)
                - fsk_type: Type of FSK ('fsk', 'gfsk', 'msk', 'gmsk')
                - constellation_size: Number of points in the constellation
                - signal_duration_in_samples_min: Minimum signal duration (samples)
                - signal_duration_in_samples_max: Maximum signal duration (samples)

        Raises:
            ValueError: If required metadata fields are missing or invalid.
        """
        super().__init__(**kwargs)
        self.required_metadata_fields = [
            "sample_rate",
            "bandwidth_min",
            "bandwidth_max",
            "fsk_type",
            "constellation_size",
            "signal_duration_in_samples_min",
            "signal_duration_in_samples_max",
        ]
        self.set_default_class_name(f"{self['constellation_size']}{self['fsk_type']}")

    def generate(self) -> Signal:
        """Generates an FSK signal based on the configured parameters.

        Returns:
            Signal: Generated FSK signal with metadata.

        Raises:
            ValueError: If required metadata fields are missing or invalid.
        """
        # Get parameters from metadata
        sample_rate = self["sample_rate"]
        num_iq_samples_signal = self.random_generator.integers(
            low=self["signal_duration_in_samples_min"],
            high=self["signal_duration_in_samples_max"] + 1,
        )
        bandwidth = self.random_generator.integers(
            low=self["bandwidth_min"], high=self["bandwidth_max"] + 1
        )
        fsk_type = self["fsk_type"]
        constellation_size = self["constellation_size"]

        # Generate signal
        signal_data = fsk_modulator(
            constellation_size,
            fsk_type,
            bandwidth,
            sample_rate,
            num_iq_samples_signal,
            self.random_generator,
        )

        return Signal(data=signal_data, center_freq=0, bandwidth=bandwidth)
