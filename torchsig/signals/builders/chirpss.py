"""Chirp Spread Spectrum Signal Generator Module"""

from __future__ import annotations

import numpy as np

from torchsig.signals.builder import BaseSignalGenerator
from torchsig.signals.builders.chirp import chirp
from torchsig.signals.signal_types import Signal
from torchsig.signals.signal_utils import random_limiting_filter_design
from torchsig.utils.dsp import (
    TorchSigComplexDataType,
    convolve,
    multistage_polyphase_resampler,
    pad_head_tail_to_length,
    slice_head_tail_to_length,
)


def get_symbol_map() -> np.ndarray:
    """Generates the symbol mapping for ChirpSS.

    Returns:
        np.ndarray: Array of 128 symbols (0 to 127).
    """
    return np.linspace(0, 2**7 - 1, 2**7)


def chirpss_modulator_baseband(
    max_num_samples: int,
    oversampling_rate_nominal: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Chirp Spread Spectrum modulator at baseband.

    Args:
        max_num_samples: Maximum number of samples to be produced.
        oversampling_rate_nominal: The amount of oversampling (sampling_rate/bandwidth).
        rng: Random number generator for reproducibility. If None, creates a new default generator.

    Returns:
        np.ndarray: Chirp SS modulated signal at baseband.

    Raises:
        ValueError: If max_num_samples or oversampling_rate_nominal are not positive.
    """
    # Input validation
    if max_num_samples <= 0:
        raise ValueError("max_num_samples must be positive")
    if oversampling_rate_nominal <= 0:
        raise ValueError("oversampling_rate_nominal must be positive")

    # Create random number generator if not provided
    if rng is None:
        rng = np.random.default_rng()

    # Modulator has implied sampling rate of 1.0
    sample_rate = 1.0
    bandwidth = sample_rate / oversampling_rate_nominal

    # Get symbol mapping
    const = get_symbol_map()

    # Randomize number of samples per symbol
    samples_per_symbol = rng.integers(low=128, high=4096)

    # Generate symbols
    symbol_nums = rng.integers(
        0, len(const), int(np.ceil(max_num_samples / samples_per_symbol))
    )
    symbols = const[symbol_nums]

    # Create chirp template
    upchirp = chirp(-bandwidth, bandwidth, samples_per_symbol)
    double_upchirp = np.concatenate((upchirp, upchirp), axis=0)

    # Pre-allocate memory for output
    modulated = np.zeros((max_num_samples,), dtype=TorchSigComplexDataType)

    # Generate modulated signal
    sym_start_index = 0
    m = const.size
    for s in symbols:
        # Calculate output time for symbol
        output_time = np.arange(sym_start_index, sym_start_index + samples_per_symbol)

        # Handle case where output exceeds available space
        if output_time[-1] >= len(modulated):
            output_time = output_time[np.where(output_time < len(modulated))[0]]

        # Calculate chirp start index
        chirp_start_index = int((s / m) * samples_per_symbol)
        input_time = np.arange(chirp_start_index, chirp_start_index + len(output_time))

        # Insert symbol into output
        modulated[output_time] = double_upchirp[input_time]
        sym_start_index += samples_per_symbol

    # Randomly apply bandwidth-limiting filter (50% chance)
    bandwidth_limit_filter_probability = 0.50
    if rng.uniform(0, 1) < bandwidth_limit_filter_probability:
        lpf = random_limiting_filter_design(bandwidth, sample_rate, rng)
        modulated = convolve(modulated, lpf)

    return modulated


def chirpss_modulator(
    bandwidth: float,
    sample_rate: float,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Chirp Spread Spectrum modulator.

    Args:
        bandwidth: Desired 3 dB bandwidth of the signal (Hz).
        sample_rate: Sampling rate for the IQ signal (Hz).
        num_samples: Number of IQ samples to produce.
        rng: Random number generator for reproducibility. If None, creates a new default generator.

    Returns:
        np.ndarray: Chirp SS modulated signal at the appropriate bandwidth.

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

    # Create random number generator if not provided
    if rng is None:
        rng = np.random.default_rng()

    # Calculate final oversampling rate
    oversampling_rate = sample_rate / bandwidth

    # Baseband modulation parameters
    oversampling_rate_baseband = 4
    resample_rate_ideal = oversampling_rate / oversampling_rate_baseband
    num_samples_baseband = int(np.floor(num_samples / resample_rate_ideal))

    # Generate baseband signal
    chirpss_signal_baseband = chirpss_modulator_baseband(
        num_samples_baseband, oversampling_rate_baseband, rng
    )

    # Apply resampling
    chirpss_mod_correct_bw = multistage_polyphase_resampler(
        chirpss_signal_baseband, resample_rate_ideal
    )

    # Adjust signal length
    if len(chirpss_mod_correct_bw) > num_samples:
        chirpss_mod_correct_bw = slice_head_tail_to_length(
            chirpss_mod_correct_bw, num_samples
        )
    else:
        chirpss_mod_correct_bw = pad_head_tail_to_length(
            chirpss_mod_correct_bw, num_samples
        )

    return chirpss_mod_correct_bw.astype(TorchSigComplexDataType)


class ChirpSSSignalGenerator(BaseSignalGenerator):
    """Chirp Spread Spectrum Signal Generator.

    Implements ChirpSS signals with configurable parameters.
    """

    def __init__(self, **kwargs: dict[str, str | float | int]) -> None:
        """Initializes ChirpSS Signal Generator.

        Args:
            **kwargs: Metadata parameters including:
                - sample_rate: Sampling rate (Hz)
                - bandwidth_min: Minimum bandwidth (Hz)
                - bandwidth_max: Maximum bandwidth (Hz)
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
            "signal_duration_in_samples_min",
            "signal_duration_in_samples_max",
        ]
        self.set_default_class_name("chirpss")

    def generate(self) -> Signal:
        """Generates a ChirpSS signal based on the configured parameters.

        Returns:
            Signal: Generated ChirpSS signal with metadata.

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

        # Generate signal
        signal_data = chirpss_modulator(
            bandwidth, sample_rate, num_iq_samples_signal, self.random_generator
        )

        return Signal(
            data=signal_data, class_name="chirpss", center_freq=0, bandwidth=bandwidth
        )
