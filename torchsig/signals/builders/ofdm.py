"""OFDM Signal Builder and Modulator Module"""

from __future__ import annotations

import numpy as np

from torchsig.signals.builder import BaseSignalGenerator
from torchsig.signals.builders.constellation_maps import all_symbol_maps
from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import (
    TorchSigComplexDataType,
    multistage_polyphase_resampler,
    pad_head_tail_to_length,
    slice_head_tail_to_length,
    slice_tail_to_length,
)


def ofdm_modulator_baseband(
    num_subcarriers: int,
    max_num_samples: int,
    oversampling_rate_nominal: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Modulates OFDM signal at baseband.

    Args:
        num_subcarriers: Number of subcarriers to use.
        max_num_samples: Maximum number of samples to produce.
        oversampling_rate_nominal: Oversampling rate (sampling_rate/bandwidth).
        rng: Random number generator for reproducibility. If None, creates a new default generator.

    Returns:
        np.ndarray: OFDM modulated signal at baseband.

    Raises:
        ValueError: If num_subcarriers, max_num_samples, or oversampling_rate_nominal are not positive.
    """
    # Input validation
    if num_subcarriers <= 0:
        raise ValueError("num_subcarriers must be positive")
    if max_num_samples <= 0:
        raise ValueError("max_num_samples must be positive")
    if oversampling_rate_nominal <= 0:
        raise ValueError("oversampling_rate_nominal must be positive")

    if rng is None:
        rng = np.random.default_rng()

    # Define oversampling rate for OFDM signal
    oversampling_rate_nominal = 4
    ifft_size = int(oversampling_rate_nominal * num_subcarriers)

    # Randomize cyclic prefix
    cyclic_prefix_probability = 0.50
    cp_len = (
        0
        if rng.uniform(0, 1) < cyclic_prefix_probability
        else rng.integers(2, int(num_subcarriers / 2))
    )
    cp_len_oversampled = cp_len * oversampling_rate_nominal

    # Calculate OFDM symbol lengths
    ofdm_symbol_length = num_subcarriers + cp_len
    ofdm_symbol_length_oversampled = ofdm_symbol_length * oversampling_rate_nominal

    # Determine number of OFDM symbols
    num_ofdm_symbols = int(np.ceil(max_num_samples / ofdm_symbol_length_oversampled))

    # Randomize subcarrier modulation
    potential_subcarrier_modulations = TorchSigSignalLists.ofdm_subcarrier_modulations
    random_index = rng.integers(0, len(potential_subcarrier_modulations))
    constellation_name = potential_subcarrier_modulations[random_index]

    # Get and normalize symbol map
    symbol_map = all_symbol_maps[constellation_name]
    symbol_map = symbol_map / np.sqrt(np.mean(np.abs(symbol_map) ** 2))

    # Generate symbols for active subcarriers
    map_index_grid = rng.integers(
        0, len(symbol_map), (num_subcarriers, num_ofdm_symbols)
    )
    symbol_grid = symbol_map[map_index_grid]

    # Create time/frequency grid
    time_frequency_grid = np.zeros(
        (ifft_size, num_ofdm_symbols), dtype=TorchSigComplexDataType
    )
    half_num_subcarriers = int(num_subcarriers / 2)
    time_frequency_grid[1 : half_num_subcarriers + 1, :] = symbol_grid[
        0:half_num_subcarriers, :
    ]
    time_frequency_grid[ifft_size - half_num_subcarriers :, :] = symbol_grid[
        half_num_subcarriers:, :
    ]

    # Perform IFFT
    modulated_grid = np.fft.ifft(time_frequency_grid, axis=0)

    # Add cyclic prefix
    cp_grid = modulated_grid[ifft_size - cp_len_oversampled :, :]
    modulated_with_cp_grid = np.concatenate((cp_grid, modulated_grid), axis=0)

    # Serialize time series
    ofdm_signal = np.ravel(np.transpose(modulated_with_cp_grid))

    # Enforce proper length
    return slice_tail_to_length(ofdm_signal, max_num_samples)


def ofdm_modulator(
    num_subcarriers: int,
    bandwidth: float,
    sample_rate: float,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Modulator for OFDM signals.

    Args:
        num_subcarriers: Number of subcarriers to use.
        bandwidth: Desired 3 dB bandwidth of the signal (Hz).
        sample_rate: Sampling rate for the IQ signal (Hz).
        num_samples: Number of IQ samples to produce.
        rng: Random number generator for reproducibility. If None, creates a new default generator.

    Returns:
        np.ndarray: OFDM modulated signal at the appropriate bandwidth.

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

    # Calculate resampling parameters
    oversampling_rate = sample_rate / bandwidth
    oversampling_rate_baseband = 4
    resample_rate_ideal = oversampling_rate / oversampling_rate_baseband

    # Calculate baseband samples
    num_samples_baseband = int(np.ceil(num_samples / resample_rate_ideal))

    # Generate and resample signal
    ofdm_signal_baseband = ofdm_modulator_baseband(
        num_subcarriers, num_samples_baseband, oversampling_rate_baseband, rng
    )

    ofdm_signal_correct_bw = multistage_polyphase_resampler(
        ofdm_signal_baseband, resample_rate_ideal
    )

    # Adjust signal length
    ofdm_signal_correct_bw = (
        slice_head_tail_to_length(ofdm_signal_correct_bw, num_samples)
        if len(ofdm_signal_correct_bw) > num_samples
        else pad_head_tail_to_length(ofdm_signal_correct_bw, num_samples)
    )

    return ofdm_signal_correct_bw.astype(TorchSigComplexDataType)


class OFDMSignalGenerator(BaseSignalGenerator):
    """OFDM Signal Generator.

    Implements OFDM waveforms with configurable parameters.
    """

    def __init__(self, **kwargs: dict[str, str | float | int]) -> None:
        """Initializes OFDM Signal Generator.

        Args:
            **kwargs: Metadata parameters including:
                - sample_rate: Sampling rate (Hz)
                - bandwidth_min: Minimum bandwidth (Hz)
                - bandwidth_max: Maximum bandwidth (Hz)
                - num_subcarriers: Number of subcarriers
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
            "num_subcarriers",
            "signal_duration_in_samples_min",
            "signal_duration_in_samples_max",
        ]
        self.set_default_class_name(f"ofdm-{self['num_subcarriers']}")

    def generate(self) -> Signal:
        """Generates an OFDM signal based on the configured parameters.

        Returns:
            Signal: Generated OFDM signal with metadata.

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
        num_subcarriers = self["num_subcarriers"]

        # Generate signal
        signal_data = ofdm_modulator(
            num_subcarriers,
            bandwidth,
            sample_rate,
            num_iq_samples_signal,
            self.random_generator,
        )

        return Signal(data=signal_data, center_freq=0, bandwidth=bandwidth)
