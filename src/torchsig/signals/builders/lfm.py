"""LFM Signal Builder and Modulator Module"""

from __future__ import annotations

from collections import OrderedDict

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


def get_symbol_map() -> OrderedDict[str, np.ndarray]:
    """Generates symbol maps for LFM signals.

    Returns:
        OrderedDict[str, np.ndarray]: Dictionary containing symbol maps for different LFM types.
    """
    return OrderedDict(
        {
            "data": np.array([-1.0, 1.0]),
            "radar": np.array([1.0]),
        }
    )


def lfm_modulator_baseband(
    lfm_type: str,
    max_num_samples: int,
    oversampling_rate_nominal: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """LFM modulator at baseband.

    Args:
        lfm_type: Type of LFM signal ('data' or 'radar').
        max_num_samples: Maximum number of samples to produce.
        oversampling_rate_nominal: Oversampling rate (sampling_rate/bandwidth).
        rng: Random number generator for reproducibility. If None, creates a new default generator.

    Returns:
        np.ndarray: LFM modulated signal at baseband.

    Raises:
        ValueError: If max_num_samples or oversampling_rate_nominal are not positive.
        ValueError: If lfm_type is not one of the supported types.
    """
    # Input validation
    if max_num_samples <= 0:
        raise ValueError("max_num_samples must be positive")
    if oversampling_rate_nominal <= 0:
        raise ValueError("oversampling_rate_nominal must be positive")

    if rng is None:
        rng = np.random.default_rng()

    # Modulator has implied sampling rate of 1.0
    sample_rate = 1.0
    bandwidth = sample_rate / oversampling_rate_nominal

    # Calculate chirp bounds
    f0 = -bandwidth / 2
    f1 = bandwidth / 2

    # Get symbol map
    symbol_map = get_symbol_map()
    if lfm_type not in symbol_map:
        raise ValueError(
            f"Unsupported LFM type: {lfm_type}. Must be one of: {list(symbol_map.keys())}"
        )

    const = symbol_map[lfm_type]

    # Randomize number of samples per symbol
    samples_per_symbol = rng.integers(low=128, high=4096)

    # Generate symbols
    symbol_nums = rng.integers(
        0, len(const), int(np.ceil(max_num_samples / samples_per_symbol))
    )
    symbols = const[symbol_nums]

    # Create chirp templates
    upchirp = chirp(f0, f1, samples_per_symbol)
    downchirp = chirp(f1, f0, samples_per_symbol)

    # Pre-allocate memory for output
    modulated = np.zeros((max_num_samples,), dtype=TorchSigComplexDataType)

    # Generate modulated signal
    sym_start_index = 0
    for s in symbols:
        # Calculate time indexing
        time_index = np.arange(sym_start_index, sym_start_index + samples_per_symbol)

        # Handle case where output exceeds available space
        if time_index[-1] >= len(modulated):
            time_index = time_index[np.where(time_index < len(modulated))[0]]

        # Store symbols
        if s > 0:
            modulated[time_index] = upchirp[0 : len(time_index)]
        else:
            modulated[time_index] = downchirp[0 : len(time_index)]

        sym_start_index += samples_per_symbol

    # Randomly apply bandwidth-limiting filter (50% chance)
    bandwidth_limiting_filter_probability = 0.50
    if rng.uniform(0, 1) < bandwidth_limiting_filter_probability:
        filter_taps = random_limiting_filter_design(bandwidth, sample_rate, rng)
        modulated = convolve(modulated, filter_taps)

    return modulated


def lfm_modulator(
    lfm_type: str,
    bandwidth: float,
    sample_rate: float,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """LFM modulator.

    Args:
        lfm_type: Type of LFM signal ('data' or 'radar').
        bandwidth: Desired 3 dB bandwidth of the signal (Hz).
        sample_rate: Sampling rate for the IQ signal (Hz).
        num_samples: Number of IQ samples to produce.
        rng: Random number generator for reproducibility. If None, creates a new default generator.

    Returns:
        np.ndarray: LFM modulated signal at the appropriate bandwidth.

    Raises:
        ValueError: If bandwidth or sample_rate are not positive.
        ValueError: If bandwidth exceeds sample_rate/2.
        ValueError: If num_samples is not positive.
        ValueError: If lfm_type is not supported.
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
    num_samples_baseband = max(1, int(np.ceil(num_samples / resample_rate_ideal)))

    # Generate and resample signal
    lfm_signal_baseband = lfm_modulator_baseband(
        lfm_type, num_samples_baseband, oversampling_rate_nominal, rng
    )

    lfm_mod_correct_bw = multistage_polyphase_resampler(
        lfm_signal_baseband, resample_rate_ideal
    )

    # Adjust signal length
    lfm_mod_correct_bw = (
        slice_head_tail_to_length(lfm_mod_correct_bw, num_samples)
        if len(lfm_mod_correct_bw) > num_samples
        else pad_head_tail_to_length(lfm_mod_correct_bw, num_samples)
    )

    return lfm_mod_correct_bw.astype(TorchSigComplexDataType)


class LFMSignalGenerator(BaseSignalGenerator):
    """LFM Signal Generator.

    Implements linear frequency modulation (LFM) waveforms with configurable parameters.
    """

    def __init__(self, **kwargs: dict[str, str | float | int]) -> None:
        """Initializes LFM Signal Generator.

        Args:
            **kwargs: Metadata parameters including:
                - sample_rate: Sampling rate (Hz)
                - bandwidth_min: Minimum bandwidth (Hz)
                - bandwidth_max: Maximum bandwidth (Hz)
                - lfm_type: Type of LFM ('data' or 'radar')
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
            "lfm_type",
            "signal_duration_in_samples_min",
            "signal_duration_in_samples_max",
        ]
        self.set_default_class_name(f"lfm-{self['lfm_type']}")

    def generate(self) -> Signal:
        """Generates an LFM signal based on the configured parameters.

        Returns:
            Signal: Generated LFM signal with metadata.

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
        lfm_type = self["lfm_type"]

        # Generate signal
        signal_data = lfm_modulator(
            lfm_type,
            bandwidth,
            sample_rate,
            num_iq_samples_signal,
            self.random_generator,
        )

        return Signal(data=signal_data, center_freq=0, bandwidth=bandwidth)
