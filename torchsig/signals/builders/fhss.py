"""Experimental frequency-hopping spread-spectrum signal generation.

This module is an in-progress prototype. Its API and waveform semantics may
change before FHSS is registered as a supported TorchSIG signal. In particular,
it does not yet expose hop timing, frequency plans, per-hop modulation, symbol
rate, occupied-bandwidth shaping, burst placement, or hop metadata. Import it
directly from :mod:`torchsig.signals.builders.fhss`; it is intentionally absent
from TorchSIG's signal lists and generator registries.
"""

from __future__ import annotations

import numpy as np

from torchsig.signals.builder import BaseSignalGenerator
from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import (
    TorchSigComplexDataType,
    frequency_hopping_offsets,
    time_varying_frequency_shift,
)

__all__ = ["FrequencyHoppingSignalGenerator", "frequency_hopping_modulator"]


def _frequency_hopping_tone(
    num_samples: int,
    sample_rate: float,
    frequency_span: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a hopped unit tone and the frequency offsets used."""
    frequency_offsets = frequency_hopping_offsets(
        num_samples,
        frequency_span,
        rng,
    )
    tone = np.ones(num_samples, dtype=TorchSigComplexDataType)
    return time_varying_frequency_shift(tone, frequency_offsets, sample_rate), frequency_offsets


def frequency_hopping_modulator(
    num_samples: int,
    sample_rate: float,
    bandwidth: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate an experimental phase-continuous frequency-hopping waveform.

    The number of equal-duration hops and every hop frequency are selected at
    random. This intentionally provides a prototype FHSS waveform rather than
    a configurable protocol implementation.

    Args:
        num_samples: Number of IQ samples to generate.
        sample_rate: IQ sample rate in Hz.
        bandwidth: Full frequency span in which hop frequencies are selected.
        rng: Random number generator used for reproducible generation.

    Returns:
        Complex unit-magnitude IQ samples.

    Raises:
        ValueError: If a numeric input is outside its valid range.

    Warning:
        This experimental API has no compatibility guarantee until the FHSS
        generator is completed and registered as a supported signal.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    if bandwidth > sample_rate:
        raise ValueError("bandwidth must not exceed sample_rate")

    if rng is None:
        rng = np.random.default_rng()

    waveform, _ = _frequency_hopping_tone(
        num_samples,
        sample_rate,
        bandwidth,
        rng,
    )
    return waveform


class FrequencyHoppingSignalGenerator(BaseSignalGenerator):
    """Generate an experimental FHSS signal with random hopping behavior.

    Warning:
        This generator is an in-progress prototype. It is intentionally absent
        from TorchSIG's signal lists, generator registries, and configuration
        choices, and its API may change before public graduation.
    """

    def __init__(self, **kwargs: dict[str, str | float | int]) -> None:
        """Initialize the generator using standard dataset metadata only.

        Args:
            **kwargs: Metadata containing ``sample_rate``, bandwidth bounds,
                and signal-duration bounds.
        """
        super().__init__(**kwargs)
        self.required_metadata_fields = [
            "sample_rate",
            "bandwidth_min",
            "bandwidth_max",
            "signal_duration_in_samples_min",
            "signal_duration_in_samples_max",
        ]
        self.set_default_class_name("fhss")

    def generate(self) -> Signal:
        """Generate an experimental waveform with random hops."""
        num_samples = int(
            self.random_generator.integers(
                self["signal_duration_in_samples_min"],
                self["signal_duration_in_samples_max"] + 1,
            )
        )
        bandwidth = float(
            self.random_generator.uniform(
                self["bandwidth_min"],
                self["bandwidth_max"],
            )
        )
        data, frequency_offsets = _frequency_hopping_tone(
            num_samples,
            self["sample_rate"],
            bandwidth,
            self.random_generator,
        )
        lower_offset = float(np.min(frequency_offsets))
        upper_offset = float(np.max(frequency_offsets))
        return Signal(
            data=data,
            center_freq=(lower_offset + upper_offset) / 2,
            bandwidth=upper_offset - lower_offset,
        )
