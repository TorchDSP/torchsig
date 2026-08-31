"""ADS-B / Mode S Signal Builder.

ADS-B (Automatic Dependent Surveillance-Broadcast) transmits aircraft position
and velocity at 1090 MHz using OOK pulse-position modulation (PPM).

Physical layer (ICAO Annex 10 / RTCA DO-260B):
    * Chip rate: 2 Mchips/s (0.5 µs per chip)
    * Preamble: 4 pulses at t = 0, 1.0, 3.5, 4.5 µs
      → 16-chip pattern: [1,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0]
    * Data encoding: PPM — bit 1 = [1,0], bit 0 = [0,1] (2 chips per bit)
    * Long frame (ADS-B / Mode S Long): 112 data bits → 224 chips
    * Short frame (Mode S Short):         56 data bits → 112 chips
    * Baseband IQ: real-valued amplitude envelope

Toy simplifications:
    * No CRC-24 or ICAO message encoding; payload bits are random.
    * "Bandwidth" scales the chip rate relative to the capture sample rate.
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

# Preamble: pulses at 0, 1.0, 3.5, 4.5 µs → chips at indices 0, 2, 7, 9
ADSB_PREAMBLE_CHIPS: np.ndarray = np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)

ADSB_LONG_FRAME_BITS: int = 112  # ADS-B / Mode S Long
ADSB_SHORT_FRAME_BITS: int = 56  # Mode S Short

_PPM_ONE: np.ndarray = np.array([1.0, 0.0], dtype=np.float32)
_PPM_ZERO: np.ndarray = np.array([0.0, 1.0], dtype=np.float32)


def _bits_to_ppm_chips(bits: np.ndarray) -> np.ndarray:
    """Encodes a bit array as PPM chip pairs: bit 1 → [1,0], bit 0 → [0,1]."""
    chips = np.empty(len(bits) * 2, dtype=np.float32)
    chips[0::2] = bits
    chips[1::2] = 1 - bits
    return chips


def build_adsb_chip_stream(num_chips: int, frame_type: str, rng: np.random.Generator) -> np.ndarray:
    """Builds a chip stream of repeated ADS-B frames (preamble + PPM data).

    Args:
        num_chips: Total chips to produce.
        frame_type: 'long' (112-bit) or 'short' (56-bit).
        rng: Random number generator.

    Returns:
        np.ndarray: Float chip stream of length num_chips, values in {0, 1}.
    """
    data_bits_per_frame = ADSB_LONG_FRAME_BITS if frame_type == "long" else ADSB_SHORT_FRAME_BITS
    data_chips = _bits_to_ppm_chips(rng.integers(0, 2, data_bits_per_frame).astype(np.float32))
    frame = np.concatenate([ADSB_PREAMBLE_CHIPS, data_chips])
    frame_len = len(frame)

    num_frames = int(np.ceil(num_chips / frame_len))
    # Regenerate random data per frame for variety
    frames = [frame]
    for _ in range(num_frames - 1):
        d = _bits_to_ppm_chips(rng.integers(0, 2, data_bits_per_frame).astype(np.float32))
        frames.append(np.concatenate([ADSB_PREAMBLE_CHIPS, d]))
    return np.concatenate(frames)[:num_chips]


def adsb_modulator_baseband(
    frame_type: str,
    max_num_samples: int,
    oversampling_rate_nominal: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """ADS-B OOK modulator at complex baseband.

    Args:
        frame_type: 'long' or 'short'.
        max_num_samples: Target number of output samples.
        oversampling_rate_nominal: Samples per chip at baseband.
        rng: Random number generator.

    Returns:
        np.ndarray: Complex baseband IQ, real-valued amplitude envelope.

    Raises:
        ValueError: If max_num_samples or oversampling_rate_nominal are not positive.
    """
    if max_num_samples <= 0:
        raise ValueError("max_num_samples must be positive")
    if oversampling_rate_nominal <= 0:
        raise ValueError("oversampling_rate_nominal must be positive")

    if rng is None:
        rng = np.random.default_rng()

    num_chips = max(1, int(np.ceil(max_num_samples / oversampling_rate_nominal)))
    chips = build_adsb_chip_stream(num_chips, frame_type, rng)

    # Each chip → osr samples (rectangular pulse per chip)
    iq = np.repeat(chips, oversampling_rate_nominal).astype(TorchSigComplexDataType)

    if len(iq) > max_num_samples:
        iq = slice_tail_to_length(iq, max_num_samples)
    elif len(iq) < max_num_samples:
        iq = pad_head_tail_to_length(iq, max_num_samples)

    return iq


def adsb_modulator(
    frame_type: str,
    bandwidth: float,
    sample_rate: float,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """ADS-B modulator: builds OOK chip stream and resamples to target bandwidth.

    Args:
        frame_type: 'long' (ADS-B) or 'short' (Mode S Short).
        bandwidth: Desired signal bandwidth (Hz).
        sample_rate: Capture sampling rate (Hz).
        num_samples: Number of IQ samples to produce.
        rng: Random number generator.

    Returns:
        np.ndarray: ADS-B IQ at the requested bandwidth, length num_samples.

    Raises:
        ValueError: If bandwidth or sample_rate are not positive, bandwidth > sample_rate/2,
            num_samples is not positive, or frame_type is unknown.
    """
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if bandwidth > sample_rate / 2:
        raise ValueError("bandwidth must be less than sample_rate/2")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if frame_type not in ("long", "short"):
        raise ValueError(f"unknown frame_type '{frame_type}'; expected 'long' or 'short'")

    if rng is None:
        rng = np.random.default_rng()

    oversampling_rate_nominal = 4
    oversampling_rate = sample_rate / bandwidth
    resample_rate_ideal = oversampling_rate / oversampling_rate_nominal
    max_num_samples = max(oversampling_rate_nominal, int(np.floor(num_samples / resample_rate_ideal)))

    baseband = adsb_modulator_baseband(frame_type, max_num_samples, oversampling_rate_nominal, rng)
    correct_bw = multistage_polyphase_resampler(baseband, resample_rate_ideal)
    correct_bw *= 1 / resample_rate_ideal

    correct_bw = slice_head_tail_to_length(correct_bw, num_samples) if len(correct_bw) > num_samples else pad_head_tail_to_length(correct_bw, num_samples)
    return correct_bw.astype(TorchSigComplexDataType)


class AdsBSignalGenerator(BaseSignalGenerator):
    """ADS-B / Mode S Signal Generator.

    Builds a toy ADS-B frame stream: fixed 4-pulse preamble followed by
    PPM-encoded random payload bits. Supports long (112-bit ADS-B) and
    short (56-bit Mode S Short) frame types.
    """

    def __init__(self, **kwargs: dict[str, str | float | int]) -> None:
        """Initializes the ADS-B Signal Generator.

        Args:
            **kwargs: Metadata parameters including:
                - sample_rate: Sampling rate (Hz)
                - bandwidth_min: Minimum bandwidth (Hz)
                - bandwidth_max: Maximum bandwidth (Hz)
                - signal_duration_in_samples_min: Minimum signal duration (samples)
                - signal_duration_in_samples_max: Maximum signal duration (samples)
                - frame_type: (optional) 'long' (default) or 'short'.
        """
        super().__init__(**kwargs)
        self.required_metadata_fields = [
            "sample_rate",
            "bandwidth_min",
            "bandwidth_max",
            "signal_duration_in_samples_min",
            "signal_duration_in_samples_max",
        ]
        try:
            frame_type = self["frame_type"]
        except AttributeError:
            frame_type = "long"
        class_name = "adsb-long" if frame_type == "long" else "adsb-short"
        self.set_default_class_name(class_name)

    def generate(self) -> Signal:
        """Generates an ADS-B signal.

        Returns:
            Signal: Generated ADS-B signal with metadata.
        """
        sample_rate = self["sample_rate"]
        num_iq_samples_signal = self.random_generator.integers(
            low=self["signal_duration_in_samples_min"],
            high=self["signal_duration_in_samples_max"] + 1,
        )
        bandwidth = self.random_generator.integers(low=self["bandwidth_min"], high=self["bandwidth_max"] + 1)
        try:
            frame_type = self["frame_type"]
        except AttributeError:
            frame_type = "long"

        signal_data = adsb_modulator(frame_type, bandwidth, sample_rate, num_iq_samples_signal, self.random_generator)
        return Signal(data=signal_data, center_freq=0, bandwidth=bandwidth)
