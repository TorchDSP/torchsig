"""LoRa (Long Range) CSS Signal Builder.

LoRa uses Chirp Spread Spectrum (CSS): each symbol is a frequency-shifted
chirp spanning the full channel bandwidth. The preamble — a fixed sequence
of up-chirps followed by down-chirp SFD symbols — makes LoRa instantly
recognizable in a spectrogram and distinct from the generic ChirpSS builder.

Physical layer (Semtech SX127x / LoRaWAN 1.1):
    * Spreading factor SF ∈ {7..12}; M = 2^SF chips per symbol
    * Bandwidth = chip rate (chip duration = 1/BW)
    * Each symbol value s ∈ [0, M) produces a chirp whose instantaneous
      frequency starts at f = s/M (normalized) and sweeps up by BW,
      wrapping around at the band edge.
    * Preamble: 8 base up-chirps, 2 sync-word chirps, 2.25 down-chirps

Toy simplifications:
    * No spreading code, no FEC, no whitening; payload chirp values are random.
    * Phase continuity between symbols is not enforced (training-data use only).
    * Sync-word symbols use fixed values representative of a LoRaWAN network.
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

LORA_SF_VALUES: tuple[int, ...] = (7, 8, 9, 10, 11, 12)
LORA_PREAMBLE_UPCHIRPS: int = 8
LORA_SYNC_CHIRPS: int = 2  # encode network sync word
LORA_SFD_DOWNCHIRPS: int = 2
LORA_SFD_QUARTER_CHIPS: int = 1  # +0.25 up-chirp after down-chirps


def _upchirp(s: int, m: int, osr: int) -> np.ndarray:
    """One LoRa up-chirp for symbol value s at oversampling rate osr."""
    k = np.arange(m * osr)
    # instantaneous freq: (s + chip_index) mod M, one osr-length plateau per chip
    chip_idx = k // osr
    f = ((s + chip_idx) % m) / m - 0.5
    phi = 2.0 * np.pi * np.cumsum(f) / osr
    return np.exp(1j * phi)


def _downchirp(m: int, osr: int) -> np.ndarray:
    """LoRa base down-chirp (conjugate sweep)."""
    k = np.arange(m * osr)
    chip_idx = k // osr
    f = ((m - chip_idx) % m) / m - 0.5
    phi = 2.0 * np.pi * np.cumsum(f) / osr
    return np.exp(1j * phi)


def build_lora_symbol_stream(
    num_samples: int,
    sf: int,
    oversampling_rate_nominal: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Builds a LoRa frame: preamble then random data chirps.

    Args:
        num_samples: Target number of baseband samples (at osr).
        sf: Spreading factor (7-12).
        oversampling_rate_nominal: Samples per chip at baseband.
        rng: Random number generator.

    Returns:
        np.ndarray: Complex baseband samples, length ≥ num_samples before trim.
    """
    if rng is None:
        rng = np.random.default_rng()

    m = 2**sf
    osr = oversampling_rate_nominal
    samples_per_sym = m * osr

    # Sync-word symbol values: representative of a standard LoRaWAN network byte
    sync_s = [m // 8, m // 4]

    preamble = np.concatenate(
        [_upchirp(0, m, osr)] * LORA_PREAMBLE_UPCHIRPS
        + [_upchirp(s, m, osr) for s in sync_s]
        + [_downchirp(m, osr)] * LORA_SFD_DOWNCHIRPS
        + [_upchirp(0, m, osr)[: samples_per_sym // 4]]  # 0.25 up-chirp
    )

    num_data_syms = max(1, int(np.ceil((num_samples - len(preamble)) / samples_per_sym)))
    data_s = rng.integers(0, m, num_data_syms)
    data = np.concatenate([_upchirp(int(s), m, osr) for s in data_s])

    return np.concatenate([preamble, data])


def lora_modulator_baseband(
    sf: int,
    max_num_samples: int,
    oversampling_rate_nominal: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """LoRa CSS modulator at complex baseband.

    Args:
        sf: Spreading factor (7-12).
        max_num_samples: Maximum output samples.
        oversampling_rate_nominal: Samples per chip at baseband.
        rng: Random number generator.

    Returns:
        np.ndarray: Baseband LoRa signal, exactly max_num_samples long.

    Raises:
        ValueError: If max_num_samples or oversampling_rate_nominal are not positive.
    """
    if max_num_samples <= 0:
        raise ValueError("max_num_samples must be positive")
    if oversampling_rate_nominal <= 0:
        raise ValueError("oversampling_rate_nominal must be positive")

    if rng is None:
        rng = np.random.default_rng()

    iq = build_lora_symbol_stream(max_num_samples, sf, oversampling_rate_nominal, rng)

    if len(iq) > max_num_samples:
        iq = slice_tail_to_length(iq, max_num_samples)
    elif len(iq) < max_num_samples:
        iq = pad_head_tail_to_length(iq, max_num_samples)

    return iq.astype(TorchSigComplexDataType)


def lora_modulator(
    sf: int,
    bandwidth: float,
    sample_rate: float,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """LoRa modulator: builds preamble + data chirps and resamples to bandwidth.

    Args:
        sf: Spreading factor (7–12).
        bandwidth: Desired signal bandwidth (Hz); equals the LoRa chip rate.
        sample_rate: Capture sampling rate (Hz).
        num_samples: Number of IQ samples to produce.
        rng: Random number generator.

    Returns:
        np.ndarray: LoRa IQ at the requested bandwidth, length num_samples.

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

    baseband = lora_modulator_baseband(sf, max_num_samples, oversampling_rate_nominal, rng)
    correct_bw = multistage_polyphase_resampler(baseband, resample_rate_ideal)
    correct_bw *= 1 / resample_rate_ideal

    correct_bw = slice_head_tail_to_length(correct_bw, num_samples) if len(correct_bw) > num_samples else pad_head_tail_to_length(correct_bw, num_samples)
    return correct_bw.astype(TorchSigComplexDataType)


class LoraSignalGenerator(BaseSignalGenerator):
    """LoRa CSS Signal Generator.

    Builds a structured LoRa frame: 8-upchirp preamble, 2-symbol sync word,
    2.25-chirp SFD, followed by random data chirps. Spreading factor is drawn
    randomly from {7, 8, 9, 10, 11, 12} each call unless fixed in metadata.
    """

    def __init__(self, **kwargs: dict[str, str | float | int]) -> None:
        """Initializes the LoRa Signal Generator.

        Args:
            **kwargs: Metadata parameters including:

                - sample_rate: Sampling rate (Hz)
                - bandwidth_min: Minimum bandwidth (Hz)
                - bandwidth_max: Maximum bandwidth (Hz)
                - signal_duration_in_samples_min: Minimum signal duration (samples)
                - signal_duration_in_samples_max: Maximum signal duration (samples)
                - sf: (optional) Fixed spreading factor. If absent, drawn randomly
                  from LORA_SF_VALUES each generate() call.
        """
        super().__init__(**kwargs)
        self.required_metadata_fields = [
            "sample_rate",
            "bandwidth_min",
            "bandwidth_max",
            "signal_duration_in_samples_min",
            "signal_duration_in_samples_max",
        ]
        self.set_default_class_name("lora")

    def generate(self) -> Signal:
        """Generates a LoRa signal.

        Returns:
            Signal: Generated LoRa signal with metadata.
        """
        sample_rate = self["sample_rate"]
        num_iq_samples_signal = self.random_generator.integers(
            low=self["signal_duration_in_samples_min"],
            high=self["signal_duration_in_samples_max"] + 1,
        )
        bandwidth = self.random_generator.integers(low=self["bandwidth_min"], high=self["bandwidth_max"] + 1)
        try:
            sf = int(self["sf"])
        except AttributeError:
            sf = int(self.random_generator.choice(LORA_SF_VALUES))

        signal_data = lora_modulator(sf, bandwidth, sample_rate, num_iq_samples_signal, self.random_generator)
        return Signal(data=signal_data, center_freq=0, bandwidth=bandwidth)
