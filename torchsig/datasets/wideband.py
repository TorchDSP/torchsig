from ast import literal_eval
from copy import deepcopy
from functools import partial
from itertools import chain
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy import signal as sp
from tqdm import tqdm

from torchsig.datasets import estimate_filter_length
from torchsig.datasets.synthetic import ConstellationDataset, FSKDataset, OFDMDataset
from torchsig.transforms import (
    AddNoise,
    Compose,
    IQImbalance,
    Normalize,
    RandAugment,
    RandomApply,
    RandomConvolve,
    RandomDropSamples,
    RandomFrequencyShift,
    RandomMagRescale,
    RandomPhaseShift,
    RandomResample,
    RandomTimeShift,
    RayleighFadingChannel,
    RollOff,
    SignalTransform,
    SpectralInversion,
)
from torchsig.transforms.functional import (
    FloatParameter,
    NumericParameter,
    to_distribution,
    uniform_continuous_distribution,
    uniform_discrete_distribution,
)
from torchsig.utils.dataset import SignalDataset
from torchsig.utils.dsp import low_pass
from torchsig.utils.types import SignalData, SignalMetadata


class SignalBurst(SignalMetadata):
    """SignalBurst is a class that inherits from the SignalDescription class but adds a
    `generate_iq` method that should be implemented by subclasses in order to
    generate the IQ for the signal described by the SignalDescription contents.
    This class should be inherited to represent several kinds of burst
    generation techniques.

    """

    def __init__(self, random_generator, **kwargs):
        super(SignalBurst, self).__init__(**kwargs)
        self.random_generator = random_generator

    def generate_iq(self):
        # meant to be implemented by sub-class
        raise NotImplementedError


class ShapedNoiseSignalBurst(SignalBurst):
    """An SignalBurst which is just shaped (filtered) Gaussian noise

    Args:
        **kwargs:

    """

    def __init__(self, **kwargs):
        super(ShapedNoiseSignalBurst, self).__init__(**kwargs)
        # Update freq values
        self.lower_frequency = self.center_frequency - self.bandwidth / 2
        self.upper_frequency = self.center_frequency + self.bandwidth / 2

    def generate_iq(self):
        real_noise = self.random_generator.randn(
            int(self.num_iq_samples * self.duration)
        )
        imag_noise = self.random_generator.randn(
            int(self.num_iq_samples * self.duration)
        )
        iq_samples = real_noise + 1j * imag_noise

        # Precompute non-aliased low,upper,center,bw freqs
        upper = 0.5 if self.upper_frequency > 0.5 else self.upper_frequency
        lower = -0.5 if self.lower_frequency < -0.5 else self.lower_frequency
        bandwidth = upper - lower
        center = lower + bandwidth / 2

        # Filter noise
        taps = low_pass(
            cutoff=bandwidth / 2, transition_bandwidth=(0.5 - bandwidth / 2) / 4
        )
        sinusoid = np.exp(
            2j * np.pi * center * np.linspace(0, len(taps) - 1, len(taps))
        )
        taps = taps * sinusoid
        iq_samples = sp.convolve(iq_samples, taps, mode="same")

        # prune to be correct size out of filter
        iq_samples = iq_samples[-int(self.num_iq_samples * self.duration) :]

        # We ultimately want E_s/N_0 to be snr. We can also express this as:
        # E_s/(N*B_n) -- N is noise energy per hertz and B_n is the noise bandwidth
        # First, to get E_s = 1, we want E[abs(x)^2] = 1
        # Multiply by sqrt(mean(abs(x)^2)) so that mean(abs(x)^2) = 1
        iq_samples = iq_samples / np.sqrt(np.mean(np.abs(iq_samples) ** 2))

        # Next, we assume that N_0 will be fixed at 1.0 because we will have something else add a uniform noise floor.
        # Also, since B_n is the bandwidth of the noise that is in the same band as our signal, B_n = self.bandwidth
        # Therefore, we multiply the signal by bandwidth so that we achieve E_s/N_0.
        # Intuitively, we are reducing our signal power the smaller bandwidth we have since we will also have
        # correspondingly less noise energy in the same band as our signal the smaller bandwidth the signal is.
        # Then we multiply by sqrt(10^(snr/10.0)) (same as 10^(snr/20.0) to force our energy per hertz to be snr
        iq_samples = (
            np.sqrt(bandwidth) * (10 ** (self.snr / 20.0)) * iq_samples / np.sqrt(2)
        )

        if iq_samples.shape[0] > 50:
            window = np.blackman(50) / np.max(np.blackman(50))
            iq_samples[:25] *= window[:25]  # burst-shape the front
            iq_samples[-25:] *= window[-25:]  # burst-shape the tail

        # zero-pad to fit num_iq_samples
        leading_silence = int(self.num_iq_samples * self.start)
        trailing_silence = self.num_iq_samples - len(iq_samples) - leading_silence
        trailing_silence = 0 if trailing_silence < 0 else trailing_silence

        iq_samples = np.pad(
            iq_samples,
            pad_width=(leading_silence, trailing_silence),
            mode="constant",
            constant_values=0,
        )
        # Prune if burst goes over
        return iq_samples[: self.num_iq_samples]


class ModulatedSignalBurst(SignalBurst):
    """A burst which is a shaped modulated signal

    Args:
        modulation (:obj: `str`, `List[str]`)
            The modulation or list of modulations to sample from for each burst

        modulation_list (:obj:`List[str]`):
            The full list of modulations for mapping class names to indices

        **kwargs

    """

    def __init__(
        self,
        modulation: Union[str, List[str]],
        modulation_list: List[str],
        **kwargs,
    ):
        super(ModulatedSignalBurst, self).__init__(**kwargs)
        # Read in full modulation list
        default_class_list = [
            "ook",
            "bpsk",
            "4pam",
            "4ask",
            "qpsk",
            "8pam",
            "8ask",
            "8psk",
            "16qam",
            "16pam",
            "16ask",
            "16psk",
            "32qam",
            "32qam_cross",
            "32pam",
            "32ask",
            "32psk",
            "64qam",
            "64pam",
            "64ask",
            "64psk",
            "128qam_cross",
            "256qam",
            "512qam_cross",
            "1024qam",
            "2fsk",
            "2gfsk",
            "2msk",
            "2gmsk",
            "4fsk",
            "4gfsk",
            "4msk",
            "4gmsk",
            "8fsk",
            "8gfsk",
            "8msk",
            "8gmsk",
            "16fsk",
            "16gfsk",
            "16msk",
            "16gmsk",
            "ofdm-64",
            "ofdm-72",
            "ofdm-128",
            "ofdm-180",
            "ofdm-256",
            "ofdm-300",
            "ofdm-512",
            "ofdm-600",
            "ofdm-900",
            "ofdm-1024",
            "ofdm-1200",
            "ofdm-2048",
        ]
        if modulation_list == "all" or modulation_list == None:
            self.class_list = default_class_list
        else:
            self.class_list = modulation_list

        # Randomized classes to sample from
        if modulation == "all" or modulation == None:
            modulation = self.class_list
        else:
            modulation = [modulation] if isinstance(modulation, str) else modulation
        self.classes = to_distribution(
            modulation,
            random_generator=self.random_generator,
        )

        # Update freq values
        assert self.center_frequency is not None
        assert self.bandwidth is not None
        self.lower_frequency = self.center_frequency - self.bandwidth / 2
        self.upper_frequency = self.center_frequency + self.bandwidth / 2

    def generate_iq(self):
        # Read mod_index to determine which synthetic dataset to read from
        self.class_name = self.classes()
        self.class_index = self.class_list.index(self.class_name)
        self.class_name = (
            self.class_name if isinstance(self.class_name, list) else [self.class_name]
        )
        approx_samp_per_sym = (
            int(np.ceil(self.bandwidth**-1))
            if self.bandwidth < 1.0
            else int(np.ceil(self.bandwidth))
        )
        approx_bandwidth = (
            approx_samp_per_sym**-1
            if self.bandwidth < 1.0
            else int(np.ceil(self.bandwidth))
        )

        # Determine if the new rate of the requested signal to determine how many samples to request
        if "ofdm" in self.class_name[0]:
            occupied_bandwidth = 0.5
        elif "g" in self.class_name[0]:
            if "m" in self.class_name[0]:
                occupied_bandwidth = approx_bandwidth * (
                    1 - 0.5 + self.excess_bandwidth
                )
            else:
                occupied_bandwidth = approx_bandwidth * (
                    1 + 0.25 + self.excess_bandwidth
                )
        elif "fsk" in self.class_name[0]:
            occupied_bandwidth = approx_bandwidth * (1 + 1)
        elif "msk" in self.class_name[0]:
            occupied_bandwidth = approx_bandwidth
        else:
            occupied_bandwidth = approx_bandwidth * (1 + self.excess_bandwidth)

        self.duration = self.stop - self.start
        new_rate = occupied_bandwidth / self.bandwidth
        num_iq_samples = int(
            np.ceil(self.num_iq_samples * self.duration / new_rate * 1.1)
        )

        # Create modulated burst
        if "ofdm" in self.class_name[0]:
            num_subcarriers = [int(self.class_name[0][5:])]
            sidelobe_suppression_methods = ("lpf", "win_start")
            modulated_burst = OFDMDataset(
                constellations=(
                    "bpsk",
                    "qpsk",
                    "16qam",
                    "64qam",
                    "256qam",
                    "1024qam",
                ),  # sub-carrier modulations
                num_subcarriers=tuple(
                    num_subcarriers
                ),  # possible number of subcarriers
                num_iq_samples=num_iq_samples,
                num_samples_per_class=1,
                random_data=True,
                sidelobe_suppression_methods=sidelobe_suppression_methods,
                dc_subcarrier=("on", "off"),
                time_varying_realism=("on", "off"),
            )
        elif "g" in self.class_name[0]:
            modulated_burst = FSKDataset(
                modulations=self.class_name,
                num_iq_samples=num_iq_samples,
                num_samples_per_class=1,
                iq_samples_per_symbol=approx_samp_per_sym,
                random_data=True,
                random_pulse_shaping=True,
            )
        elif "fsk" in self.class_name[0] or "msk" in self.class_name[0]:
            modulated_burst = FSKDataset(
                modulations=self.class_name,
                num_iq_samples=num_iq_samples,
                num_samples_per_class=1,
                iq_samples_per_symbol=approx_samp_per_sym,
                random_data=True,
                random_pulse_shaping=False,
            )
        else:
            modulated_burst = ConstellationDataset(
                constellations=self.class_name,
                num_iq_samples=num_iq_samples,
                num_samples_per_class=1,
                iq_samples_per_symbol=approx_samp_per_sym,
                random_data=True,
                random_pulse_shaping=True,
            )

        # Extract IQ samples from dataset example
        iq_samples = modulated_burst[0][0]

        # Resample to target bandwidth * oversample to avoid freq wrap during shift
        if (
            self.center_frequency + self.bandwidth / 2 > 0.4
            or self.center_frequency - self.bandwidth / 2 < -0.4
        ):
            oversample = 2 if self.bandwidth < 1.0 else int(np.ceil(self.bandwidth * 2))
        else:
            oversample = 1
        up_rate = np.floor(new_rate * 100 * oversample).astype(np.int32)
        down_rate = 100
        iq_samples = sp.resample_poly(iq_samples, up_rate, down_rate)

        # Freq shift to desired center freq
        time_vector = np.arange(iq_samples.shape[0], dtype=float)
        iq_samples = iq_samples * np.exp(
            2j * np.pi * self.center_frequency / oversample * time_vector
        )

        if oversample == 1:
            # Prune to length
            iq_samples = iq_samples[-int(self.num_iq_samples * self.duration) :]
        else:
            # Pre-prune to reduce filtering cost
            iq_samples = iq_samples[
                -int(self.num_iq_samples * self.duration * oversample) :
            ]
            taps = low_pass(
                cutoff=1 / oversample / 2,
                transition_bandwidth=(0.5 - 1 / oversample / 2) / 4,
            )
            iq_samples = sp.convolve(iq_samples, taps, mode="same")

            # Decimate back down to correct sample rate
            iq_samples = sp.resample_poly(iq_samples, 1, oversample)
            iq_samples = iq_samples[-int(self.num_iq_samples * self.duration) :]

        # Set power
        iq_samples = iq_samples / np.sqrt(np.mean(np.abs(iq_samples) ** 2))
        iq_samples = (
            np.sqrt(self.bandwidth)
            * (10 ** (self.snr / 20.0))
            * iq_samples
            / np.sqrt(2)
        )

        if iq_samples.shape[0] > 50:
            window = np.blackman(50) / np.max(np.blackman(50))
            iq_samples[:25] *= window[:25]
            iq_samples[-25:] *= window[-25:]

        # Zero-pad to fit num_iq_samples
        leading_silence = int(self.num_iq_samples * self.start)
        trailing_silence = self.num_iq_samples - len(iq_samples) - leading_silence
        trailing_silence = 0 if trailing_silence < 0 else trailing_silence

        iq_samples = np.pad(
            np.array(iq_samples),
            pad_width=(leading_silence, trailing_silence),
            mode="constant",
            constant_values=0,
        )
        return iq_samples[: self.num_iq_samples]


class SignalOfInterestSignalBurst(SignalBurst):
    """A burst which is a generic class, reading in its IQ generation function

    Args:
        soi_gen_iq: (:obj: `Callable`):
            A function that generates the SOI's IQ data. Note that in order for
            the randomized bandwidths to function with the
            `SyntheticBurstSource`, the generation function must input a
            bandwidth argument.

        soi_gen_bw: (:obj:`float`):
            A float parameter informing the `SignalOfInterestSignalBurst` object
            what the SOI's bandwidth was generated at within the `soi_gen_iq`
            function. Defaults to 0.5, signifying half-bandwidth or 2x over-
            sampled generation, which is sufficient for most signals.

        soi_class: (:obj:`str`):
            The class of the SOI

        soi_class_list: (:obj:`List[str]`):
            The class list from which the SOI belongs

        **kwargs

    """

    def __init__(
        self,
        soi_gen_iq: Callable,
        soi_class: str,
        soi_class_list: List[str],
        soi_gen_bw: float = 0.5,
        **kwargs,
    ):
        super(SignalOfInterestSignalBurst, self).__init__(**kwargs)
        self.soi_gen_iq = soi_gen_iq
        self.soi_gen_bw = soi_gen_bw
        self.class_name = soi_class if soi_class else "soi0"
        self.class_list = soi_class_list if soi_class_list else ["soi0"]
        self.class_index = self.class_list.index(self.class_name)
        assert self.center_frequency is not None
        assert self.bandwidth is not None
        self.lower_frequency = self.center_frequency - self.bandwidth / 2
        self.upper_frequency = self.center_frequency + self.bandwidth / 2

    def generate_iq(self):
        # Generate the IQ from the provided SOI generator
        iq_samples = self.soi_gen_iq()

        # Resample to target bandwidth * 2 to avoid freq wrap during shift
        new_rate = self.soi_gen_bw / self.bandwidth
        up_rate = np.floor(new_rate * 100 * 2).astype(np.int32)
        down_rate = 100
        iq_samples = sp.resample_poly(iq_samples, up_rate, down_rate)

        # Freq shift to desired center freq
        time_vector = np.arange(iq_samples.shape[0], dtype=float)
        iq_samples = iq_samples * np.exp(
            2j * np.pi * self.center_frequency / 2 * time_vector
        )

        # Filter around center
        taps = low_pass(cutoff=1 / 4, transition_bandwidth=(0.5 - 1 / 4) / 4)
        iq_samples = sp.convolve(iq_samples, taps, mode="same")

        # Decimate back down to correct sample rate
        iq_samples = sp.resample_poly(iq_samples, 1, 2)
        iq_samples = iq_samples[-int(self.num_iq_samples * self.duration) :]

        # Set power
        iq_samples = iq_samples / np.sqrt(np.mean(np.abs(iq_samples) ** 2))
        iq_samples = (
            np.sqrt(self.bandwidth)
            * (10 ** (self.snr / 20.0))
            * iq_samples
            / np.sqrt(2)
        )

        if iq_samples.shape[0] > 50:
            window = np.blackman(50) / np.max(np.blackman(50))
            iq_samples[:25] *= window[:25]
            iq_samples[-25:] *= window[-25:]

        # Zero-pad to fit num_iq_samples
        leading_silence = int(self.num_iq_samples * self.start)
        trailing_silence = self.num_iq_samples - len(iq_samples) - leading_silence
        trailing_silence = 0 if trailing_silence < 0 else trailing_silence

        iq_samples = np.pad(
            iq_samples,
            pad_width=(leading_silence, trailing_silence),
            mode="constant",
            constant_values=0,
        )
        return iq_samples[: self.num_iq_samples]


class FileSignalBurst(SignalBurst):
    """A burst which reads previously-extracted bursts from individual files
    that contain the IQ data for each burst.

    Args:
        file_path (:obj: `str`, :obj:`list`):
            Specify the file path from which to read the IQ data
            * If string, file_path is fixed at the value provided
            * If list, file_path is randomly sampled from the input list

        file_reader (:obj: `Callable`):
            A function that instructs the `FileSignalBurst` class how to read the
            IQ data from the file(s) along with the class name and occupied
            bandwidth within the file

        class_list (:obj: `List[str]`):
            A list of classes to map the read class name to the respective
            class index

        **kwargs

    """

    def __init__(
        self,
        file_path: Union[str, List],
        file_reader: Callable,
        class_list: List[str],
        **kwargs,
    ):
        super(FileSignalBurst, self).__init__(**kwargs)
        self.file_path = to_distribution(
            file_path,
            random_generator=self.random_generator,
        )
        self.file_reader = file_reader
        self.class_list = class_list
        assert self.center_frequency is not None
        assert self.bandwidth is not None
        self.lower_frequency = self.center_frequency - self.bandwidth / 2
        self.upper_frequency = self.center_frequency + self.bandwidth / 2

    def generate_iq(self):
        # Read the IQ from the file_path using the file_reader
        file_path = (
            self.file_path if isinstance(self.file_path, str) else self.file_path()
        )
        iq_samples, class_name, file_bw = self.file_reader(file_path)

        # Assign read class information to SignalBurst
        self.class_name = class_name
        self.class_index = self.class_list.index(self.class_name)

        # Resample to target bandwidth * 2 to avoid freq wrap during shift
        new_rate = file_bw / self.bandwidth
        up_rate = np.floor(new_rate * 100 * 2).astype(np.int32)
        down_rate = 100
        iq_samples = sp.resample_poly(iq_samples, up_rate, down_rate)

        # Freq shift to desired center freq
        time_vector = np.arange(iq_samples.shape[0], dtype=float)
        iq_samples = iq_samples * np.exp(
            2j * np.pi * self.center_frequency / 2 * time_vector
        )

        # Filter around center
        taps = low_pass(cutoff=1 / 4, transition_bandwidth=(0.5 - 1 / 4) / 4)
        iq_samples = sp.convolve(iq_samples, taps, mode="same")

        # Decimate back down to correct sample rate
        iq_samples = sp.resample_poly(iq_samples, 1, 2)

        # Inspect/set duration
        if iq_samples.shape[0] < self.num_iq_samples * self.duration:
            self.duration = iq_samples.shape[0] / self.num_iq_samples
            self.stop = self.start + self.duration
        iq_samples = iq_samples[-int(self.num_iq_samples * self.duration) :]

        # Set power
        iq_samples = iq_samples / np.sqrt(np.mean(np.abs(iq_samples) ** 2))
        iq_samples = (
            np.sqrt(self.bandwidth)
            * (10 ** (self.snr / 20.0))
            * iq_samples
            / np.sqrt(2)
        )

        if iq_samples.shape[0] > 50:
            window = np.blackman(50) / np.max(np.blackman(50))
            iq_samples[:25] *= window[:25]
            iq_samples[-25:] *= window[-25:]

        # Zero-pad to fit num_iq_samples
        leading_silence = int(self.num_iq_samples * self.start)
        trailing_silence = self.num_iq_samples - len(iq_samples) - leading_silence
        trailing_silence = 0 if trailing_silence < 0 else trailing_silence

        iq_samples = np.pad(
            iq_samples,
            pad_width=(leading_silence, trailing_silence),
            mode="constant",
            constant_values=0,
        )
        return iq_samples[: self.num_iq_samples]


class BurstSourceDataset(SignalDataset):
    """Abstract Base Class for sources of bursts.

    Args:
        num_iq_samples (int, optional): [description]. Defaults to 512*512.
        num_samples (int, optional): [description]. Defaults to 100.

    """

    def __init__(
        self, num_iq_samples: int = 512 * 512, num_samples: int = 1000, **kwargs
    ):
        super(BurstSourceDataset, self).__init__(**kwargs)
        self.num_iq_samples = num_iq_samples
        self.num_samples = num_samples
        self.index: List[Tuple[Any, ...]] = []

    def __getitem__(self, item: int) -> Tuple[np.ndarray, Any]:
        burst_collection = self.index[item][0]
        iq_samples = np.zeros((self.num_iq_samples,), dtype=np.complex128)
        for burst_idx, burst in enumerate(burst_collection):
            iq_samples += burst.generate_iq()

        # Format into single SignalData object
        signal_data = SignalData(
            data=iq_samples.tobytes(),
            item_type=np.dtype(np.float64),
            data_type=np.dtype(np.complex128),
            signal_description=burst_collection,
        )

        # Apply transforms
        signal_data = self.transform(signal_data) if self.transform else signal_data
        target = (
            self.target_transform(signal_data.signal_description)
            if self.target_transform
            else signal_data.signal_description
        )
        iq_data = signal_data.iq_data
        assert iq_data is not None

        return iq_data, target

    def __len__(self) -> int:
        return len(self.index)


class SyntheticBurstSourceDataset(BurstSourceDataset):
    """SyntheticBurstSourceDataset is a Dataset that is meant to represent a set of bursts presumably coming from the same
    or similar kinds of sources. It could represent a single Wi-Fi or bluetooth device, for example. This was made
    so that it could be its own dataset, if necessary.

    """

    def __init__(
        self,
        burst_class: SignalBurst,
        bandwidths: FloatParameter = (0.01, 0.1),
        center_frequencies: FloatParameter = (-0.25, 0.25),
        burst_durations: FloatParameter = (0.2, 0.2),
        silence_durations: FloatParameter = (0.01, 0.3),
        snrs_db: NumericParameter = (-5, 15),
        start: FloatParameter = (0.0, 0.9),
        num_iq_samples: int = 512 * 512,
        num_samples: int = 20,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super(SyntheticBurstSourceDataset, self).__init__(**kwargs)
        self.random_generator = np.random.RandomState(seed)
        self.num_iq_samples = num_iq_samples
        self.num_samples = num_samples
        self.burst_class = burst_class
        self.bandwidths = to_distribution(
            bandwidths, random_generator=self.random_generator
        )
        self.center_frequencies = to_distribution(
            center_frequencies, random_generator=self.random_generator
        )
        self.burst_durations = to_distribution(
            burst_durations, random_generator=self.random_generator
        )
        self.silence_durations = to_distribution(
            silence_durations, random_generator=self.random_generator
        )
        self.snrs_db = to_distribution(snrs_db, random_generator=self.random_generator)
        self.start = to_distribution(start, random_generator=self.random_generator)

        # Generate the index by creating a set of bursts.
        self.index = [
            (collection, idx)
            for idx, collection in enumerate(self._generate_burst_collections())
        ]

    def _generate_burst_collections(self) -> List[List[SignalBurst]]:
        dataset = []
        for sample_idx in range(self.num_samples):
            sample_burst_collection = []
            start = self.start()
            while start < 0.95:  # Avoid bursts of durations < 0.05 at end
                burst_duration = self.burst_durations()
                silence_duration = self.silence_durations()
                center_frequency = self.center_frequencies()
                bandwidth = self.bandwidths()
                snr = self.snrs_db()

                # Boundary checks
                stop = start + burst_duration
                if stop > 1.0:
                    burst_duration = 1.0 - start

                sample_burst_collection.append(
                    self.burst_class(  # type: ignore
                        num_iq_samples=self.num_iq_samples,
                        start=0 if start < 0 else start,
                        stop=start + burst_duration,
                        center_frequency=center_frequency,
                        bandwidth=bandwidth,
                        snr=snr,
                        random_generator=self.random_generator,
                    )
                )
                start = start + burst_duration + silence_duration
            dataset.append(sample_burst_collection)
        return dataset


class WidebandDataset(SignalDataset):
    """WidebandDataset is an SignalDataset that contains several SignalSourceDataset
    objects. Each sample from this dataset includes bursts from each contained
    SignalSourceDataset as well as a collection of SignalDescriptions which
    includes all meta-data about the bursts.

    Args:
        signal_sources (:obj:`list` of :py:class:`SignalSource`):
            List of SignalSource objects from which to sample bursts and add to an overall signal

        num_iq_samples (:obj:`int`):
            number of IQ samples to produce

        num_samples (:obj:`int`):
            number of dataset samples to produce

        transform (:class:`Callable`, optional):
            A function/transform that takes in an IQ vector and returns a transformed version.

    """

    def __init__(
        self,
        signal_sources: List[BurstSourceDataset],
        num_iq_samples: int,
        num_samples: int,
        pregenerate: bool = False,
        **kwargs,
    ):
        super(WidebandDataset, self).__init__(**kwargs)
        self.signal_sources = signal_sources
        self.num_iq_samples = num_iq_samples
        self.num_samples = num_samples

        self.index = []
        self.pregenerate = False
        if pregenerate:
            print("Pregenerating dataset...")
            for idx in tqdm(range(self.num_samples)):
                self.index.append(self.__getitem__(idx))
        self.pregenerate = pregenerate

    def __getitem__(self, item: int) -> Tuple[np.ndarray, Any]:
        if self.pregenerate:
            return self.index[item]
        # Retrieve data & metadata from all signal sources
        iq_data: Optional[np.ndarray] = None
        signal_description_collection = []
        for source_idx in range(len(self.signal_sources)):
            signal_iq_data, signal_description = self.signal_sources[source_idx][item]
            iq_data = signal_iq_data if iq_data else iq_data + signal_iq_data
            signal_description = (
                [signal_description]
                if isinstance(signal_description, SignalMetadata)
                else signal_description
            )
            signal_description_collection.extend(signal_description)

        # Format into single SignalData object
        assert iq_data is not None
        signal_data = SignalData(
            data=iq_data.tobytes(),
            item_type=np.dtype(np.float64),
            data_type=np.dtype(np.complex128),
            signal_description=signal_description_collection,
        )

        # Apply transforms
        signal_data = self.transform(signal_data) if self.transform else signal_data
        target = (
            self.target_transform(signal_data.signal_description)
            if self.target_transform
            else signal_data.signal_description
        )
        assert signal_data.iq_data is not None
        iq_data = signal_data.iq_data

        return iq_data, target

    def __len__(self) -> int:
        return self.num_samples


class WidebandModulationsDataset(SignalDataset):
    """The `WidebandModulationsDataset` is an `SignalDataset` that creates
    multiple, non-overlapping, realistic wideband modulated signals whenever
    a data sample is requested. The `__gen_metadata__` method is responsible
    for any inter-modulation relationships, currently hard-coded such that OFDM
    signals are handled differently than the remaining modulations.

    Args:
        modulation_list (:obj: `List[str]`):
            The list of modulations to include in the wideband samples

        level (:obj: `int`):
            Set the difficulty level of the dataset with levels 0-2

        num_iq_samples (:obj: `int`):
            Set the requested number of IQ samples for each dataset example

        num_samples (:obj: `int`):
            Set the number of samples for the dataset to contain

        transform (:class:`Callable`, optional):
            A function/transform that takes in an IQ vector and returns a transformed version.

        target_transform (:class:`Callable`, optional):
            A function/transform that takes in a list of SignalDescription objects and returns a transformed target.

        seed (:obj: `int`, optional):
            A seed for reproducibility

        **kwargs

    """

    default_modulations: List[str] = [
        "ook",
        "bpsk",
        "4pam",
        "4ask",
        "qpsk",
        "8pam",
        "8ask",
        "8psk",
        "16qam",
        "16pam",
        "16ask",
        "16psk",
        "32qam",
        "32qam_cross",
        "32pam",
        "32ask",
        "32psk",
        "64qam",
        "64pam",
        "64ask",
        "64psk",
        "128qam_cross",
        "256qam",
        "512qam_cross",
        "1024qam",
        "2fsk",
        "2gfsk",
        "2msk",
        "2gmsk",
        "4fsk",
        "4gfsk",
        "4msk",
        "4gmsk",
        "8fsk",
        "8gfsk",
        "8msk",
        "8gmsk",
        "16fsk",
        "16gfsk",
        "16msk",
        "16gmsk",
        "ofdm-64",
        "ofdm-72",
        "ofdm-128",
        "ofdm-180",
        "ofdm-256",
        "ofdm-300",
        "ofdm-512",
        "ofdm-600",
        "ofdm-900",
        "ofdm-1024",
        "ofdm-1200",
        "ofdm-2048",
    ]

    def __init__(
        self,
        modulation_list: Optional[List] = None,
        level: int = 0,
        num_iq_samples: int = 262144,
        num_samples: int = 10,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super(WidebandModulationsDataset, self).__init__(**kwargs)
        self.random_generator = np.random.RandomState(seed)
        self.seed = seed
        self.modulation_list = (
            self.default_modulations if modulation_list is None else modulation_list
        )
        self.level = level
        self.metadata = self.__gen_metadata__(self.modulation_list)
        self.num_modulations = len(self.metadata)
        # Bump up OFDM ratio slightly due to its higher bandwidth and lack of bursty nature
        # This helps make the number of OFDM signals closer to the others
        self.ofdm_ratio = (self.num_ofdm / self.num_modulations) * 2.0
        self.num_iq_samples = num_iq_samples
        self.num_samples = num_samples

        # Set level-specific parameters
        if level == 0:
            num_signals = (1, 1)
            snrs = (40, 40)
            self.transform = Compose(
                [
                    AddNoise(
                        noise_power_db=(0, 0), input_noise_floor_db=-100
                    ),  # Set input noise floor very low because this transform sets the floor
                    Normalize(norm=np.inf),
                ]
            )
        elif level == 1:
            num_signals = (1, 6)
            snrs = (20, 40)
            self.transform = Compose(
                [
                    AddNoise(
                        noise_power_db=(0, 0), input_noise_floor_db=-100
                    ),  # Set input noise floor very low because this transform sets the floor
                    AddNoise(
                        noise_power_db=(-40, -20), input_noise_floor_db=0
                    ),  # Then add minimal noise without affecting SNR
                    Normalize(norm=np.inf),
                ]
            )
        elif level == 2:
            num_signals = (1, 6)
            snrs = (0, 30)
            self.transform = Compose(
                transforms=[
                    RandomApply(
                        RandomTimeShift(
                            shift=(-int(num_iq_samples / 2), int(num_iq_samples / 2))
                        ),
                        0.25,
                    ),
                    RandomApply(RandomFrequencyShift(freq_shift=(-0.2, 0.2)), 0.25),
                    RandomApply(
                        RandomResample(
                            rate_ratio=(0.8, 1.2), num_iq_samples=num_iq_samples
                        ),
                        0.25,
                    ),
                    RandomApply(SpectralInversion(), 0.5),
                    AddNoise(
                        noise_power_db=(0, 0), input_noise_floor_db=-100
                    ),  # Set input noise floor very low because this transform sets the floor
                    AddNoise(
                        noise_power_db=(-40, -20), input_noise_floor_db=0
                    ),  # Then add minimal noise without affecting SNR
                    RandAugment(
                        [
                            RandomApply(
                                RandomMagRescale(start=(0, 0.9), scale=(-4, 4)), 0.5
                            ),
                            RollOff(
                                low_freq=(0.00, 0.05),
                                upper_freq=(0.95, 1.00),
                                low_cut_apply=0.5,
                                upper_cut_apply=0.5,
                                order=(6, 20),
                            ),
                            RandomConvolve(num_taps=(2, 5), alpha=(0.1, 0.4)),
                            RayleighFadingChannel((0.001, 0.01)),
                            RandomDropSamples(
                                drop_rate=0.01,
                                size=(1, 1),
                                fill=["ffill", "bfill", "mean", "zero"],
                            ),
                            RandomPhaseShift((-1, 1)),
                            IQImbalance(
                                (-3, 3),
                                (-np.pi * 1.0 / 180.0, np.pi * 1.0 / 180.0),
                                (-0.1, 0.1),
                            ),
                        ],
                        num_transforms=2,
                    ),
                    Normalize(norm=np.inf),
                ]
            )
        else:
            raise ValueError(
                "Input level expected to be either 0, 1, or 2. Found {}".format(
                    self.level
                )
            )

        if transform is not None:
            self.transform = Compose(
                [
                    self.transform,
                    transform,
                ]
            )
        self.target_transform = target_transform

        self.num_signals = to_distribution(
            num_signals, random_generator=self.random_generator
        )
        self.snrs = to_distribution(snrs, random_generator=self.random_generator)

    def __gen_metadata__(self, modulation_list: List) -> pd.DataFrame:
        """This method defines the parameters of the modulations to be inserted
        into the wideband data. The values below are hardcoded; however, if
        new datasets are desired with different modulation relationships, the
        below data can be parameterized or updated to new values.

        """
        self.num_ofdm = 0
        column_names = [
            "index",
            "modulation",
            "bursty_prob",
            "burst_duration",
            "silence_multiple",
            "freq_hopping_prob",
            "freq_hopping_channels",
        ]
        metadata = []
        for index, modulation in enumerate(modulation_list):
            if "ofdm" in modulation:
                self.num_ofdm += 1
                bursty_prob = 0.0
                burst_duration = "(0.05,0.10)"
                silence_multiple = "(1,1)"
                freq_hopping_prob = 0.0
                freq_hopping_channels = "(1,1)"
            else:
                bursty_prob = 0.2
                burst_duration = "(0.05,0.20)"
                silence_multiple = "(1,3)"
                freq_hopping_prob = 0.2
                freq_hopping_channels = "(2,16)"

            metadata.append(
                [
                    index,
                    modulation,
                    bursty_prob,
                    burst_duration,
                    silence_multiple,
                    freq_hopping_prob,
                    freq_hopping_channels,
                ]
            )

        return pd.DataFrame(metadata, columns=column_names)

    def __getitem__(self, item: int) -> Tuple[np.ndarray, Any]:
        # Initialize empty list of signal sources & signal descriptors
        signal_sources: List[SyntheticBurstSourceDataset] = []

        # Randomly decide how many signals in capture
        num_signals = int(self.num_signals())

        # Randomly decide if OFDM signals are in capture
        ofdm_present = True if self.random_generator.rand() < self.ofdm_ratio else False

        # Loop through signals to add
        sig_counter = 0
        overlap_counter = 0
        while sig_counter < num_signals and overlap_counter < 5:
            if ofdm_present:
                if sig_counter == 0:
                    # Randomly sample from OFDM options (assumes OFDM at end)
                    meta_idx = self.random_generator.randint(
                        self.num_modulations - self.num_ofdm, self.num_modulations
                    )
                    modulation = self.metadata.iloc[meta_idx].modulation
                else:
                    # Randomly select signal from full metadata list
                    meta_idx = self.random_generator.randint(self.num_modulations)
                    modulation = self.metadata.iloc[meta_idx].modulation
            else:
                # Randomly sample from all but OFDM (assumes OFDM at end)
                meta_idx = self.random_generator.randint(
                    self.num_modulations - self.num_ofdm
                )
                modulation = self.metadata.iloc[meta_idx].modulation

            # Random bandwidth based on signal modulation and num_signals
            if ofdm_present:
                if "ofdm" in modulation:
                    if num_signals == 1:
                        bandwidth = self.random_generator.uniform(0.2, 0.7)
                    else:
                        bandwidth = self.random_generator.uniform(0.3, 0.5)
                else:
                    bandwidth = self.random_generator.uniform(0.025, 0.1)
            else:
                if num_signals == 1:
                    bandwidth = self.random_generator.uniform(0.05, 0.4)
                else:
                    bandwidth = self.random_generator.uniform(0.05, 0.15)

            # Random center frequency
            center_freq = self.random_generator.uniform(-0.4, 0.4)

            # Determine if continuous or bursty
            burst_random_var = self.random_generator.rand()
            hop_random_var = self.random_generator.rand()
            if (
                burst_random_var < self.metadata.iloc[meta_idx].bursty_prob
                or hop_random_var < self.metadata.iloc[meta_idx].freq_hopping_prob
            ):
                # Signal is bursty
                bursty = True
                burst_duration = to_distribution(
                    literal_eval(self.metadata.iloc[meta_idx].burst_duration),
                    random_generator=self.random_generator,
                )()
                silence_multiple = to_distribution(
                    literal_eval(self.metadata.iloc[meta_idx].silence_multiple),
                    random_generator=self.random_generator,
                )()
                stops_in_frame = False
                if hop_random_var < self.metadata.iloc[meta_idx].freq_hopping_prob:
                    # override bandwidth with smaller options for freq hoppers
                    if ofdm_present:
                        bandwidth = self.random_generator.uniform(0.0125, 0.025)
                    else:
                        bandwidth = self.random_generator.uniform(0.025, 0.05)

                    silence_duration = burst_duration * (silence_multiple - 1)
                    freq_channels = to_distribution(
                        literal_eval(
                            self.metadata.iloc[meta_idx].freq_hopping_channels
                        ),
                        random_generator=self.random_generator,
                    )()

                    # Convert channel count to list of center frequencies
                    center_freq_array = np.arange(
                        center_freq,
                        center_freq + (bandwidth * freq_channels),
                        bandwidth,
                    )
                    center_freq_array = center_freq_array - (
                        freq_channels / 2 * bandwidth
                    )
                    center_freq_array = center_freq_array[center_freq_array < 0.5]
                    center_freq_array = center_freq_array[center_freq_array > -0.5]
                    center_freq_list = center_freq_array.tolist()
                    if len(center_freq_list) == 1 and silence_duration == 0:
                        # If all but one band outside freq range, ensure nonzero silence duration
                        silence_duration = burst_duration

                    low_freq = min(center_freq_list) - bandwidth / 2
                    high_freq = max(center_freq_list) + bandwidth / 2

                else:
                    silence_duration = burst_duration * silence_multiple
                    low_freq = center_freq - bandwidth / 2
                    high_freq = center_freq + bandwidth / 2

            else:
                # Signal is continous
                bursty = False
                burst_duration = 1.0
                silence_duration = 1.0
                low_freq = center_freq - bandwidth / 2
                high_freq = center_freq + bandwidth / 2

                # Randomly determine if the signal should stop in the frame
                if self.random_generator.rand() < 0.2:
                    stops_in_frame = True
                    burst_duration = self.random_generator.uniform(0.05, 0.95)
                else:
                    stops_in_frame = False

            # Randomly determine if the signal should start in the frame
            if self.random_generator.rand() < 0.2 and not stops_in_frame:
                start = self.random_generator.uniform(0, 0.95)
                stop = 1.0
            else:
                start = 0.0
                stop = burst_duration
            if bursty:
                start = start + self.random_generator.rand() * burst_duration
                stop = 1.0

            # Handle overlaps
            overlap = False
            minimum_freq_spacing = 0.05
            for source in signal_sources:
                for signal in source.index[0][0]:
                    # Check time overlap
                    if (
                        (start > signal.start and start < signal.stop)
                        or (start + burst_duration > signal.stop and stop < signal.stop)
                        or (signal.start > start and signal.start < stop)
                        or (signal.stop > start and signal.stop < stop)
                        or (start == 0.0 and signal.start == 0.0)
                        or (stop == 1.0 and signal.stop == 1.0)
                    ):
                        # Check freq overlap
                        if (
                            (
                                low_freq > signal.lower_frequency - minimum_freq_spacing
                                and low_freq
                                < signal.upper_frequency + minimum_freq_spacing
                            )
                            or (
                                high_freq
                                > signal.lower_frequency - minimum_freq_spacing
                                and high_freq
                                < signal.upper_frequency + minimum_freq_spacing
                            )
                            or (
                                signal.lower_frequency - minimum_freq_spacing > low_freq
                                and signal.lower_frequency - minimum_freq_spacing
                                < high_freq
                            )
                            or (
                                signal.upper_frequency + minimum_freq_spacing > low_freq
                                and signal.upper_frequency + minimum_freq_spacing
                                < high_freq
                            )
                        ):
                            # Overlaps in both time and freq, skip
                            overlap = True
            if overlap:
                overlap_counter += 1
                continue

            # Add signal to signal sources
            signal_sources.append(
                SyntheticBurstSourceDataset(
                    bandwidths=bandwidth,
                    center_frequencies=center_freq,
                    burst_durations=burst_duration,
                    silence_durations=silence_duration,
                    snrs_db=self.snrs(),
                    start=start,
                    burst_class=partial(  # type: ignore
                        ModulatedSignalBurst,
                        modulation=modulation,
                        modulation_list=self.modulation_list,
                    ),
                    num_iq_samples=self.num_iq_samples,
                    num_samples=1,
                    transform=None,
                    seed=self.seed + item * 53 if self.seed else None,
                ),
            )
            sig_counter += 1

        iq_data = None
        signal_description_collection = []
        for source_idx in range(len(signal_sources)):
            signal_iq_data, signal_description = signal_sources[source_idx][0]
            iq_data = signal_iq_data if iq_data is None else iq_data + signal_iq_data
            signal_description = (
                [signal_description]
                if isinstance(signal_description, SignalMetadata)
                else signal_description
            )
            signal_description_collection.extend(signal_description)

        # If no signal sources present, add noise
        if iq_data is None:
            real_noise = np.random.randn(
                self.num_iq_samples,
            )
            imag_noise = np.random.randn(
                self.num_iq_samples,
            )
            iq_data = real_noise + 1j * imag_noise

        # Format into single SignalData object
        signal_data = SignalData(
            data=iq_data.tobytes(),
            item_type=np.dtype(np.float64),
            data_type=np.dtype(np.complex128),
            signal_description=signal_description_collection,
        )

        # Apply transforms
        signal_data = self.transform(signal_data) if self.transform else signal_data
        target = (
            self.target_transform(signal_data.signal_description)
            if self.target_transform
            else signal_data.signal_description
        )
        iq_data = signal_data.iq_data
        assert iq_data is not None

        return iq_data, target

    def __len__(self) -> int:
        return self.num_samples


class Interferers(SignalTransform):
    """SignalTransform that inputs burst sources to add as unlabeled interferers

    Args:
        burst_sources :obj:`BurstSourceDataset`:
            Burst source dataset defining interferers to be added

        num_iq_samples :obj:`int`:
            Number of IQ samples in requested dataset & interferer examples

        num_samples :obj:`int`:
            Number of unique interfer examples

        interferer_transform :obj:`SignalTransform`:
            SignalTransforms to be applied to the interferers

    """

    def __init__(
        self,
        burst_sources: BurstSourceDataset,
        num_iq_samples: int = 262144,
        num_samples: int = 10,
        interferer_transform: Optional[SignalTransform] = None,
    ):
        super(Interferers, self).__init__()
        self.num_samples = num_samples
        self.interferers = WidebandDataset(
            signal_sources=[burst_sources],
            num_iq_samples=num_iq_samples,
            num_samples=self.num_samples,
            transform=interferer_transform,
            target_transform=None,
        )

    def __call__(self, data: Any) -> Any:
        idx = np.random.randint(self.num_samples)
        if isinstance(data, SignalData):
            data.iq_data = data.iq_data + self.interferers[idx][0]
        else:
            data = data + self.interferers[idx][0]
        return data


class RandomSignalInsertion(SignalTransform):
    """RandomSignalInsertion reads the input SignalData's occupied frequency
    bands from the SignalDescription objects and then randomly generates and
    inserts a new continuous or bursty single carrier signal into a randomly
    selected unoccupied frequency band, such that no signal overlap occurs

    Args:
        modulation_list :obj:`list`:
            Optionally pass in a list of modulations to sample from for the
            inserted signal. If None or omitted, the default full list of
            modulations will be used.

    """

    default_modulation_list: List[str] = [
        "ook",
        "bpsk",
        "4pam",
        "4ask",
        "qpsk",
        "8pam",
        "8ask",
        "8psk",
        "16qam",
        "16pam",
        "16ask",
        "16psk",
        "32qam",
        "32qam_cross",
        "32pam",
        "32ask",
        "32psk",
        "64qam",
        "64pam",
        "64ask",
        "64psk",
        "128qam_cross",
        "256qam",
        "512qam_cross",
        "1024qam",
        "2fsk",
        "2gfsk",
        "2msk",
        "2gmsk",
        "4fsk",
        "4gfsk",
        "4msk",
        "4gmsk",
        "8fsk",
        "8gfsk",
        "8msk",
        "8gmsk",
        "16fsk",
        "16gfsk",
        "16msk",
        "16gmsk",
        "ofdm-64",
        "ofdm-72",
        "ofdm-128",
        "ofdm-180",
        "ofdm-256",
        "ofdm-300",
        "ofdm-512",
        "ofdm-600",
        "ofdm-900",
        "ofdm-1024",
        "ofdm-1200",
        "ofdm-2048",
    ]

    def __init__(self, modulation_list: Optional[List[str]] = None):
        super(RandomSignalInsertion, self).__init__()
        self.modulation_list: List[str] = (
            modulation_list if modulation_list else self.default_modulation_list
        )

    def __call__(self, data: Any) -> Any:
        if isinstance(data, SignalData):
            assert data.iq_data is not None
            assert data.signal_description is not None

            # Create new SignalData object for transformed data
            new_data = SignalData(
                data=None,
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[],
            )
            num_iq_samples = data.iq_data.shape[0]

            # Read existing SignalDescription for unoccupied freq bands
            new_signal_description = deepcopy(data.signal_description)
            new_signal_description_list: List[SignalMetadata] = (
                [new_signal_description]
                if isinstance(new_signal_description, SignalMetadata)
                else new_signal_description
            )
            occupied_bands = []
            for new_signal_desc in new_signal_description_list:
                assert new_signal_desc.lower_frequency is not None
                assert new_signal_desc.upper_frequency is not None
                occupied_bands.append(
                    [
                        int((new_signal_desc.lower_frequency + 0.5) * 100),
                        int((new_signal_desc.upper_frequency + 0.5) * 100),
                    ]
                )
            occupied_bands = sorted(occupied_bands)
            flat = chain((0 - 1,), chain.from_iterable(occupied_bands), (100 + 1,))
            unoccupied_bands = [
                ((x + 1) / 100 - 0.5, (y - 1) / 100 - 0.5)
                for x, y in zip(flat, flat)
                if x + 6 < y
            ]
            if len(unoccupied_bands) > 0:
                max_bandwidth = min([y - x for x, y in unoccupied_bands])
                bandwidth = np.random.uniform(0.05, max_bandwidth)
                center_freqs: List[Tuple[float, float]] = [
                    (x + bandwidth / 2, y - bandwidth / 2) for x, y in unoccupied_bands
                ]
                rand_band_idx = np.random.randint(len(center_freqs))
                center_freqs_dist = to_distribution(center_freqs[rand_band_idx])
                center_freq = center_freqs_dist()
                bursty = True if np.random.rand() < 0.5 else False
                burst_duration = np.random.uniform(0.05, 1.0) if bursty else 1.0
                silence_duration = burst_duration if bursty else 1.0
                if bandwidth < 0.2:
                    modulation_list = []
                    for mod in self.modulation_list:
                        if "ofdm" not in mod:
                            modulation_list.append(mod)
                else:
                    modulation_list = self.modulation_list
                num_samples = int(1 / burst_duration + 2) if bursty else 1

                signal_sources = [
                    SyntheticBurstSourceDataset(
                        bandwidths=bandwidth,
                        center_frequencies=center_freq,
                        burst_durations=burst_duration,
                        silence_durations=silence_duration,
                        snrs_db=20,
                        start=(-0.05, 0.95),
                        burst_class=partial(  # type: ignore
                            ModulatedSignalBurst,
                            modulation=modulation_list,
                            modulation_list=modulation_list,
                        ),
                        num_iq_samples=num_iq_samples,
                        num_samples=num_samples,
                        transform=None,
                    ),
                ]
                signal_dataset = WidebandDataset(
                    signal_sources=signal_sources,  # type: ignore
                    num_iq_samples=num_iq_samples,
                    num_samples=num_samples,
                    transform=Normalize(norm=np.inf),
                    target_transform=None,
                )

                new_signal_data, new_signal_signal_desc = signal_dataset[0]
                new_data.iq_data = data.iq_data + new_signal_data

                # Update the SignalDescription
                new_signal_description.extend(new_signal_signal_desc)  # type: ignore
                new_data.signal_description = new_signal_description

            else:
                new_data.iq_data = data.iq_data

            return new_data

        else:
            num_iq_samples = data.shape[0]
            num_samples = int(1 / 0.05 + 2)
            signal_sources = [
                SyntheticBurstSourceDataset(
                    bandwidths=(0.05, 0.8),
                    center_frequencies=(-0.4, 0.4),
                    burst_durations=(0.05, 1.0),
                    silence_durations=(0.05, 1.0),
                    snrs_db=20,
                    start=(-0.05, 0.95),
                    burst_class=partial(  # type: ignore
                        ModulatedSignalBurst,
                        modulation=self.modulation_list,
                        modulation_list=self.modulation_list,
                    ),
                    num_iq_samples=num_iq_samples,
                    num_samples=num_samples,
                    transform=None,
                ),
            ]
            signal_dataset = WidebandDataset(
                signal_sources=signal_sources,  # type: ignore
                num_iq_samples=num_iq_samples,
                num_samples=num_samples,
                transform=Normalize(norm=np.inf),
                target_transform=None,
            )
            output = data + signal_dataset[0][0]

            return output
