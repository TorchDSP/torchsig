"""Wideband Dataset Generation Tools
"""
from torchsig.utils.types import (
    create_signal_data,
    create_modulated_rf_metadata,
    is_signal_data,
)
from torchsig.datasets.synthetic import ConstellationDataset, FSKDataset, OFDMDataset, AMDataset, FMDataset, LFMDataset, ChirpSSDataset
from torchsig.transforms import *
from torchsig.transforms.functional import (
    FloatParameter,
    NumericParameter,
    to_distribution,
)
import os
from torchsig.utils.types import SignalData, SignalMetadata, Signal
from torchsig.utils.dataset import SignalDataset
from torchsig.utils.dsp import low_pass
from torchsig.datasets.signal_classes import torchsig_signals
from typing import Any, Callable, List, Optional, Tuple, Union
from ast import literal_eval
from functools import partial
from itertools import chain
from scipy import signal as sp
from tqdm import tqdm
import pandas as pd
import numpy as np
import copy


class SignalBurst:
    """SignalBurst is a class that contains from the SignalMetadata class but adds a
    `generate_iq` method that should be implemented by subclasses in order to
    generate the IQ for the signal described by the SignalMetadata contents.
    This class should be inherited to represent several kinds of burst
    generation techniques.

    """

    def __init__(self, meta: SignalMetadata, random_generator: np.random.Generator):
        super(SignalBurst, self).__init__()
        self.meta = meta
        self.random_generator = random_generator

    def generate_iq(self):
        raise NotImplementedError


class ShapedNoiseSignalBurst(SignalBurst):
    """An SignalBurst which is just shaped (filtered) Gaussian noise

    Args:
        **kwargs:

    """

    def __init__(self, **kwargs):
        super(ShapedNoiseSignalBurst, self).__init__(**kwargs)
        # Update freq values
        self.meta["lower_freq"] = self.meta["center_freq"] - self.meta["bandwidth"] / 2
        self.meta["upper_freq"] = self.meta["center_freq"] + self.meta["bandwidth"] / 2
        self.snr = self.meta["snr"]

    def generate_iq(self):
        real_noise = self.random_generator.standard_normal(
            int(self.meta["num_samples"] * self.meta["duration"])
        )
        imag_noise = self.random_generator.standard_normal(
            int(self.meta["num_samples"] * self.meta["duration"])
        )
        iq_samples = real_noise + 1j * imag_noise

        # Precompute non-aliased low,upper,center,bw freqs
        upper = 0.5 if self.meta["upper_freq"] > 0.5 else self.meta["upper_freq"]
        lower = -0.5 if self.meta["lower_freq"] < -0.5 else self.meta["lower_freq"]
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
        filtered = sp.convolve(iq_samples, taps, mode="full")
        lidx = (len(filtered) - len(iq_samples)) // 2
        ridx = lidx + len(iq_samples)
        iq_samples = filtered[lidx:ridx]

        # We ultimately want E_s/N_0 to be snr. We can also express this as:
        # E_s/(N*B_n) -- N is noise energy per hertz and B_n is the noise bandwidth
        # First, to get E_s = 1, we want E[abs(x)^2] = 1
        # Multiply by sqrt(mean(abs(x)^2)) so that mean(abs(x)^2) = 1
        iq_samples = iq_samples / np.sqrt(np.mean(np.abs(iq_samples) ** 2))

        # Next, we assume that N_0 will be fixed at 1.0 because we will have something else add a uniform noise floor.
        # Also, since B_n is the bandwidth of the noise that is in the same band as our signal, B_n = self.meta["bandwidth"]
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
        leading_silence = int(self.meta["num_samples"] * self.meta["start"])
        trailing_silence = self.meta["num_samples"] - len(iq_samples) - leading_silence
        trailing_silence = 0 if trailing_silence < 0 else trailing_silence

        iq_samples = np.pad(
            iq_samples,
            pad_width=(leading_silence, trailing_silence),
            mode="constant",
            constant_values=0,
        )
        # Prune if burst goes over
        return iq_samples[: self.meta["num_samples"]]


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

        if modulation_list == "all" or modulation_list == None:
            self.class_list = torchsig_signals.class_list
        else:
            self.class_list = modulation_list

        # Randomized classes to sample from
        if modulation == "all" or modulation == None:
            modulation = self.class_list
        else:
            modulation = [modulation] if isinstance(modulation, str) else modulation
        self.classes = to_distribution(modulation, random_generator=self.random_generator)

        # Update freq values
        assert self.meta["center_freq"] is not None
        assert self.meta["bandwidth"] is not None
        self.meta["lower_freq"] = self.meta["center_freq"] - self.meta["bandwidth"] / 2
        self.meta["upper_freq"] = self.meta["center_freq"] + self.meta["bandwidth"] / 2

    def generate_iq(self):
        self.meta["class_name"] = self.classes()
        self.meta["class_index"] = self.class_list.index(self.meta["class_name"])

        # estimate the approximate samples per symbol (used as an oversampling estimate for ConstellationDataset and FSK/MSK modulations)
        approx_samp_per_sym = int(np.ceil(self.meta["bandwidth"] ** -1)) if self.meta["bandwidth"] < 1.0 else int(np.ceil(self.meta["bandwidth"])) 
        # self.meta["num_samples"] is the total number of IQ samples used in the snapshot. the duration (self.meta["duration"]) represents
        # what proportion of the snapshot that the modulated waveform will occupy. the duration is on a range of [0,1]. for example, a 
        # duration of 0.75 means that the modulated waveform will occupy 75% of the total length of the snapshot.
        self.meta["duration"] = self.meta["stop"] - self.meta["start"]
        # calculate how many IQ samples are needed from the modulator based on the duration
        num_iq_samples = int(np.ceil(self.meta["num_samples"] * self.meta["duration"]) )

        # Create modulated burst
        if self.meta["class_name"] in torchsig_signals.ofdm_signals:
            num_subcarriers = [int(self.meta["class_name"][5:])]
            sidelobe_suppression_methods = ("lpf", "win_start")
            modulated_burst = OFDMDataset(
                constellations=torchsig_signals.ofdm_subcarrier_modulations, # sub-carrier modulations
                num_subcarriers=tuple(num_subcarriers),  # possible number of subcarriers
                num_iq_samples=num_iq_samples,
                num_samples_per_class=1,
                random_data=True,
                sidelobe_suppression_methods=sidelobe_suppression_methods,
                dc_subcarrier=("on", "off"),
                time_varying_realism=("on", "off"),
                center_freq=self.meta["center_freq"],
                bandwidth=self.meta["bandwidth"]
            )
        elif self.meta["class_name"] in torchsig_signals.fsk_signals: # FSK, GFSK, MSK, GMSK
            modulated_burst = FSKDataset(
                modulations=[self.meta["class_name"]],
                num_iq_samples=num_iq_samples,
                num_samples_per_class=1,
                iq_samples_per_symbol=approx_samp_per_sym,
                random_data=True,
                random_pulse_shaping=True,
                center_freq=self.meta["center_freq"],
                bandwidth=self.meta["bandwidth"]
            )
        elif self.meta["class_name"] in torchsig_signals.constellation_signals: # QAM, PSK, OOK, PAM, ASK
            modulated_burst = ConstellationDataset(
                constellations=[self.meta["class_name"]],
                num_iq_samples=num_iq_samples,
                num_samples_per_class=1,
                iq_samples_per_symbol=approx_samp_per_sym,
                random_data=True,
                random_pulse_shaping=False, #True, TODO fix pulse shaping code.
                center_freq=self.meta["center_freq"],
            )
        elif self.meta["class_name"] in torchsig_signals.am_signals: # AM-DSB, AM-DSB-SC, AM-USB, AM-LSB
            modulated_burst = AMDataset(
                modulations=[self.meta["class_name"]],
                num_iq_samples=num_iq_samples,
                num_samples_per_class=1,
                random_data=True,
                center_freq=self.meta["center_freq"],
                bandwidth=self.meta["bandwidth"]
            )
        elif self.meta["class_name"] in torchsig_signals.fm_signals: # FM
            modulated_burst = FMDataset(
                num_iq_samples=num_iq_samples,
                num_samples_per_class=1,
                random_data=True,
                center_freq=self.meta["center_freq"],
                bandwidth=self.meta["bandwidth"]
            )
        elif self.meta["class_name"] in torchsig_signals.lfm_signals: # LFM data, LFM radar
            modulated_burst = LFMDataset(
                num_iq_samples=num_iq_samples,
                num_samples_per_class=1,
                random_data=True,
                center_freq=self.meta["center_freq"],
                bandwidth=self.meta["bandwidth"]
            )
        elif self.meta["class_name"] in torchsig_signals.chirpss_signals: # chirp SS
            modulated_burst = ChirpSSDataset(
                num_iq_samples=num_iq_samples,
                num_samples_per_class=1,
                random_data=True,
                center_freq=self.meta["center_freq"],
                bandwidth=self.meta["bandwidth"]
            )

        # Extract IQ samples from dataset example
        iq_samples = modulated_burst[0][0]

        # limit the number of samples to the desired duration
        iq_samples = iq_samples[:int(self.meta["num_samples"] * self.meta["duration"])]

        # Set power
        iq_samples = iq_samples / np.sqrt(np.mean(np.abs(iq_samples) ** 2))
        iq_samples = np.sqrt(self.meta["bandwidth"]) * (10 ** (self.meta["snr"] / 20.0)) * iq_samples / np.sqrt(2)

        # Zero-pad to fit num_iq_samples
        leading_silence = int(self.meta["num_samples"] * self.meta["start"])
        trailing_silence = self.meta["num_samples"] - len(iq_samples) - leading_silence
        trailing_silence = 0 if trailing_silence < 0 else trailing_silence

        iq_samples = np.pad(np.array(iq_samples), pad_width=(leading_silence, trailing_silence), mode="constant", constant_values=0.,)
        return iq_samples[: self.meta["num_samples"]]


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
        self.meta["class_name"] = soi_class if soi_class else "soi0"
        self.class_list = soi_class_list if soi_class_list else ["soi0"]
        self.meta["class_index"] = self.class_list.index(self.meta["class_name"])
        assert self.meta["center_freq"] is not None
        assert self.meta["bandwidth"] is not None
        self.meta["lower_freq"] = self.meta["center_freq"] - self.meta["bandwidth"] / 2
        self.meta["upper_freq"] = self.meta["center_freq"] + self.meta["bandwidth"] / 2

    def generate_iq(self):
        # Generate the IQ from the provided SOI generator
        iq_samples = self.soi_gen_iq()

        # Resample to target bandwidth * 2 to avoid freq wrap during shift
        new_rate = self.soi_gen_bw / self.meta["bandwidth"]
        up_rate = np.floor(new_rate * 100 * 2).astype(np.int32)
        down_rate = 100
        iq_samples = sp.resample_poly(iq_samples, up_rate, down_rate)

        # Freq shift to desired center freq
        time_vector = np.arange(iq_samples.shape[0], dtype=float)
        iq_samples = iq_samples * np.exp(2j * np.pi * self.meta["center_freq"] / 2 * time_vector)

        # Filter around center
        taps = low_pass(cutoff=1 / 4, transition_bandwidth=(0.5 - 1 / 4) / 4)
        filtered = sp.convolve(iq_samples, taps, mode="full")
        lidx = (len(filtered) - len(iq_samples)) // 2
        ridx = lidx + len(iq_samples)
        iq_samples = filtered[lidx:ridx]

        # Decimate back down to correct sample rate
        iq_samples = sp.resample_poly(iq_samples, 1, 2)

        # Set power
        iq_samples = iq_samples / np.sqrt(np.mean(np.abs(iq_samples) ** 2))
        iq_samples = (np.sqrt(self.meta["bandwidth"]) * (10 ** (self.meta["snr"] / 20.0)) * iq_samples / np.sqrt(2) )

        if iq_samples.shape[0] > 50:
            window = np.blackman(50) / np.max(np.blackman(50))
            iq_samples[:25] *= window[:25]
            iq_samples[-25:] *= window[-25:]

        # Zero-pad to fit num_iq_samples
        leading_silence = int(self.meta["num_samples"] * self.meta["start"])
        trailing_silence = self.meta["num_samples"] - len(iq_samples) - leading_silence
        trailing_silence = 0 if trailing_silence < 0 else trailing_silence

        iq_samples = np.pad(
            iq_samples,
            pad_width=(leading_silence, trailing_silence),
            mode="constant",
            constant_values=0,
        )
        return iq_samples[: self.meta["num_samples"]]


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
        self.file_path = to_distribution(file_path, random_generator=self.random_generator)
        self.file_reader = file_reader
        self.class_list = class_list
        assert self.meta["center_freq"] is not None
        assert self.meta["bandwidth"] is not None
        self.meta["lower_freq"] = self.meta["center_freq"] - self.meta["bandwidth"] / 2
        self.meta["upper_freq"] = self.meta["center_freq"] + self.meta["bandwidth"] / 2

    def generate_iq(self):
        # Read the IQ from the file_path using the file_reader
        file_path = self.file_path if isinstance(self.file_path, str) else self.file_path()
        iq_samples, class_name, file_bw = self.file_reader(file_path)

        # Assign read class information to SignalBurst
        self.meta["class_name"] = class_name
        self.meta["class_index"] = self.class_list.index(self.meta["class_name"])

        # Resample to target bandwidth * 2 to avoid freq wrap during shift
        new_rate = file_bw / self.meta["bandwidth"]
        up_rate = np.floor(new_rate * 100 * 2).astype(np.int32)
        down_rate = 100
        iq_samples = sp.resample_poly(iq_samples, up_rate, down_rate)

        # Freq shift to desired center freq
        time_vector = np.arange(iq_samples.shape[0], dtype=float)
        iq_samples = iq_samples * np.exp(2j * np.pi * self.meta["center_freq"] / 2 * time_vector)

        # Filter around center
        taps = low_pass(cutoff=1 / 4, transition_bandwidth=(0.5 - 1 / 4) / 4)
        filtered = sp.convolve(iq_samples, taps, mode="full")
        lidx = (len(filtered) - len(iq_samples)) // 2
        ridx = lidx + len(iq_samples)
        iq_samples = filtered[lidx:ridx]

        # Decimate back down to correct sample rate
        iq_samples = sp.resample_poly(iq_samples, 1, 2)

        # Inspect/set duration
        if iq_samples.shape[0] < self.meta["num_samples"] * self.meta["duration"]:
            self.meta["duration"] = iq_samples.shape[0] / self.meta["num_samples"]
            self.meta["stop"] = self.meta["start"] + self.meta["duration"]

        # Set power
        iq_samples = iq_samples / np.sqrt(np.mean(np.abs(iq_samples) ** 2))
        iq_samples = (np.sqrt(self.meta["bandwidth"]) * (10 ** (self.meta["snr"] / 20.0)) * iq_samples / np.sqrt(2))

        if iq_samples.shape[0] > 50:
            window = np.blackman(50) / np.max(np.blackman(50))
            iq_samples[:25] *= window[:25]
            iq_samples[-25:] *= window[-25:]

        # Zero-pad to fit num_iq_samples
        leading_silence = int(self.meta["num_samples"] * self.meta["start"])
        trailing_silence = self.meta["num_samples"] - len(iq_samples) - leading_silence
        trailing_silence = 0 if trailing_silence < 0 else trailing_silence

        iq_samples = np.pad(
            iq_samples,
            pad_width=(leading_silence, trailing_silence),
            mode="constant",
            constant_values=0,
        )
        return iq_samples[: self.meta["num_samples"]]


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

    def __getitem__(self, item: int):
        burst_collection: List[SignalBurst] = self.index[item][0]
        iq_samples = np.zeros((self.num_iq_samples,), dtype=np.complex128)
        for burst in burst_collection:
            iq_samples += burst.generate_iq()

        # Format into single SignalData object
        signal_data = create_signal_data(samples=iq_samples)
        signal_meta = [b.meta for b in burst_collection]
        signal = Signal(data=signal_data, metadata=signal_meta)

        # Apply transforms
        if self.transform:
            signal = self.transform(signal)

        return signal["data"], signal["metadata"]

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
        self.random_generator = np.random.default_rng(seed)
        self.num_iq_samples = num_iq_samples
        self.num_samples = num_samples
        self.burst_class = burst_class
        self.bandwidths = to_distribution(bandwidths, random_generator=self.random_generator)
        self.center_frequencies = to_distribution(center_frequencies, random_generator=self.random_generator)
        self.burst_durations = to_distribution(burst_durations, random_generator=self.random_generator)
        # self.burst_durations = burst_durations
        self.silence_durations = to_distribution(silence_durations, random_generator=self.random_generator)
        # self.silence_durations = silence_durations
        self.snrs_db = to_distribution(snrs_db, random_generator=self.random_generator)
        self.start = to_distribution(start, random_generator=self.random_generator)
        self.start_test = start

        # Generate the index by creating a set of bursts.
        self.index = [(collection, idx) for idx, collection in enumerate(self._generate_burst_collections())]

    def _generate_burst_collections(self) -> List[List[SignalBurst]]:
        dataset = []
        for _ in range(self.num_samples):
            sample_burst_collection = []
            start = self.start()  # could get negative values
            while start < 0.95:  # Avoid bursts of durations < 0.05 at end
                burst_duration = self.burst_durations()               
                silence_duration = self.silence_durations()
                center_frequency = self.center_frequencies()
                bandwidth = self.bandwidths()
                snr = self.snrs_db()
                start = 0 if start < 0 else start
                stop = start + burst_duration                
                if stop > 1.0:
                    burst_duration = 1.0 - start
                    stop = 1.
                
                sample_burst_collection.append(
                    self.burst_class(  # type: ignore
                        meta=create_modulated_rf_metadata(
                            num_samples=self.num_iq_samples, 
                            start=start,
                            stop=stop,
                            duration=burst_duration,
                            center_freq=center_frequency,
                            bandwidth=bandwidth,
                            snr=snr,
                        ),
                        random_generator=self.random_generator,
                    )
                )
                start = start + burst_duration + silence_duration
            dataset.append(sample_burst_collection)

        return dataset


class WidebandDataset(SignalDataset):
    """WidebandDataset is an SignalDataset that contains several SignalSourceDataset
    objects. Each sample from this dataset includes bursts from each contained
    SignalSourceDataset as well as a collection of SignalMetadatas which
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
            for idx in tqdm(range(self.num_samples)):
                self.index.append(self.__getitem__(idx))
        self.pregenerate = pregenerate

    def __getitem__(self, item: int) -> Tuple[np.ndarray, Any]:
        if self.pregenerate:
            return self.index[item]

        # Retrieve data & metadata from all signal sources
        iq_data: Optional[np.ndarray] = None
        metadata = []
        for source_idx in range(len(self.signal_sources)):
            data, meta = self.signal_sources[source_idx][item]
            iq_data = data if iq_data else iq_data + data
            metadata.extend(meta)

        # Format into single SignalData object
        assert iq_data is not None
        signal = Signal(data=create_signal_data(samples=iq_data), metadata=metadata)

        # Apply transforms
        signal = self.transform(signal) if self.transform else signal

        target = signal["metadata"]
        if self.target_transform:
            target = self.target_transform(signal["metadata"])

        return signal["data"]["samples"], target

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
            A function/transform that takes in a list of SignalMetadata objects and returns a transformed target.

        seed (:obj: `int`, optional):
            A seed for reproducibility

        overlap_prob (Optional[float], optional): 
            Set the signal overlap probability. Defaults to 0.

        **kwargs

    """

    default_modulations: List[str] = torchsig_signals.class_list

    def __init__(
        self,
        modulation_list: Optional[List] = None,
        level: int = 0,
        num_iq_samples: int = 262144,
        num_samples: int = 10,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: Optional[int] = None,
        overlap_prob: Optional[float] = 0,
        **kwargs,
    ):
        super(WidebandModulationsDataset, self).__init__(**kwargs)
        self.random_generator = np.random.default_rng(seed)
        self.update_rng = False
        self.seed = seed
        self.modulation_list = (self.default_modulations if modulation_list is None else modulation_list)
        self.level = level
        self.metadata = self.__gen_metadata__(self.modulation_list)
        self.num_modulations = len(self.metadata)
        # Bump up OFDM ratio slightly due to its higher bandwidth and lack of bursty nature
        # This helps make the number of OFDM signals closer to the others
        self.ofdm_ratio = (self.num_ofdm / self.num_modulations) * 2.0
        self.num_iq_samples = num_iq_samples
        self.num_samples = num_samples
        self.overlap_prob = overlap_prob

        # Set level-specific parameters
        if level == 0:
            num_signals = (1, 1)
            snrs = (40, 40)
            self.transform = Compose(
                [
                    AddNoise(noise_power_db=(0, 0), input_noise_floor_db=-100),  # Set input noise floor very low because this transform sets the floor
                    Normalize(norm=np.inf),
                ]
            )
        elif level == 1:
            num_signals = (1, 6)
            snrs = (20, 40)
            self.transform = Compose(
                [
                    AddNoise(noise_power_db=(0, 0), input_noise_floor_db=-100),  # Set input noise floor very low because this transform sets the floor
                    AddNoise(noise_power_db=(-40, -20), input_noise_floor_db=0),  # Then add minimal noise without affecting SNR
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
        self.num_signals = to_distribution(num_signals, random_generator=self.random_generator)
        self.snrs = to_distribution(snrs, random_generator=self.random_generator)

    def ret_transforms(self):
        return self.transform

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
    
    @staticmethod
    def iter_cf_bw(bw, cf, edge_val=.495):
        cf_temp  = cf if isinstance(cf, list) else [cf]            
        band_edges = (np.max(cf) + bw / 2, np.min(cf) - bw / 2)
        bw_check = band_edges[0] > edge_val or band_edges[1] < -edge_val
        while bw_check:
            if band_edges[0] > edge_val:
                center_freq_shift = np.max(cf_temp) - (edge_val - bw /  2)                     
            else:
                center_freq_shift = np.min(cf_temp) + (edge_val - bw /  2) 
            cf_temp = [value - center_freq_shift for value in cf_temp]

            band_edges = (np.max(cf_temp) + bw / 2, np.min(cf_temp) - bw / 2)
            bw_check = band_edges[0] > edge_val or band_edges[1] < -edge_val
            if bw_check:
                bw *=.95

        ret_cf = cf_temp if isinstance(cf, list) else cf_temp[0]
        return bw, ret_cf

    def __getitem__(self, item: int) -> Tuple[np.ndarray, Any]:
        # Initialize empty list of signal sources & signal descriptors
        if not self.update_rng:
            self.random_generator = np.random.default_rng(os.getpid())
            self.update_rng = True
        signal_sources: List[SyntheticBurstSourceDataset] = []

        # Randomly decide how many signals in capture
        num_signals = int(self.num_signals())

        # Randomly decide if OFDM signals are in capture
        ofdm_present = True if self.random_generator.random() < self.ofdm_ratio else False
        # Loop through signals to add
        sig_counter = 0
        overlap_counter = 0
        while sig_counter < num_signals and overlap_counter < 5:
            if ofdm_present:
                if sig_counter == 0:
                    # Randomly sample from OFDM excluding options (assumes OFDM at end)
                    meta_idx = self.random_generator.integers(self.num_modulations - self.num_ofdm, self.num_modulations)
                    modulation = self.metadata.iloc[meta_idx].modulation
                else:
                    # Randomly select signal from full metadata list
                    meta_idx = self.random_generator.integers(self.num_modulations)
                    modulation = self.metadata.iloc[meta_idx].modulation
            else:
                # Randomly sample from all but OFDM (assumes OFDM at end)
                meta_idx = self.random_generator.integers(self.num_modulations - self.num_ofdm)
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
            center_freq_list = []
            # Determine if continuous or bursty
            burst_random_var = self.random_generator.random()
            hop_random_var = self.random_generator.random()
            if (burst_random_var < self.metadata.iloc[meta_idx].bursty_prob or hop_random_var < self.metadata.iloc[meta_idx].freq_hopping_prob):
                # Signal is bursty
                bursty = True
                burst_duration = to_distribution(literal_eval(self.metadata.iloc[meta_idx].burst_duration), random_generator=self.random_generator)()
                silence_multiple = to_distribution(literal_eval(self.metadata.iloc[meta_idx].silence_multiple), random_generator=self.random_generator)()
                stops_in_frame = False
                if hop_random_var < self.metadata.iloc[meta_idx].freq_hopping_prob:
                    # override bandwidth with smaller options for freq hoppers
                    if ofdm_present:
                        bandwidth = self.random_generator.uniform(0.0125, 0.025)
                    else:
                        bandwidth = self.random_generator.uniform(0.025, 0.05)

                    silence_duration = burst_duration * (silence_multiple - 1)
                    freq_channels = to_distribution(literal_eval(self.metadata.iloc[meta_idx].freq_hopping_channels), random_generator=self.random_generator)()
                    # Convert channel count to list of center frequencies
                    center_freq_array = np.arange(center_freq, center_freq + (bandwidth * freq_channels), bandwidth)
                    center_freq_array = center_freq_array - (freq_channels / 2 * bandwidth)
                    center_freq_array = center_freq_array[center_freq_array < 0.5]
                    center_freq_array = center_freq_array[center_freq_array > -0.5]
                    center_freq_list = center_freq_array.tolist()
                    bandwidth, center_freq_list = WidebandModulationsDataset.iter_cf_bw(bandwidth, center_freq_list)
                    low_freq = min(center_freq_list) - bandwidth / 2
                    high_freq = max(center_freq_list) + bandwidth / 2
                else:
                    bandwidth, center_freq = WidebandModulationsDataset.iter_cf_bw(bandwidth, center_freq)
                    silence_duration = burst_duration * silence_multiple
                    low_freq = center_freq - bandwidth / 2
                    high_freq = center_freq + bandwidth / 2
            else:
                # Signal is continous
                bandwidth, center_freq = WidebandModulationsDataset.iter_cf_bw(bandwidth, center_freq)
                bursty = False
                burst_duration = 1.0
                silence_duration = 1.0
                low_freq = center_freq - bandwidth / 2
                high_freq = center_freq + bandwidth / 2

                # Randomly determine if the signal should stop in the frame
                if self.random_generator.random() < 0.2:
                    stops_in_frame = True
                    burst_duration = self.random_generator.uniform(0.05, 0.95)
                else:
                    stops_in_frame = False

            # Randomly determine if the signal should start in the frame
            if self.random_generator.random() < 0.2 and not stops_in_frame:
                start = self.random_generator.uniform(0, 0.95)
                stop = np.clip(start + burst_duration, 0., 1.)
            else:
                start = 0.0
                stop = burst_duration

            # Handle overlaps
            # Add signal to signal sources
            center_freqs = center_freq if len(center_freq_list) == 0 else center_freq_list
            snrs_db = self.snrs()
            signal_sources.append(
                SyntheticBurstSourceDataset(
                    bandwidths=bandwidth,
                    center_frequencies=center_freqs,
                    burst_durations=burst_duration,
                    silence_durations=silence_duration,
                    snrs_db=snrs_db,
                    start=start,
                    burst_class=partial(  # type: ignore
                        ModulatedSignalBurst,
                        modulation=modulation,
                        modulation_list=self.modulation_list,
                        meta=create_modulated_rf_metadata(
                            bandwidth=bandwidth,
                            center_freq=center_freq,
                            start=start,
                            stop=stop
                        ),
                    ),
                    num_iq_samples=self.num_iq_samples,
                    num_samples=1,
                    transform=None,
                    seed=self.seed + item * 53 if self.seed else None,
                ),
            )
            
            overlap = False
            minimum_freq_spacing = 0.05
            for curr_sig in signal_sources[-1].index[0][0]: 
                curr_sig_meta = curr_sig.meta
                start = curr_sig_meta["start"]
                stop = curr_sig_meta["stop"]
                low_freq = curr_sig_meta["lower_freq"]
                high_freq = curr_sig_meta["upper_freq"]

                for source in signal_sources[:-1]:
                    for signal in source.index[0][0]:
                        meta = signal.meta
                        # Check time overlap
                        if (
                            (start >= meta["start"] and start <= meta["stop"])
                            or (stop >= meta["start"] and stop <= meta["stop"])
                            or (meta["start"] > start and meta["start"] < stop)
                            or (meta["stop"] > start and meta["stop"] < stop)
                            or (start == 0.0 and meta["start"] == 0.0)
                            or (stop == 1.0 and meta["stop"] == 1.0)
                        ):
                            # Check freq overlap
                            if ((low_freq > (meta["lower_freq"] - minimum_freq_spacing) and low_freq < (meta["upper_freq"] + minimum_freq_spacing))
                                or (high_freq > (meta["lower_freq"] - minimum_freq_spacing) and high_freq < (meta["upper_freq"] + minimum_freq_spacing))
                                or ((meta["lower_freq"] - minimum_freq_spacing) > low_freq and (meta["lower_freq"] - minimum_freq_spacing) < high_freq)
                                or ((meta["upper_freq"] + minimum_freq_spacing) > low_freq and (meta["upper_freq"] + minimum_freq_spacing) < high_freq)):
                                # Overlaps in both time and freq, skip
                                overlap = True

                            
            if overlap:
                overlap_draw = self.random_generator.uniform(0., 1.0)
                if overlap_draw < (1 - self.overlap_prob):
                    overlap_counter += 1
                    signal_sources.pop()
                    continue
            
            sig_counter += 1

        iq_data = None
        metadata = []
        for source_idx in range(len(signal_sources)):
            data, meta = signal_sources[source_idx][0]
            iq_data = data["samples"] if iq_data is None else iq_data + data["samples"]
            metadata.extend(meta)

        # If no signal sources present, add noise
        if iq_data is None:
            real_noise = np.random.randn(self.num_iq_samples,)
            imag_noise = np.random.randn(self.num_iq_samples,)
            iq_data = real_noise + 1j * imag_noise

        # Format into single SignalData object
        signal = Signal(data=SignalData(samples=iq_data), metadata=metadata)

        # Apply transforms
        signal = self.transform(signal) if self.transform else signal
        target = signal["metadata"]
        if self.target_transform:
            target = self.target_transform(signal["metadata"])

        return signal["data"]["samples"], target

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
        
        import ipdb
        ipdb.set_trace()
        if is_signal_data(data):
            data["data"]["samples"] = data["data"]["samples"] + self.interferers[idx][0]
        else:
            data = data + self.interferers[idx][0]
        return data


class RandomSignalInsertion(SignalTransform):
    """RandomSignalInsertion reads the input SignalData's occupied frequency
    bands from the SignalMetadata objects and then randomly generates and
    inserts a new continuous or bursty single carrier signal into a randomly
    selected unoccupied frequency band, such that no signal overlap occurs

    Args:
        modulation_list :obj:`list`:
            Optionally pass in a list of modulations to sample from for the
            inserted signal. If None or omitted, the default full list of
            modulations will be used.

    """

    default_modulation_list: List[str] = torchsig_signals.class_list

    def __init__(self, modulation_list: Optional[List[str]] = None):
        super(RandomSignalInsertion, self).__init__()
        self.modulation_list: List[str] = (
            modulation_list if modulation_list else self.default_modulation_list
        )

    def transform_meta(self, signal: Signal, params: tuple) -> Signal:
        occupied_bands = []
        for meta in signal["metadata"]:
            occupied_bands.append(
                [
                    int((meta["lower_freq"] + 0.5) * 100),
                    int((meta["upper_freq"] + 0.5) * 100),
                ]
            )
        occupied_bands = sorted(occupied_bands)
        flat = chain((0 - 1,), chain.from_iterable(occupied_bands), (100 + 1,))
        unoccupied_bands = [((x + 1) / 100 - 0.5, (y - 1) / 100 - 0.5) for x, y in zip(flat, flat) if x + 6 < y]
        if len(unoccupied_bands) == 0:
            return signal

        max_bandwidth = min([y - x for x, y in unoccupied_bands])
        bandwidth = np.random.uniform(0.05, max_bandwidth)
        center_freqs: List[Tuple[float, float]] = [(x + bandwidth / 2, y - bandwidth / 2) for x, y in unoccupied_bands]
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

        num_iq_samples = signal["data"]["samples"].shape[0]
        signal_sources = [
            SyntheticBurstSourceDataset(
                bandwidths=bandwidth,
                center_frequencies=center_freq,
                burst_durations=burst_duration,
                silence_durations=silence_duration,
                snrs_db=20,
                seed=self.seed,
                start=(0.0, 0.95),
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

        iq_data, metadata = signal_dataset[0]
        signal["data"]["samples"] += iq_data

        # Update the SignalMetadata
        signal["metadata"].extend(metadata)  # type: ignore
        return signal
