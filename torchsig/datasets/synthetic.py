"""Synthetic Dataset Generation Tools
"""

from torchsig.utils.types import SignalData, SignalMetadata, Signal, ModulatedRFMetadata
from torchsig.utils.types import (
    create_signal_metadata,
    create_rf_metadata,
    create_modulated_rf_metadata,
)
from torchsig.utils.dsp import convolve, gaussian_taps, low_pass, rrc_taps, rational_rate_resampler
from torchsig.transforms.functional import FloatParameter, IntParameter
from torchsig.utils.dataset import SignalDataset
from torchsig.datasets.signal_classes import torchsig_signals
from torchsig.utils.dsp import estimate_filter_length, MAX_SIGNAL_UPPER_EDGE_FREQ, MAX_SIGNAL_LOWER_EDGE_FREQ
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.utils.data import ConcatDataset
from scipy import signal as sp
from collections import OrderedDict
import numpy as np
import itertools
import pickle


def remove_corners(const):
    spacing = 2.0 / (np.sqrt(len(const)) - 1)
    cutoff = spacing * (np.sqrt(len(const)) / 6 - 0.5)
    return [
        p
        for p in const
        if np.abs(np.real(p)) < 1.0 - cutoff or np.abs(np.imag(p)) < 1.0 - cutoff
    ]


default_const_map = OrderedDict(
    {
        "ook": np.add(*map(np.ravel, np.meshgrid(np.linspace(0, 1, 2), 0j))),
        "bpsk": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 2), 0j))),
        "4pam": np.add(*map(np.ravel, np.meshgrid(np.linspace(0, 1, 4), 0j))),
        "4ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 4), 0j))),
        "qpsk": np.add(
            *map(
                np.ravel, np.meshgrid(np.linspace(-1, 1, 2), 1j * np.linspace(-1, 1, 2))
            )
        ),
        "8pam": np.add(*map(np.ravel, np.meshgrid(np.linspace(0, 1, 8), 0j))),
        "8ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 8), 0j))),
        "8psk": np.exp(2j * np.pi * np.linspace(0, 7, 8) / 8.0),
        "16qam": np.add(
            *map(
                np.ravel, np.meshgrid(np.linspace(-1, 1, 4), 1j * np.linspace(-1, 1, 4))
            )
        ),
        "16pam": np.add(*map(np.ravel, np.meshgrid(np.linspace(0, 1, 16), 0j))),
        "16ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 16), 0j))),
        "16psk": np.exp(2j * np.pi * np.linspace(0, 15, 16) / 16.0),
        "32qam": np.add(
            *map(
                np.ravel, np.meshgrid(np.linspace(-1, 1, 4), 1j * np.linspace(-1, 1, 8))
            )
        ),
        "32qam_cross": remove_corners(
            np.add(
                *map(
                    np.ravel,
                    np.meshgrid(np.linspace(-1, 1, 6), 1j * np.linspace(-1, 1, 6)),
                )
            )
        ),
        "32pam": np.add(*map(np.ravel, np.meshgrid(np.linspace(0, 1, 32), 0j))),
        "32ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 32), 0j))),
        "32psk": np.exp(2j * np.pi * np.linspace(0, 31, 32) / 32.0),
        "64qam": np.add(
            *map(
                np.ravel, np.meshgrid(np.linspace(-1, 1, 8), 1j * np.linspace(-1, 1, 8))
            )
        ),
        "64pam": np.add(*map(np.ravel, np.meshgrid(np.linspace(0, 1, 64), 0j))),
        "64ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 64), 0j))),
        "64psk": np.exp(2j * np.pi * np.linspace(0, 63, 64) / 64.0),
        "128qam_cross": remove_corners(
            np.add(
                *map(
                    np.ravel,
                    np.meshgrid(np.linspace(-1, 1, 12), 1j * np.linspace(-1, 1, 12)),
                )
            )
        ),
        "256qam": np.add(
            *map(
                np.ravel,
                np.meshgrid(np.linspace(-1, 1, 16), 1j * np.linspace(-1, 1, 16)),
            )
        ),
        "512qam_cross": remove_corners(
            np.add(
                *map(
                    np.ravel,
                    np.meshgrid(np.linspace(-1, 1, 24), 1j * np.linspace(-1, 1, 24)),
                )
            )
        ),
        "1024qam": np.add(
            *map(
                np.ravel,
                np.meshgrid(np.linspace(-1, 1, 32), 1j * np.linspace(-1, 1, 32)),
            )
        ),
    }
)


class ModulateNarrowbandDataset(ConcatDataset):
    """Digital Modulation Dataset

    Args:
        modulations (:obj:`list` or :obj:`tuple`):
            Sequence of strings representing the constellations that should be included.

        num_iq_samples (:obj:`int`):
            number of samples to read from each file in the database

        num_samples_per_class (:obj:`int`):
            number of samples to be kept for each class

        iq_samples_per_symbol (:obj:`Optional[int]`):
            number of IQ samples per symbol

        random_data (:obj:`bool`):
            whether the modulated binary utils should be random each time, or seeded by index

        random_pulse_shaping (:obj:`bool`):
            boolean to enable/disable randomized pulse shaping

    """

    def __init__(
        self,
        modulations: Optional[Union[List, Tuple]] = torchsig_signals.class_list,
        num_iq_samples: int = 100,
        num_samples_per_class: int = 100,
        iq_samples_per_symbol: Optional[int] = None,
        random_data: bool = False,
        random_pulse_shaping: bool = False,
        **kwargs,
    ) -> None:
        modulations = (
            torchsig_signals.class_list
            if modulations is None
            else modulations
        )

        constellation_list = [m for m in map(str.lower, modulations) if m in torchsig_signals.constellation_signals]
        fsk_list = [m for m in map(str.lower, modulations) if m in torchsig_signals.fsk_signals]
        fm_list = [m for m in map(str.lower, modulations) if m in torchsig_signals.fm_signals]
        am_list = [m for m in map(str.lower, modulations) if m in torchsig_signals.am_signals]
        lfm_list = [m for m in map(str.lower, modulations) if m in torchsig_signals.lfm_signals]
        chirpss_list = [m for m in map(str.lower, modulations) if m in torchsig_signals.chirpss_signals]
        cw_list = [m for m in map(str.lower, modulations) if m in torchsig_signals.cw_signals]

        const_dataset = ConstellationDataset(
            constellations=constellation_list,
            num_iq_samples=num_iq_samples,
            num_samples_per_class=num_samples_per_class,
            iq_samples_per_symbol=2
            if iq_samples_per_symbol is None
            else iq_samples_per_symbol,
            random_data=random_data,
            random_pulse_shaping=random_pulse_shaping,
            **kwargs,
        )

        fsk_dataset = FSKDataset(
            modulations=fsk_list,
            num_iq_samples=num_iq_samples,
            num_samples_per_class=num_samples_per_class,
            iq_samples_per_symbol=8,
            random_data=random_data,
            random_pulse_shaping=random_pulse_shaping,
            **kwargs,
        )

        fm_dataset = FMDataset(
            num_iq_samples=num_iq_samples,
            num_samples_per_class=num_samples_per_class,
            random_data=random_data,
            **kwargs,
        )

        am_dataset = AMDataset(
            num_iq_samples=num_iq_samples,
            num_samples_per_class=num_samples_per_class,
            random_data=random_data,
            **kwargs,
        )

        lfm_dataset = LFMDataset(
            num_iq_samples=num_iq_samples,
            num_samples_per_class=num_samples_per_class,
            random_data=random_data,
            **kwargs,
        )

        chirpss_dataset = ChirpSSDataset(
            num_iq_samples=num_iq_samples,
            num_samples_per_class=num_samples_per_class,
            random_data=random_data,
            **kwargs,
        )
        cw_dataset = CarrierWaveSpikeDataset(            
            num_iq_samples=num_iq_samples,
            num_samples_per_class=num_samples_per_class,
            random_data=random_data,
            **kwargs,
        )

        super(ModulateNarrowbandDataset, self).__init__([const_dataset, fsk_dataset, fm_dataset, am_dataset, lfm_dataset, chirpss_dataset, cw_dataset])

class SyntheticDataset(SignalDataset):
    def __init__(self, **kwargs) -> None:
        super(SyntheticDataset, self).__init__(**kwargs)
        self.index: List[Tuple[Any, ...]] = []

    def __getitem__(self, index: int) -> Tuple[Union[SignalData, np.ndarray], Any]:
        signal_meta = self.index[index][-1]
        signal_data = SignalData(samples=self._generate_samples(self.index[index]))
        signal = Signal(data=signal_data, metadata=signal_meta)

        if self.transform:
            signal = self.transform(signal)

        target = signal["metadata"]

        if self.target_transform:
            target = self.target_transform(signal["metadata"])

        return signal["data"]["samples"], target

    def __len__(self) -> int:
        return len(self.index)

    def _generate_samples(self, item: Tuple) -> np.ndarray:
        raise NotImplementedError

def ConstellationBasebandModulator ( class_name, excess_bandwidth, iq_samples_per_symbol, num_iq_samples ):

    # get the constellation maps
    const_map = default_const_map
    # normalize the constellation map to unit energy
    const = const_map[class_name] / np.mean(np.abs(const_map[class_name]))
    # compute the symbols to index into the symbol map
    symbol_nums = np.random.randint(0, len(const), int(num_iq_samples / iq_samples_per_symbol))
    # compute symbols
    symbols = const[symbol_nums]
    # zero-pad the symbols
    zero_padded = np.zeros((iq_samples_per_symbol * len(symbols),), dtype=np.complex64)
    zero_padded[::iq_samples_per_symbol] = symbols
    # design the pulse shaping filter:
    #   excess bandwidth is defined in porportion to signal bandwidth, not sampling rate,
    #   thus needs to be scaled by the samples per symbol
    pulse_shape_filter_length = estimate_filter_length(excess_bandwidth / iq_samples_per_symbol)
    pulse_shape_filter_span = int((pulse_shape_filter_length - 1) / (2*iq_samples_per_symbol))  # convert filter length into the span
    pulse_shape_filter = rrc_taps(iq_samples_per_symbol, pulse_shape_filter_span, excess_bandwidth,)
    # apply pulse shaping filter 
    filtered = sp.convolve(zero_padded, pulse_shape_filter, 'full')
    # remove transition periods
    lidx = (len(filtered) - num_iq_samples) // 2
    ridx = lidx + num_iq_samples
    filtered = filtered[lidx:ridx]
    return filtered


class ConstellationDataset(SyntheticDataset):
    """Constellation Dataset

    Args:
        constellations (:obj:`list` or :obj:`tuple`):
            Sequence of strings representing the constellations that should be included.

        num_iq_samples (:obj:`int`):
            number of samples to read from each file in the database

        num_samples_per_class (:obj:`int`):
            number of samples to be kept for each class

        iq_samples_per_symbol (:obj:`int`):
            number of IQ samples per symbol

        pulse_shape_filter (:obj:`int`):
            Pulse shape filter to apply to the up-sampled symbols. Default is RRC spanning 11 symbols.

        random_data (:obj:`bool`):
            whether the modulated binary utils should be random each time, or seeded by index

        center_freq (:obj:`float`):
            center frequency of the signal, will be upconverted internally

    """

    def __init__(
        self,
        constellations: Optional[Union[List, Tuple]] = torchsig_signals.constellation_signals,
        num_iq_samples: int = 100,
        num_samples_per_class: int = 100,
        iq_samples_per_symbol: int = 2,
        pulse_shape_filter: Optional[Union[bool, np.ndarray]] = None,
        random_pulse_shaping: bool = False,
        random_data: bool = False,
        center_freq: float = 0,
        **kwargs,
    ):
        super(ConstellationDataset, self).__init__(**kwargs)
        self.const_map: Dict[str, np.ndarray] = default_const_map
        self.constellations = (
            list(torchsig_signals.constellation_signals) if constellations is None else constellations
        )
        self.num_iq_samples = num_iq_samples
        self.iq_samples_per_symbol = iq_samples_per_symbol
        self.num_samples_per_class = num_samples_per_class
        self.random_pulse_shaping = random_pulse_shaping

        num_constellations = len(self.constellations)
        total_num_samples = int(num_constellations * self.num_samples_per_class)

        #if pulse_shape_filter is None:
        if 1:
            #if self.random_pulse_shaping:
            if 0:
                alphas = np.random.uniform(0.15, 0.6, size=total_num_samples)
            else:
                alphas = np.ones(total_num_samples) * 0.35
        else:
            self.pulse_shape_filter = pulse_shape_filter

        self.random_data = random_data
        self.index = []

        for const_idx, const_name in enumerate(map(str.lower, self.constellations)):
            for idx in range(self.num_samples_per_class):
                meta = create_modulated_rf_metadata(
                    num_samples=self.num_iq_samples,
                    bits_per_symbol=np.log2(len(self.const_map[const_name])),
                    samples_per_symbol=iq_samples_per_symbol,
                    class_name=const_name,
                    class_index=const_idx,
                    excess_bandwidth=alphas[int(const_idx * self.num_samples_per_class + idx)],
                    center_freq=center_freq
                )
                self.index.append(
                    (
                        const_name,
                        const_idx * self.num_samples_per_class + idx,
                        [meta],
                    )
                )

    def _generate_samples(self, item: Tuple) -> np.ndarray:
        class_name = item[0]
        index = item[1]
        meta = item[2][0]

        center_freq = meta["center_freq"]
        bandwidth = 1/self.iq_samples_per_symbol

        orig_state = np.random.get_state()
        if not self.random_data:
            np.random.seed(index)

        # apply baseband signal modulator
        filtered = ConstellationBasebandModulator ( class_name, meta["excess_bandwidth"], self.iq_samples_per_symbol, self.num_iq_samples )

        # apply frequency shifting
        filtered *= np.exp(2j*np.pi*center_freq*np.arange(0,len(filtered)))

        # determine the boundaries for where the signal currently resides.
        # these values are used to determine if aliasing has occured
        upperSignalEdge = center_freq + (bandwidth/2)
        lowerSignalEdge = center_freq - (bandwidth/2)

        # check to see if aliasing has occured due to upconversion. if so, then apply
        # a filter to minimize it
        if ( upperSignalEdge > 0.5 or lowerSignalEdge < -0.5):

            # the signal has overlaped either the -fs/2 or +fs/2 boundary and therefore
            # a BPF filter will be applied to attenuate the portion of the signal that
            # is overlapping the -fs/2 or +fs/2 boundary to minimize aliasing
            filtered = upconversionAntiAliasingFilter ( filtered, center_freq, bandwidth )

        if not self.random_data:
            np.random.set_state(orig_state)  # return numpy back to its previous state

        return filtered[0:self.num_iq_samples]


class OFDMDataset(SyntheticDataset):
    """OFDM Dataset

    Args:
        constellations (:obj:`list` or :obj:`tuple`):
            Sequence of strings representing the set of possible sub-carrier modulations. All sub-carriers are modulated
            with the same modulation or they are randomly modulated per subcarrier with equal probability.

        num_subcarriers (:obj:`list` or :obj:`tuple`):
            Sequence of ints that represent the different number of subcarriers to include.

        cyclic_prefix_ratio (:obj:`list` or :obj:`tuple`):
            Sequence of possible cyclic_prefix_ratios to use. The cyclic prefix length will be an
            element cyclic_prefix_ratio times an element of num_subcarriers

        num_iq_samples (:obj:`int`):
            number of samples to produce for each waveform

        num_samples_per_class (:obj:`int`):
            number of samples to be kept for each class

        random_data (:obj:`bool`):
            whether the modulated binary utils should be random each time, or seeded by index

        sidelobe_suppression_methods (:obj:`tuple`):
            Tuple of possible sidelobe suppression methods. The options are:
                - `none` ~ Perform no sidelobe suppression methods
                - `lpf` ~ Apply a static low pass filter to the OFDM signal
                - `rand_lpf` ~ Apply a low pass filter with a randomized cutoff frequency to the OFDM signal
                - `win_start` ~ Apply a windowing method starting at the symbol boundary
                - `win_center` ~ Apply a windowing method centered at the symbol boundary
            For more details on the windowing method options, please see:
            http://zone.ni.com/reference/en-XX/help/373725J-01/wlangen/windowing/

        dc_subcarrier (:obj:`tuple`):
            Tuple of possible DC subcarrier options:
                - `(on,)` ~ Always leave the DC subcarrier on
                - `(off,)` ~ Always turn the DC subcarrier off
                - `(on, off)` ~ Half with DC subcarrier on and half off

        time_varying_realism (:obj:`tuple`):
            Tuple of on/off/both options for adding time-varying realistic effects in the form of
            bursts, pilot carriers, and resource blocks. Options:
                - `(on,)` ~ Leave the time-varying effects on, with half under full bursty effects and half under partial
                - `(off,)` ~ Always leave the time-varying effects off
                - `(on, off)` ~ One third with full bursty effects, one third with partial, and one third off
                - `(full_bursty,)` ~ All signals are bursty with consistent pattern throughout
                - `(partial_bursty,)` ~ All signals are mixed with bursty and continuous regions
            Note: The partial bursty behavior occurs prior to time slicing, and as such, is more interesting in longer
            duration examples

        center_freq (:obj:`float`):
            center frequency of the signal, will be upconverted internally

        bandwidth (:obj:`float`):
            bandwidth of the signal, will be resampled internally

        transform (:obj:`Callable`, optional):
            A function/transform that takes in an IQ vector and returns a transformed version.

    """

    def __init__(
        self,
        constellations: Union[List, Tuple] = ("bpsk", "qpsk"),
        num_subcarriers: List[int] = [64, 128, 256, 512, 1024, 2048],
        cyclic_prefix_ratios: FloatParameter = (0.125, 0.25),
        num_iq_samples: int = 100,
        num_samples_per_class: int = 100,
        random_data: bool = False,
        sidelobe_suppression_methods: tuple = (
            "none",
            "lpf",
            "rand_lpf",
            "win_start",
            "win_center",
        ),
        dc_subcarrier: tuple = ("on", "off"),
        time_varying_realism: tuple = ("off",),
        center_freq: float = 0,
        bandwidth: float = 0.5,
        **kwargs,
    ):
        super(OFDMDataset, self).__init__(**kwargs)
        self.constellations = constellations
        self.num_iq_samples = num_iq_samples
        self.num_samples_per_class = num_samples_per_class
        self.random_data = random_data
        self.sidelobe_suppression_methods = sidelobe_suppression_methods
        self.index = []
        if "lpf" in sidelobe_suppression_methods:
            cutoff = 0.3
            self.taps = low_pass(cutoff=cutoff, transition_bandwidth=(0.5 - cutoff) / 4)

        # Precompute all possible random symbols for speed at sample generation
        self.random_symbols = []
        for const_name in self.constellations:
            const = default_const_map[const_name] / np.mean(
                np.abs(default_const_map[const_name])
            )
            self.random_symbols.append(const)

        subcarrier_modulation_types = ("fixed", "random")
        if "on" in time_varying_realism:
            if "off" in time_varying_realism:
                time_varying_realism = ("off", "full_bursty", "partial_bursty")
            else:
                time_varying_realism = ("full_bursty", "partial_bursty")
        combinations = list(
            itertools.product(
                constellations,
                subcarrier_modulation_types,
                cyclic_prefix_ratios,  # type: ignore
                sidelobe_suppression_methods,
                dc_subcarrier,
                time_varying_realism,
            )
        )

        for class_idx, num_subcarrier in enumerate(num_subcarriers):
            class_name = "ofdm-{}".format(num_subcarrier)
            for idx in range(self.num_samples_per_class):
                (
                    const_name,
                    mod_type,
                    cyclic_prefix_ratio,
                    sidelobe_suppression_method,
                    dc_subcarrier,
                    time_varying_realism,
                ) = combinations[np.random.randint(len(combinations))]
                meta = ModulatedRFMetadata(
                    sample_rate=0.0,
                    num_samples=self.num_iq_samples,
                    complex=True,
                    lower_freq=center_freq-(bandwidth/2),
                    upper_freq=center_freq+(bandwidth/2),
                    center_freq=center_freq,
                    bandwidth=bandwidth,
                    start=0.0,
                    stop=1.0,
                    duration=1.0,
                    snr=0.0,
                    bits_per_symbol=2.0,
                    samples_per_symbol=0.0,
                    class_name=class_name,
                    class_index=class_idx,
                    excess_bandwidth=0.0,
                )
                self.index.append(
                    (
                        class_name,
                        class_idx * self.num_samples_per_class + idx,
                        num_subcarrier,
                        cyclic_prefix_ratio * num_subcarrier,
                        const_name,
                        mod_type,
                        sidelobe_suppression_method,
                        dc_subcarrier,
                        time_varying_realism,
                        [meta],
                    )
                )

    def _generate_samples(self, item: Tuple) -> np.ndarray:
        index = item[1]
        num_subcarriers = item[2]
        cyclic_prefix_len = item[3]
        const_name = item[4]
        mod_type = item[5]
        sidelobe_suppression_method = item[6]
        dc_subcarrier = item[7]
        time_varying_realism = item[8]
        meta = item[9][0]
        center_freq = meta["center_freq"]
        bandwidth = meta["bandwidth"]

        orig_state = np.random.get_state()
        if not self.random_data:
            np.random.seed(index)

        if mod_type == "random":
            symbols_idxs = np.random.randint(0, 1024, size=self.num_iq_samples)
            const_idxes = np.random.choice(range(len(self.random_symbols)), size=num_subcarriers)
            symbols = np.zeros(self.num_iq_samples, dtype=np.complex128)
            for subcarrier_idx, const_idx in enumerate(const_idxes):
                begin_idx = (self.num_iq_samples) * subcarrier_idx
                end_idx = (self.num_iq_samples) * (subcarrier_idx + 1)
                symbols[begin_idx:end_idx] = self.random_symbols[const_idx][
                    np.mod(
                        symbols_idxs[begin_idx:end_idx],
                        len(self.random_symbols[const_idx]),
                    )
                ]
        else:
            # Fixed modulation across all subcarriers
            const_name = np.random.choice(self.constellations)
            const = default_const_map[const_name] / np.mean(
                np.abs(default_const_map[const_name])
            )
            symbol_nums = np.random.randint(0, len(const), int(self.num_iq_samples))
            symbols = const[symbol_nums]
        divisible_index = -(len(symbols) % num_subcarriers)
        if divisible_index != 0:
            symbols = symbols[:divisible_index]

        # now sub-carrier modulate
        serial_to_parallel = symbols.reshape(num_subcarriers, -1)
        zero_pad = np.pad(
            serial_to_parallel,
            ((int(num_subcarriers / 2), int(num_subcarriers / 2)), (0, 0)),
            "constant",
            constant_values=0,
        )

        # Turn off DC subcarrier
        if dc_subcarrier == "off":
            dc_center = int(zero_pad.shape[0] // 2)
            zero_pad[dc_center, :] = np.zeros((zero_pad.shape[1]))

        # Add time-varying realism with randomized bursts, pilots, and resource blocks
        burst_dur = 1
        original_on = False
        if (
            time_varying_realism == "full_bursty"
            or time_varying_realism == "partial_bursty"
        ):
            # Bursty
            if time_varying_realism == "full_bursty":
                burst_region_start = 0.0
                burst_region_stop = zero_pad.shape[1]
            else:
                burst_region_start = np.random.uniform(0.0, 0.9)
                burst_region_dur = min(1.0 - burst_region_start, np.random.uniform(0.25, 1.0))
                burst_region_start = int(burst_region_start * zero_pad.shape[1] // 4)
                burst_region_dur = int(burst_region_dur * zero_pad.shape[1] // 4)
                burst_region_stop = burst_region_start + burst_region_dur
            # bursty = deepcopy(zero_pad)
            bursty = pickle.loads(
                pickle.dumps(zero_pad, -1)
            )  # no random hangs like deepcopy

            burst_dur = np.random.choice([1, 2, 4])
            original_on = True if np.random.rand() <= 0.5 else False
            for subcarrier_idx in range(bursty.shape[0]):
                on = original_on
                for time_idx in range(bursty.shape[1]):
                    if time_idx % burst_dur == 0:
                        on = not on
                    if (not on) and (
                        time_idx >= burst_region_start and time_idx <= burst_region_stop
                    ):
                        bursty[subcarrier_idx, time_idx] = 0 + 1j * 0

            # Pilots
            min_num_pilots = 4
            max_num_pilots = int(num_subcarriers // 8)
            num_pilots = np.random.randint(min_num_pilots, max_num_pilots)
            pilot_indices = np.random.choice(
                range(num_subcarriers), num_pilots, replace=False
            )
            bursty[pilot_indices + num_subcarriers // 2, :] = zero_pad[pilot_indices + num_subcarriers // 2, :]

            # Resource blocks
            min_num_blocks = 2
            max_num_blocks = 16
            num_blocks = np.random.randint(min_num_blocks, max_num_blocks)
            for _ in range(num_blocks):
                block_start = np.random.uniform(0.0, 0.9)
                block_dur = np.random.uniform(0.05, 1.0 - block_start)
                block_start = int(block_start * zero_pad.shape[1])
                block_dur = int(block_dur * zero_pad.shape[1] // 4)
                block_stop = block_start + block_dur

                block_low_carrier = np.random.randint(0, num_subcarriers - 4)
                block_num_carriers = np.random.randint(1, num_subcarriers // 8)
                block_high_carrier = min(block_low_carrier + block_num_carriers, num_subcarriers)

                bursty[
                    block_low_carrier
                    + num_subcarriers // 2 : block_high_carrier
                    + num_subcarriers // 2,
                    block_start:block_stop,
                ] = zero_pad[
                    block_low_carrier
                    + num_subcarriers // 2 : block_high_carrier
                    + num_subcarriers // 2,
                    block_start:block_stop,
                ]
            zero_pad = bursty

        ofdm_symbols = np.fft.ifft(np.fft.ifftshift(zero_pad, axes=0), axis=0)
        symbol_dur = ofdm_symbols.shape[0]
        cyclic_prefixed = np.pad(ofdm_symbols, ((int(cyclic_prefix_len), 0), (0, 0)), "wrap")

        if sidelobe_suppression_method == "none":
            output = cyclic_prefixed.T.flatten()

        elif sidelobe_suppression_method == "lpf":
            flattened = cyclic_prefixed.T.flatten()
            # Apply pre-computed LPF
            output = convolve(flattened, self.taps)  #[:-50]

        elif sidelobe_suppression_method == "rand_lpf":
            flattened = cyclic_prefixed.T.flatten()
            # Generate randomized LPF
            cutoff = np.random.uniform(0.25, 0.475)
            taps = low_pass(cutoff=cutoff, transition_bandwidth=(0.5 - cutoff) / 4)
            # Apply random LPF
            output = convolve(flattened, taps) #[: -len(taps)]
        else:
            # Apply appropriate windowing technique
            window_len = cyclic_prefix_len
            half_window_len = int(window_len / 2)
            if sidelobe_suppression_method == "win_center":
                windowed = np.pad(
                    cyclic_prefixed,
                    ((half_window_len, half_window_len), (0, 0)),
                    "constant",
                    constant_values=0,
                )
                windowed[-half_window_len:, :] = windowed[
                    int(half_window_len)
                    + int(cyclic_prefix_len) : int(half_window_len)
                    + int(cyclic_prefix_len)
                    + int(half_window_len),
                    :,
                ]
                windowed[:half_window_len, :] = windowed[
                    int(half_window_len)
                    + int(cyclic_prefix_len)
                    + int(symbol_dur) : int(half_window_len)
                    + int(cyclic_prefix_len)
                    + int(symbol_dur)
                    + int(half_window_len),
                    :,
                ]
            elif sidelobe_suppression_method == "win_start":
                windowed = np.pad(
                    cyclic_prefixed,
                    ((0, int(window_len)), (0, 0)),
                    "constant",
                    constant_values=0,
                )
                windowed[-int(window_len) :, :] = windowed[
                    int(cyclic_prefix_len) : int(cyclic_prefix_len) + int(window_len), :
                ]
            else:
                raise ValueError(
                    "Expected window method to be: none, win_center, or win_start. Received: {}".format(
                        self.sidelobe_suppression_methods
                    )
                )

            # window the tails
            front_window = np.blackman(int(window_len * 2))[: int(window_len)].reshape(
                -1, 1
            )
            tail_window = np.blackman(int(window_len * 2))[-int(window_len) :].reshape(
                -1, 1
            )
            windowed[: int(window_len), :] = (
                front_window * windowed[: int(window_len), :]
            )
            windowed[-int(window_len) :, :] = (
                tail_window * windowed[-int(window_len) :, :]
            )

            combined = np.zeros((windowed.shape[0] * windowed.shape[1],), dtype=complex)
            start_idx: int = 0
            for symbol_idx in range(windowed.shape[1]):
                combined[start_idx : start_idx + windowed.shape[0]] += windowed[
                    :, symbol_idx
                ]
                start_idx += int(symbol_dur) + int(window_len)
            output = combined[: int(cyclic_prefixed.shape[0] * cyclic_prefixed.shape[1])]

        # Randomize the start index (while bypassing the initial windowing if present)
        if num_subcarriers * 4 * burst_dur < self.num_iq_samples:
            start_idx = np.random.randint(0, output.shape[0] - self.num_iq_samples)
        else:
            if "win" in sidelobe_suppression_method:
                start_idx = np.random.randint(
                    window_len, int(symbol_dur * burst_dur) + window_len
                )
            else:
                start_idx = np.random.randint(0, int(symbol_dur * burst_dur))
            # if original_on:
            #     lower: int = int(
            #         max(0, int(symbol_dur * burst_dur) - self.num_iq_samples * 0.7)
            #     )
            #     upper: int = int(
            #         min(
            #             int(symbol_dur * burst_dur),
            #             output.shape[0] - self.num_iq_samples,
            #         )
            #     )
            #     start_idx = np.random.randint(lower, upper)
            # elif "win" in sidelobe_suppression_method:
            #     start_idx = np.random.randint(
            #         window_len, int(symbol_dur * burst_dur) + window_len
            #     )
            # else:
            #     start_idx = np.random.randint(0, int(symbol_dur * burst_dur))


        # determine the fine resampling rate required after signal is modulated,
        # OFDM modulator uses a nominal oversampling rate of 2.
        resamplerRate = (1/2)/bandwidth

        # apply resampling
        output = rational_rate_resampler ( output, resamplerRate )

        # apply frequency shifting
        output *= np.exp(2j*np.pi*center_freq*np.arange(0,len(output)))

        # determine the boundaries for where the signal currently resides.
        # these values are used to determine if aliasing has occured
        upperSignalEdge = center_freq + (bandwidth/2)
        lowerSignalEdge = center_freq - (bandwidth/2)

        # check to see if aliasing has occured due to upconversion. if so, then apply
        # a filter to minimize it
        if ( upperSignalEdge > 0.5 or lowerSignalEdge < -0.5):

            # the signal has overlaped either the -fs/2 or +fs/2 boundary and therefore
            # a BPF filter will be applied to attenuate the portion of the signal that
            # is overlapping the -fs/2 or +fs/2 boundary to minimize aliasing
            output = upconversionAntiAliasingFilter ( output, center_freq, bandwidth )

        if not self.random_data:
            np.random.set_state(orig_state)  # return numpy back to its previous state

        return output[0:self.num_iq_samples]


def getFSKFreqMap ( ):
    freq_map = OrderedDict(
        {
            "2fsk": np.linspace(-1 + (1 / 2), 1 - (1 / 2), 2, endpoint=True),
            "2gfsk": np.linspace(-1 + (1 / 2), 1 - (1 / 2), 2, endpoint=True),
            "2msk": np.linspace(-1 + (1 / 2), 1 - (1 / 2), 2, endpoint=True),
            "2gmsk": np.linspace(-1 + (1 / 2), 1 - (1 / 2), 2, endpoint=True),
            "4fsk": np.linspace(-1 + (1 / 4), 1 - (1 / 4), 4, endpoint=True),
            "4gfsk": np.linspace(-1 + (1 / 4), 1 - (1 / 4), 4, endpoint=True),
            "4msk": np.linspace(-1 + (1 / 4), 1 - (1 / 4), 4, endpoint=True),
            "4gmsk": np.linspace(-1 + (1 / 4), 1 - (1 / 4), 4, endpoint=True),
            "8fsk": np.linspace(-1 + (1 / 8), 1 - (1 / 8), 8, endpoint=True),
            "8gfsk": np.linspace(-1 + (1 / 8), 1 - (1 / 8), 8, endpoint=True),
            "8msk": np.linspace(-1 + (1 / 8), 1 - (1 / 8), 8, endpoint=True),
            "8gmsk": np.linspace(-1 + (1 / 8), 1 - (1 / 8), 8, endpoint=True),
            "16fsk": np.linspace(-1 + (1 / 16), 1 - (1 / 16), 16, endpoint=True),
            "16gfsk": np.linspace(-1 + (1 / 16), 1 - (1 / 16), 16, endpoint=True),
            "16msk": np.linspace(-1 + (1 / 16), 1 - (1 / 16), 16, endpoint=True),
            "16gmsk": np.linspace(-1 + (1 / 16), 1 - (1 / 16), 16, endpoint=True),
        }
    )
    return freq_map

def getFSKModIndex( const_name ):
    # returns the modulation index based on the modulation
    if "gfsk" in const_name:
        # bluetooth
        mod_idx = 0.32
    elif "msk" in const_name:
        # MSK, GMSK
        mod_idx = 0.5
    else: # FSK
        # 50% chance to use mod index of 1 (orthogonal) ...
        if (np.random.uniform(0,1) < 0.5):
            mod_idx = 1
        else: # ... or something else (non-orthogonal)
            mod_idx = np.random.uniform(0.7,1)
    return mod_idx


def FSKBasebandModulator ( const_name, mod_idx, oversampling_rate, num_iq_samples ):

    # get the FSK frequency symbol map
    freq_map = getFSKFreqMap()

    # get the constellation to modulate
    const = freq_map[const_name]

    # calculate the modulation order, ex: the "4" in "4-FSK"
    mod_order = len(const)

    # determine how many samples are in each symbol
    samples_per_symbol_recalculated = int(mod_order * oversampling_rate)

    # scale the frequency map by the oversampling rate such that the tones
    # are packed tighter around f=0 the larger the oversampling rate
    const_oversampled = const / oversampling_rate

    # calculate the indexes into symbol table
    symbol_nums = np.random.randint(0, len(const_oversampled), int(np.ceil((num_iq_samples / samples_per_symbol_recalculated) * oversampling_rate)))

    # produce data symbols
    symbols = const_oversampled[symbol_nums]

    # rectangular pulse shape
    pulse_shape = np.ones(samples_per_symbol_recalculated)

    if "g" in const_name: # GMSK, GFSK
        # design the gaussian pulse shape with the bandwidth as dictated by the
        # oversampling rate, which will then be fine-tuned into the proper 'bandwidth'
        # by the resampling stage
        preresample_bandwidth = 1/oversampling_rate
        taps = gaussian_taps(samples_per_symbol_recalculated, preresample_bandwidth)
        pulse_shape = np.convolve(taps,pulse_shape)

    # upsample symbols and apply pulse shaping
    filtered = sp.upfirdn(pulse_shape,symbols,up=samples_per_symbol_recalculated,down=1)

    # insert a zero at first sample to start at zero phase
    filtered = np.insert(filtered, 0, 0)

    phase = np.cumsum(np.array(filtered) * 1j * mod_idx * np.pi)
    modulated = np.exp(phase)

    return modulated



class FSKDataset(SyntheticDataset):
    """FSK Dataset

    Args:
        modulations (:obj:`list` or :obj:`tuple`):
            Sequence of strings representing the modulations that should be included

        num_iq_samples (:obj:`int`):
            number of samples to read from each file in the database

        num_samples_per_class (:obj:`int`):
            number of samples to be kept for each class

        iq_samples_per_symbol (:obj:`int`):
            number of IQ samples per symbol

        random_data (:obj:`bool`):
            whether the modulated binary utils should be random each time, or seeded by index

        center_freq (:obj:`float`):
            center frequency of the signal, will be upconverted internally

        bandwidth (:obj:`float`):
            bandwidth of the signal, will be resampled internally

        transform (:obj:`Callable`, optional):
            A function/transform that takes in an IQ vector and returns a transformed version.

    """

    def __init__(
        self,
        modulations: Optional[Union[List, Tuple]] = torchsig_signals.fsk_signals,
        num_iq_samples: int = 100,
        num_samples_per_class: int = 100,
        iq_samples_per_symbol: int = 2,
        random_data: bool = False,
        random_pulse_shaping: bool = False,
        center_freq: float = 0,
        bandwidth: float = 0.5,
        **kwargs,
    ):
        super(FSKDataset, self).__init__(**kwargs)
        self.modulations = list(torchsig_signals.fsk_signals) if modulations is None else modulations
        self.num_iq_samples = num_iq_samples
        self.num_samples_per_class = num_samples_per_class
        self.iq_samples_per_symbol = iq_samples_per_symbol
        self.random_data = random_data
        self.random_pulse_shaping = random_pulse_shaping
        self.index = []
        self.freq_map = getFSKFreqMap() # TODO: this needs to be removed

        for freq_idx, freq_name in enumerate(map(str.lower, self.modulations)):
            for idx in range(self.num_samples_per_class):
                # modulation index scales the bandwidth of the signal, and
                # iq_samples_per_symbol is used as an oversampling rate in
                # FSKDataset class, therefore the signal bandwidth can be
                # approximated by mod_idx/iq_samples_per_symbol.
                #mod_idx = self._mod_index(freq_name)
                #bandwidth_cutoff = mod_idx / self.iq_samples_per_symbol
                #bandwidth = np.random.uniform(bandwidth_cutoff, 0.5 - bandwidth_cutoff) #if self.random_pulse_shaping else 0.0
                meta = ModulatedRFMetadata(
                    sample_rate=0.0,
                    num_samples=float(self.num_iq_samples),
                    complex=True,
                    lower_freq=center_freq-(bandwidth/2),
                    upper_freq=center_freq+(bandwidth/2),
                    center_freq=center_freq,
                    bandwidth=bandwidth,
                    start=0.0,
                    stop=1.0,
                    duration=1.0,
                    snr=0.0,
                    bits_per_symbol=np.log2(len(self.freq_map[freq_name])), # TODO: this needs to be removed
                    samples_per_symbol=float(iq_samples_per_symbol),
                    class_name=freq_name,
                    class_index=freq_idx,
                    excess_bandwidth=0,
                )
                self.index.append((freq_name, freq_idx * self.num_samples_per_class + idx, [meta])
                )

    def _generate_samples(self, item: Tuple) -> np.ndarray:
        const_name = item[0]
        index = item[1]
        metadata = item[2][0]
        center_freq = metadata["center_freq"]
        bandwidth = metadata["bandwidth"]

        # samples per symbol presumably used as a bandwidth measure (ex: BW=1/SPS),
        # but does not work for FSK. samples per symbol is redefined into
        # the "oversampling rate", and samples per symbol is instead derived
        # from the modulation order
        oversampling_rate = np.copy(self.iq_samples_per_symbol)

        # control RNG
        orig_state = np.random.get_state()
        if not self.random_data:
            np.random.seed(index)

        # determine modulation index
        mod_idx = getFSKModIndex(const_name)

        # modulate the FSK signal at complex baseband
        modulated = FSKBasebandModulator ( const_name, mod_idx, oversampling_rate, self.num_iq_samples )

        if self.random_pulse_shaping:
            taps = low_pass(cutoff=bandwidth / 2, transition_bandwidth=(0.5 - bandwidth / 2) / 4)
            # apply the filter
            modulated = convolve(modulated, taps)

        # calculate the resampling rate to convert from the oversampling rate specified by
        # self.iq_samples_per_symbol into the proper bandwidth
        resampleRate = bandwidth*mod_idx/(1/oversampling_rate)

        # apply resampling
        modulated = rational_rate_resampler ( modulated, resampleRate )

        # apply center frequency shifting
        modulated *= np.exp(2j*np.pi*center_freq*np.arange(0,len(modulated)))

        # determine the boundaries for where the signal currently resides.
        # these values are used to determine if aliasing has occured
        upperSignalEdge = center_freq + (bandwidth/2)
        lowerSignalEdge = center_freq - (bandwidth/2)

        # check to see if aliasing has occured due to upconversion. if so, then apply
        # a filter to minimize it
        if ( upperSignalEdge > 0.5 or lowerSignalEdge < -0.5):

            # the signal has overlaped either the -fs/2 or +fs/2 boundary and therefore
            # a BPF filter will be applied to attenuate the portion of the signal that
            # is overlapping the -fs/2 or +fs/2 boundary to minimize aliasing
            modulated = upconversionAntiAliasingFilter ( modulated, center_freq, bandwidth )

        if not self.random_data:
            np.random.set_state(orig_state)  # return numpy back to its previous state

        return modulated[0:self.num_iq_samples]




class AMDataset(SyntheticDataset):
    """AM Dataset

    Args:
        transform (:obj:`Callable`, optional):
            A function/transform that takes in an IQ vector and returns a transformed version.

    """

    def __init__(
        self,
        modulations: Optional[Union[List, Tuple]] = torchsig_signals.am_signals,
        num_iq_samples: int = 100,
        num_samples_per_class: int = 100,
        random_data: bool = False,
        center_freq: float = 0,
        bandwidth: float = 0.5,
        **kwargs,
    ):
        super(AMDataset, self).__init__(**kwargs)
        self.num_iq_samples = num_iq_samples
        self.num_samples_per_class = num_samples_per_class
        self.modulations = modulations
        self.random_data = random_data
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self.index = []

        for class_idx, class_name in enumerate(self.modulations):
            meta = ModulatedRFMetadata(
                sample_rate=0.0,
                num_samples=self.num_iq_samples,
                complex=True,
                lower_freq=-0.25,
                upper_freq=0.25,
                center_freq=0.0,
                bandwidth=0.5,
                start=0.0,
                stop=1.0,
                duration=1.0,
                snr=0.0,
                bits_per_symbol=0.0,
                samples_per_symbol=0.0,
                class_name=class_name,
                class_index=class_idx,
                excess_bandwidth=0.0,
            )
            for idx in range(self.num_samples_per_class):
                self.index.append(
                    (
                        class_name,
                        class_idx * self.num_samples_per_class + idx,
                        [meta],
                    )
                )

    def __len__(self) -> int:
        return len(self.index)

    def _generate_samples(self, item: Tuple) -> np.ndarray:
        const_name = item[0]
        index = item[1]
        orig_state = np.random.get_state()
        if not self.random_data:
            np.random.seed(index)

        if ("lsb" in const_name or "usb" in const_name):
            num_samples = 2*self.num_iq_samples
        else:
            num_samples = self.num_iq_samples

        # generate the random message
        message = np.random.randn(num_samples) + 0j
        # generate bandwidth-limiting LPF
        LPF = low_pass(cutoff=self.bandwidth/2, transition_bandwidth=self.bandwidth/4)
        # scale LPF in order to increase power due to balance reduction in bandwidth
        LPF *= 1/self.bandwidth
        # apply bandwidth-limiting filter
        shapedMessage = sp.convolve(message, LPF, "full")
        # remove transients
        lidx = (len(shapedMessage) - num_samples) // 2
        ridx = lidx + num_samples
        shapedMessage = shapedMessage[lidx:ridx]
        if (const_name == "am-dsb-sc"):
            basebandSignal = shapedMessage
        elif (const_name == "am-dsb"):
            # build carrier
            carrier = np.ones(len(shapedMessage))
            # randomly determine modulation index
            modulationIndex = np.random.uniform(0.1,1)
            basebandSignal = (modulationIndex*shapedMessage) + carrier
        elif (const_name == "am-lsb"):
            # upconvert signal to bandwidth/2
            LSBMixer = np.exp(2j*np.pi*(self.bandwidth/2)*np.arange(0,len(shapedMessage)))
            DSBUpconverted = LSBMixer*shapedMessage
            # the existing BW limiting filter can be be repurposed to discard upper band
            LSBSignalAtIF = np.convolve(DSBUpconverted,LPF)
            # remove transients
            lidx = (len(LSBSignalAtIF) - num_samples) // 2
            ridx = lidx + num_samples
            LSBSignalAtIF = LSBSignalAtIF[lidx:ridx]
            # mix LSB back down to baseband
            basebandSignalOversampled = LSBSignalAtIF*np.exp(-2j*np.pi*(self.bandwidth/4)*np.arange(0,len(LSBSignalAtIF)))
            # since threw away 1/2 the bandwidth to only retain LSB, then downsample by 2 in order to match
            # the requested self.bandwidth
            basebandSignal = rational_rate_resampler ( basebandSignalOversampled, resampler_rate=0.5 )
            basebandSignal = basebandSignal[0:self.num_iq_samples]
        elif (const_name == "am-usb"):
            # downconvert signal to -bandwidth/2
            USBMixer = np.exp(-2j*np.pi*(self.bandwidth/2)*np.arange(0,len(shapedMessage)))
            DSBDownconverted = USBMixer*shapedMessage
            # the existing BW limiting filter can be be repurposed to discard upper band
            USBSignalAtIF = np.convolve(DSBDownconverted,LPF)
            # remove transients
            lidx = (len(USBSignalAtIF) - num_samples) // 2
            ridx = lidx + num_samples
            USBSignalAtIF = USBSignalAtIF[lidx:ridx]
            # mix USB back up to baseband
            basebandSignalOversampled = USBSignalAtIF*np.exp(2j*np.pi*(self.bandwidth/4)*np.arange(0,len(USBSignalAtIF)))
            # since threw away 1/2 the bandwidth to only retain USB, then downsample by 2 in order to match
            # the requested self.bandwidth
            basebandSignal = rational_rate_resampler ( basebandSignalOversampled, resampler_rate=0.5 )
            basebandSignal = basebandSignal[0:self.num_iq_samples]

        # generate mixer
        mixer = np.exp(2j * np.pi * self.center_freq * np.arange(self.num_iq_samples))
        # apply upconversion to center frequency
        modulated = mixer*basebandSignal

        # determine the boundaries for where the signal currently resides.
        # these values are used to determine if aliasing has occured
        upperSignalEdge = self.center_freq + (self.bandwidth/2)
        lowerSignalEdge = self.center_freq - (self.bandwidth/2)

        # check to see if aliasing has occured due to upconversion. if so, then apply
        # a filter to minimize it
        if ( upperSignalEdge > 0.5 or lowerSignalEdge < -0.5):

            # the signal has overlaped either the -fs/2 or +fs/2 boundary and therefore
            # a BPF filter will be applied to attenuate the portion of the signal that
            # is overlapping the -fs/2 or +fs/2 boundary to minimize aliasing
            modulated = upconversionAntiAliasingFilter ( modulated, self.center_freq, self.bandwidth )

        if not self.random_data:
            np.random.set_state(orig_state)  # return numpy back to its previous state

        return modulated[0:self.num_iq_samples]


class FMDataset(SyntheticDataset):
    """FM Dataset

    Args:
        transform (:obj:`Callable`, optional):
            A function/transform that takes in an IQ vector and returns a transformed version.

    """

    def __init__(
        self,
        num_iq_samples: int = 100,
        num_samples_per_class: int = 100,
        random_data: bool = False,
        center_freq: float = 0,
        bandwidth: float = 0.5,
        **kwargs,
    ):
        super(FMDataset, self).__init__(**kwargs)
        self.num_iq_samples = num_iq_samples
        self.num_samples_per_class = num_samples_per_class
        self.classes = torchsig_signals.fm_signals
        self.random_data = random_data
        self.index = []
        self.center_freq = center_freq
        self.bandwidth = bandwidth

        for class_idx, class_name in enumerate(self.classes):
            meta = ModulatedRFMetadata(
                sample_rate=0.0,
                num_samples=self.num_iq_samples,
                complex=True,
                lower_freq=center_freq-(bandwidth/2),
                upper_freq=center_freq+(bandwidth/2),
                center_freq=center_freq,
                bandwidth=bandwidth,
                start=0.0,
                stop=1.0,
                duration=1.0,
                snr=0.0,
                bits_per_symbol=0.0,
                samples_per_symbol=0.0,
                class_name=class_name,
                class_index=class_idx,
                excess_bandwidth=0.0,
            )
            for idx in range(self.num_samples_per_class):
                self.index.append(
                    (
                        class_name,
                        class_idx * self.num_samples_per_class + idx,
                        [meta],
                    )
                )

    def __len__(self) -> int:
        return len(self.index)

    def _generate_samples(self, item: Tuple) -> np.ndarray:
        # class_name = item[0]
        index = item[1]
        meta = item[2]
        orig_state = np.random.get_state()
        if not self.random_data:
            np.random.seed(index)

        # randomly determine modulation index
        mod_index = np.random.uniform(1,10)
        # calculate the frequency deviation using Carson's Rule
        fdev = (self.bandwidth/2)/(1 + (1/mod_index))
        # calculate the maximum deviation
        fmax = fdev/mod_index
        # compute input message
        message = np.random.normal(0,1,self.num_iq_samples)
        # design LPF to limit frequencies based on fmax
        LPF = low_pass(cutoff=fmax,transition_bandwidth=fmax)
        # apply the LPF to noise to limit the bandwidth prior to modulation
        source = np.convolve(message,LPF)
        # normalize maximum amplitude to 1
        source = source/np.max(np.abs(source))
        # apply FM modulation
        modulated = np.exp(2j * np.pi * np.cumsum(source) * fdev)
        # frequency shift to center_freq
        modulated *= np.exp(2j*np.pi*self.center_freq*np.arange(0,len(modulated)))
        
        # determine the boundaries for where the signal currently resides.
        # these values are used to determine if aliasing has occured
        upperSignalEdge = self.center_freq + (self.bandwidth/2)
        lowerSignalEdge = self.center_freq - (self.bandwidth/2)
        
        # check to see if aliasing has occured due to upconversion. if so, then apply
        # a filter to minimize it
        if ( upperSignalEdge > 0.5 or lowerSignalEdge < -0.5):
        
            # the signal has overlaped either the -fs/2 or +fs/2 boundary and therefore
            # a BPF filter will be applied to attenuate the portion of the signal that
            # is overlapping the -fs/2 or +fs/2 boundary to minimize aliasing
            modulated = upconversionAntiAliasingFilter ( modulated, self.center_freq, self.bandwidth )

        if not self.random_data:
            np.random.set_state(orig_state)  # return numpy back to its previous state

        return modulated[0:self.num_iq_samples]


class ToneDataset(SyntheticDataset):
    """Tone Dataset

    Args:
        transform (:obj:`Callable`, optional):
            A function/transform that takes in an IQ vector and returns a transformed version.

    """

    def __init__(
        self,
        num_iq_samples: int = 100,
        num_samples_per_class: int = 100,
        random_data: bool = False,
        center_freq: float = 0,
        bandwidth: float = 0.5,
        **kwargs,
    ):
        super(ToneDataset, self).__init__(**kwargs)
        self.num_iq_samples = num_iq_samples
        self.num_samples_per_class = num_samples_per_class
        self.classes = ["tone"]
        self.random_data = random_data
        self.index = []
        self.center_freq = center_freq

        for class_idx, class_name in enumerate(self.classes):
            meta = ModulatedRFMetadata(
                sample_rate=0.0,
                num_samples=self.num_iq_samples,
                complex=True,
                lower_freq=center_freq,
                upper_freq=center_freq,
                center_freq=center_freq,
                bandwidth=0.0,
                start=0.0,
                stop=1.0,
                duration=1.0,
                snr=0.0,
                bits_per_symbol=0.0,
                samples_per_symbol=0.0,
                class_name=class_name,
                class_index=class_idx,
                excess_bandwidth=0.0,
            )
            for idx in range(self.num_samples_per_class):
                self.index.append(
                    (
                        class_name,
                        class_idx * self.num_samples_per_class + idx,
                        [meta],
                    )
                )

    def __len__(self) -> int:
        return len(self.index)

    def _generate_samples(self, item: Tuple) -> np.ndarray:
        # class_name = item[0]
        index = item[1]
        meta = item[2]
        orig_state = np.random.get_state()
        if not self.random_data:
            np.random.seed(index)

        # compute a random phase offset
        phaseOffset = np.random.uniform(0,2*np.pi)
        # compute time indices
        n = np.arange(0,self.num_iq_samples)
        # create tone
        modulated = np.exp(2j*np.pi*self.center_freq*n)*np.exp(1j*phaseOffset)

        if not self.random_data:
            np.random.set_state(orig_state)  # return numpy back to its previous state

        return modulated[-self.num_iq_samples :]


class ChirpSSDataset(SyntheticDataset):
    """Frequency Shift Chirp Spread Spectrum Modulated Dataset

    Args:
        num_iq_samples (:obj:`int`):
            number of iq samples in record, pads record with trailing zeros

        num_samples_per_class (:obj:`int`):
            number of samples of each class

        iq_samples_per_symbol (:obj:`Optional[int]`):
            number of IQ samples per symbol
       
        random_data (:obj:`bool`):
            uses numpy random values

        center_freq (:obj:`float`):
            center frequency of the signal

        bandwidth (:obj:`float`):
            bandwidth of the signal            

    """

    def __init__(
        self,
        constellations: Optional[Union[List, Tuple]] = torchsig_signals.chirpss_signals,
        num_iq_samples : int = 10000,
        num_samples_per_class: int = 20,
        iq_samples_per_symbol: int = 1000,
        random_data: bool = False,
        center_freq: float = 0.,
        bandwidth: float = 0.5,        
        **kwargs,
    ):
        super(ChirpSSDataset, self).__init__(**kwargs)
        self.symbol_map: Dict[str, np.ndarray] = self.get_symbol_map()
        self.constellations = (
            list(torchsig_signals.constellation_signals) if constellations is None else constellations
        )
        self.num_iq_samples = num_iq_samples 
        self.num_samples_per_class = num_samples_per_class
        self.iq_samples_per_symbol = iq_samples_per_symbol
        self.random_data = random_data
        self.index = []

        for const_idx, const_name in enumerate(map(str.lower, self.constellations)):
            for idx in range(self.num_samples_per_class):
                meta = create_modulated_rf_metadata(
                    num_samples=self.num_iq_samples,
                    bits_per_symbol=1,
                    samples_per_symbol=iq_samples_per_symbol,
                    class_name=const_name,
                    class_index=const_idx,
                    center_freq=center_freq,
                    bandwidth=bandwidth
                )
                self.index.append(
                    (
                        const_name,
                        const_idx * self.num_samples_per_class + idx,
                        [meta],
                    )
                )

        # design filter
        transitionBandwidth = bandwidth/8
        cutoff = bandwidth/2 + (transitionBandwidth/2)
        self.LPFWeights = low_pass(cutoff=cutoff,transition_bandwidth=transitionBandwidth)

    def __len__(self) -> int:
        return len(self.index)

    def chirp(self, t0, t1, f0, f1, phi=0) -> np.ndarray:
        t = np.linspace(t0, t1, 2*self.iq_samples_per_symbol)
        b = (f1 - f0) / (t1 - t0)
        phase = 2 * np.pi * (f0 * t + 0.5 * b * t * t) # Linear FM
        phi *= np.pi / 180
        return np.exp(1j*(phase+phi))

    def _generate_samples(self, item: Tuple) -> np.ndarray:
        class_name = item[0]
        index = item[1]
        metadata = item[2][0]
        center_freq = metadata["center_freq"]
        bandwidth = metadata["bandwidth"]
        
        orig_state = np.random.get_state()
        if not self.random_data:
            np.random.seed(index)

        # symbol mapping and padding
        const = self.symbol_map[class_name] 
        symbol_nums = np.random.randint(
            0, len(const), int(self.num_iq_samples / self.iq_samples_per_symbol)
        )
        symbols = const[symbol_nums]
        modulated = np.zeros((self.num_iq_samples,), dtype=np.complex128)
        
        # construct template symbol
        upchirp = self.chirp(0,self.iq_samples_per_symbol,-bandwidth,bandwidth)
        double_upchirp = np.concatenate((upchirp, upchirp), axis=0)

        # modulate 
        sym_start_index = 0
        M = const.size
        for s in symbols:
            chirp_start_index = int((s/M)*self.iq_samples_per_symbol)
            modulated[sym_start_index:(sym_start_index+self.iq_samples_per_symbol)] = \
                double_upchirp[chirp_start_index:(chirp_start_index+self.iq_samples_per_symbol)] 
            sym_start_index = sym_start_index + self.iq_samples_per_symbol # 100% duty cycle
        
        modulated = np.convolve(self.LPFWeights, modulated)

        # apply center frequency shifting
        modulated *= np.exp(2j*np.pi*center_freq*np.arange(0,len(modulated)))

        # determine the boundaries for where the signal currently resides.
        # these values are used to determine if aliasing has occured
        upperSignalEdge = center_freq + (bandwidth/2)
        lowerSignalEdge = center_freq - (bandwidth/2)

        # check to see if aliasing has occured due to upconversion. if so, then apply
        # a filter to minimize it
        if ( upperSignalEdge > 0.5 or lowerSignalEdge < -0.5):

            # the signal has overlaped either the -fs/2 or +fs/2 boundary and therefore
            # a BPF filter will be applied to attenuate the portion of the signal that
            # is overlapping the -fs/2 or +fs/2 boundary to minimize aliasing
            modulated = upconversionAntiAliasingFilter ( modulated, center_freq, bandwidth )

        if not self.random_data:
            np.random.set_state(orig_state)  # return numpy back to its previous state

        return modulated[:self.num_iq_samples]

    def get_symbol_map ( self ):
        chirpss_symbol_map = OrderedDict(
            {
                'chirpss': np.linspace(0,2**7-1,2**7),
            })
        return chirpss_symbol_map

class LFMDataset(SyntheticDataset):
    """Linear Frequency Modulated (LFM) Dataset: 
    Calculates number of LFM chirp symbols that can fit in specified length, 
    then modulates random data upchirp/downchirp symbols based on a custom or
    default provided constellation map 

    Args:
        num_iq_samples (:obj:`int`):
            number of iq samples in record, pads record with trailing zeros 

        num_samples_per_class (:obj:`int`):
            number of samples of each class

        iq_samples_per_symbol (:obj:`Optional[int]`):
            number of IQ samples per symbol 
       
        random_data (:obj:`bool`):
            uses numpy random values

        center_freq (:obj:`float`):
            center frequency of the signal

        bandwidth (:obj:`float`):
            bandwidth of the signal

    """

    def __init__(
        self,
        constellations: Optional[Union[List, Tuple]] = torchsig_signals.lfm_signals,
        num_iq_samples : int = 10000,
        num_samples_per_class: int = 20,
        iq_samples_per_symbol: int = 1000,
        random_data: bool = False,
        center_freq: float = 0.,
        bandwidth: float = 0.5,
        **kwargs,
    ):

        super(LFMDataset, self).__init__(**kwargs)
        self.symbol_map: Dict[str, np.ndarray] = self.get_symbol_map()
        self.constellations = (
            list(torchsig_signals.constellation_signals) if constellations is None else constellations
        )
        self.num_iq_samples = num_iq_samples 
        self.num_samples_per_class = num_samples_per_class
        self.iq_samples_per_symbol = iq_samples_per_symbol
        self.random_data = random_data
        self.index = []

        for const_idx, const_name in enumerate(map(str.lower, self.constellations)):
            for idx in range(self.num_samples_per_class):
                meta = create_modulated_rf_metadata(
                    num_samples=self.num_iq_samples,
                    bits_per_symbol=1,
                    samples_per_symbol=iq_samples_per_symbol,
                    class_name=const_name,
                    class_index=const_idx,
                    center_freq=center_freq,
                    bandwidth=bandwidth
                )
                self.index.append(
                    (
                        const_name,
                        const_idx * self.num_samples_per_class + idx,
                        [meta],
                    )
                )

    def __len__(self) -> int:
        return len(self.index)

    def chirp(self, t0, t1, f0, f1, phi=0) -> np.ndarray:
        t = np.linspace(t0, t1, self.iq_samples_per_symbol)
        b = (f1 - f0) / (t1 - t0)
        phase = 2 * np.pi * (f0 * t + 0.5 * b * t * t) # Linear FM
        phi *= np.pi / 180
        return np.exp(1j*(phase+phi))

    def _generate_samples(self, item: Tuple) -> np.ndarray:
        class_name = item[0]
        index = item[1]
        metadata = item[2][0]
        center_freq = metadata["center_freq"]
        bandwidth = metadata["bandwidth"]   
        f0 = center_freq - bandwidth / 2
        f1 = center_freq + bandwidth / 2

        orig_state = np.random.get_state()
        if not self.random_data:
            np.random.seed(index)

        # symbol mapping and padding
        const = self.symbol_map[class_name] 
        symbol_nums = np.random.randint(
            0, len(const), int(self.num_iq_samples / self.iq_samples_per_symbol)
        )
        symbols = const[symbol_nums]
        modulated = np.zeros((self.num_iq_samples,), dtype=np.complex128)
        upchirp = self.chirp(0,self.iq_samples_per_symbol,f0,f1)
        downchirp = self.chirp(0,self.iq_samples_per_symbol,f1,f0)

        sym_start_index = 0
        for s in symbols:
            if s > 0:
                modulated[sym_start_index:(sym_start_index+self.iq_samples_per_symbol)] = upchirp
            else:
                modulated[sym_start_index:(sym_start_index+self.iq_samples_per_symbol)] = downchirp
            sym_start_index = sym_start_index + self.iq_samples_per_symbol

        # determine the boundaries for where the signal currently resides.
        # these values are used to determine if aliasing has occured
        upperSignalEdge = center_freq + (bandwidth/2)
        lowerSignalEdge = center_freq - (bandwidth/2)

        # check to see if aliasing has occured due to upconversion. if so, then apply
        # a filter to minimize it
        if ( upperSignalEdge > 0.5 or lowerSignalEdge < -0.5):

            # the signal has overlaped either the -fs/2 or +fs/2 boundary and therefore
            # a BPF filter will be applied to attenuate the portion of the signal that
            # is overlapping the -fs/2 or +fs/2 boundary to minimize aliasing
            modulated = upconversionAntiAliasingFilter ( modulated, center_freq, bandwidth )

        if not self.random_data:
            np.random.set_state(orig_state)  # return numpy back to its previous state

        return modulated[:self.num_iq_samples]

    def get_symbol_map ( self ):
        lfm_symbol_map = OrderedDict(
            {
                'lfm_data': np.array([-1.,1.]),
                'lfm_radar': np.array([1.]),
            })
        return lfm_symbol_map
    

#TODO: flesh out and test this
class CarrierWaveSpikeDataset(SyntheticDataset):
    """Frequency Shift Carrier Wave Spike Modulated Dataset

    Args:
        num_iq_samples (:obj:`int`):
            number of iq samples in record, pads record with trailing zeros

        num_samples_per_class (:obj:`int`):
            number of samples of each class

        iq_samples_per_symbol (:obj:`Optional[int]`):
            number of IQ samples per symbol
       
        random_data (:obj:`bool`):
            uses numpy random values

        center_freq (:obj:`float`):
            center frequency of the signal

        bandwidth (:obj:`float`):
            bandwidth of the signal            

    """
    def __init__(
        self,
        CWs: Optional[Union[List, Tuple]] = torchsig_signals.cw_signals,
        num_iq_samples : int = 10000,
        num_samples_per_class: int = 20,
        iq_samples_per_symbol: int = 1000,
        random_data: bool = False,
        center_freq: float = 0.,
        bandwidth: float = 0.5,        
        **kwargs,
    ):
        super(CarrierWaveSpikeDataset, self).__init__(**kwargs)
        # self.symbol_map: Dict[str, np.ndarray] = self.get_symbol_map()
        self.CWs = (
            list(torchsig_signals.cw_signals) if CWs is None else CWs
        )
        self.num_iq_samples = num_iq_samples 
        self.num_samples_per_class = num_samples_per_class
        self.iq_samples_per_symbol = iq_samples_per_symbol
        self.random_data = random_data
        self.index = []

        for const_idx, const_name in enumerate(map(str.lower, self.CWs)):
            for idx in range(self.num_samples_per_class):
                meta = create_modulated_rf_metadata(
                    num_samples=self.num_iq_samples,
                    bits_per_symbol=1,
                    samples_per_symbol=iq_samples_per_symbol,
                    class_name=const_name,
                    class_index=const_idx,
                    center_freq=center_freq,
                    bandwidth=bandwidth
                )
                self.index.append(
                    (
                        const_name,
                        const_idx * self.num_samples_per_class + idx,
                        [meta],
                    )
                )

        # design filter
        transitionBandwidth = bandwidth/8
        cutoff = bandwidth/2 + (transitionBandwidth/2)
        self.LPFWeights = low_pass(cutoff=cutoff,transition_bandwidth=transitionBandwidth)

    def __len__(self) -> int:
        return len(self.index)
    
    #TODO: here is where the cw needs to be generated below:::: mode is for hopping ect...
    def spike(self, t0, t1, f0, f1, duration = 1 , phi=0, mode=None) -> np.ndarray:
        t = np.linspace(t0, t1, 2*self.iq_samples_per_symbol)
        b = (f1 - f0) / (t1 - t0)
        # if mode == 'hopping':
            # if f0 is None or hop_duration is None:
            #     raise ValueError("frequencies and hop_duration must be provided for hopping CW signal")

            # Generate the hopping CW signal TODO:
            # hopping_signal = np.zeros(self.num_iq_samples, dtype=complex)
            # num_hops = int(duration / hop_duration)

            # for i in range(num_hops):
            #     start_index =  int(i * hop_duration * sampling_rate)
            #     end_index = int((i + 1) * hop_duration * sampling_rate)
            #     # TODO: select some random frequencies here
            #     if i < len(frequencies):
            #         frequency = frequencies[i]
            #     else:
            #         frequency = frequencies[-1]  # Use the last frequency if not enough frequencies are provided
            #     hopping_signal[start_index:end_index] = np.exp(2j * np.pi * frequency * t[start_index:end_index])
            # signal = hopping_signal

        # TODO: check this calc for cw # mode == 'static'
        # phase = 2 * np.pi * (f0 * t + 0.5 * b * t * t) # Linear FM
        phase = np.cos(2 * np.pi * f0 * t) + 1j * np.sin(2 * np.pi * f0 * t)
        phi *= np.pi / 180
        return np.exp(1j*(phase+phi))

    def _generate_samples(self, item: Tuple) -> np.ndarray:
        print("CarrierWaveSpikeDataset._generate_samples called", flush=True)
        class_name = item[0]
        index = item[1]
        metadata = item[2][0]
        center_freq = metadata["center_freq"]
        bandwidth = metadata["bandwidth"]
        
        orig_state = np.random.get_state()
        if not self.random_data:
            np.random.seed(index)

        # symbol mapping and padding
        # const = self.symbol_map[class_name] 
        # symbol_nums = np.random.randint(
        #     0, len(const), int(self.num_iq_samples / self.iq_samples_per_symbol)
        # )
        # symbols = const[symbol_nums]
        modulated = np.zeros((self.num_iq_samples,), dtype=np.complex128)
        
        # construct template symbol
        upspike = self.spike(0,self.iq_samples_per_symbol,-bandwidth,bandwidth)
        double_upspike = np.concatenate((upspike, upspike), axis=0)

        # modulated prob not needed for cw spike - mtt TODO:
        # modulate 
        # sym_start_index = 0
        # M = const.size
        # for s in symbols:
        #     chirp_start_index = int((s/M)*self.iq_samples_per_symbol)
        #     modulated[sym_start_index:(sym_start_index+self.iq_samples_per_symbol)] = \
        #         double_upchirp[chirp_start_index:(chirp_start_index+self.iq_samples_per_symbol)] 
        #     sym_start_index = sym_start_index + self.iq_samples_per_symbol # 100% duty cycle
        # modulated = np.convolve(self.LPFWeights, modulated)

        # apply center frequency shifting
        upspike *= np.exp(2j*np.pi*center_freq*np.arange(0,len(modulated)))

        # determine the boundaries for where the signal currently resides.
        # these values are used to determine if aliasing has occured
        upperSignalEdge = center_freq + (bandwidth/2)
        lowerSignalEdge = center_freq - (bandwidth/2)
        # TODO: this prob not needed but not sure
        # check to see if aliasing has occured due to upconversion. if so, then apply
        # a filter to minimize it
        # if ( upperSignalEdge > 0.5 or lowerSignalEdge < -0.5):

        #     # the signal has overlaped either the -fs/2 or +fs/2 boundary and therefore
        #     # a BPF filter will be applied to attenuate the portion of the signal that
        #     # is overlapping the -fs/2 or +fs/2 boundary to minimize aliasing
        #     modulated = upconversionAntiAliasingFilter ( modulated, center_freq, bandwidth )

        if not self.random_data:
            np.random.set_state(orig_state)  # return numpy back to its previous state
        print("SPIKEDataset", flush=True)
        return upspike[:self.num_iq_samples]



# apply an anti-aliasing filter to a signal which has aliased and wrapped around the
# -fs/2 or +fs/2 boundary due to upconversion
def upconversionAntiAliasingFilter ( input_signal, center_freq, bandwidth ):

    # determine the boundaries for where the signal currently resides.
    # these values are used to determine if aliasing has occured
    upperSignalEdge = center_freq + (bandwidth/2)
    lowerSignalEdge = center_freq - (bandwidth/2)

    # define the boundary for the upper and lower frequencies
    # upon which a BPF will be designed to limit aliasing
    upperBoundary = MAX_SIGNAL_UPPER_EDGE_FREQ
    lowerBoundary = MAX_SIGNAL_LOWER_EDGE_FREQ

    # determine if aliasing has occured, and if so, which direction,
    # either +fs/2 or -fs/2
    if ( upperSignalEdge > 0.5): # aliasing occurs across +fs/2
        slicedUpperSignalEdge = upperBoundary
        slicedLowerSignalEdge = -0.5+center_freq
    elif ( lowerSignalEdge < -0.5): # aliasing occurs across -fs/2
        slicedLowerSignalEdge = lowerBoundary
        slicedUpperSignalEdge = 0.5+center_freq
    else: # no aliasing occurs
        slicedUpperSignalEdge = upperSignalEdge
        slicedLowerSignalEdge = lowerSignalEdge

    # compute the bandwidth and center frequency after the BPF is applied
    slicedBandwidth = slicedUpperSignalEdge - slicedLowerSignalEdge
    slicedCenterFreq = slicedLowerSignalEdge + (slicedBandwidth/2)

    # design a LPF then upconvert it to a BPF
    # 
    # calculate the transition bandwidth for the LPF with proportion to the
    # signal's bandwidth. a fixed ratio is used here in order to keep the 
    # transition bandwidth small
    transitionBandwidth = slicedBandwidth/16
    # the passband edge of the LPF is 1/2 of the post-filtered bandwidth. this
    # pushes the cutoff frequency past the 3 dB point in the signal bandwidth 
    # as to minimize the distortion of the underlying signal
    fPass = slicedBandwidth/2
    # calculate the filter cutoff location
    cutoff = fPass + (transitionBandwidth/2)
    # design the LPF
    LPFWeights = low_pass(cutoff=cutoff,transition_bandwidth=transitionBandwidth)
    # modulate the LPF to BPF
    numLPFWeights = len(LPFWeights)
    n = np.arange(-int(numLPFWeights-1)/2,((numLPFWeights-1)/2)+1)
    BPFWeights = LPFWeights * np.exp(2j*np.pi*slicedCenterFreq*n)
    # apply BPF
    output = np.convolve(BPFWeights,input_signal)
    return output
