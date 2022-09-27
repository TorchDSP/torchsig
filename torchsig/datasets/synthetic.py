import itertools
import numpy as np
from scipy import signal as sp
from collections import OrderedDict
from torch.utils.data import ConcatDataset
from typing import Tuple, Any, List, Union, Optional

from torchsig.utils.dataset import SignalDataset
from torchsig.utils.types import SignalData, SignalDescription
from torchsig.transforms.functional import IntParameter, FloatParameter


def remove_corners(const):
    spacing = 2.0 / (np.sqrt(len(const)) - 1)
    cutoff = spacing * (np.sqrt(len(const)) / 6 - .5)
    return [p for p in const if np.abs(np.real(p)) < 1.0 - cutoff or np.abs(np.imag(p)) < 1.0 - cutoff]


const_map = OrderedDict({
    "ook": np.add(*map(np.ravel, np.meshgrid(np.linspace(0, 1, 2), 0j))),
    "bpsk": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 2), 0j))),
    "4pam": np.add(*map(np.ravel, np.meshgrid(np.linspace(0, 1, 4), 0j))),
    "4ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 4), 0j))),
    "qpsk": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 2), 1j * np.linspace(-1, 1, 2)))),
    "8pam": np.add(*map(np.ravel, np.meshgrid(np.linspace(0, 1, 8), 0j))),
    "8ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 8), 0j))),
    "8psk": np.exp(2j * np.pi * np.linspace(0, 7, 8) / 8.0),
    "16qam": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 4), 1j * np.linspace(-1, 1, 4)))),
    "16pam": np.add(*map(np.ravel, np.meshgrid(np.linspace(0, 1, 16), 0j))),
    "16ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 16), 0j))),
    "16psk": np.exp(2j * np.pi * np.linspace(0, 15, 16) / 16.0),
    "32qam": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 4), 1j * np.linspace(-1, 1, 8)))),
    "32qam_cross":
        remove_corners(np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 6), 1j * np.linspace(-1, 1, 6))))),
    "32pam": np.add(*map(np.ravel, np.meshgrid(np.linspace(0, 1, 32), 0j))),
    "32ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 32), 0j))),
    "32psk": np.exp(2j * np.pi * np.linspace(0, 31, 32) / 32.0),
    "64qam": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 8), 1j * np.linspace(-1, 1, 8)))),
    "64pam": np.add(*map(np.ravel, np.meshgrid(np.linspace(0, 1, 64), 0j))),
    "64ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 64), 0j))),
    "64psk": np.exp(2j * np.pi * np.linspace(0, 63, 64) / 64.0),
    "128qam_cross":
        remove_corners(np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 12), 1j * np.linspace(-1, 1, 12))))),
    "256qam": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 16), 1j * np.linspace(-1, 1, 16)))),
    "512qam_cross":
        remove_corners(np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 24), 1j * np.linspace(-1, 1, 24))))),
    "1024qam": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 32), 1j * np.linspace(-1, 1, 32))))
})

# This is probably redundant.
freq_map = OrderedDict({
    "2fsk": np.linspace(-1, 1, 2, endpoint=True),
    "2gfsk": np.linspace(-1, 1, 2, endpoint=True),
    "2msk": np.linspace(-1, 1, 2, endpoint=True),
    "2gmsk": np.linspace(-1, 1, 2, endpoint=True),
    "4fsk": np.linspace(-1, 1, 4, endpoint=True),
    "4gfsk": np.linspace(-1, 1, 4, endpoint=True),
    "4msk": np.linspace(-1, 1, 4, endpoint=True),
    "4gmsk": np.linspace(-1, 1, 4, endpoint=True),
    "8fsk": np.linspace(-1, 1, 8, endpoint=True),
    "8gfsk": np.linspace(-1, 1, 8, endpoint=True),
    "8msk": np.linspace(-1, 1, 8, endpoint=True),
    "8gmsk": np.linspace(-1, 1, 8, endpoint=True),
    "16fsk": np.linspace(-1, 1, 16, endpoint=True),
    "16gfsk": np.linspace(-1, 1, 16, endpoint=True),
    "16msk": np.linspace(-1, 1, 16, endpoint=True),
    "16gmsk": np.linspace(-1, 1, 16, endpoint=True),
})


class DigitalModulationDataset(ConcatDataset):
    """Digital Modulation Dataset

    Args:
        modulations (:obj:`list` or :obj:`tuple`):
            Sequence of strings representing the constellations that should be included.

        num_iq_samples (:obj:`int`):
            number of samples to read from each file in the database

        num_samples_per_class (:obj:`int`):
            number of samples to be kept for each class

        random_data (:obj:`bool`):self.num_samples_per_class/num_subcarriers/len(cycle_prefix_ratios)
            whether the modulated binary utils should be random each time, or seeded by index

        transform (:obj:`Callable`, optional):
            A function/transform that takes in an IQ vector and returns a transformed version.

    """
    def __init__(
        self,
        modulations: Optional[Union[List, Tuple]] = ("bpsk", "2gfsk"),
        num_iq_samples: int = 100,
        num_samples_per_class: int = 100,
        iq_samples_per_symbol: Optional[int] = None,
        random_data: bool = False,
        random_pulse_shaping: bool = False,
        **kwargs
    ):
        modulations = list(const_map.keys()) + list(freq_map.keys()) if modulations is None else modulations
        constellations = [m for m in map(str.lower, modulations) if m in const_map.keys()]
        freqs = [m for m in map(str.lower, modulations) if m in freq_map.keys()]
        const_dataset = ConstellationDataset(
            constellations=constellations,
            num_iq_samples=num_iq_samples,
            num_samples_per_class=num_samples_per_class,
            iq_samples_per_symbol=2 if iq_samples_per_symbol is None else iq_samples_per_symbol,
            random_data=random_data,
            random_pulse_shaping=random_pulse_shaping,
            **kwargs
        )

        # FSK signals with the Gaussian pulse shaping filter are handled differently than without
        fsks = []
        gfsks = []
        for freq_mod in freqs:
            if "g" in freq_mod:
                gfsks.append(freq_mod)
            else:
                fsks.append(freq_mod)
        fsk_dataset = FSKDataset(
            modulations=fsks,
            num_iq_samples=num_iq_samples,
            num_samples_per_class=num_samples_per_class,
            iq_samples_per_symbol=8,
            random_data=random_data,
            random_pulse_shaping=random_pulse_shaping,
            **kwargs
        )
        gfsks_dataset = FSKDataset(
            modulations=gfsks,
            num_iq_samples=num_iq_samples,
            num_samples_per_class=num_samples_per_class,
            iq_samples_per_symbol=8 if iq_samples_per_symbol is None else iq_samples_per_symbol,
            random_data=random_data,
            random_pulse_shaping=random_pulse_shaping,
            **kwargs
        )
        super(DigitalModulationDataset, self).__init__([const_dataset, fsk_dataset, gfsks_dataset])


class SyntheticDataset(SignalDataset):
    def __init__(self, **kwargs):
        super(SyntheticDataset, self).__init__(**kwargs)
        self.index = []

    def __getitem__(self, index: int) -> Tuple[SignalData, Any]:
        signal_description = self.index[index][-1]
        signal_data = SignalData(
            data=self._generate_samples(self.index[index]).tobytes(),
            item_type=np.dtype(np.float64),
            data_type=np.dtype(np.complex128),
            signal_description=signal_description
        )

        if self.transform:
            signal_data = self.transform(signal_data)

        if self.target_transform:
            target = self.target_transform(signal_data.signal_description)
        else:
            target = signal_description

        return signal_data.iq_data, target

    def __len__(self) -> int:
        return len(self.index)

    def _generate_samples(self, item: Tuple) -> np.ndarray:
        raise NotImplementedError


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

    """
    def __init__(
        self,
        constellations: Optional[Union[List, Tuple]] = ("bpsk", "qpsk"),
        num_iq_samples: int = 100,
        num_samples_per_class: int = 100,
        iq_samples_per_symbol: int = 2,
        pulse_shape_filter: bool = None,
        random_pulse_shaping: bool = False,
        random_data: bool = False,
        **kwargs
    ):
        super(ConstellationDataset, self).__init__(**kwargs)
        self.constellations = list(const_map.keys()) if constellations is None else constellations
        self.num_iq_samples = num_iq_samples
        self.iq_samples_per_symbol = iq_samples_per_symbol
        self.num_samples_per_class = num_samples_per_class
        self.random_pulse_shaping = random_pulse_shaping
        
        num_constellations = len(self.constellations)
        total_num_samples = int(num_constellations*self.num_samples_per_class)
        
        if pulse_shape_filter is None:
            if self.random_pulse_shaping:
                alphas = np.random.uniform(0.15,0.6,size=total_num_samples)
            else:
                alphas = np.ones(total_num_samples)*0.35
        else:
            self.pulse_shape_filter = pulse_shape_filter
    
        self.random_data = random_data
        self.index = []

        for const_idx, const_name in enumerate(map(str.lower, self.constellations)):
            for idx in range(self.num_samples_per_class):
                signal_description = SignalDescription(
                    sample_rate=0,
                    bits_per_symbol=np.log2(len(const_map[const_name])),
                    samples_per_symbol=iq_samples_per_symbol,
                    class_name=const_name,
                    excess_bandwidth=alphas[int(const_idx*self.num_samples_per_class+idx)],
                )
                self.index.append((const_name, const_idx*self.num_samples_per_class + idx, signal_description))

    def _generate_samples(self, item: Tuple) -> np.ndarray:
        class_name = item[0]
        index = item[1]
        signal_description = item[2]
        orig_state = np.random.get_state()
        if not self.random_data:
            np.random.seed(index)

        const = const_map[class_name] / np.mean(np.abs(const_map[class_name]))
        symbol_nums = np.random.randint(0, len(const), 2 * int(self.num_iq_samples / self.iq_samples_per_symbol))
        symbols = const[symbol_nums]
        zero_padded = np.zeros((self.iq_samples_per_symbol * len(symbols),), dtype=np.complex64)
        zero_padded[::self.iq_samples_per_symbol] = symbols
        self.pulse_shape_filter = self._rrc_taps(11, signal_description.excess_bandwidth)
        filtered = np.convolve(zero_padded, self.pulse_shape_filter, "same")

        if not self.random_data:
            np.random.set_state(orig_state)  # return numpy back to its previous state

        return filtered[-self.num_iq_samples:]

    def _rrc_taps(self, size_in_symbols: int, alpha: float = .35) -> np.ndarray:
        # this could be made into a transform
        M = size_in_symbols
        Ns = float(self.iq_samples_per_symbol)
        n = np.arange(-M * Ns, M * Ns + 1)
        taps = np.zeros(int(2 * M * Ns + 1))
        for i in range(int(2 * M * Ns + 1)):
            if abs(1 - 16 * alpha ** 2 * (n[i] / Ns) ** 2) <= np.finfo(np.float64).eps / 2:
                taps[i] = 1 / 2. * ((1 + alpha) * np.sin((1 + alpha) * np.pi / (4. * alpha)) - (1 - alpha) * np.cos(
                    (1 - alpha) * np.pi / (4. * alpha)) + (4 * alpha) / np.pi * np.sin(
                    (1 - alpha) * np.pi / (4. * alpha)))
            else:
                taps[i] = 4 * alpha / (np.pi * (1 - 16 * alpha ** 2 * (n[i] / Ns) ** 2))
                taps[i] = taps[i] * (np.cos((1 + alpha) * np.pi * n[i] / Ns) + np.sinc(
                    (1 - alpha) * n[i] / Ns) * (1 - alpha) * np.pi / (
                                             4. * alpha))
        return taps


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
             * `none` ~ Perform no sidelobe suppression methods
             * `lpf` ~ Apply a static low pass filter to the OFDM signal
             * `rand_lpf` ~ Apply a low pass filter with a randomized cutoff frequency to the OFDM signal
             * `win_start` ~ Apply a windowing method starting at the symbol boundary
             * `win_center` ~ Apply a windowing method centered at the symbol boundary
             
            For more details on the windowing method options, please see:
            http://zone.ni.com/reference/en-XX/help/373725J-01/wlangen/windowing/
            
        dc_subcarrier (:obj:`tuple`):
            Tuple of possible DC subcarrier options:
             * `on` ~ Always leave the DC subcarrier on
             * `off` ~ Always turn the DC subcarrier off

        transform (:obj:`Callable`, optional):
            A function/transform that takes in an IQ vector and returns a transformed version.

    """
    def __init__(
        self,
        constellations: Optional[Union[List, Tuple]] = ("bpsk", "qpsk"),
        num_subcarriers: IntParameter = (64, 128, 256, 512, 1024, 2048),
        cyclic_prefix_ratios: FloatParameter = (.125, .25),
        num_iq_samples: int = 100,
        num_samples_per_class: int = 100,
        random_data: bool = False,
        sidelobe_suppression_methods: tuple = ('none', 'lpf', 'rand_lpf', 'win_start', 'win_center'),
        dc_subcarrier: tuple = ('on', 'off'),
        **kwargs
    ):
        super(OFDMDataset, self).__init__(**kwargs)
        self.constellations = constellations
        self.num_iq_samples = num_iq_samples
        self.num_samples_per_class = num_samples_per_class
        self.random_data = random_data
        self.index = []
        if 'lpf' in sidelobe_suppression_methods:
            # Precompute LPF
            num_taps = 50
            cutoff = 0.6
            self.taps = sp.firwin(
                num_taps,
                cutoff,
                width=cutoff * .02,
                window=sp.get_window("blackman", num_taps),
                scale=True
            )
        
        # Precompute all possible random symbols for speed at sample generation
        self.random_symbols = []
        for const_name in self.constellations:
            const = const_map[const_name] / np.mean(np.abs(const_map[const_name]))
            self.random_symbols.append(const)
        
        subcarrier_modulation_types = ("fixed", "random")
        combinations = list(itertools.product(constellations, subcarrier_modulation_types, cyclic_prefix_ratios, sidelobe_suppression_methods, dc_subcarrier))

        for class_idx, num_subcarrier in enumerate(num_subcarriers):
            class_name = "ofdm-{}".format(num_subcarrier)
            for idx in range(self.num_samples_per_class):
                const_name, mod_type, cyclic_prefix_ratio, sidelobe_suppression_method, dc_subcarrier = combinations[np.random.randint(len(combinations))]
                signal_description = SignalDescription(
                    sample_rate=0,
                    bits_per_symbol=2,
                    samples_per_symbol=2, # Not accurate, but useful in calculating effective half bandwidth target
                    class_name=class_name,
                )
                self.index.append((
                    class_name,
                    class_idx*self.num_samples_per_class + idx,
                    num_subcarrier,
                    cyclic_prefix_ratio*num_subcarrier,
                    const_name,
                    mod_type,
                    sidelobe_suppression_method,
                    dc_subcarrier,
                    signal_description
                ))
                
    def _generate_samples(self, item: Tuple) -> np.ndarray:
        index = item[1]
        num_subcarriers = item[2]
        cyclic_prefix_len = item[3]
        const_name = item[4]
        mod_type = item[5]
        sidelobe_suppression_method = item[6]
        dc_subcarrier = item[7]
        orig_state = np.random.get_state()
        if not self.random_data:
            np.random.seed(index)

        if mod_type == "random":
            # Randomized subcarrier modulations
            symbols = []
            for subcarrier_idx in range(num_subcarriers):
                curr_const = np.random.randint(len(self.random_symbols))
                symbols.extend(np.random.choice(self.random_symbols[curr_const], size=int(2*self.num_iq_samples/num_subcarriers)))
            symbols = np.asarray(symbols)
        else:
            # Fixed modulation across all subcarriers
            const_name = np.random.choice(self.constellations)
            const = const_map[const_name] / np.mean(np.abs(const_map[const_name]))
            symbol_nums = np.random.randint(0, len(const), 2 * self.num_iq_samples)
            symbols = const[symbol_nums]
        divisible_index = -(len(symbols) % num_subcarriers)
        if divisible_index != 0:
            symbols = symbols[:divisible_index]

        # now sub-carrier modulate
        serial_to_parallel = symbols.reshape(num_subcarriers, -1)
        zero_pad = np.pad(
            serial_to_parallel,
            ((int(num_subcarriers / 2), int(num_subcarriers / 2)), (0, 0)),
            'constant',
            constant_values=0,
        )
        
        # Turn off DC subcarrier
        if dc_subcarrier == 'off':
            dc_center = int(zero_pad.shape[0]//2)
            zero_pad[dc_center,:] = np.zeros((zero_pad.shape[1]))

        ofdm_symbols = np.fft.ifft(np.fft.ifftshift(zero_pad, axes=0), axis=0)
        symbol_dur = ofdm_symbols.shape[0]
        
        cyclic_prefixed = np.pad(ofdm_symbols, ((int(cyclic_prefix_len), 0), (0, 0)), 'wrap')
        
        if sidelobe_suppression_method == 'none':
            # randomize the start index
            start_idx = np.random.randint(0,symbol_dur)
            return cyclic_prefixed.T.flatten()[start_idx:start_idx+self.num_iq_samples]
        
        elif sidelobe_suppression_method == 'lpf':
            flattened = cyclic_prefixed.T.flatten()
            # Apply pre-computed LPF
            filtered = sp.fftconvolve(flattened, self.taps, mode="same")
            # randomize the start index
            start_idx = np.random.randint(0,symbol_dur)
            return filtered[start_idx:start_idx+self.num_iq_samples]
            
        elif sidelobe_suppression_method == 'rand_lpf':
            flattened = cyclic_prefixed.T.flatten()
            # Generate randomized LPF
            cutoff = np.random.uniform(0.50,0.95)
            num_taps = int(np.ceil(50*2*np.pi/cutoff/.125/22)) # fred harris rule of thumb
            taps = sp.firwin(
                num_taps,
                cutoff,
                width=cutoff * .02,
                window=sp.get_window("blackman", num_taps),
                scale=True
            )
            # Apply random LPF
            filtered = sp.fftconvolve(flattened, taps, mode="same")
            # randomize the start index
            start_idx = np.random.randint(0,symbol_dur)
            return filtered[start_idx:start_idx+self.num_iq_samples]
            
        else:
            # Apply appropriate windowing technique
            window_len = cyclic_prefix_len
            half_window_len = int(window_len / 2)
            if sidelobe_suppression_method == 'win_center':
                windowed = np.pad(cyclic_prefixed, ((half_window_len, half_window_len), (0, 0)), 'constant', constant_values=0) 
                windowed[-half_window_len:, :] = windowed[
                    int(half_window_len)+int(cyclic_prefix_len):int(half_window_len)+int(cyclic_prefix_len)+int(half_window_len),
                    :
                ]
                windowed[:half_window_len, :] = windowed[
                    int(half_window_len)+int(cyclic_prefix_len)+int(symbol_dur):int(half_window_len)+int(cyclic_prefix_len)+int(symbol_dur)+int(half_window_len),
                    :
                ]
            elif sidelobe_suppression_method == 'win_start':
                windowed = np.pad(cyclic_prefixed, ((0, int(window_len)), (0, 0)), 'constant', constant_values=0) 
                windowed[-int(window_len):,:] = windowed[int(cyclic_prefix_len):int(cyclic_prefix_len)+int(window_len),:]
            else:
                raise ValueError('Expected window method to be: none, center, or start. Received: {}'.format(self.window_method))

            # window the tails
            front_window = np.blackman(int(window_len*2))[:int(window_len)].reshape(-1, 1)
            tail_window = np.blackman(int(window_len*2))[-int(window_len):].reshape(-1, 1)
            windowed[:int(window_len), :] = front_window * windowed[:int(window_len), :]
            windowed[-int(window_len):, :] = tail_window * windowed[-int(window_len):, :]

            if not self.random_data:
                np.random.set_state(orig_state)  # return numpy back to its previous state

            combined = np.zeros((windowed.shape[0]*windowed.shape[1],), dtype=complex)
            start_idx = 0
            for symbol_idx in range(windowed.shape[1]):
                combined[start_idx:start_idx+windowed.shape[0]] += windowed[:,symbol_idx]
                start_idx += (symbol_dur+int(window_len))

            # randomize the start index while bypassing the initial windowing
            start_idx = np.random.randint(window_len,symbol_dur+window_len)

            return combined[start_idx:start_idx+self.num_iq_samples]

        
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

        transform (:obj:`Callable`, optional):
            A function/transform that takes in an IQ vector and returns a transformed version.

    """
    def __init__(
        self,
        modulations: Optional[Union[List, Tuple]] = ("2fsk", "2gmsk"),
        num_iq_samples: int = 100,
        num_samples_per_class: int = 100,
        iq_samples_per_symbol: int = 2,
        random_data: bool = False,
        random_pulse_shaping: bool = False,
        **kwargs
    ):
        super(FSKDataset, self).__init__(**kwargs)
        self.modulations = list(freq_map.keys()) if modulations is None else modulations
        self.num_iq_samples = num_iq_samples
        self.num_samples_per_class = num_samples_per_class
        self.iq_samples_per_symbol = iq_samples_per_symbol
        self.random_data = random_data
        self.random_pulse_shaping = random_pulse_shaping
        self.index = []

        for freq_idx, freq_name in enumerate(map(str.lower, self.modulations)):
            for idx in range(self.num_samples_per_class):
                if "g" in freq_name:
                    bandwidth = np.random.uniform(0.1, 0.5) if self.random_pulse_shaping else 0.35
                else:
                    bandwidth = np.random.uniform(
                        (1 / self.iq_samples_per_symbol) * 1.25,
                        (1 / self.iq_samples_per_symbol) * 3.75,
                    ) if self.random_pulse_shaping else 0.0
                signal_description = SignalDescription(
                    sample_rate=0,
                    bits_per_symbol=np.log2(len(freq_map[freq_name])),
                    samples_per_symbol=iq_samples_per_symbol,
                    class_name=freq_name,
                    excess_bandwidth=bandwidth,
                )
                self.index.append((freq_name, freq_idx*self.num_samples_per_class + idx, bandwidth, signal_description))

    def _generate_samples(self, item: Tuple) -> np.ndarray:
        const_name = item[0]
        index = item[1]
        bandwidth = item[2]
        signal_description = item[3]
    
        orig_state = np.random.get_state()
        if not self.random_data:
            np.random.seed(index)

        const = freq_map[const_name]
        symbol_nums = np.random.randint(0, len(const), int(self.num_iq_samples)) # Create extra symbols and truncate later
        symbols = const[symbol_nums]
        symbols_repeat = np.repeat(symbols, self.iq_samples_per_symbol)
        filtered = symbols_repeat
        if "g" in const_name:
            taps = self._gaussian_taps(bandwidth)
            signal_description.excess_bandwidth = bandwidth
            filtered = np.convolve(symbols_repeat, taps, "same")

        mod_idx = 1.0 if "fsk" in const_name else .5
        phase = np.cumsum(filtered * 1j / self.iq_samples_per_symbol * mod_idx * np.pi)
        modulated = np.exp(phase)

        if "g" not in const_name and self.random_pulse_shaping:
            # Apply a randomized LPF simulating a noisy detector/burst extractor, then downsample to ~fs/2 bw
            lpf_bandwidth = bandwidth
            num_taps = int(np.ceil(50 * 2 * np.pi / lpf_bandwidth / .125 / 22))
            taps = sp.firwin(
                num_taps,
                lpf_bandwidth,
                width=lpf_bandwidth * .02,
                window=sp.get_window("blackman", num_taps),
                scale=True
            )
            modulated = sp.fftconvolve(modulated, taps, mode="same")
            new_rate = lpf_bandwidth * 2
            modulated = sp.resample_poly(
                modulated, 
                up=np.floor(new_rate*100).astype(np.int32), 
                down=100,
            )
            signal_description.samples_per_symbol = 2 # Effective samples per symbol at half bandwidth
            signal_description.excess_bandwidth = 0 # Reset excess bandwidth due to LPF
        
        if not self.random_data:
            np.random.set_state(orig_state)  # return numpy back to its previous state
            
        return modulated[-self.num_iq_samples:]

    def _gaussian_taps(self, BT: float = 0.35) -> np.ndarray:
        # pre-modulation Bb*T product which sets the bandwidth of the Gaussian lowpass filter
        M = 4  # duration in symbols
        Ns = self.iq_samples_per_symbol
        n = np.arange(-M * Ns, M * Ns + 1)
        p = np.exp(-2 * np.pi ** 2 * BT ** 2 / np.log(2) * (n / float(Ns)) ** 2)
        p = p / np.sum(p)
        return p


class AMDataset(SyntheticDataset):
    """AM Dataset

    Args:
        transform (:obj:`Callable`, optional):
            A function/transform that takes in an IQ vector and returns a transformed version.

    """
    def __init__(
        self,
        num_iq_samples: int = 100,
        num_samples_per_class: int = 100,
        random_data: bool = False,
        **kwargs
    ):
        super(AMDataset, self).__init__(**kwargs)
        self.num_iq_samples = num_iq_samples
        self.num_samples_per_class = num_samples_per_class
        self.classes = ["am", "am-ssb", "am-dsb"]
        self.random_data = random_data
        self.index = []

        for class_idx, class_name in enumerate(self.classes):
            signal_description = SignalDescription(sample_rate=0)
            for idx in range(self.num_samples_per_class):
                self.index.append((class_name, class_idx*self.num_samples_per_class + idx, signal_description))
    
    def __len__(self) -> int:
        return len(self.index)

    def _generate_samples(self, item: Tuple) -> np.ndarray:
        const_name = item[0]
        index = item[1]
        orig_state = np.random.get_state()
        if not self.random_data:
            np.random.seed(index)

        source = np.random.randn(self.num_iq_samples) + 0j
        taps = sp.firwin(
            100,  # num taps
            .5 if "ssb" not in const_name else .25,
            .5 / 16 if "ssb" not in const_name else .25 / 4,
            window="blackman"
        )
        filtered = np.convolve(source, taps, "same")
        sinusoid = np.exp(2j * np.pi * .125 * np.arange(self.num_iq_samples))
        filtered *= np.ones_like(filtered) if "ssb" not in const_name else sinusoid
        filtered += 5 if const_name == "am" else 0

        if not self.random_data:
            np.random.set_state(orig_state)  # return numpy back to its previous state

        return filtered[-self.num_iq_samples:]

    
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
        **kwargs
    ):
        super(FMDataset, self).__init__(**kwargs)
        self.num_iq_samples = num_iq_samples
        self.num_samples_per_class = num_samples_per_class
        self.classes = ["fm"]
        self.random_data = random_data
        self.index = []

        for class_idx, class_name in enumerate(self.classes):
            signal_description = SignalDescription(sample_rate=0)
            for idx in range(self.num_samples_per_class):
                self.index.append((class_name, class_idx*self.num_samples_per_class + idx, signal_description))

    def __len__(self) -> int:
        return len(self.index)
    
    def _generate_samples(self, item: Tuple) -> np.ndarray:
        # class_name = item[0]
        index = item[1]
        orig_state = np.random.get_state()
        if not self.random_data:
            np.random.seed(index)

        source = np.random.randn(self.num_iq_samples) + 0j
        modulated = np.exp(1j * np.pi / 2 * np.cumsum(source) / 2.0)

        if not self.random_data:
            np.random.set_state(orig_state)  # return numpy back to its previous state

        return modulated[-self.num_iq_samples:]
