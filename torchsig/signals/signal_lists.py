"""Config File that includes dictionaries of signal classes. Includes the default modulations.
"""
# TorchSig
from torchsig.signals.signal_utils import check_signal_class

# Built-In
from dataclasses import dataclass
from typing import ClassVar, Dict




CLASS_FAMILY_DICT: Dict[str, str] = {
    "ook": "ook",
    "4ask": "ask",
    "8ask": "ask",
    "16ask": "ask",
    "32ask": "ask",
    "64ask": "ask",
    "2fsk": "fsk",
    "2gfsk": "fsk",
    "2msk": "fsk",
    "2gmsk": "fsk",
    "4fsk": "fsk",
    "4gfsk": "fsk",
    "4msk": "fsk",
    "4gmsk": "fsk",
    "8fsk": "fsk",
    "8gfsk": "fsk",
    "8msk": "fsk",
    "8gmsk": "fsk",
    "16fsk": "fsk",
    "16gfsk": "fsk",
    "16msk": "fsk",
    "16gmsk": "fsk",
    "bpsk": "psk",
    "qpsk": "psk",
    "8psk": "psk",
    "16psk": "psk",
    "32psk": "psk",
    "64psk": "psk",
    "16qam": "qam",
    "32qam": "qam",
    "32qam_cross": "qam",
    "64qam": "qam",
    "128qam_cross": "qam",
    "256qam": "qam",
    "512qam_cross": "qam",
    "1024qam": "qam",
    "ofdm-64": "ofdm",
    "ofdm-72": "ofdm",
    "ofdm-128": "ofdm",
    "ofdm-180": "ofdm",
    "ofdm-256": "ofdm",
    "ofdm-300": "ofdm",
    "ofdm-512": "ofdm",
    "ofdm-600": "ofdm",
    "ofdm-900": "ofdm",
    "ofdm-1024": "ofdm",
    "ofdm-1200": "ofdm",
    "ofdm-2048": "ofdm",
    "fm": "fm",
    "am-dsb-sc": "am",
    "am-dsb": "am",
    "am-lsb": "am",
    "am-usb": "am",
    "lfm_data": "chirp",
    "lfm_radar": "chirp",
    "chirpss": "chirp",
    "tone": "tone"
}

SIGNALS_SHARED_LIST: list = [
        "ook",
        "bpsk",
        "4ask",
        "qpsk",
        "8ask",
        "8psk",
        "16qam",
        "16ask",
        "16psk",
        "32qam",
        "32qam_cross",
        "32ask",
        "32psk",
        "64qam",
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
        "fm",
        "am-dsb-sc",
        "am-dsb",
        "am-lsb",
        "am-usb",
        "lfm_data",
        "lfm_radar",
        "chirpss",
        "tone",
    ]

FAMILY_SHARED_LIST: list = sorted(list(set(CLASS_FAMILY_DICT.values())))

# list for radio ML 2018 dataset
FAMILY_SHARED_LIST_RADIOML: list = [
        "OOK",
        "4ASK",
        "8ASK",
        "BPSK",
        "QPSK",
        "8PSK",
        "16PSK",
        "32PSK",
        "16APSK",
        "32APSK",
        "64APSK",
        "128APSK",
        "16QAM",
        "32QAM",
        "64QAM",
        "128QAM",
        "256QAM",
        "AM-SSB-WC",
        "AM-SSB-SC",
        "AM-DSB-WC",
        "AM-DSB-SC",
        "FM",
        "GMSK",
        "OQPSK",
    ]

@dataclass
class TorchSigSignalLists():
    """Various lists of signals available within TorchSig

    Attributes:
        all_signals (list[str]): List of all signal types.
        family_dict (dict[str, str]]: List dict which contains all signal types and their associated signal family
        family_list: (list[str]): List of all the values from family_dict

        fsk_signals (list[str]): Frequency shift keying and FSK-related signals; FSK, GFSK, MSK, GMSK
        ofdm_signals (list[str]): Orthogonal frequency division multiplexing signals OFDM-64, OFDM-600, OFDM-1024, and others.
        constellation_signals (list[str]): Linearly modulated constellation-based signals, contains QAM, PSK, ASK and OOK signals.
        am_signals (list[str]): Amplitude modulation-based signals: AM-DSB, AM-DSB-SC, AM-USB, AM-LSB
        fm_signals (list[str]): Frequency Modulated signal
        lfm_signals (list[str]): Linearly frequency modulated signals, includes LFM data and LFM radar
        chirpss_signals (list[str]): Chirp spread-spectrum signal
        tone_signals (list[str]): Tone signal

    Example:
        Access this list::
            >>> from torchsig.signals.signal_lists import TorchSigSignalLists
            >>> TorchSigSignalLists.all_signals
            >>> TorchSigSignalLists.family_dict
    """

    all_signals: ClassVar[list[str]] = SIGNALS_SHARED_LIST
    family_dict: ClassVar[Dict[str, str]] = CLASS_FAMILY_DICT
    family_list: ClassVar[list[str]] = FAMILY_SHARED_LIST

    fsk_signals = []
    ofdm_signals = []
    constellation_signals = []
    am_signals = []
    fm_signals = []
    lfm_signals = []
    chirpss_signals = []
    tone_signals = []

    fsk_names = ['fsk', 'msk']
    ofdm_names = ['ofdm']
    constellation_names = ['ask', 'qam', 'psk', 'ook']
    am_names = ['am-dsb', 'am-lsb', 'am-usb']
    lfm_names = ['lfm_']
    
    # automatic grouping of each signal into a specific class
    for name in all_signals:
        if check_signal_class(name, fsk_names):
            fsk_signals.append(name)
        elif check_signal_class(name, ofdm_names):
            ofdm_signals.append(name)
        elif check_signal_class(name, constellation_names):
            constellation_signals.append(name)
        elif check_signal_class(name, am_names):
            am_signals.append(name)
        elif 'fm' == name:
            fm_signals.append(name)
        elif check_signal_class(name, lfm_names):
            lfm_signals.append(name)
        elif 'chirpss' == name:
            chirpss_signals.append(name)
        elif 'tone' == name:
            tone_signals.append(name)

    # specifically designed lists
    ofdm_subcarrier_modulations = ["bpsk", "qpsk", "16qam", "64qam", "256qam", "1024qam"]


@dataclass
class RadioML2018():
    """Radio ML 2016 dataclass, containing family class names list `family_class_list`
    
    Example:
        Access this list::
            >>> from torchsig.signals.signal_lists import radioml2018
            >>> radioml2018.family_class_list
    """
    family_class_list: ClassVar[list[str]] = FAMILY_SHARED_LIST_RADIOML
