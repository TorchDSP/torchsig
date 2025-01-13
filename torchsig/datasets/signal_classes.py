"""Config File that includes dictionaries of signal classes. Includes the default 53 modulations.
"""
from dataclasses import dataclass
from typing import ClassVar, Dict, List

CLASS_FAMILY_DICT: Dict[str, str] = {
    "4ask": "ask",
    "8ask": "ask",
    "16ask": "ask",
    "32ask": "ask",
    "64ask": "ask",
    "ook": "pam",
    "4pam": "pam",
    "8pam": "pam",
    "16pam": "pam",
    "32pam": "pam",
    "64pam": "pam",
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
    "cw_spike":"cw"
}

SIGNALS_SHARED_LIST: list = [
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
        "fm",
        "am-dsb-sc",
        "am-dsb",
        "am-lsb",
        "am-usb",
        "lfm_data",
        "lfm_radar",
        "chirpss",
        "cw_spike"
    ]

# list for radio ML 2018 dataset
FAMILY_SHARED_LIST: list = [
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
        "CW"
    ]

@dataclass
class torchsig_signals():
    """TorchSigNarrowband dataclass, containing class modulation names list `class_list`
    
    Example:
        Access this list::
            >>> from torchsig.datasets.signal_classes import torchsig_signals
            >>> torchsig_signals.class_list
            >>> torchsig_signals.family_dict
    """
    class_list: ClassVar[list[str]] = SIGNALS_SHARED_LIST
    family_dict: ClassVar[Dict[str, str]] = CLASS_FAMILY_DICT

    fsk_signals = []
    ofdm_signals = []
    constellation_signals = []
    am_signals = []
    fm_signals = []
    lfm_signals = []
    chirpss_signals = []
    cw_signals = []
    
    # automatic grouping of each signal into a specific class
    for name in class_list:
        if ('fsk' in name or 'msk' in name):
            fsk_signals.append(name)
        elif ('ofdm' in name):
            ofdm_signals.append(name)
        elif ('pam' in name or 'ask' in name or 'qam' in name or 'psk' in name or 'ook' == name):
            constellation_signals.append(name)
        elif ('am-dsb' in name or 'am-lsb' == name or 'am-usb' == name):
            am_signals.append(name)
        elif ('fm' == name):
            fm_signals.append(name)
        elif ('lfm_' in name):
            lfm_signals.append(name)
        elif ('chirpss' == name):
            chirpss_signals.append(name)
        elif ('cw' in name):
            cw_signals.append(name)

    # specifically designed lists
    ofdm_subcarrier_modulations = ["bpsk", "qpsk", "16qam", "64qam", "256qam", "1024qam"]

@dataclass
class sig53():
    """Legacy Sig53 dataclass
    """

    class_list: ClassVar[list[str]] = SIGNALS_SHARED_LIST[:53]
    family_dict: ClassVar[Dict[str, str]] = CLASS_FAMILY_DICT



@dataclass
class radioml2018():
    """Radio ML 2016 dataclass, containing family class names list `family_class_list`
    
    Example:
        Access this list::
            >>> from torchsig.datasets.signal_classes import radioml2018
            >>> radioml2018.family_class_list
    """
    family_class_list: ClassVar[list[str]] = FAMILY_SHARED_LIST
