"""Config File that includes dictionaries of signal classes. Includes the default modulations."""

# TorchSig
# Built-In
from dataclasses import dataclass
from typing import ClassVar, Final

from torchsig.signals.signal_utils import check_signal_class

# Signal class to signal family mapping
CLASS_FAMILY_DICT: Final[dict[str, str]] = {
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
    "tone": "tone",
}

# Derived lists
SIGNALS_SHARED_LIST: Final[list[str]] = list(CLASS_FAMILY_DICT.keys())
FAMILY_SHARED_LIST: Final[list[str]] = sorted(set(CLASS_FAMILY_DICT.values()))

# Constants
TORCHSIG_NUM_SIGNALS: Final[int] = len(SIGNALS_SHARED_LIST)
TORCHSIG_NUM_FAMILIES: Final[int] = len(FAMILY_SHARED_LIST)


@dataclass
class TorchSigSignalLists:
    """Various lists of signals available within TorchSig.

    Attributes:
        all_signals: List of all signal types.
        family_dict: Dictionary containing all signal types and their associated signal family.
        family_list: List of all signal families.
        fsk_signals: Frequency shift keying and FSK-related signals.
        ofdm_signals: Orthogonal frequency division multiplexing signals.
        constellation_signals: Linearly modulated constellation-based signals.
        am_signals: Amplitude modulation-based signals.
        fm_signals: Frequency Modulated signal.
        lfm_signals: Linearly frequency modulated signals.
        chirpss_signals: Chirp spread-spectrum signal.
        tone_signals: Tone signal.
    """

    all_signals: ClassVar[list[str]] = SIGNALS_SHARED_LIST
    family_dict: ClassVar[dict[str, str]] = CLASS_FAMILY_DICT
    family_list: ClassVar[list[str]] = FAMILY_SHARED_LIST

    fsk_signals: ClassVar[list[str]] = []
    ofdm_signals: ClassVar[list[str]] = []
    constellation_signals: ClassVar[list[str]] = []
    am_signals: ClassVar[list[str]] = []
    fm_signals: ClassVar[list[str]] = []
    lfm_signals: ClassVar[list[str]] = []
    chirpss_signals: ClassVar[list[str]] = []
    tone_signals: ClassVar[list[str]] = []

    # Signal family identifiers
    fsk_names: ClassVar[list[str]] = ["fsk", "msk"]
    ofdm_names: ClassVar[list[str]] = ["ofdm"]
    constellation_names: ClassVar[list[str]] = ["ask", "qam", "psk", "ook"]
    am_names: ClassVar[list[str]] = ["am-"]
    lfm_names: ClassVar[list[str]] = ["lfm_"]
    ofdm_subcarrier_modulations: ClassVar[list[str]] = [
        "bpsk",
        "qpsk",
        "16qam",
        "64qam",
        "256qam",
        "1024qam",
    ]

    def __post_init__(self) -> None:
        """Automatically groups each signal into its specific class."""
        for name in self.all_signals:
            if check_signal_class(name, self.fsk_names):
                self.fsk_signals.append(name)
            elif check_signal_class(name, self.ofdm_names):
                self.ofdm_signals.append(name)
            elif check_signal_class(name, self.constellation_names):
                self.constellation_signals.append(name)
            elif check_signal_class(name, self.am_names):
                self.am_signals.append(name)
            elif name == "fm":
                self.fm_signals.append(name)
            elif check_signal_class(name, self.lfm_names):
                self.lfm_signals.append(name)
            elif name == "chirpss":
                self.chirpss_signals.append(name)
            elif name == "tone":
                self.tone_signals.append(name)
