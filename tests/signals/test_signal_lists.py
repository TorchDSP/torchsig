"""Unit Tests for signals/signal_lists.py

Classes:
- TorchSigSignalLists
- radioml2018

Variables:
- CLASS_FAMILY_DICT
- SIGNALS_SHARED_LIST
- FAMILY_SHARED_LIST
- FAMILY_SHARED_LIST_RADIOML
"""
from torchsig.signals.signal_lists import (
    CLASS_FAMILY_DICT,
    SIGNALS_SHARED_LIST,
    FAMILY_SHARED_LIST,
    TorchSigSignalLists
)
from torchsig.utils.verify import (
    verify_dict,
    verify_list
)

import pytest

@pytest.mark.parametrize("l, name", [
    (SIGNALS_SHARED_LIST, "SIGNAL_SHARED_LIST"),
    (FAMILY_SHARED_LIST, "FAMILY_SHARED_LIST"),
    (TorchSigSignalLists.all_signals, "TorchSigSignalLists.all_signals"),
    (TorchSigSignalLists.family_list, "TorchSigSignalLists.family_list"),
    (TorchSigSignalLists.fsk_signals, "TorchSigSignalLists.fsk_signals"),
    (TorchSigSignalLists.ofdm_signals, "TorchSigSignalLists.ofdm_signals"),
    (TorchSigSignalLists.constellation_signals, "TorchSigSignalLists.constellation_signals"),
    (TorchSigSignalLists.am_signals, "TorchSigSignalLists.am_signals"),
    (TorchSigSignalLists.fm_signals, "TorchSigSignalLists.fm_signals"),
    (TorchSigSignalLists.lfm_signals, "TorchSigSignalLists.lfm_signals"),
    (TorchSigSignalLists.chirpss_signals, "TorchSigSignalLists.chirpss_signals"),
    (TorchSigSignalLists.tone_signals, "TorchSigSignalLists.tone_signals"),
    (TorchSigSignalLists.ofdm_subcarrier_modulations, "TorchSigSignalLists.ofdm_subcarrier_modulations"),
])
def test_signal_lists(l: list, name: str):
    # no duplicates
    verify_list(
        l,
        name = name,
        no_duplicates = True,
        data_type = str
    )

    # all lower case
    for s in l:
        if any(char.isupper() for char in s):
            raise ValueError(f"{name} has uppercase characters: {s}")

@pytest.mark.parametrize("l, name, required_strs", [
    (TorchSigSignalLists.fsk_signals, "TorchSigSignalLists.fsk_signals", ['fsk', 'msk']),
    (TorchSigSignalLists.ofdm_signals, "TorchSigSignalLists.ofdm_signals", ['ofdm']),
    (TorchSigSignalLists.constellation_signals, "TorchSigSignalLists.constellation_signals", ['ask', 'qam', 'psk', 'ook']),
    (TorchSigSignalLists.am_signals, "TorchSigSignalLists.am_signals", ['am-dsb', 'am-lsb','am-usb']),
    (TorchSigSignalLists.fm_signals, "TorchSigSignalLists.fm_signals", ['fm']),
    (TorchSigSignalLists.lfm_signals, "TorchSigSignalLists.lfm_signals", ['lfm_']),
    (TorchSigSignalLists.chirpss_signals, "TorchSigSignalLists.chirpss_signals", ['chirpss']),
    (TorchSigSignalLists.tone_signals, "TorchSigSignalLists.tone_signals", ['tone']),
    (TorchSigSignalLists.ofdm_subcarrier_modulations, "TorchSigSignalLists.ofdm_subcarrier_modulations", ["bpsk", "qpsk", "16qam", "64qam", "256qam", "1024qam"]),
])
def test_signal_groups_lists(l: list, name: str, required_strs: list):
    for signal_name in l:
        if not any(req_str in signal_name for req_str in required_strs):
            raise ValueError(f"{name} has signal that seems to not belong in the group {required_strs}: {signal_name}")

def test_CLASS_FAMILY_DICT():
    verify_dict(
        CLASS_FAMILY_DICT,
        name = "CLASS_FAMILY_DICT",
        required_keys = SIGNALS_SHARED_LIST,
        required_types = [str] * len(SIGNALS_SHARED_LIST)
    )

    # check all values in family class list are in the dict
    family_classes_from_dict = CLASS_FAMILY_DICT.values()
    for family_name in FAMILY_SHARED_LIST:
        if family_name not in family_classes_from_dict:
            raise ValueError(f"CLASS_FAMILY_DICT is missing family name {family_name} OR unknown family name {family_name} in FAMILY_SHARED_LIST")

def test_signal_list_dicts():
    verify_dict(
        TorchSigSignalLists.family_dict,
        name = "TorchSigSignalLists.family_dict,",
        required_keys = TorchSigSignalLists.all_signals,
        required_types = [str] * len(TorchSigSignalLists.all_signals)
    )

    # check all values in family class list are in the dict
    family_classes_from_dict = TorchSigSignalLists.family_dict.values()
    for family_name in TorchSigSignalLists.family_list:
        if family_name not in family_classes_from_dict:
            raise ValueError(f"TorchSigSignalLists.family_dict is missing family name {family_name} OR unknown family name {family_name} in TorchSigSignalLists.family_list")

    
