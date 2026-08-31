from torchsig.signals.signal_lists import CLASS_FAMILY_DICT
from torchsig.utils.signal_building import signal_generator_lookup_table


def test_signal_generator_keys_match_class_family():
    # CLASS_FAMILY_DICT.keys() should be a subset of signal_generator_lookup_table.keys()
    assert set(signal_generator_lookup_table.keys()) >= set(CLASS_FAMILY_DICT.keys())
