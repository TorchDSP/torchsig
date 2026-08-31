from dataclasses import fields

from torchsig.signals.signal_lists import CLASS_FAMILY_DICT, TorchSigSignalLists
from torchsig.utils.signal_building import lookup_signal_generator_by_string

CONTROL_FRAME_GENERATORS = {"80211a_ack", "80211a_cts", "80211a_rts"}


def _grouped_signals(signal_lists: TorchSigSignalLists) -> list[str]:
    """Collect every per-category signal list declared on TorchSigSignalLists."""
    return [signal_name for field in fields(signal_lists) for signal_name in getattr(signal_lists, field.name)]


def test_all_generators_match_public_signal_classes_exactly():
    """The aggregate generator contains exactly the public signal classes."""
    all_generator = lookup_signal_generator_by_string("all")
    generated_class_names = [generator.class_name for generator in all_generator.signal_generators]

    assert len(generated_class_names) == len(set(generated_class_names))
    assert set(generated_class_names) == set(CLASS_FAMILY_DICT)


def test_wifi_control_frame_generators_remain_available():
    for generator_name in CONTROL_FRAME_GENERATORS:
        generator = lookup_signal_generator_by_string(generator_name)

        assert generator.class_name == generator_name


def test_torchsig_signal_lists_groups_expected_signal_families():
    signal_lists = TorchSigSignalLists()

    assert "fm" in signal_lists.fm_signals
    assert "tone" in signal_lists.tone_signals
    assert "chirpss" in signal_lists.chirpss_signals

    assert all("ofdm" in name for name in signal_lists.ofdm_signals)
    assert all(any(key in name for key in ["fsk", "msk"]) for name in signal_lists.fsk_signals)
    assert all(any(key in name for key in ["ask", "qam", "psk", "ook"]) for name in signal_lists.constellation_signals)
    assert all("am-" in name for name in signal_lists.am_signals)
    assert all("lfm-" in name for name in signal_lists.lfm_signals)


def test_torchsig_signal_lists_assigns_every_known_signal_to_exactly_one_group():
    signal_lists = TorchSigSignalLists()
    grouped_signals = _grouped_signals(signal_lists)

    assert len(grouped_signals) == len(set(grouped_signals))
    assert set(grouped_signals) == set(signal_lists.all_signals)
    assert CONTROL_FRAME_GENERATORS.isdisjoint(grouped_signals)


def test_torchsig_signal_lists_repeated_init_does_not_duplicate_entries():
    first = TorchSigSignalLists()
    first_grouped_count = len(_grouped_signals(first))

    second = TorchSigSignalLists()
    second_grouped_count = len(_grouped_signals(second))

    assert second_grouped_count == first_grouped_count
