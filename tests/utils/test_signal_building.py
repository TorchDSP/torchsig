import pytest

from torchsig.signals.builder import ConcatSignalGenerator
from torchsig.signals.builders.am import AMSignalGenerator
from torchsig.signals.builders.chirpss import ChirpSSSignalGenerator
from torchsig.signals.builders.constellation import ConstellationSignalGenerator
from torchsig.signals.builders.constellation_maps import all_symbol_maps
from torchsig.signals.builders.fm import FMSignalGenerator
from torchsig.signals.builders.fsk import FSKSignalGenerator
from torchsig.signals.builders.lfm import LFMSignalGenerator
from torchsig.signals.builders.ofdm import OFDMSignalGenerator
from torchsig.signals.builders.tone import ToneSignalGenerator
from torchsig.utils import signal_building
from torchsig.utils.signal_building import signal_generator_lookup_table


def test_exports_expected_public_symbols():
    assert set(signal_building.__all__) == {
        "signal_generator_lookup_table",
        "num_subcarrier_values",
        "family_names",
        "lookup_signal_generator_by_string",
    }


def test_tone_generator_entry():
    assert signal_building.signal_generator_lookup_table["tone"] == (
        ToneSignalGenerator,
        {},
    )


@pytest.mark.parametrize("num_subcarriers", signal_building.num_subcarrier_values)
def test_ofdm_generator_entries(num_subcarriers):
    assert signal_building.signal_generator_lookup_table[f"ofdm-{num_subcarriers}"] == (
        OFDMSignalGenerator,
        {"num_subcarriers": num_subcarriers},
    )


@pytest.mark.parametrize(
    ("name", "metadata"),
    [
        ("lfm-data", {"lfm_type": "data"}),
        ("lfm-radar", {"lfm_type": "radar"}),
    ],
)
def test_lfm_generator_entries(name, metadata):
    assert signal_building.signal_generator_lookup_table[name] == (
        LFMSignalGenerator,
        metadata,
    )


@pytest.mark.parametrize("fsk_type", ["fsk", "gfsk", "msk", "gmsk"])
@pytest.mark.parametrize("constellation_size", [2, 4, 8, 16])
def test_fsk_generator_entries(fsk_type, constellation_size):
    name = f"{constellation_size}{fsk_type}"

    assert signal_building.signal_generator_lookup_table[name] == (
        FSKSignalGenerator,
        {
            "fsk_type": fsk_type,
            "constellation_size": constellation_size,
        },
    )


def test_fm_generator_entry():
    assert signal_building.signal_generator_lookup_table["fm-data"] == (
        FMSignalGenerator,
        {},
    )


@pytest.mark.parametrize("constellation_name", all_symbol_maps)
def test_constellation_generator_entries(constellation_name):
    assert signal_building.signal_generator_lookup_table[constellation_name] == (
        ConstellationSignalGenerator,
        {"constellation_name": constellation_name},
    )


def test_chirpss_generator_entry():
    assert signal_building.signal_generator_lookup_table["chirpss"] == (
        ChirpSSSignalGenerator,
        {},
    )


@pytest.mark.parametrize("am_mode", ["dsb", "dsb-sc", "usb", "lsb"])
def test_am_generator_entries(am_mode):
    assert signal_building.signal_generator_lookup_table[f"am-{am_mode}"] == (
        AMSignalGenerator,
        {"am_mode": am_mode},
    )


def test_all_generator_contains_only_public_concrete_generators():
    generator_cls, generator_list, metadata = signal_building.signal_generator_lookup_table["all"]

    assert generator_cls is ConcatSignalGenerator
    assert metadata == {}

    public_concrete_names = [
        name
        for name in signal_building.signal_generator_lookup_table
        if name
        not in {
            "all",
            *signal_building.family_names,
            *signal_building.non_public_generator_names,
        }
    ]

    expected_generator_list = [signal_building.signal_generator_lookup_table[name] for name in public_concrete_names]

    assert generator_list == expected_generator_list


def test_non_public_generators_are_available_but_excluded_from_all():
    _, generator_list, _ = signal_building.signal_generator_lookup_table["all"]

    for name in signal_building.non_public_generator_names:
        assert name in signal_building.signal_generator_lookup_table
        assert signal_building.signal_generator_lookup_table[name] not in generator_list


@pytest.mark.parametrize("family_name", signal_building.family_names)
def test_family_entries_are_concat_generators(family_name):
    generator_cls, generator_list, metadata = signal_building.signal_generator_lookup_table[family_name]

    assert generator_cls is ConcatSignalGenerator
    assert metadata == {"family_name": family_name}
    assert len(generator_list) > 0


@pytest.mark.parametrize("family_name", signal_building.family_names)
def test_family_entries_are_selected_by_exact_family_name(family_name):
    _, generator_list, _ = signal_building.signal_generator_lookup_table[family_name]

    concrete_names = [name for name in signal_building.signal_generator_lookup_table if name not in {"all", *signal_building.family_names}]

    expected_generator_list = [signal_building.signal_generator_lookup_table[name] for name in concrete_names if signal_building._family_name(name) == family_name]

    assert generator_list == expected_generator_list


def test_am_family_does_not_include_qam_generators_regression():
    _, am_generators, _ = signal_building.signal_generator_lookup_table["am"]

    constellation_names = [metadata["constellation_name"] for _, metadata in am_generators if "constellation_name" in metadata]

    assert not any("qam" in name for name in constellation_names)


def test_fm_family_does_not_include_ofdm_generators_regression():
    _, fm_generators, _ = signal_building.signal_generator_lookup_table["fm"]

    assert fm_generators == [
        signal_building.signal_generator_lookup_table["fm-data"],
    ]


@pytest.mark.parametrize(
    ("signal_name", "expected_family"),
    [
        ("ofdm-64", "ofdm"),
        ("am-dsb", "am"),
        ("lfm-data", "lfm"),
        ("fm-data", "fm"),
        ("2fsk", "fsk"),
        ("4gfsk", "fsk"),
        ("8msk", "msk"),
        ("16gmsk", "msk"),
        ("tone", None),
        ("chirpss", None),
    ],
)
def test_family_name_classification(signal_name, expected_family):
    assert signal_building._family_name(signal_name) == expected_family


def test_lookup_signal_generator_by_string_instantiates_simple_generator(monkeypatch):
    class DummyGenerator:
        def __init__(self, metadata):
            self.metadata = metadata

    monkeypatch.setattr(
        signal_building,
        "signal_generator_lookup_table",
        {
            "dummy": (
                DummyGenerator,
                {"value": 1},
            )
        },
    )

    generator = signal_building.lookup_signal_generator_by_string("dummy")

    assert isinstance(generator, DummyGenerator)
    assert generator.metadata == {"value": 1}


def test_lookup_signal_generator_by_string_instantiates_concat_generator(monkeypatch):
    class DummyGenerator:
        def __init__(self, metadata):
            self.metadata = metadata

    class DummyConcatGenerator:
        def __init__(self, signal_generators, metadata):
            self.signal_generators = signal_generators
            self.metadata = metadata

    monkeypatch.setattr(signal_building, "ConcatSignalGenerator", DummyConcatGenerator)
    monkeypatch.setattr(
        signal_building,
        "signal_generator_lookup_table",
        {
            "dummy-family": (
                DummyConcatGenerator,
                [
                    (DummyGenerator, {"value": 1}),
                    (DummyGenerator, {"value": 2}),
                ],
                {"family_name": "dummy"},
            )
        },
    )

    generator = signal_building.lookup_signal_generator_by_string("dummy-family")

    assert isinstance(generator, DummyConcatGenerator)
    assert generator.metadata == {"family_name": "dummy"}
    assert [child.metadata for child in generator.signal_generators] == [
        {"value": 1},
        {"value": 2},
    ]


def test_lookup_signal_generator_by_string_raises_value_error_for_missing_name():
    with pytest.raises(ValueError, match="could not instantiate signal generator: 'missing'"):
        signal_building.lookup_signal_generator_by_string("missing")


def test_lookup_signal_generator_by_string_raises_value_error_for_bad_lookup_data(
    monkeypatch,
):
    monkeypatch.setattr(
        signal_building,
        "signal_generator_lookup_table",
        {
            "bad": (
                object,
                [],
                {},
            )
        },
    )

    with pytest.raises(ValueError, match="could not instantiate signal generator: 'bad'"):
        signal_building.lookup_signal_generator_by_string("bad")


def test_am_family_does_not_include_qam_generators():
    _, am_generators, metadata = signal_generator_lookup_table["am"]

    assert metadata == {"family_name": "am"}
    assert len(am_generators) > 0

    am_constellation_names = [generator_metadata["constellation_name"] for _, generator_metadata in am_generators if "constellation_name" in generator_metadata]

    assert not any("qam" in name for name in am_constellation_names)
