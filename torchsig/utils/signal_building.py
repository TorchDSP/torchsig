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

from typing import Any, Dict, List, Tuple, Union

# Stores generator class and metadata for generators to make per label
signal_generator_lookup_table: dict[str,
    tuple[type, dict[str, any]] |
    tuple[type, list[tuple[type, dict[str, Any]]], dict[str, Any]]
] = {}

# Initialize lookup table with signal generators
signal_generator_lookup_table["tone"] = (ToneSignalGenerator, {})
num_subcarrier_values = [64, 72, 128, 180, 256, 300, 512, 600, 900, 1024, 1200, 2048]
for num_subcarriers in num_subcarrier_values:
    signal_generator_lookup_table["ofdm-" + str(num_subcarriers)] = (
        OFDMSignalGenerator,
        {"num_subcarriers": num_subcarriers},
    )
signal_generator_lookup_table["lfm-data"] = (LFMSignalGenerator, {"lfm_type": "data"})
signal_generator_lookup_table["lfm-radar"] = (LFMSignalGenerator, {"lfm_type": "radar"})
for fsk_type in ["fsk", "gfsk", "msk", "gmsk"]:
    for constellation_size in [2, 4, 8, 16]:
        signal_generator_lookup_table[str(constellation_size) + str(fsk_type)] = (
            FSKSignalGenerator,
            {"fsk_type": fsk_type, "constellation_size": constellation_size},
        )
signal_generator_lookup_table["fm"] = (FMSignalGenerator, {})
for constellation_name in all_symbol_maps:
    signal_generator_lookup_table[constellation_name] = (
        ConstellationSignalGenerator,
        {"constellation_name": constellation_name},
    )
signal_generator_lookup_table["chirpss"] = (ChirpSSSignalGenerator, {})
for am_mode in ["dsb", "dsb-sc", "usb", "lsb"]:
    signal_generator_lookup_table["am-" + am_mode] = (
        AMSignalGenerator,
        {"am_mode": am_mode},
    )
signal_generator_lookup_table["all"] = (
    ConcatSignalGenerator,
    [
        signal_generator_lookup_table[key]
        for key in signal_generator_lookup_table
    ],
    {},
)
family_names = ["ofdm", "am", "fm", "fsk", "psk", "qam", "ask", "lfm", "msk"]
for family_name in family_names:
    signal_generator_lookup_table[family_name] = (
        ConcatSignalGenerator,
        [
            signal_generator_lookup_table[key]
            for key in signal_generator_lookup_table
            if family_name in key
        ],
        {"family_name": family_name},
    )


def lookup_signal_generator_by_string(signal_generator_name: str) -> Any:
    """Look up and instantiate a signal generator by its name.

    This function searches the signal_generator_lookup_table for the given name
    and returns an instantiated signal generator. It handles both simple generators
    and concatenated generators (ConcatSignalGenerator).

    Args:
        signal_generator_name: The name of the signal generator to instantiate.

    Returns:
        An instantiated signal generator object.

    Raises:
        ValueError: If the signal generator name is not found in the lookup table
            or if there's an error instantiating the generator.
    """
    try:
        lookup_value = signal_generator_lookup_table[signal_generator_name]
        if len(lookup_value) == 2:
            generator_init, metadata = lookup_value
            return generator_init(metadata=metadata)
        if len(lookup_value) == 3 and lookup_value[0] == ConcatSignalGenerator:
            generator_init, generator_list, metadata = lookup_value
            return generator_init(
                signal_generators=[el[0](metadata=el[1]) for el in generator_list],
                metadata=metadata,
            )
        raise KeyError("bad data found in generator lookup table")
    except KeyError:
        raise ValueError(
            "could not instantiate signal generator: '"
            + str(signal_generator_name)
            + "'"
        )
