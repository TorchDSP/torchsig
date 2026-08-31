from typing import Any

from torchsig.signals.builder import ConcatSignalGenerator
from torchsig.signals.builders.adsb import AdsBSignalGenerator
from torchsig.signals.builders.am import AMSignalGenerator
from torchsig.signals.builders.btle import BTLESignalGenerator
from torchsig.signals.builders.cellular import GSMSignalGenerator
from torchsig.signals.builders.chirpss import ChirpSSSignalGenerator
from torchsig.signals.builders.constellation import ConstellationSignalGenerator
from torchsig.signals.builders.constellation_maps import all_symbol_maps
from torchsig.signals.builders.dvb import DVBS2SignalGenerator
from torchsig.signals.builders.fm import FMSignalGenerator
from torchsig.signals.builders.fsk import FSKSignalGenerator
from torchsig.signals.builders.lfm import LFMSignalGenerator
from torchsig.signals.builders.lmr import DMRSignalGenerator, P25SignalGenerator
from torchsig.signals.builders.lora import LoraSignalGenerator
from torchsig.signals.builders.ofdm import OFDMSignalGenerator
from torchsig.signals.builders.tone import ToneSignalGenerator
from torchsig.signals.builders.wifi import Wifi80211aSignalGenerator
from torchsig.signals.builders.zigbee import ZigBeeSignalGenerator

__all__ = ["family_names", "lookup_signal_generator_by_string", "num_subcarrier_values", "signal_generator_lookup_table"]

# Stores generator class and metadata for generators to make per label
SignalGeneratorSpec = tuple[type, dict[str, Any]] | tuple[type, list[tuple[type, dict[str, Any]]], dict[str, Any]]

non_public_generator_names = {"80211a_ack", "80211a_cts", "80211a_rts"}

signal_generator_lookup_table: dict[str, SignalGeneratorSpec] = {}


def _add_signal_generator(
    name: str,
    generator_cls: type,
    metadata: dict[str, Any],
) -> None:
    signal_generator_lookup_table[name] = (generator_cls, metadata)


def _family_name(signal_name: str) -> str | None:
    """Return the modulation family for a concrete signal generator name."""
    if signal_name.startswith("ofdm-"):
        return "ofdm"
    if signal_name.startswith("am-"):
        return "am"
    if signal_name.startswith("lfm-"):
        return "lfm"
    if signal_name.startswith("fm-"):
        return "fm"
    if signal_name.endswith("fsk") or signal_name.endswith("gfsk"):
        return "fsk"
    if signal_name.endswith("msk") or signal_name.endswith("gmsk"):
        return "msk"
    if "psk" in signal_name:
        return "psk"
    if "qam" in signal_name:
        return "qam"
    if "ask" in signal_name:
        return "ask"
    if "adsb" in signal_name:
        return "adsb"
    if "btle" in signal_name:
        return "btle"
    if "gsm" in signal_name:
        return "cellular"
    if "dvb" in signal_name:
        return "dvb"
    if ("dmr" in signal_name) or ("p25" in signal_name):
        return "lmr"
    if "lora" in signal_name:
        return "lora"
    if "80211a" in signal_name:
        return "wifi"
    if "zigbee" in signal_name:
        return "zigbee"
    return None


# Initialize lookup table with signal generators
_add_signal_generator("tone", ToneSignalGenerator, {})

num_subcarrier_values = [64, 72, 128, 180, 256, 300, 512, 600, 900, 1024, 1200, 2048]
for num_subcarriers in num_subcarrier_values:
    _add_signal_generator(
        f"ofdm-{num_subcarriers}",
        OFDMSignalGenerator,
        {"num_subcarriers": num_subcarriers},
    )

_add_signal_generator("lfm-data", LFMSignalGenerator, {"lfm_type": "data"})
_add_signal_generator("lfm-radar", LFMSignalGenerator, {"lfm_type": "radar"})

for fsk_type in ["fsk", "gfsk", "msk", "gmsk"]:
    for constellation_size in [2, 4, 8, 16]:
        _add_signal_generator(
            f"{constellation_size}{fsk_type}",
            FSKSignalGenerator,
            {"fsk_type": fsk_type, "constellation_size": constellation_size},
        )

_add_signal_generator("fm-data", FMSignalGenerator, {})

for constellation_name in all_symbol_maps:
    _add_signal_generator(
        constellation_name,
        ConstellationSignalGenerator,
        {"constellation_name": constellation_name},
    )

_add_signal_generator("chirpss", ChirpSSSignalGenerator, {})

for am_mode in ["dsb", "dsb-sc", "usb", "lsb"]:
    _add_signal_generator(
        f"am-{am_mode}",
        AMSignalGenerator,
        {"am_mode": am_mode},
    )

_add_signal_generator("adsb-long", AdsBSignalGenerator, {"frame_type": "long"})
_add_signal_generator("adsb-short", AdsBSignalGenerator, {"frame_type": "short"})
_add_signal_generator("btle", BTLESignalGenerator, {})
_add_signal_generator("gsm", GSMSignalGenerator, {})
_add_signal_generator("dvbs2", DVBS2SignalGenerator, {})
_add_signal_generator("dmr", DMRSignalGenerator, {})
_add_signal_generator("p25", P25SignalGenerator, {})
_add_signal_generator("lora", LoraSignalGenerator, {})
_add_signal_generator("80211a", Wifi80211aSignalGenerator, {"frame_type": "data"})
_add_signal_generator("80211a_ack", Wifi80211aSignalGenerator, {"frame_type": "ack"})
_add_signal_generator("80211a_cts", Wifi80211aSignalGenerator, {"frame_type": "cts"})
_add_signal_generator("80211a_rts", Wifi80211aSignalGenerator, {"frame_type": "rts"})
_add_signal_generator("zigbee", ZigBeeSignalGenerator, {})

# convert to list
concrete_generator_names = list(signal_generator_lookup_table)
public_generator_names = [name for name in concrete_generator_names if name not in non_public_generator_names]

signal_generator_lookup_table["all"] = (
    ConcatSignalGenerator,
    [signal_generator_lookup_table[name] for name in public_generator_names],
    {},
)

family_names = ["ofdm", "am", "fm", "fsk", "psk", "qam", "ask", "lfm", "msk", "adsb"]
for family_name in family_names:
    signal_generator_lookup_table[family_name] = (
        ConcatSignalGenerator,
        [signal_generator_lookup_table[name] for name in concrete_generator_names if _family_name(name) == family_name],
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
        raise ValueError("could not instantiate signal generator: '" + str(signal_generator_name) + "'")
