from typing import List, Optional, TypedDict
from torch import Tensor
import numpy as np

n_type = (float, int, Tensor, np.float16)


# --------------------------------------------------------------------------------- #
# SignalMetadata
# --------------------------------------------------------------------------------- #
class SignalMetadata(TypedDict):
    sample_rate: float
    num_samples: int


def create_signal_metadata(
    sample_rate: float = 0.0, num_samples: int = 0
) -> SignalMetadata:
    return SignalMetadata(sample_rate=sample_rate, num_samples=num_samples)


def is_signal_metadata(d: SignalMetadata) -> bool:
    if "sample_rate" not in d.keys() or "num_samples" not in d.keys():
        return False

    return isinstance(d["sample_rate"], n_type) and isinstance(d["num_samples"], n_type)  # type: ignore


# --------------------------------------------------------------------------------- #
# RFMetadata
# --------------------------------------------------------------------------------- #
class RFMetadata(SignalMetadata):
    complex: bool
    lower_freq: float
    upper_freq: float
    center_freq: float
    bandwidth: float
    start: float
    stop: float
    duration: float


def create_rf_metadata(
    sample_rate: float = 0,
    num_samples: int = 0,
    complex: bool = True,
    lower_freq: float = -0.25,
    upper_freq: float = 0.25,
    center_freq: float = 0.0,
    bandwidth: float = 0.5,
    start: float = 0.0,
    stop: float = 1.0,
    duration: float = 1.0,
) -> RFMetadata:
    return RFMetadata(
        sample_rate=sample_rate,
        num_samples=num_samples,
        complex=complex,
        lower_freq=lower_freq,
        upper_freq=upper_freq,
        center_freq=center_freq,
        bandwidth=bandwidth,
        start=start,
        stop=stop,
        duration=duration,
    )


def is_rf_metadata(d: SignalMetadata) -> bool:
    rf_keys = (
        "complex",
        "lower_freq",
        "upper_freq",
        "center_freq",
        "bandwidth",
        "start",
        "stop",
        "duration",
    )
    rf_types = (bool, n_type, n_type, n_type, n_type, n_type, n_type, n_type)
    if not all(k in d for k in rf_keys):
        return False

    if not is_signal_metadata(d):
        return False

    return all(isinstance(d[k], t) for k, t in zip(rf_keys, rf_types))  # type: ignore


def has_rf_metadata(metadata: List[SignalMetadata]) -> bool:
    return any(is_rf_metadata(m) for m in metadata)


def meta_bound_frequency(meta: SignalMetadata) -> SignalMetadata:
    # Check bounds for partial signals
    meta["lower_freq"] = np.clip(meta["lower_freq"], a_min=-0.5, a_max=0.5)
    meta["upper_freq"] = np.clip(meta["upper_freq"], a_min=-0.5, a_max=0.5)
    meta["bandwidth"] = meta["upper_freq"] - meta["lower_freq"]
    meta["center_freq"] = (meta["lower_freq"] + meta["upper_freq"]) * 0.5
    return meta


def meta_pad_height(
    meta: SignalMetadata, height: float, pixel_height: float, pad_start: float
):
    meta["lower_freq"] = (
        (meta["lower_freq"] + 0.5) * height + pad_start
    ) / pixel_height - 0.5
    meta["upper_freq"] = (
        (meta["upper_freq"] + 0.5) * height + pad_start
    ) / pixel_height - 0.5
    meta["center_freq"] = (
        (meta["center_freq"] + 0.5) * height + pad_start
    ) / pixel_height - 0.5
    meta["bandwidth"] = meta["upper_freq"] - meta["lower_freq"]


# --------------------------------------------------------------------------------- #
# ModulatedRFMetadata
# --------------------------------------------------------------------------------- #
class ModulatedRFMetadata(RFMetadata):
    snr: float
    bits_per_symbol: float
    samples_per_symbol: float
    excess_bandwidth: float
    class_name: str
    class_index: int


def create_modulated_rf_metadata(
    sample_rate: float = 0.0,
    num_samples: int = 0,
    complex: bool = True,
    lower_freq: float = -0.25,
    upper_freq: float = 0.25,
    center_freq: float = 0.0,
    bandwidth: float = 0.5,
    start: float = 0.0,
    stop: float = 1.0,
    duration: float = 1.0,
    snr: float = 0.0,
    bits_per_symbol: float = 0.0,
    samples_per_symbol: float = 0.0,
    excess_bandwidth: float = 0.0,
    class_name: str = "",
    class_index: int = 0,
) -> RFMetadata:
    return ModulatedRFMetadata(
        sample_rate=sample_rate,
        num_samples=num_samples,
        complex=complex,
        lower_freq=lower_freq,
        upper_freq=upper_freq,
        center_freq=center_freq,
        bandwidth=bandwidth,
        start=start,
        stop=stop,
        duration=duration,
        snr=snr,
        bits_per_symbol=bits_per_symbol,
        samples_per_symbol=samples_per_symbol,
        excess_bandwidth=excess_bandwidth,
        class_name=class_name,
        class_index=class_index,
    )


def is_rf_modulated_metadata(d: SignalMetadata) -> bool:
    mod_keys = (
        "snr",
        "bits_per_symbol",
        "samples_per_symbol",
        "excess_bandwidth",
        "class_name",
        "class_index",
    )
    mod_types = (n_type, n_type, n_type, n_type, str, int)
    if not all(k in d for k in mod_keys):
        return False

    if not is_rf_metadata(d):
        return False

    return all(isinstance(d[k], t) for k, t in zip(mod_keys, mod_types))  # type: ignore


def has_modulated_rf_metadata(metadata: List[SignalMetadata]) -> bool:
    return any([is_rf_modulated_metadata(m) for m in metadata])


# --------------------------------------------------------------------------------- #
# SignalData
# --------------------------------------------------------------------------------- #
class SignalData(TypedDict):
    samples: np.ndarray


def create_signal_data(samples: np.ndarray = np.empty((1,))) -> SignalData:
    return SignalData(samples=samples)


def is_signal_data(d: SignalData) -> bool:
    if "samples" not in d.keys():
        return False

    return isinstance(d["samples"], np.ndarray)


def data_shape(data: SignalData) -> tuple:
    return data["samples"].shape


# --------------------------------------------------------------------------------- #
# Signal
#
# --------------------------------------------------------------------------------- #
class Signal(TypedDict):
    data: SignalData
    metadata: List[SignalMetadata]


def create_signal(data: SignalData, metadata: List[SignalMetadata]) -> Signal:
    return Signal(data=data, metadata=metadata)


def is_signal(d: Signal) -> bool:
    if not isinstance(d, dict):
        return False

    if "data" not in d.keys() or "metadata" not in d.keys():
        return False

    if not isinstance(d["metadata"], list):
        return False

    return is_signal_data(d["data"]) and all(
        [is_signal_metadata(m) for m in d["metadata"]]
    )


# TODO[GV] Very niche class, probably can refactor this out.
class SignalCapture:
    def __init__(
        self,
        absolute_path: str,
        num_bytes: int,
        item_type: np.dtype,
        is_complex: bool,
        byte_offset: int = 0,
        metadata: Optional[SignalMetadata] = None,
    ) -> None:
        self.absolute_path = absolute_path
        self.num_bytes = num_bytes
        self.item_type = item_type
        self.is_complex = is_complex
        self.byte_offset = byte_offset
        self.num_samples = self.num_bytes // self.item_type.itemsize
        self.num_samples //= 2 if is_complex else 1
        self.meta = metadata if metadata else create_signal_metadata()
