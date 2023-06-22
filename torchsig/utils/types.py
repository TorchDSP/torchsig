from typing import List, Optional, TypedDict
import numpy as np


class SignalMetadata(TypedDict):
    sample_rate: int
    num_samples: int


class SignalData(TypedDict):
    samples: np.ndarray


class Signal(TypedDict):
    data: SignalData
    metadata: List[SignalMetadata]


class RFMetadata(SignalMetadata):
    complex: bool
    lower_freq: float
    upper_freq: float
    center_freq: float
    bandwidth: float
    start: float
    stop: float
    duration: float
    snr: float


class ModulatedRFMetadata(RFMetadata):
    bits_per_symbol: int
    samples_per_symbol: float
    excess_bandwidth: float
    class_name: str
    class_index: int


# TODO[GV] Very niche class, probably can refactor this out.
class SignalCapture:
    def __init__(
        self,
        absolute_path: str,
        num_bytes: int,
        item_type: np.dtype,
        is_complex: bool,
        byte_offset: int = 0,
        signal_description: Optional[SignalMetadata] = None,
    ) -> None:
        self.absolute_path = absolute_path
        self.num_bytes = num_bytes
        self.item_type = item_type
        self.is_complex = is_complex
        self.byte_offset = byte_offset
        self.num_samples = self.num_bytes // self.item_type.itemsize
        self.num_samples //= 2 if is_complex else 1
        self.signal_description = signal_description
