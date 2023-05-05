import numpy as np
from typing import Optional, List, Union


class SignalDescription:
    """A class containing typically necessary details to understand the data

    Args:
        sample_rate (:obj:`Optional[int]`):
            Sample rate of signal data
        num_iq_samples (:obj:`Optional[int]`):
            Number of IQ samples in the full example
        lower_frequency (:obj:`Optional[float]`):
            Lower frequency of signal data within example
        upper_frequency (:obj:`Optional[float]`):
            Upper frequency of signal data within example
        center_frequency (:obj:`Optional[float]`):
            Center frequency of signal data within example
        bandwidth (:obj:`Optional[float]`):
            Bandwidth of signal data within example
        start (:obj:`Optional[float]`):
            In [0.0, 1.0], the start of the signal data should be at `start*num_iq_samples`
        stop (:obj:`Optional[float]`):
            In [0.0, 1.0], the stop of the signal data should be at `stop*num_iq_samples`
        duration (:obj:`Optional[float]`):
            In [0.0, 1.0], the duration of the signal data should be `duration*num_iq_samples`
        snr (:obj:`Optional[float]`):
            Signal-to-noise ratio (SNR) of signal data in dB
        bits_per_symbol (:obj:`Optional[int]`):
            Bits per symbol of signal data
        samples_per_symbol (:obj:`Optional[float]`):
            IQ samples per symbol of the signal data
        excess_bandwidth (:obj:`Optional[float]`):
            Excess bandwidth of pulse shaping filter for signal data
        class_name (:obj:`Optional[str]`):
            Name of the signal's class
        class_index (:obj:`Optional[int]`):
            Index of the signal's class

    """

    def __init__(
        self,
        sample_rate: Optional[int] = 1,
        num_iq_samples: Optional[int] = 4096,
        lower_frequency: Optional[float] = -0.25,
        upper_frequency: Optional[float] = 0.25,
        center_frequency: Optional[float] = None,
        bandwidth: Optional[float] = None,
        start: Optional[float] = 0.0,
        stop: Optional[float] = 1.0,
        duration: Optional[float] = None,
        snr: Optional[float] = 0,
        bits_per_symbol: Optional[int] = 0,
        samples_per_symbol: Optional[float] = 0.0,
        excess_bandwidth: Optional[float] = 0.0,
        class_name: Optional[str] = None,
        class_index: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.num_iq_samples = num_iq_samples
        self.lower_frequency = (
            lower_frequency if lower_frequency else center_frequency - bandwidth / 2
        )
        self.upper_frequency = (
            upper_frequency if upper_frequency else center_frequency + bandwidth / 2
        )
        self.bandwidth = bandwidth if bandwidth else upper_frequency - lower_frequency
        self.center_frequency = (
            center_frequency
            if center_frequency
            else lower_frequency + self.bandwidth / 2
        )
        self.start = start
        self.stop = stop
        self.duration = duration if duration else stop - start
        self.snr = snr
        self.bits_per_symbol = bits_per_symbol
        self.samples_per_symbol = samples_per_symbol
        self.excess_bandwidth = excess_bandwidth
        self.class_name = class_name
        self.class_index = class_index


class SignalData:
    """A class representing signal data and typical meta-data to be used when
    applying signal transforms

    Args:
        data: bytes
            Signal data
        item_type: np.dtype
            Underlying real-valued precision of original data
        data_type: np.dtype
            Target real or complex-valued precision
        signal_description: Optional[Union[List[SignalDescription], SignalDescription]]
            Either a SignalDescription of signal data or a list of multiple
            SignalDescription objects describing multiple signals

    """

    def __init__(
        self,
        data: Optional[bytes],
        item_type: np.dtype,
        data_type: np.dtype,
        signal_description: Optional[
            Union[List[SignalDescription], SignalDescription]
        ] = None,
    ):
        self.iq_data = None
        self.signal_description = signal_description
        if data is not None:
            # No matter the underlying item type, we convert to double-precision
            self.iq_data = (
                np.frombuffer(data, dtype=item_type).astype(np.float64).view(data_type)
            )

        if not isinstance(signal_description, list):
            self.signal_description = [signal_description]


class SignalCapture:
    def __init__(
        self,
        absolute_path: str,
        num_bytes: int,
        item_type: np.dtype,
        is_complex: bool,
        byte_offset: int = 0,
        signal_description: Optional[SignalDescription] = None,
    ):
        self.absolute_path = absolute_path
        self.num_bytes = num_bytes
        self.item_type = item_type
        self.is_complex = is_complex
        self.byte_offset = byte_offset
        self.num_samples = self.num_bytes // self.item_type.itemsize
        self.num_samples //= 2 if is_complex else 1
        self.signal_description = signal_description
