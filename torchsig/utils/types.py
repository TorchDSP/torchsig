from typing import List, Optional, Union
import numpy as np


class RandomDistribution:
    rng = np.random.default_rng()

    @staticmethod
    def to_distribution(dist):
        if isinstance(dist, RandomDistribution):
            return dist

        if isinstance(dist, (int, float)):
            return ConstantRD(dist)

        if isinstance(dist, tuple):
            if len(dist) == 1:
                return ConstantRD(dist[0])

            return UniformContinuousRD(dist[0], dist[1])

        if isinstance(dist, list):
            if len(dist) == 1:
                return ConstantRD(dist[0])

            return UniformDiscreteRD(dist)

        if isinstance(dist, np.ndarray):
            if dist.shape[0] == 1:
                return ConstantRD(dist[0])

            return UniformDiscreteRD(dist)

    def __call__(self, num: int = 1):
        raise NotImplementedError


class ConstantRD(RandomDistribution):
    def __init__(self, constant: float) -> None:
        super(ConstantRD, self).__init__()
        self.constant = constant

    def __call__(self, num: int = 1):
        if num == 1:
            return self.constant

        return np.repeat(self.constant, repeats=num)


class UniformContinuousRD(RandomDistribution):
    def __init__(self, low: float, high: float) -> None:
        super(UniformContinuousRD, self).__init__()
        self.low = low
        self.high = high

    def __call__(self, num: int = 1) -> np.ndarray:
        return RandomDistribution.rng.uniform(low=self.low, high=self.high, size=num)


class UniformDiscreteRD(RandomDistribution):
    def __init__(self, choices: np.ndarray) -> None:
        super(UniformDiscreteRD, self).__init__()
        self.choices = choices

    def __call__(self, num: int = 1) -> np.ndarray:
        return RandomDistribution.rng.choice(self.choices, size=num)


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
        sample_rate: Optional[float] = 1,
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
    ) -> None:
        self.sample_rate = sample_rate
        self.num_iq_samples = num_iq_samples
        if center_frequency and bandwidth:
            self.lower_frequency: Optional[float] = (
                lower_frequency if lower_frequency else center_frequency - bandwidth / 2
            )
            self.upper_frequency: Optional[float] = (
                upper_frequency if upper_frequency else center_frequency + bandwidth / 2
            )
        else:
            self.lower_frequency = lower_frequency
            self.upper_frequency = upper_frequency
        if lower_frequency and upper_frequency:
            self.bandwidth: Optional[float] = (
                bandwidth if bandwidth else upper_frequency - lower_frequency
            )
            self.center_frequency: Optional[float] = (
                center_frequency
                if center_frequency
                else lower_frequency + self.bandwidth / 2
            )
        else:
            self.bandwidth = bandwidth
            self.center_frequency = center_frequency
        self.start = start
        self.stop = stop
        self.duration: Optional[float] = stop - start if start and stop else duration
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
    ) -> None:
        self.iq_data: Optional[np.ndarray] = None
        self.signal_description: Optional[
            Union[List[SignalDescription], SignalDescription]
        ] = signal_description
        if data is not None:
            # No matter the underlying item type, we convert to double-precision
            self.iq_data = (
                np.frombuffer(data, dtype=item_type).astype(np.float64).view(data_type)
            )

        self.signal_description = (
            [signal_description]
            if not isinstance(signal_description, list) and signal_description
            else signal_description
        )


class SignalCapture:
    def __init__(
        self,
        absolute_path: str,
        num_bytes: int,
        item_type: np.dtype,
        is_complex: bool,
        byte_offset: int = 0,
        signal_description: Optional[SignalDescription] = None,
    ) -> None:
        self.absolute_path = absolute_path
        self.num_bytes = num_bytes
        self.item_type = item_type
        self.is_complex = is_complex
        self.byte_offset = byte_offset
        self.num_samples = self.num_bytes // self.item_type.itemsize
        self.num_samples //= 2 if is_complex else 1
        self.signal_description = signal_description
