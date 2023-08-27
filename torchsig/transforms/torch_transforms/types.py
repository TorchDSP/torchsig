###############################################################################
# This file should be moved/merged with the existing types.py within
# torchsig/utils/ once the torch transforms are properly supported.
#
###############################################################################

from typing import List, Optional, TypedDict

import torch


class SignalMetadata:
    def __init__(self, sample_rate: float, num_samples: int) -> None:
        self.sample_rate = sample_rate
        self.num_samples = num_samples


class Signal:
    def __init__(self, data: torch.Tensor, metadata: List[SignalMetadata],) -> None:
        self.data = data
        self.metadata = metadata


class RFMetadata(SignalMetadata):
    def __init__(
        self,
        sample_rate: float,
        num_samples: int,
        complex: bool,
        lower_freq: float,
        upper_freq: float,
        center_freq: float,
        bandwidth: float,
        start: float,
        stop: float,
        duration: float,
        snr: float,
    ) -> None:
        super(RFMetadata, self).__init__(
            sample_rate=sample_rate, num_samples=num_samples,
        )
        self.complex = complex
        self.lower_freq = (lower_freq,)
        self.upper_freq = (upper_freq,)
        self.center_freq = (center_freq,)
        self.bandwidth = (bandwidth,)
        self.start = (start,)
        self.stop = (stop,)
        self.duration = (duration,)
        self.snr = (snr,)


class ModulatedRFMetadata(RFMetadata):
    def __init__(
        self,
        sample_rate: float,
        num_samples: int,
        complex: bool,
        lower_freq: float,
        upper_freq: float,
        center_freq: float,
        bandwidth: float,
        start: float,
        stop: float,
        duration: float,
        snr: float,
        bits_per_symbol: int,
        samples_per_symbol: float,
        excess_bandwidth: float,
        class_name: str,
        class_index: int,
    ) -> None:
        super(ModulatedRFMetadata, self).__init__(
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
        )
        self.bits_per_symbol = bits_per_symbol
        self.samples_per_symbol = samples_per_symbol
        self.excess_bandwidth = excess_bandwidth
        self.class_name = class_name
        self.class_index = class_index
