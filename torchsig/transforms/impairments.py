"""Dataset Transform/Impairment class

Impairments are transforms applied to Signal objects, after the Signal Builder generates an isolated signal.
Transforms are applied to DatasetSignal objects, after isolated signals are placed on an IQ cut of noise.

Example:
    >>> impairments = Impairments(level = 2, dataset_metadata=dm)
    >>> iq_samples = <random noise>
    >>> metadatas = []
    >>> for i in range(3): # 3 signals in wideband sample
    >>>     sb = SignalBuilder(...)
    >>>     new_signal = sb.build()
    >>>     impaired_new_signal = impairments(new_signal)
    >>>     iq_samples[start:stop] += new_signal.data
    >>>     metadatas.append(impaired_new_signal.metadata)

    >>> new_dataset_signal = DatasetSignal(data=iq_samples, metadata=metadatas)

    >>> transforms = WidebandTransforms(level = 2, dataset_metadata=dm)
    >>> transformed_dataset_signal = transforms(new_dataset_signal)

"""

# TorchSig
from torchsig.transforms.base_transforms import Compose, Transform
from torchsig.transforms.transforms import SignalTransform

from torchsig.transforms.base_transforms import (
    RandomApply,
    RandAugment
)

from torchsig.transforms.transforms import (
    AddSlope,
    CarrierFrequencyDrift,
    CarrierPhaseNoise,
    CarrierPhaseOffset,
    ChannelSwap,
    CoarseGainChange,
    DigitalAGC,
    Fading,
    IntermodulationProducts,
    IQImbalance,
    NonlinearAmplifier,
    Quantize,
    RandomDropSamples,
    Spurs,
    SpectralInversion,
    Spurs,
    TimeReversal,
)


# Built-In
from typing import List
from copy import copy

class Impairments(Transform):
    """Applies signal and dataset transformations at specific impairment levels.

    This class applies a set of signal and dataset transforms based on a given impairment
    level. The impairment level must be between 0 and 2, where each level corresponds to
    different sets of transformations for signals and datasets.
    * Level 0: Perfect
    * Level 1: Cabled enviornment
    * Level 2: Wireless environment

    Args:
        level (int): The impairment level (must be between 0 and 2).
        **kwargs: Additional keyword arguments passed to the parent class `Transform`.

    Raises:
        ValueError: If the provided impairment level is outside the valid range (0, 1, 2).

    Attributes:
        level (int): The specified impairment level.
        signal_transforms (Compose): The composed signal transformations corresponding to the given impairment level.
        dataset_transforms (Compose): The composed dataset transformations corresponding to the given impairment level.
    """
    def __init__(
        self, 
        level: int, 
        **kwargs
    ): 
        super().__init__(**kwargs)

        self.level = level

        # listing of transmit and receive HW impairments
        tx_hw_impairments = [
            RandomApply(Quantize(),0.75),
            # RandomApply(,), # clock jitter
            # RandomApply(,), # clock drift
            # RandomApply(,), # passband ripple
            RandomApply(CarrierPhaseNoise(),0.75),
            RandomApply(CarrierFrequencyDrift(),0.75),
            RandomApply(CarrierPhaseOffset(),1.0),
            RandomApply(IntermodulationProducts(),0.5),
            RandomApply(IQImbalance(),0.25),
            RandomApply(NonlinearAmplifier(),0.75),
            RandomApply(Spurs(),0.75),
            RandomApply(SpectralInversion(),0.25),
        ]

        rx_hw_impairments = [
            # RandomApply(,), # image rejection
            RandomApply(NonlinearAmplifier(),0.75),
            RandomApply(CoarseGainChange(),0.25),
            RandomApply(Spurs(),0.75),
            RandomApply(CarrierPhaseNoise(),0.75),
            RandomApply(CarrierFrequencyDrift(),0.75),
            RandomApply(CarrierPhaseOffset(),1.0),
            RandomApply(IntermodulationProducts(),0.5),
            RandomApply(IQImbalance(),0.5),
            # RandomApply(,), # passband ripple
            # RandomApply(,), # clock jitter
            # RandomApply(,), # clock drift
            RandomApply(Quantize(),0.75),
            RandomApply(DigitalAGC(),0.25),
        ]

        # define ML transforms
        ml_transforms = [
            RandAugment(
                transforms= [
                    RandomDropSamples(
                        drop_rate = 0.01,
                        size = (1,1),
                        fill = ["ffill", "bfill", "mean", "zero"]
                    ),
                    ChannelSwap(),
                    TimeReversal(),
                    AddSlope()
                ],
                choose=2,
                replace=False
            )
        ]

        # listing of channel models
        channel_models = [
            RandomApply(Fading(),0.25),
        ]

        # Signal (TX) Transforms
        ST_level_0 = []                                # None
        ST_level_1 = copy(tx_hw_impairments)           # TX impairments
        ST_level_2 = copy(ST_level_1) + channel_models # TX impairments + channel models

        ST_all_levels = [
            ST_level_0,
            ST_level_1,
            ST_level_2
        ]

        # Dataset (RX) Transforms
        DT_level_0 = copy(ml_transforms)            # ML Transforms
        DT_level_1 = DT_level_0 + rx_hw_impairments # ML transforms + HW impairments
        DT_level_2 = copy(DT_level_1)               # ML transforms + HW impairments

        DT_all_levels = [
            DT_level_0,
            DT_level_1,
            DT_level_2
        ]

        self.signal_transforms = Compose(transforms = ST_all_levels[self.level])
        self.signal_transforms.add_parent(self)

        self.dataset_transforms = Compose(transforms = DT_all_levels[self.level])
        self.dataset_transforms.add_parent(self)
