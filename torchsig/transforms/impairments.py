"""Dataset Transform/Impairment class

Impairments are transforms applied to Signal objects, after the Signal Builder generates an isolated signal.
Transforms are applied to Signal objects, after isolated signals are placed on an IQ cut of noise.
"""

# TorchSig
# Built-In
from copy import copy

from torchsig.transforms.base_transforms import Compose, RandAugment, RandomApply, Transform
from torchsig.transforms.transforms import (
    AddSlope,
    CarrierFrequencyDrift,
    CarrierPhaseNoise,
    CarrierPhaseOffset,
    ChannelSwap,
    ClockDrift,
    ClockJitter,
    CoarseGainChange,
    DigitalAGC,
    Fading,
    IntermodulationProducts,
    IQImbalance,
    NonlinearAmplifier,
    PassbandRipple,
    Quantize,
    RandomDropSamples,
    SpectralInversion,
    Spurs,
    TimeReversal,
)


class Impairments(Transform):
    """Applies signal and dataset transformations at specific impairment levels.

    This class applies a set of signal and dataset transforms based on a given impairment
    level. The impairment level must be between 0 and 2, where each level corresponds to
    different sets of transformations for signals and datasets.

    * Level 0: Perfect (no impairments)
    * Level 1: Cabled environment (transmit impairments only)
    * Level 2: Wireless environment (transmit impairments + channel models)

    Args:
        level: The impairment level (must be between 0 and 2).
        **kwargs: Additional keyword arguments passed to the parent class `Transform`.

    Raises:
        ValueError: If the provided impairment level is outside the valid range (0, 1, 2).

    Attributes:
        level: The specified impairment level.
        signal_transforms: The composed signal transformations corresponding to the given impairment level.
        dataset_transforms: The composed dataset transformations corresponding to the given impairment level.
    """

    def __init__(self, level: int, **kwargs):
        """Initialize the Impairments class.

        Args:
            level: The impairment level (must be between 0 and 2).
            **kwargs: Additional keyword arguments passed to the parent class `Transform`.
        """
        super().__init__(**kwargs)

        self.level = level

        # listing of transmit and receive HW impairments
        tx_hw_impairments = [
            RandomApply(Quantize(), 0.75),
            RandomApply(ClockDrift(), 0.75),
            RandomApply(ClockJitter(), 0.75),
            RandomApply(PassbandRipple(), 0.75),
            RandomApply(IQImbalance(), 0.25),
            RandomApply(CarrierPhaseNoise(), 0.75),
            RandomApply(CarrierFrequencyDrift(), 0.75),
            RandomApply(CarrierPhaseOffset(), 1.0),
            RandomApply(IntermodulationProducts(), 0.5),
            RandomApply(NonlinearAmplifier(), 0.75),
            RandomApply(Spurs(), 0.75),
            RandomApply(SpectralInversion(), 0.25),
        ]

        rx_hw_impairments = [
            RandomApply(IntermodulationProducts(), 0.5),
            RandomApply(NonlinearAmplifier(), 0.75),
            RandomApply(CoarseGainChange(), 0.25),
            RandomApply(Spurs(), 0.75),
            RandomApply(IQImbalance(), 0.5),
            RandomApply(CarrierPhaseNoise(), 0.75),
            RandomApply(CarrierFrequencyDrift(), 0.75),
            RandomApply(CarrierPhaseOffset(), 1.0),
            RandomApply(PassbandRipple(), 0.75),
            RandomApply(ClockDrift(), 0.75),
            RandomApply(ClockJitter(), 0.75),
            RandomApply(Quantize(), 0.75),
            RandomApply(DigitalAGC(), 0.25),
        ]

        # define ML transforms
        ml_transforms = [
            RandAugment(
                transforms=[
                    RandomDropSamples(
                        drop_rate=0.01,
                        size=(1, 1),
                        fill=["ffill", "bfill", "mean", "zero"],
                    ),
                    ChannelSwap(),
                    TimeReversal(),
                    AddSlope(),
                ],
                choose=2,
                replace=False,
            )
        ]

        # listing of channel models
        channel_models = [
            RandomApply(Fading(), 0.25),
        ]

        # Signal (TX) Transforms
        st_level_0 = []  # None
        st_level_1 = copy(tx_hw_impairments)  # TX impairments
        st_level_2 = (
            copy(st_level_1) + channel_models
        )  # TX impairments + channel models

        st_all_levels = [st_level_0, st_level_1, st_level_2]

        # Dataset (RX) Transforms
        dt_level_0 = copy(ml_transforms)  # ML Transforms
        dt_level_1 = dt_level_0 + rx_hw_impairments  # ML transforms + HW impairments
        dt_level_2 = copy(dt_level_1)  # ML transforms + HW impairments

        dt_all_levels = [dt_level_0, dt_level_1, dt_level_2]

        self.signal_transforms = Compose(transforms=st_all_levels[self.level])
        self.signal_transforms.add_parent(self)

        self.dataset_transforms = Compose(transforms=dt_all_levels[self.level])
        self.dataset_transforms.add_parent(self)

    def get_signal_transforms(self) -> list[Transform]:
        """Get the signal transforms for this impairment level.

        Returns:
            List of signal transforms configured for the current impairment level.
        """
        return self.signal_transforms.transforms

    def get_dataset_transforms(self) -> list[Transform]:
        """Get the dataset transforms for this impairment level.

        Returns:
            List of dataset transforms configured for the current impairment level.
        """
        return self.dataset_transforms.transforms
