"""Narrowband Transforms and Impairments for Impairment Levels 0-2

Impairments are transforms applied to Signal objects, after the Signal Builder generates an isolated signal.
Transforms are applied to DatasetSignal objects, after isolated signals are placed on an IQ cut of noise.

Example:
    >>> impairments = NarrowbandImpairments(level = 2, dataset_metadata=dm)
    >>> sb = SignalBuilder(...)
    >>> new_signal = sb.build()
    >>> impaired_new_signal = impairments(new_signal)

    >>> iq_samples = <random noise>
    >>> iq_samples[start:stop] += new_signal.data
    >>> new_dataset_signal = DatasetSignal(data=iq_samples, metadata=[impaired_new_signal.metadata])

    >>> transforms = NarrowbandTransforms(level = 2, dataset_metadata=dm)
    >>> transformed_dataset_signal = transforms(new_dataset_signal)
"""

# TorchSig
from torchsig.transforms.impairments import Impairments
from torchsig.transforms.base_transforms import RandomApply
from torchsig.transforms.transforms import (
    CarrierFrequencyDrift,
    CarrierPhaseNoise,
    CarrierPhaseOffset,    
    CoarseGainChange,
    DigitalAGC,
    Fading,
    IntermodulationProducts,
    IQImbalance,
    Quantize,
    SpectralInversion,
    Spurs
)

# Third Party
import numpy as np

class NarrowbandImpairments(Impairments):
    """Applies impairments to Narrowband dataset

    """
    def __init__(self, level: int, **kwargs):
        """Initializes narrowband impairments

        Args:
            level (int): Impairment level (0-2).
        """

        # Narrowband Signal Transforms
        ST_level_0 = []
        ST_level_1 = [
            IQImbalance(),
            CarrierPhaseOffset()
        ]
        ST_level_2 = [
            RandomApply(IQImbalance(),0.25),
            RandomApply(Fading(), 0.5),
            RandomApply(CarrierPhaseOffset(),1.0),
            RandomApply(SpectralInversion(), 0.25),
            RandomApply(CarrierPhaseNoise(), 0.5),
            RandomApply(CarrierFrequencyDrift(), 0.5),
            RandomApply(Quantize(), 0.5),
            RandomApply(IntermodulationProducts(), 0.5),
        ]
        
        ST_all_levels = [
            ST_level_0, 
            ST_level_1, 
            ST_level_2
        ]
        
        # Narrowband Dataset Transforms
        DT_level_0 = [
        
        ]
        DT_level_1 = [
            IQImbalance(),
            CarrierPhaseOffset()
        ]
        DT_level_2 = [
            RandomApply(IQImbalance(),0.5),
            RandomApply(CarrierPhaseOffset(), 1.0),
            RandomApply(SpectralInversion(), 0.5),
            RandomApply(CarrierPhaseNoise(), 0.5),
            RandomApply(CarrierFrequencyDrift(), 0.5),
            RandomApply(Quantize(), 1.0),
            RandomApply(CoarseGainChange(),0.1),
            RandomApply(DigitalAGC(),0.2),
            RandomApply(Spurs,0.5)
        ]
        
        DT_all_levels = [
            DT_level_0,
            DT_level_1,
            DT_level_2
        ]

        super().__init__(
            all_levels_signal_transforms = ST_all_levels,
            all_levels_dataset_transforms = DT_all_levels,
            level = level,
            **kwargs
        )
    
