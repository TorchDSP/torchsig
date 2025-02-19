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
from torchsig.transforms.signal_transforms import (
    IQImbalanceSignalTransform,
    CarrierPhaseOffsetSignalTransform,
    Fading,
    SpectralInversionSignalTransform
)
from torchsig.transforms.dataset_transforms import (
    IQImbalanceDatasetTransform,
    CarrierPhaseOffsetDatasetTransform,
    SpectralInversionDatasetTransform
)

# Third Party
import numpy as np



# Narrowband Signal Transforms
ST_level_0 = []
ST_level_1 = [
    IQImbalanceSignalTransform(
        amplitude_imbalance = (-1., 1.), 
        phase_imbalance = (-5.0 * np.pi / 180.0, 5.0 * np.pi / 180.0), 
        dc_offset = ((-0.1, 0.1),(-0.1, 0.1))
    ),
    CarrierPhaseOffsetSignalTransform()
]
ST_level_2 = [
    RandomApply(
        IQImbalanceSignalTransform(
            amplitude_imbalance = (-1., 1.), 
            phase_imbalance = (-5.0 * np.pi / 180.0, 5.0 * np.pi / 180.0), 
            dc_offset = ((-0.1, 0.1),(-0.1, 0.1)),
        ),
        0.9
    ),
    RandomApply(Fading(), 0.5),
    RandomApply(SpectralInversionSignalTransform(), 0.5)
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
    IQImbalanceDatasetTransform(
        amplitude_imbalance = (-1., 1.), 
        phase_imbalance = (-5.0 * np.pi / 180.0, 5.0 * np.pi / 180.0), 
        dc_offset = ((-0.1, 0.1),(-0.1, 0.1))
    ),
    CarrierPhaseOffsetDatasetTransform()
]
DT_level_2 = [
    RandomApply(
        IQImbalanceDatasetTransform(
            amplitude_imbalance = (-1., 1.), 
            phase_imbalance = (-5.0 * np.pi / 180.0, 5.0 * np.pi / 180.0), 
            dc_offset = ((-0.1, 0.1),(-0.1, 0.1))
        ), 
        0.9
    ),
    RandomApply(CarrierPhaseOffsetDatasetTransform(), 0.9),
    RandomApply(SpectralInversionDatasetTransform(), 0.5),
]

DT_all_levels = [
    DT_level_0,
    DT_level_1,
    DT_level_2
]



class NarrowbandImpairments(Impairments):
    """Applies impairments to Narrowband dataset

    """
    def __init__(self, level: int, **kwargs):
        """Initializes narrowband impairments

        Args:
            level (int): Impairment level (0-2).
        """        
        super().__init__(
            all_levels_signal_transforms = ST_all_levels,
            all_levels_dataset_transforms = DT_all_levels,
            level = level,
            **kwargs
        )