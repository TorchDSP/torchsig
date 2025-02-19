"""Wideband Transforms and Impairments for Impairment Levels 0-2

Impairments are transforms applied to Signal objects, after the Signal Builder generates an isolated signal.
Transforms are applied to DatasetSignal objects, after isolated signals are placed on an IQ cut of noise.

Example:
    >>> impairments = WidebandImpairments(level = 2, dataset_metadata=dm)
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
from torchsig.transforms.impairments import Impairments
from torchsig.transforms.base_transforms import (
    RandomApply,
    RandAugment
)
from torchsig.transforms.signal_transforms import (
    IQImbalanceSignalTransform,
    CarrierPhaseOffsetSignalTransform,
    Fading,
)
from torchsig.transforms.dataset_transforms import (
    IQImbalanceDatasetTransform,
    CarrierPhaseOffsetDatasetTransform,
    RandomDropSamples,
    ChannelSwap,
    TimeReversal,
    AddSlope,
)

# Third Party
import numpy as np



# Wideband Signal Transforms
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
            dc_offset = ((-0.1, 0.1),(-0.1, 0.1))
        ), 
        0.9),
    RandomApply(
        CarrierPhaseOffsetSignalTransform(), 
        0.9
    ),
    RandomApply(
        Fading(
            coherence_bandwidth = (0.001, 0.01)
        ), 
        0.5
    )
]

ST_all_levels = [
    ST_level_0,
    ST_level_1,
    ST_level_2
]

# Wideband Dataset Transforms
DT_level_0 = []
DT_level_1 = [
    IQImbalanceDatasetTransform(
        amplitude_imbalance = (-1., 1.),
        phase_imbalance = (-5.0 * np.pi / 180.0, 5.0 * np.pi / 180.0),
        dc_offset = ((-0.1, 0.1), (-0.1, 0.1))
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
    # RandomApply(AGC(), TBD),
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
DT_all_levels = [
    DT_level_0,
    DT_level_1,
    DT_level_2
]



class WidebandImpairments(Impairments):
    """Applies impairements to Wideband dataset

    """
    def __init__(self, level: int, **kwargs):
        """Initializes wideband impairments

        Args:
            level (int): Impairment level (0-2).
        """        
        super().__init__(
            all_levels_signal_transforms = ST_all_levels,
            all_levels_dataset_transforms = DT_all_levels,
            level = level,
            **kwargs
        )
