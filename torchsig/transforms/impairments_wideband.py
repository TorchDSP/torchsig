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
    LocalOscillatorPhaseNoiseSignalTransform,
    LocalOscillatorFrequencyDriftSignalTransform,
    QuantizeSignalTransform,
    IntermodulationProductsSignalTransform
)
from torchsig.transforms.dataset_transforms import (
    IQImbalanceDatasetTransform,
    CarrierPhaseOffsetDatasetTransform,
    LocalOscillatorPhaseNoiseDatasetTransform,
    LocalOscillatorFrequencyDriftDatasetTransform,
    QuantizeDatasetTransform,
    RandomDropSamples,
    ChannelSwap,
    TimeReversal,
    AddSlope,
)

# Third Party
import numpy as np

class WidebandImpairments(Impairments):
    """Applies impairements to Wideband dataset

    """
    def __init__(self, level: int, **kwargs):
        """Initializes wideband impairments

        Args:
            level (int): Impairment level (0-2).
        """

        # Wideband Signal Transforms
        ST_level_0 = []
        ST_level_1 = [
            IQImbalanceSignalTransform(),
            CarrierPhaseOffsetSignalTransform()
        ]
        ST_level_2 = [
            RandomApply(IQImbalanceSignalTransform(),0.25),
            RandomApply(CarrierPhaseOffsetSignalTransform(),1.0),
            RandomApply(Fading(coherence_bandwidth = (0.001, 0.01)),0.5),
            RandomApply(LocalOscillatorPhaseNoiseSignalTransform(), 0.5),
            RandomApply(LocalOscillatorFrequencyDriftSignalTransform(), 0.5),
            RandomApply(QuantizeSignalTransform(), 0.5),
            RandomApply(IntermodulationProductsSignalTransform(), 0.5),
        ]
        
        ST_all_levels = [
            ST_level_0,
            ST_level_1,
            ST_level_2
        ]
        
        # Wideband Dataset Transforms
        DT_level_0 = []
        DT_level_1 = [
            IQImbalanceDatasetTransform(),
            CarrierPhaseOffsetDatasetTransform()
        ]
        DT_level_2 = [
            RandomApply(IQImbalanceDatasetTransform(),0.5),
            RandomApply(CarrierPhaseOffsetDatasetTransform(), 1.0),
            RandomApply(LocalOscillatorPhaseNoiseDatasetTransform(),0.5),
            RandomApply(LocalOscillatorFrequencyDriftDatasetTransform(),0.5),
            RandomApply(QuantizeDatasetTransform(),0.5),
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

        super().__init__(
            all_levels_signal_transforms = ST_all_levels,
            all_levels_dataset_transforms = DT_all_levels,
            level = level,
            **kwargs
        )
