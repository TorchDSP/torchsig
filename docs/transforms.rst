Transforms
======================

.. currentmodule:: torchsig.transforms

Transforms are common signal transformations. They can be chained together using :class:`Compose`.
Additionally, there is the :mod:`torchsig.transforms.functional` module.
Functional transforms give fine-grained control over the transformations.
This is useful if you have to build a more complex transformation pipeline

.. contents:: Transforms
    :local:

Transforms
----------
.. currentmodule:: torchsig.transforms.transforms

Transform
^^^^^^^^^
.. autoclass:: Transform

Compose
^^^^^^^^^
.. autoclass:: Compose

Identity
^^^^^^^^^
.. autoclass:: Identity

Lambda
^^^^^^^^^
.. autoclass:: Lambda

FixedRandom
^^^^^^^^^^^^^
.. autoclass:: FixedRandom

RandomApply
^^^^^^^^^^^^^
.. autoclass:: RandomApply

SignalTransform
^^^^^^^^^^^^^^^^^
.. autoclass:: SignalTransform

Concatenate
^^^^^^^^^^^^^
.. autoclass:: Concatenate

TargetConcatenate
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: TargetConcatenate

InterleaveComplex
^^^^^^^^^^^^^^^^^
.. autoclass:: InterleaveComplex

ComplexTo2D
^^^^^^^^^^^^^
.. autoclass:: ComplexTo2D

Real
^^^^^^^^^
.. autoclass:: Real

Imag
^^^^^^^^^
.. autoclass:: Imag

ComplexMagnitude
^^^^^^^^^^^^^^^^^
.. autoclass:: ComplexMagnitude

WrappedPhase
^^^^^^^^^^^^^
.. autoclass:: WrappedPhase

DiscreteFourierTransform
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DiscreteFourierTransform

ChannelConcatIQDFT
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: ChannelConcatIQDFT

Spectrogram
^^^^^^^^^^^^^
.. autoclass:: Spectrogram

ContinuousWavelet
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: ContinuousWavelet

ReshapeTransform
^^^^^^^^^^^^^^^^^
.. autoclass:: ReshapeTransform

RandAugment
^^^^^^^^^^^^^
.. autoclass:: RandAugment

Normalize
^^^^^^^^^
.. autoclass:: Normalize


Augmentations
-------------
.. currentmodule:: torchsig.transforms.transforms

DatasetBasebandMixUp
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DatasetBasebandMixUp

DatasetBasebandCutMix
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DatasetBasebandCutMix

CutOut
^^^^^^^^^
.. autoclass:: CutOut

PatchShuffle
^^^^^^^^^^^^^
.. autoclass:: PatchShuffle

DatasetWidebandMixUp
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DatasetWidebandMixUp

DatasetWidebandCutMix
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DatasetWidebandCutMix

SpectrogramRandomResizeCrop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SpectrogramRandomResizeCrop

RandomResample
^^^^^^^^^^^^^^^^^
.. autoclass:: RandomResample

RandomTimeShift
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: RandomTimeShift

TimeCrop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: TimeCrop

TimeReversal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: TimeReversal

AmplitudeReversal 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AmplitudeReversal 

RandomFrequencyShift
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: RandomFrequencyShift

RandomDelayedFrequencyShift
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: RandomDelayedFrequencyShift

LocalOscillatorDrift
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: LocalOscillatorDrift

GainDrift
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GainDrift

AutomaticGainControl
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AutomaticGainControl

IQImbalance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: IQImbalance

RollOff
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: RollOff

AddSlope
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AddSlope

SpectralInversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SpectralInversion

ChannelSwap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: ChannelSwap

RandomMagRescale 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: RandomMagRescale

RandomDropSamples  
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: RandomDropSamples

Quantize   
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Quantize 

Clip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Clip 

RandomConvolve 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: RandomConvolve 

TargetSNR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: TargetSNR

AddNoise
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AddNoise

TimeVaryingNoise 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: TimeVaryingNoise 

RayleighFadingChannel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: RayleighFadingChannel

ImpulseInterferer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: ImpulseInterferer

RandomPhaseShift
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: RandomPhaseShift

SpectrogramDropSamples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SpectrogramDropSamples

SpectrogramPatchShuffle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SpectrogramPatchShuffle

SpectrogramTranslation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SpectrogramTranslation

SpectrogramMosaicCrop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SpectrogramMosaicCrop

SpectrogramMosaicDownsample
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SpectrogramMosaicDownsample


Target Transforms
-----------------
.. currentmodule:: torchsig.transforms.target_transforms

DescToClassName
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToClassName

DescToClassNameSNR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToClassNameSNR

DescToClassIndex
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToClassIndex

DescToClassIndexSNR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToClassIndexSNR

DescToMask
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToMask

DescToMaskSignal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToMaskSignal

DescToMaskFamily
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToMaskFamily

DescToMaskClass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToMaskClass

DescToSemanticClass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToSemanticClass

DescToBBox
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToBBox

DescToAnchorBoxes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToAnchorBoxes

DescPassThrough
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescPassThrough

DescToBinary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToBinary

DescToCustom
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToCustom

DescToClassEncoding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToClassEncoding

DescToWeightedMixUp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToWeightedMixUp

DescToWeightedCutMix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToWeightedCutMix

DescToBBoxDict
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToBBoxDict

DescToBBoxSignalDict
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToBBoxSignalDict

DescToBBoxFamilyDict
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToBBoxFamilyDict

DescToInstMaskDict
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToInstMaskDict

DescToSignalInstMaskDict
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToSignalInstMaskDict

DescToSignalFamilyInstMaskDict
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToSignalFamilyInstMaskDict

DescToListTuple
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DescToListTuple

ListTupleToDesc
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: ListTupleToDesc

LabelSmoothing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: LabelSmoothing

