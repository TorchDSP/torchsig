Transforms
======================

.. currentmodule:: torchsig.transforms

Transforms are common signal transformations. They can be chained together using :class:`Compose`.
Additionally, there is the :mod:`torchsig.transforms.functional` module.
Functional transforms give fine-grained control over the transformations.
This is useful if you have to build a more complex transformation pipeline

.. contents:: Transforms
    :local:

General Transforms
------------------
.. currentmodule:: torchsig.transforms

Transform
^^^^^^^^^
.. autoclass:: Transform

Compose
^^^^^^^^^
.. autoclass:: Compose

NoTransform
^^^^^^^^^^^^^
.. autoclass:: NoTransform

Lamda
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

RandAugment
^^^^^^^^^^^^^
.. autoclass:: RandAugment


Deep Learning Techniques
------------------------
.. currentmodule:: torchsig.transforms.deep_learning_techniques.dlt

DatasetBasebandMixUp
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DatasetBasebandMixUp

DatasetBasebandCutMix
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DatasetBasebandCutMix

CutOut
^^^^^^^^^
.. autoclass:: CutOut


Expert Feature Transforms
-------------------------
.. currentmodule:: torchsig.transforms.expert_feature.eft

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


Signal Processing Transforms
----------------------------
.. currentmodule:: torchsig.transforms.signal_processing.sp

Normalize
^^^^^^^^^
.. autoclass:: Normalize

RandomResample
^^^^^^^^^^^^^^^^^
.. autoclass:: RandomResample


System Impairment Transforms
-----------------------------
.. currentmodule:: torchsig.transforms.system_impairment.si

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


Wireless Channel Transforms
----------------------------
.. currentmodule:: torchsig.transforms.wireless_channel.wce

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
