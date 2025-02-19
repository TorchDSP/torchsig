Models
====================

Note: We are depreciating support on models, but will continue support for our pretrained models.

Modeled after the torchvision.models module, this module contains model classes
used for loaded neural network architectures useful for experimenting in the
complex-valued signals domain. The module mirrors torchvision capabilities for
loading pretrained models for direct use and/or finetuning.

The following utilities are available:

.. contents:: Model Utilities
    :local:


IQ Models
------------------


XCiT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchsig.models.iq_models.xcit.xcit

.. autoclass:: xcit_nano

.. autoclass:: xcit_tiny12


DETR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchsig.models.spectrogram_models.detr.detr

.. autoclass:: detr_b0_nano

.. autoclass:: detr_b2_nano

.. autoclass:: detr_b4_nano

.. autoclass:: detr_b0_nano_mod_family

.. autoclass:: detr_b2_nano_mod_family

.. autoclass:: detr_b4_nano_mod_family