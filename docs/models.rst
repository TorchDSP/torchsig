Models
====================

Modeled after the torchvision.models module, this module contains model classes
used for loaded neural network architectures useful for experimenting in the
complex-valued signals domain. The module mirrors torchvision capabilities for
loading pretrained models for direct use and/or finetuning.

The following utilities are available:

.. contents:: Model Utilities
    :local:


IQ Models
------------------


EfficientNet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchsig.models.iq_models.efficientnet.efficientnet

.. autoclass:: efficientnet_b0

.. autoclass:: efficientnet_b2

.. autoclass:: efficientnet_b4


XCiT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchsig.models.iq_models.xcit.xcit

.. autoclass:: xcit_nano

.. autoclass:: xcit_tiny12


Spectrogram Models
------------------


.. YOLOv5
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. .. currentmodule:: torchsig.models.spectrogram_models.yolov5

.. .. autoclass:: yolov5p

.. .. autoclass:: yolov5n

.. .. autoclass:: yolov5s

.. .. autoclass:: yolov5p_mod_family

.. .. autoclass:: yolov5n_mod_family

.. .. autoclass:: yolov5s_mod_family


DETR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchsig.models.spectrogram_models.detr.detr

.. autoclass:: detr_b0_nano

.. autoclass:: detr_b2_nano

.. autoclass:: detr_b4_nano

.. autoclass:: detr_b0_nano_mod_family

.. autoclass:: detr_b2_nano_mod_family

.. autoclass:: detr_b4_nano_mod_family


.. PSPNet
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. .. currentmodule:: torchsig.models.spectrogram_models.pspnet

.. .. autoclass:: pspnet_b0

.. .. autoclass:: pspnet_b2

.. .. autoclass:: pspnet_b4

.. .. autoclass:: pspnet_b0_mod_family

.. .. autoclass:: pspnet_b2_mod_family

.. .. autoclass:: pspnet_b4_mod_family


.. Mask2Former
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. .. currentmodule:: torchsig.models.spectrogram_models.mask2former

.. .. autoclass:: mask2former_b0

.. .. autoclass:: mask2former_b2

.. .. autoclass:: mask2former_b4

.. .. autoclass:: mask2former_b0_mod_family

.. .. autoclass:: mask2former_b2_mod_family

.. .. autoclass:: mask2former_b4_mod_family
