Transforms
======================

.. currentmodule:: torchsig.transforms.tranforms

Transforms are applied to signals or samples to emulate transmitter and reciever effects, as well as tools for machine learning.
There are four types of transforms, that differ in purpose and scope.

1. :class:`torchsig.transforms.signal_transforms.SignalTransform` - applied to isolated signals from the signal builder, and typically represent transmitter effects.
2. :class:`torchsig.transforms.dataset_transforms.DatasetTransform` - applied to samples, after isolated signals are placed onto a noise floor. Typically represents reciever effects and other machine learning transforms.
3. Functionals - core logic of both Signal Transforms and Dataset Transforms. Users can use for more fine-grained control of the transform.
4. :class:`torchsig.transforms.impairments.DatasetImpairments` - a collection of Signal Transforms and Dastaset Transforms that represent an environment, such as wireless.

.. contents:: Transforms
    :local:

Transforms
----------------------

Base Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: torchsig.transforms.base_transforms
    :members:
    :undoc-members:
    :show-inheritance:

Signal Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: torchsig.transforms.signal_transforms
    :members:
    :undoc-members:
    :show-inheritance:

Dataset Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: torchsig.transforms.dataset_transforms
    :members:
    :undoc-members:
    :show-inheritance:


Impairments
----------------------

Base Impairments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: torchsig.transforms.impairments
    :members:
    :undoc-members:
    :show-inheritance:

Narrowband Impairments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: torchsig.transforms.impairments_narrowband
    :members:
    :undoc-members:
    :show-inheritance:

Wideband Impairments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: torchsig.transforms.impairments_wideband
    :members:
    :undoc-members:
    :show-inheritance:


Functional Transforms
----------------------
.. automodule:: torchsig.transforms.functional
    :members:
    :undoc-members:
    :show-inheritance: