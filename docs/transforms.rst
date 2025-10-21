Transforms
======================

.. currentmodule:: torchsig.transforms.tranforms

Transforms are applied to signals or samples to emulate transmitter and reciever effects, as well as tools for machine learning.
There are four types of transforms, that differ in purpose and scope.

1. :class:`torchsig.transforms.transforms.Transform` - may be applied to isolated signals from the signal builder (typically representing transmitter effects), or may be applied to samples, after isolated signals are placed onto a noise floor (typically represents receiver effects and other machine learning transforms).
2. :class:`torchsig.transforms.impairments.Impairments` - special collections of Transforms that represent an environment. 
3. Functionals - core logic of Transforms. Users can use for more fine-grained control of the transform.

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
.. automodule:: torchsig.transforms.transforms
    :members:
    :undoc-members:
    :show-inheritance:


Impairments
----------------------
.. automodule:: torchsig.transforms.impairments
    :members:
    :undoc-members:
    :show-inheritance:


Functional Transforms
----------------------
.. automodule:: torchsig.transforms.functional
    :members:
    :undoc-members:
    :show-inheritance: