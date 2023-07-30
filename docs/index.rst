.. torchsig documentation master file, created by
   sphinx-quickstart 
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TorchSig
===========

.. image:: logo.png
    :width: 400
    :alt: TorchSig
    :target: https://torchsig.com
    :align: center

`TorchSig <https://torchsig.com>`_ is an open-source signal processing machine learning toolkit based on the PyTorch data handling pipeline. The user-friendly toolkit simplifies common digital signal processing operations, augmentations, and transformations when dealing with both real and complex-valued signals. TorchSig streamlines the integration process of these signal processing tools building on PyTorch, enabling faster and easier development and research for machine learning techniques applied to signals data, particularly within (but not limited to) the radio frequency domain. An example dataset, :mod:`Sig53`, based on many unique communication signal modulations is included to accelerate the field of modulation classification. Additionally, an example wideband dataset, :mod:`WidebandSig53`, is also included that extends :mod:`Sig53` with larger data example sizes containing multiple signals enabling accelerated research in the fields of wideband signal detection and recognition.

*TorchSig is currently in beta.*


.. toctree::
    :caption: Dataset Tutorials
    :name: datasettutorialtoc
    :maxdepth: 2
    
    00_dataset_tutorials/00_Sig53DatasetTutorial
    00_dataset_tutorials/01_WidebandSig53DatasetTutorial
    00_dataset_tutorials/02_RadioMLDatasetTutorial

.. toctree::
    :caption: Transform Tutorials
    :name: transformtutorialtoc
    :maxdepth: 2

    01_transform_tutorials/00_DataTransformTutorial
    01_transform_tutorials/01_TargetTransformTutorial
    01_transform_tutorials/02_DataAugmentationTutorial
    
.. toctree::
    :caption: ML Tutorials
    :name: mltutorialtoc
    :maxdepth: 2

    02_ml_tutorials/00_SignalClassificationTrainingTutorial
    02_ml_tutorials/01_SignalClassificationInferenceTutorial
    02_ml_tutorials/02_SignalClassificationFinetuningTutorial
    02_ml_tutorials/03_SignalDetectionTrainingTutorial
    02_ml_tutorials/04_SignalDetectionInferenceTutorial
    02_ml_tutorials/05_SignalDetectionFinetuningTutorial
    02_ml_tutorials/06_SignalRecognitionTrainingTutorial
    02_ml_tutorials/07_SignalRecognitionInferenceTutorial
    02_ml_tutorials/08_SignalRecognitionFinetuningTutorial
   
.. toctree::
    :caption: Code API
    :name: mastertoc
    :maxdepth: 2

    datasets
    transforms
    models
    utils

.. automodule:: torchsig
     :members:

