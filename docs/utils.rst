Signal Data Utilities
======================

Modeled after the torch.utils.data module, this module contains the Data classes including Dataset abstract class
and the DataLoader class which combines a dataset and a sampler, and provides single- or multi-process iterators 
over the dataset. This module may also include other utilities for sampling or combining Dataset objects.

The following utilities are available:

.. contents:: Data Utilities
    :local:


Signal Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchsig.utils.dataset

.. autoclass:: SignalDataset

.. autoclass:: SignalFileDataset

.. autoclass:: SignalTensorDataset


Signal Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchsig.utils.types

.. autoclass:: SignalDescription

.. autoclass:: SignalData

.. autoclass:: SignalCapture


Signal Visualizers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchsig.utils.visualize

.. autoclass:: Visualizer

.. autoclass:: SpectrogramVisualizer

.. autoclass:: WaveletVisualizer

.. autoclass:: ConstellationVisualizer

.. autoclass:: IQVisualizer

.. autoclass:: TimeSeriesVisualizer

.. autoclass:: ImageVisualizer