Datasets
====================

All generic datasets are subclasses of :class:`torchsig.data.SignalDataset`
i.e, they have ``__getitem__`` and ``__len__`` methods implemented.
Hence, they can all be passed to a :class:`torch.data.DataLoader`
which can load multiple samples parallelly using multiprocessing workers.
For example: ::

    signal_data = torchsig.datasets.ModulationsDataset()
    data_loader = torch.data.DataLoader(signal_data, batch_size=4,shuffle=True)

The following datasets are available:

.. contents:: SignalDatasets
    :local:

All the datasets have almost similar API. They all have a common argument:
``transform`` to transform the input data.


.. currentmodule:: torchsig.datasets


Sig53
~~~~~~~~~~~~~~

.. autoclass:: Sig53


ModulationsDataset
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ModulationsDataset


DigitalModulationDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DigitalModulationDataset


ConstellationDataset
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ConstellationDataset


OFDMDataset
~~~~~~~~~~~~~~

.. autoclass:: OFDMDataset


FSKDataset
~~~~~~~~~~~~~~

.. autoclass:: FSKDataset


AMDataset
~~~~~~~~~~~~~~

.. autoclass:: AMDataset


FMDataset
~~~~~~~~~~~~~~

.. autoclass:: FMDataset


RadioML2016
~~~~~~~~~~~~~~

.. autoclass:: RadioML2016


RadioML2018
~~~~~~~~~~~~~~

.. autoclass:: RadioML2018