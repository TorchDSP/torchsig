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


TorchSigNarrowband
----------------
.. automodule:: torchsig.datasets.torchsig_narrowband
    :members:
    :undoc-members:
    :show-inheritance:


TorchSigWideband
----------------
.. automodule:: torchsig.datasets.torchsig_wideband
    :members:
    :undoc-members:
    :show-inheritance:




Modulations Dataset
------------------
.. automodule:: torchsig.datasets.modulations
    :members:
    :undoc-members:
    :show-inheritance:



Wideband Datasets
------------------
.. automodule:: torchsig.datasets.wideband
    :members:
    :undoc-members:
    :show-inheritance:


Synthetic Datasets
------------------
.. automodule:: torchsig.datasets.synthetic
    :members:
    :undoc-members:
    :show-inheritance:



Radio ML Datasets
------------------
.. currentmodule:: torchsig.datasets.radioml
Radio ML 2016
~~~~~~~~~~~~~~
.. autoclass:: RadioML2016
    :members:
    :undoc-members:
    :show-inheritance:

Radio ML 2018
~~~~~~~~~~~~~~
.. autoclass:: RadioML2018
    :members:
    :undoc-members:
    :show-inheritance:



File Datasets
---------------
.. automodule:: torchsig.datasets.file_datasets
    :members:
    :undoc-members:
    :show-inheritance: