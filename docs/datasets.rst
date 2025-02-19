Datasets
====================

There are two main types of datasets: :class:`torchsig.datasets.datasets.NewDataset` and :class:`torchsig.datasets.datasets.StaticDataset`.

`NewDataset` and its counterparts :class:`torchsig.datasets.narrowband.NewDataset` and :class:`torchsig.datasets.wideband.NewWideband` are for generating synthetic data in memory (infinitely).
Samples are not saved after being returned, and previous samples are inaccesible. 

To then save a dataset to disk, use a :class:`torchsig.utils.writer.DatasetCreator` which accepts a `NewDataset`` object.

`StaticDataset` (:class:`torchsig.datasets.narrowband.StaticNarrowband` and :class:`torchsig.datasets.wideband.StaticWideband`) are for loading a saved dataset to disk.
Samples can be accessed in any order and previously generated samples are accesible.

Note: If a `NewDataset` is written to disk with no transforms and target transforms, it is considered `raw`. 
Otherwise, it is considered to `processed`.
`raw` means when the dataset is loaded back in using a `StaticDataset` object, users can define transforms and target transforms to be applied.
When a `processed` dataset is loaded back in, users cannot define any transforms and target transform to be applied.


.. contents:: Datasets
    :local:


Base Classes
---------------------

TorchSig Datasets
**********************
.. automodule:: torchsig.datasets.datasets
    :members:
    :undoc-members:
    :show-inheritance:

Dataset Metadata
**********************
.. automodule:: torchsig.datasets.dataset_metadata
    :members:
    :undoc-members:
    :show-inheritance:


Narrowband
---------------------
.. automodule:: torchsig.datasets.narrowband
    :members:
    :undoc-members:
    :show-inheritance:


Wideband
---------------------
.. automodule:: torchsig.datasets.wideband
    :members:
    :undoc-members:
    :show-inheritance:

Datamodules
---------------------
.. automodule:: torchsig.datasets.datamodules
    :members:
    :undoc-members:
    :show-inheritance: