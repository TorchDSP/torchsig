Datasets
====================

There are two main types of datasets: :class:`torchsig.datasets.datasets.TorchSigIterableDataset` and :class:`torchsig.datasets.datasets.StaticDataset`.

`TorchSigIterableDataset` is for generating synthetic data in memory (infinitely).

To then save a dataset to disk, use a :class:`torchsig.utils.writer.DatasetCreator` which accepts a `TorchSigIterableDataset` object.

`StaticTorchSigDataset` (:class:`torchsig.datasets.StaticTorchSigDataset`) is for loading a saved dataset from disk.
Samples can be accessed in any order and previously generated samples are accesible.

Note: If a `TorchSigIterableDataset` is written to disk with no transforms and target transforms, it is considered `raw`. 
Otherwise, it is considered to `processed`.
`raw` means when the dataset is loaded back in using a `StaticTorchSigDataset` object, users can define transforms and target transforms to be applied.
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

Datamodules
---------------------
.. automodule:: torchsig.datasets.datamodules
    :members:
    :undoc-members:
    :show-inheritance: