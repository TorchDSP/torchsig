HDF5 Readers
============

TorchSig provides three HDF5 readers. The reader must match the layout used
when the dataset was written; the layouts are not interchangeable.

.. list-table:: Reader selection
   :header-rows: 1
   :widths: 22 25 28 25

   * - Reader
     - Top-level arrays
     - Components and metadata
     - Recommended use
   * - :class:`~torchsig.utils.file_handlers.hdf5.HDF5Reader`
     - Shapes and dtypes may vary.
     - Preserves component and parent hierarchy using the legacy
       object-per-record layout.
     - Existing legacy datasets.
   * - :class:`~torchsig.utils.file_handlers.packed_hdf5.PackedHDF5Reader`
     - Shapes and dtypes may vary.
     - Variable components and hierarchical metadata are supported.
     - New datasets with heterogeneous top-level arrays.
   * - :class:`~torchsig.utils.file_handlers.homogeneous_hdf5.HomogeneousHDF5Reader`
     - Every top-level array has one shared shape and dtype.
     - Variable components are supported; inherited metadata is flattened.
     - Fixed-size model inputs and efficient contiguous batches.

The array content is not restricted to IQ. Packed and homogeneous files can
contain narrowband IQ, wideband IQ, spectrograms, and other non-object NumPy
arrays. Homogeneous files require only that all *top-level* arrays have the
same shape and dtype. Their component counts, shapes, and dtypes may vary.

Using a reader directly
-----------------------

All readers take the dataset directory as ``root`` and open ``root/data.h5``
lazily. Call ``reader.teardown()`` when the reader is no longer needed.

.. code-block:: python

   from torchsig.utils.file_handlers.packed_hdf5 import PackedHDF5Reader

   reader = PackedHDF5Reader("/path/to/dataset")
   try:
       print(len(reader))
       signal = reader.read(0)
       samples = signal.data
       components = signal.component_signals
   finally:
       reader.teardown()

Using ``StaticTorchSigDataset``
-------------------------------

Pass the matching reader class when loading a saved dataset through
:class:`~torchsig.datasets.datasets.StaticTorchSigDataset`.

.. code-block:: python

   from torchsig.datasets.datasets import StaticTorchSigDataset
   from torchsig.utils.file_handlers.homogeneous_hdf5 import (
       HomogeneousHDF5Reader,
   )

   dataset = StaticTorchSigDataset(
       root="/path/to/dataset",
       file_handler_class=HomogeneousHDF5Reader,
   )
   signal = dataset[0]

Contiguous DataLoader index batches automatically use
:meth:`~torchsig.utils.file_handlers.homogeneous_hdf5.HomogeneousHDF5Reader.read_signals_batch`.
Shuffled or non-contiguous index batches fall back to individual reads.

Format compatibility
--------------------

Packed files use the frozen identifier ``torchsig-packed`` and schema version
``1.0``. Readers reject unsupported major versions and unknown required
features. Homogeneous files use ``torchsig-homogeneous`` and schema version
``1``; other versions are rejected.

The legacy layout has no equivalent schema identifier. Keep the legacy reader
for existing files, and use packed or homogeneous storage for new datasets.

Reader details
--------------

.. toctree::
   :maxdepth: 1

   hdf5_legacy
   hdf5_packed
   hdf5_homogeneous
