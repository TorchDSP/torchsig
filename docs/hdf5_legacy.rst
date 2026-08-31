Legacy HDF5 Reader
==================

:class:`torchsig.utils.file_handlers.hdf5.HDF5Reader` reads TorchSig's
original object-per-record HDF5 layout. Each signal array, metadata object,
component relationship, and index entry is represented by a separate HDF5
object.

Use this reader for datasets previously written by
:class:`~torchsig.utils.file_handlers.hdf5.HDF5Writer`. For new datasets,
prefer the :doc:`hdf5_packed` or :doc:`hdf5_homogeneous` layout.

.. code-block:: python

   from torchsig.utils.file_handlers.hdf5 import HDF5Reader

   reader = HDF5Reader("/path/to/legacy-dataset")
   try:
       signal = reader.read(0)
       print(signal.data.shape)
       print(len(signal.component_signals))
   finally:
       reader.teardown()

The reader reconstructs signal metadata, parent metadata, and component
signals. It supports random access through ``read(index)`` but does not expose
a native batch-read API. The object-heavy layout generally requires more HDF5
lookups per sample than the packed and homogeneous layouts.

API reference
-------------

.. autoclass:: torchsig.utils.file_handlers.hdf5.HDF5Reader
   :members:
   :no-index:
   :show-inheritance:
