Packed HDF5 Reader
==================

:class:`torchsig.utils.file_handlers.packed_hdf5.PackedHDF5Reader` is the
general-purpose reader for newly generated TorchSig HDF5 datasets. Arrays are
stored in dtype-specific streams with separate record, shape, component, and
metadata tables.

Use packed HDF5 when top-level signal shapes or NumPy dtypes can vary. This
includes mixed-length IQ, wideband arrays, spectrograms, and datasets that mix
representations. NumPy object arrays are not supported.

.. code-block:: python

   from torchsig.utils.file_handlers.packed_hdf5 import PackedHDF5Reader

   reader = PackedHDF5Reader("/path/to/packed-dataset")
   try:
       signal = reader.read(0)
       print(signal.data.dtype, signal.data.shape)
       for component in signal.component_signals:
           print(component.data.dtype, component.data.shape)
   finally:
       reader.teardown()

Components and metadata
-----------------------

Component counts, shapes, and dtypes may vary for every sample. Nested
component relationships and hierarchical parent metadata are represented by
record IDs and reconstructed by the reader. Do not rely on Python object
identity being shared between Signals returned by separate ``read`` calls.

Metadata uses the versioned ``torchsig-json`` codec. It supports JSON scalar
values, bytes, complex values, tuples, NumPy scalars, NumPy arrays, lists, and
dictionaries with string keys.

Validation and compatibility
----------------------------

The reader validates the embedded schema, required datasets, physical dtypes,
record ranges, component links, parent links, and the file completion marker
before serving records. The frozen format identifier is ``torchsig-packed``
and the current schema is ``1.0``.

Packed HDF5 does not currently provide a native contiguous batch method.
Applications read complete Signals with ``read(index)``.

API reference
-------------

.. autoclass:: torchsig.utils.file_handlers.packed_hdf5.PackedHDF5Reader
   :members:
   :no-index:
   :show-inheritance:
