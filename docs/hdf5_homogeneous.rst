Homogeneous HDF5 Reader
=======================

:class:`torchsig.utils.file_handlers.homogeneous_hdf5.HomogeneousHDF5Reader`
reads datasets whose top-level arrays all share one NumPy dtype and shape.
Those arrays are stored as one native
``(number_of_signals, *signal_shape)`` HDF5 dataset.

The format works for fixed-length IQ, wideband IQ, spectrograms, and other
fixed-shape non-object NumPy arrays. Only top-level arrays must be homogeneous:
component counts, component shapes, and component dtypes may vary.

Complete Signal reads
---------------------

Use ``read`` for one complete Signal or ``read_signals_batch`` for a contiguous
half-open range. Both methods decode metadata and reconstruct components.

.. code-block:: python

   from torchsig.utils.file_handlers.homogeneous_hdf5 import (
       HomogeneousHDF5Reader,
   )

   reader = HomogeneousHDF5Reader("/path/to/homogeneous-dataset")
   try:
       signal = reader.read(0)
       signals = reader.read_signals_batch(0, 32)
   finally:
       reader.teardown()

Array-only batch reads
----------------------

Use ``read_batch`` when only the top-level arrays are needed:

.. code-block:: python

   reader = HomogeneousHDF5Reader("/path/to/homogeneous-dataset")
   try:
       arrays = reader.read_batch(0, 32)
       assert arrays.shape[0] == 32
   finally:
       reader.teardown()

``read_batch`` returns a NumPy array and deliberately does not decode metadata
or reconstruct component signals. This is the lowest-overhead contiguous read
path.

Metadata and limitations
------------------------

Inherited parent metadata is flattened into each Signal when the file is
written. The effective values are available after reading, but parent object
identity and hierarchy are not reconstructed. Nested component signals are
not supported.

The reader validates the format identifier, schema version, completion marker,
dataset structure, component offsets, shapes, dtypes, and data ranges. The
frozen identifier is ``torchsig-homogeneous`` and the schema version is ``1``.

API reference
-------------

.. autoclass:: torchsig.utils.file_handlers.homogeneous_hdf5.HomogeneousHDF5Reader
   :members:
   :no-index:
   :show-inheritance:
