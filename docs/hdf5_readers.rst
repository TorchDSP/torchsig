HDF5 Readers
============

TorchSig provides four HDF5 readers. The reader must match the layout used
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
   * - :class:`~torchsig.utils.file_handlers.structured_hdf5.StructuredHDF5Reader`
     - Each model-facing leaf has one fixed shape and dtype.
     - Preserves nested tuples, lists, and string-keyed mappings rather than
       ``Signal`` objects.
     - Precomputed model inputs and targets; see :doc:`hdf5_structured`.

The array content is not restricted to IQ. Packed and homogeneous files can
contain narrowband IQ, wideband IQ, spectrograms, and other non-object NumPy
arrays. Homogeneous files require only that all *top-level* arrays have the
same shape and dtype. Their component counts, shapes, and dtypes may vary.
Structured files instead require a fixed container and leaf schema across all
model-facing samples.

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

Unified raw-I/O benchmark
-------------------------

``benchmarks/optional/benchmark_hdf5_readers_unified.py`` creates equivalent
legacy, packed, homogeneous, and structured datasets and validates their array
values, shapes, and dtypes before timing them. It compares individual
sequential and random reads with contiguous and shuffled batch requests for
complex IQ and floating-point image-like samples.

.. code-block:: console

   python benchmarks/optional/benchmark_hdf5_readers_unified.py \
       --output-dir .benchmarks/hdf5-readers \
       --results .benchmarks/hdf5-reader-results.json \
       --samples 256 --reads 128 --repetitions 3 \
       --compressions none,lzf --chunk-sizes 1,8,32

The benchmark writes JSON and CSV results containing first-read latency,
warm-cache samples per second, file size, environment versions, and native
batch-read capability. Legacy and packed writers accept compression but do not
expose a sample-count chunk control; their ``chunk_samples`` result is therefore
``null`` rather than implying that a requested chunk size was used.

These are raw reader measurements, not end-to-end model throughput. The
homogeneous contiguous fast path returns top-level arrays without decoding
metadata, while legacy and packed readers reconstruct complete ``Signal``
objects. Structured reads reconstruct the fixed model-facing sample grammar.
Interpret those differences as the cost of each supported reader path, and use
the Stage-2 benchmark when deciding whether materialization benefits training.
Filesystem-cache state is warm during steady-state measurements.

The structured reader also provides ``read_indices(indices)`` for DataLoader
batch requests. It validates, sorts, and deduplicates coalescible multi-field
requests, reads adjacent runs with slices, and restores sampler order and
duplicates. It adaptively retains direct reads for sparse or single-leaf
requests when batching would add overhead without reducing storage operations.

Reader details
--------------

.. toctree::
   :maxdepth: 1

   hdf5_legacy
   hdf5_packed
   hdf5_homogeneous
   hdf5_structured
   hdf5_structured_findings
