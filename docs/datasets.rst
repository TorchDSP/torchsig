Datasets
====================

There are two main types of datasets: :class:`torchsig.datasets.datasets.TorchSigIterableDataset` and :class:`torchsig.datasets.datasets.StaticTorchSigDataset`.

`TorchSigIterableDataset` is for generating synthetic data in memory (infinitely).

To then save a dataset to disk, use a :class:`torchsig.utils.writer.DatasetCreator` which accepts a `TorchSigIterableDataset` object.

`StaticTorchSigDataset` (:class:`torchsig.datasets.StaticTorchSigDataset`) is for loading a saved dataset from disk.
Samples can be accessed in any order and previously generated samples are accessible.

Note: If a `TorchSigIterableDataset` is written to disk with no transforms and target transforms, it is considered `raw`.
Otherwise, it is considered `processed`.
`raw` means when the dataset is loaded back in using a `StaticTorchSigDataset` object, users can define transforms and target transforms to be applied.
When a `processed` dataset is loaded back in, users cannot define any transforms and target transform to be applied.


.. contents:: Datasets
    :local:


SNR calibration in generated datasets
-------------------------------------

For each component, TorchSig takes the mean linear STFT power over the sampled
time frames at each frequency and uses the strongest resulting bin. The
component's ``snr_db`` is that bin's power in dB relative to ``noise_power_db``.
The background uses the mean linear power over all sampled bins and frames.
For colored backgrounds this is a global reference. Calibration occurs before
frequency placement, mixing and sample-level transforms.

Earlier generators averaged dB values. Regenerating data with the same seed
and settings can therefore change IQ amplitudes, especially for swept or
intermittent components. Previously saved samples retain their original scaling.


Base Classes
---------------------

TorchSig Datasets
**********************
.. automodule:: torchsig.datasets.datasets
    :members:
    :no-index:
    :undoc-members:
    :show-inheritance:

Datamodules
---------------------
.. automodule:: torchsig.datasets.datamodules
    :members:
    :no-index:
    :undoc-members:
    :show-inheritance:


How to: select signal configurations outside the defaults
---------------------------------------------------------

The convenience aggregate ``signal_generators="all"`` selects a fixed, curated
menu of signal classes. For OFDM that menu is a discrete set of subcarrier
counts -- ``[64, 72, 128, 180, 256, 300, 512, 600, 900, 1024, 1200, 2048]`` (see
``torchsig.utils.signal_building.num_subcarrier_values``) -- whose largest value
is **2048**. The ``ofdm`` builder itself, however, accepts any count in the range
``[4, 8192]`` (``MIN_NUM_SUBCARRIERS`` to ``MAX_NUM_SUBCARRIERS`` in
``torchsig/signals/builders/ofdm.py``), so values such as ``8192`` are legal but
not present in the ``"all"`` menu.

A subcarrier count is a property of the *generator class*, not a per-signal
metadata field, so it cannot be set via ``per_signal_metadata`` (which only
overrides ``snr_db_min``/``snr_db_max``, ``bandwidth_min``/``bandwidth_max``,
``signal_duration_in_samples_min``/``max``, and ``num_iq_samples_dataset``).

To use an OFDM subcarrier count that is not in the ``"all"`` menu, name the
generator directly. Every ``ofdm-{N}`` with ``4 <= N <= 8192`` is a valid
generator name and is resolved by
:func:`torchsig.utils.signal_building.lookup_signal_generator_by_string`, so a
list of name strings can be passed straight to ``signal_generators`` (dataset
metadata is required so each generator can validate its required keys)::

    from torchsig.datasets.datasets import TorchSigIterableDataset
    from torchsig.utils.defaults import TorchSigDefaults

    metadata = TorchSigDefaults().default_dataset_metadata
    metadata.update({"num_iq_samples_dataset": 8192, "sample_rate": 10_000_000})

    dataset = TorchSigIterableDataset(
        signal_generators=["ofdm-8192"],  # not in the "all" menu (max 2048)
        metadata=metadata,
    )

Because each name is resolved to a generator whose ``class_name`` is
``"ofdm-8192"``, that name can also be used as a key in ``per_signal_metadata``
to override the per-signal fields (snr, bandwidth, duration) for that specific
count::

    from torchsig.datasets.datasets import TorchSigIterableDataset
    from torchsig.utils.defaults import TorchSigDefaults

    metadata = TorchSigDefaults().default_dataset_metadata
    metadata.update({"num_iq_samples_dataset": 8192, "sample_rate": 10_000_000})

    dataset = TorchSigIterableDataset(
        signal_generators=["ofdm-8192"],
        metadata=metadata,
        per_signal_metadata={
            "ofdm-8192": {"snr_db_min": 10, "snr_db_max": 10, "bandwidth_min": 4e6, "bandwidth_max": 4e6},
        },
    )

Several non-default counts can be selected at once by listing them together,
mixing freely with other signal families::

    dataset = TorchSigIterableDataset(
        signal_generators=["ofdm-8000", "ofdm-8192", "qpsk"],
        metadata=metadata,
    )

For the ``ofdm`` family aggregate, a non-default set of subcarrier counts can also
be supplied programmatically via the ``ofdm_num_subcarriers`` argument of
:func:`~torchsig.utils.signal_building.lookup_signal_generator_by_string` (or
equivalently through the dataset's ``experiment_config``); see the ticket
``docs/development/tickets/ofdm-num-subcarriers-in-per-signal-metadata.md`` for a
proposed convenience that would allow the count itself to be set through
``per_signal_metadata``.

A runnable end-to-end example -- including a disk round-trip through
:class:`torchsig.utils.writer.DatasetCreator` and
:class:`~torchsig.datasets.datasets.StaticTorchSigDataset` -- lives in
``scripts/dev/demonstrate_ofdm_near_max_subcarriers.py``::

    python -m scripts.dev.demonstrate_ofdm_near_max_subcarriers