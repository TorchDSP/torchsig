Metadata Debugging
==================

TorchSig provides opt-in diagnostics for hierarchical metadata. Debugging is
disabled by default and uses the ``torchsig.metadata`` Python logger. Applications
are responsible for configuring a logging handler and enabling the ``DEBUG``
level.

Explaining metadata resolution
------------------------------

Use ``explain_metadata`` to determine whether a key is local, inherited,
overridden, or missing without exposing its value::

    resolution = signal.explain_metadata("sample_rate")
    print(resolution.source, resolution.depth, resolution.owner_type)

Event logging
-------------

Enable structured records on any ``HierarchicalMetadataObject``::

    dataset.enable_metadata_debug(
        keys={"sample_rate", "bandwidth_min", "bandwidth_max"},
        events={"lookup", "set", "delete"},
        max_events=100,
        include_values=False,
    )

``keys`` and ``events`` exclude unwanted records before they are constructed.
``max_events`` limits emitted records. Values are omitted unless
``include_values=True`` is selected, and their representations are bounded by
``value_repr_limit``.

Call ``disable_metadata_debug`` to emit a summary and stop logging. Summary
statistics distinguish emitted records, selected records suppressed by a rate
limit or logger level, and records intentionally excluded by filters.

Correlation context
-------------------

Use ``metadata_logging_context`` to attach application-specific correlation
information::

    from torchsig.utils.metadata_logging import metadata_logging_context

    with metadata_logging_context(
        session_id="training-debug",
        fields={"split": "train"},
    ):
        sample = next(dataset)

Iterable datasets automatically add dataset ID, sample index, worker ID, and
the ``generate`` or ``transform`` stage. Records also contain process and thread
identifiers. Nested contexts inherit outer values and restore them on exit.

Completed metadata snapshots
----------------------------

Select only the ``snapshot`` event to log one completed metadata object instead
of individual accesses::

    dataset.enable_metadata_debug(
        events={"snapshot"},
        include_values=True,
        value_repr_limit=160,
    )

``TorchSigIterableDataset`` and ``SafeTorchSigIterableDataset`` automatically
emit the completed sample snapshot after the transform stage. Wideband
component signals remain separate in ``metadata_component_snapshots``. Sample
arrays are never logged; records contain only their shape and dtype.

Snapshots can also be emitted explicitly::

    dataset.log_metadata_snapshot(sample, include_components=True)

The ``keys`` filter applies to both sample and component mappings. Snapshot
construction reads the hierarchy without producing additional lookup events.

DataModule and DataLoader pipelines
-----------------------------------

Both TorchSig DataModules can enable metadata debugging on the iterable
datasets used to create their static datasets. Pass ``metadata_debug=True`` to
use the default event configuration::

    datamodule = TorchSigDataModule(
        ...,
        metadata_debug=True,
    )

Pass a mapping to forward options to ``enable_metadata_debug``::

    datamodule = SplitTorchSigDataModule(
        ...,
        metadata_debug={
            "events": {"snapshot"},
            "include_values": True,
            "max_events": 100,
        },
    )

Debugging is enabled before the creation DataLoader starts, so worker copies
inherit the debug session. Configure the ``torchsig.metadata`` logger in the
main application before starting workers. Records include process and worker
identifiers and may arrive interleaved when ``create_num_workers`` is greater
than zero. Use ``metadata_debug=False`` (the default) for normal generation.

Examples
--------

The developer script demonstrates snapshot debugging through dataset creation
and a DataModule-provided loader:

.. code-block:: console

    python examples/scripts/dev/debug_datamodule_pipeline.py

Pass ``--root PATH`` to retain the generated demonstration dataset. Without
that argument, the script uses and removes a temporary directory.

Value safety
------------

Metadata values can contain sensitive or unusually large objects. Keep value
logging disabled unless it is required, select only relevant keys where
possible, and use a conservative ``value_repr_limit``. Completed snapshots do
not include IQ samples or spectrogram arrays.
