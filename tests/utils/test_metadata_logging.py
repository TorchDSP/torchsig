"""Tests for metadata logging correlation context."""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.metadata_logging import (
    MetadataDebugFormatter,
    MetadataLoggingContext,
    MetadataSnapshotFormatter,
    get_metadata_logging_context,
    metadata_logging_context,
)


def test_metadata_logging_context_is_empty_by_default():
    assert get_metadata_logging_context() == MetadataLoggingContext()


def test_metadata_logging_context_sets_named_and_custom_fields():
    with metadata_logging_context(
        session_id="session-1",
        dataset_id="wideband-train",
        sample_index=42,
        worker_id=2,
        fields={"split": "train", "attempt": 3},
    ) as context:
        assert context.session_id == "session-1"
        assert context.dataset_id == "wideband-train"
        assert context.sample_index == 42
        assert context.worker_id == 2
        assert dict(context.fields) == {"attempt": 3, "split": "train"}
        assert get_metadata_logging_context() is context

    assert get_metadata_logging_context() == MetadataLoggingContext()


def test_metadata_logging_context_generates_session_id():
    with metadata_logging_context() as context:
        assert isinstance(context.session_id, str)
        assert context.session_id


def test_nested_metadata_logging_context_inherits_and_overrides():
    with metadata_logging_context(
        session_id="outer-session",
        dataset_id="dataset",
        sample_index=1,
        fields={"split": "train", "attempt": 1},
    ) as outer:
        with metadata_logging_context(
            sample_index=2,
            worker_id=4,
            fields={"attempt": 2},
        ) as inner:
            assert inner.session_id == "outer-session"
            assert inner.dataset_id == "dataset"
            assert inner.sample_index == 2
            assert inner.worker_id == 4
            assert dict(inner.fields) == {"attempt": 2, "split": "train"}

        assert get_metadata_logging_context() is outer


def test_metadata_logging_context_restores_after_exception():
    with (
        pytest.raises(
            RuntimeError,
            match="test error",
        ),
        metadata_logging_context(session_id="session"),
    ):
        raise RuntimeError("test error")

    assert get_metadata_logging_context() == MetadataLoggingContext()


def test_metadata_logging_context_is_isolated_from_new_thread():
    with metadata_logging_context(session_id="main-thread"):
        with ThreadPoolExecutor(max_workers=1) as executor:
            child_context = executor.submit(get_metadata_logging_context).result()

        assert child_context == MetadataLoggingContext()
        assert get_metadata_logging_context().session_id == "main-thread"


def test_metadata_logging_context_is_isolated_between_async_tasks():
    async def observe_context(sample_index: int) -> int:
        with metadata_logging_context(sample_index=sample_index):
            await asyncio.sleep(0)
            return get_metadata_logging_context().sample_index

    async def run_tasks() -> list[int]:
        return await asyncio.gather(observe_context(1), observe_context(2))

    assert asyncio.run(run_tasks()) == [1, 2]
    assert get_metadata_logging_context() == MetadataLoggingContext()


def test_metadata_logging_context_is_pickle_safe():
    context = MetadataLoggingContext(
        session_id="session",
        dataset_id="dataset",
        sample_index=1,
        worker_id=2,
        correlation_fields=(("split", "train"),),
    )

    restored = pickle.loads(pickle.dumps(context))  # noqa: S301

    assert restored == context


def test_metadata_logging_context_bounds_string_values():
    long_value = "x" * 300

    with metadata_logging_context(
        session_id=long_value,
        dataset_id=long_value,
        fields={"long": long_value},
    ) as context:
        assert len(context.session_id) == 200
        assert context.session_id.endswith("...")
        assert len(context.dataset_id) == 200
        assert len(context.fields["long"]) == 200


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"session_id": 1}, "session_id must be"),
        ({"dataset_id": 1}, "dataset_id must be"),
        ({"sample_index": []}, "sample_index.*must be a scalar"),
        ({"worker_id": {}}, "worker_id.*must be a scalar"),
        ({"fields": []}, "fields must be a mapping"),
        ({"fields": {1: "value"}}, "field names must be"),
        ({"fields": {"": "value"}}, "field names must be"),
        ({"fields": {"field": []}}, "field.*must be a scalar"),
    ],
)
def test_metadata_logging_context_rejects_invalid_values(kwargs, match):
    with pytest.raises(TypeError, match=match), metadata_logging_context(**kwargs):
        pass


def test_metadata_records_include_correlation_and_execution_fields(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    obj = HierarchicalMetadataObject(metadata={"field": "value"})
    obj.enable_metadata_debug()

    with metadata_logging_context(
        session_id="session",
        dataset_id="dataset",
        sample_index=7,
        worker_id=3,
        fields={"split": "validation"},
    ):
        assert obj["field"] == "value"

    record = caplog.records[-1]
    assert record.metadata_session_id == "session"
    assert record.metadata_dataset_id == "dataset"
    assert record.metadata_sample_index == 7
    assert record.metadata_worker_id == 3
    assert record.metadata_process_id == os.getpid()
    assert record.metadata_thread_id == threading.get_ident()
    assert record.metadata_correlation_fields == {"split": "validation"}


def test_metadata_summary_uses_context_active_when_disabled(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    obj = HierarchicalMetadataObject(metadata={"field": "value"})
    obj.enable_metadata_debug()

    with metadata_logging_context(session_id="summary-session"):
        assert obj["field"] == "value"
        obj.disable_metadata_debug()

    summary = caplog.records[-1]
    assert summary.metadata_event == "summary"
    assert summary.metadata_session_id == "summary-session"


def test_metadata_debug_formatter_formats_operation_record(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    obj = HierarchicalMetadataObject(metadata={"field": "value"})
    obj.enable_metadata_debug(include_values=True)

    with metadata_logging_context(
        session_id="session",
        sample_index=7,
        worker_id=2,
        fields={"stage": "generate"},
    ):
        assert obj["field"] == "value"

    formatted = MetadataDebugFormatter().format(caplog.records[-1])

    assert formatted == ("session=session sample=7 worker=2 stage=generate event=lookup key=field source=local depth=0 value='value'")


def test_metadata_debug_formatter_handles_disabled_values_and_summary(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    obj = HierarchicalMetadataObject(metadata={"field": "value"})
    obj.enable_metadata_debug()
    assert obj["field"] == "value"
    operation = caplog.records[-1]
    obj.disable_metadata_debug()
    summary = caplog.records[-1]

    formatted_operation = MetadataDebugFormatter().format(operation)
    formatted_summary = MetadataDebugFormatter().format(summary)

    assert formatted_operation.endswith("value=<values disabled>")
    assert formatted_summary == ("metadata debug summary: emitted=1 suppressed=0 filtered=0")


def test_metadata_snapshot_includes_complete_object_and_components(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    dataset = HierarchicalMetadataObject(metadata={"sample_rate": 10_000_000, "unused": "filtered"})
    component = Signal(
        data=np.ones(4, dtype=np.complex64),
        parent=dataset,
        class_name="qpsk",
    )
    sample = Signal(
        data=np.ones((2, 2), dtype=np.float32),
        component_signals=[component],
        parent=dataset,
        complete=True,
    )
    dataset.enable_metadata_debug(
        keys={"sample_rate", "class_name", "complete"},
        events={"snapshot"},
        include_values=True,
    )

    dataset.log_metadata_snapshot(sample, include_components=True)

    record = caplog.records[-1]
    assert record.metadata_event == "snapshot"
    assert record.metadata_snapshot == {
        "sample_rate": "10000000",
        "complete": "True",
    }
    assert record.metadata_component_snapshots == ({"sample_rate": "10000000", "class_name": "'qpsk'"},)
    assert record.metadata_component_count == 1
    assert record.metadata_data_shape == (2, 2)
    assert record.metadata_data_dtype == "float32"
    assert "unused" not in record.metadata_snapshot


def test_metadata_snapshot_formatter_formats_snapshot_record(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    obj = HierarchicalMetadataObject(metadata={"field": "value"})
    obj.enable_metadata_debug(events={"snapshot"}, include_values=True)

    with metadata_logging_context(
        session_id="session",
        dataset_id="dataset",
        sample_index=3,
        worker_id=1,
        fields={"stage": "transform"},
    ):
        obj.log_metadata_snapshot()

    formatted = MetadataSnapshotFormatter().format(caplog.records[-1])

    assert formatted.startswith("completed metadata snapshot:\n")
    assert "'session_id': 'session'" in formatted
    assert "'dataset_id': 'dataset'" in formatted
    assert "'stage': 'transform'" in formatted
    assert "'metadata': {'field': \"'value'\"}" in formatted
    assert "'component_metadata': ()" in formatted


def test_metadata_snapshot_formatter_handles_disabled_values_and_summary(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    obj = HierarchicalMetadataObject(metadata={"field": "value"})
    obj.enable_metadata_debug(events={"snapshot"})
    obj.log_metadata_snapshot()
    snapshot = caplog.records[-1]
    obj.disable_metadata_debug()
    summary = caplog.records[-1]

    formatted_snapshot = MetadataSnapshotFormatter().format(snapshot)
    formatted_summary = MetadataSnapshotFormatter().format(summary)

    assert "'metadata': None" in formatted_snapshot
    assert formatted_summary == "metadata snapshot summary: emitted=1 suppressed=0"


def test_metadata_snapshot_requires_value_logging_to_include_values(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    obj = HierarchicalMetadataObject(metadata={"field": "value"})
    obj.enable_metadata_debug(events={"snapshot"})

    obj.log_metadata_snapshot()

    record = caplog.records[-1]
    assert record.metadata_snapshot_keys == ("field",)
    assert not hasattr(record, "metadata_snapshot")


def test_metadata_snapshot_does_not_generate_lookup_events(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    parent = HierarchicalMetadataObject(metadata={"inherited": 1})
    child = HierarchicalMetadataObject(parent=parent, metadata={"local": 2})
    child.enable_metadata_debug(
        events={"lookup", "snapshot"},
        include_values=True,
    )

    child.log_metadata_snapshot()

    assert [record.metadata_event for record in caplog.records] == ["snapshot"]
    assert caplog.records[0].metadata_snapshot == {
        "inherited": "1",
        "local": "2",
    }


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        ((object(),), {}, "metadata_object must be"),
        ((), {"include_components": 1}, "include_components must be"),
    ],
)
def test_metadata_snapshot_rejects_invalid_arguments(args, kwargs, match):
    obj = HierarchicalMetadataObject()

    with pytest.raises(TypeError, match=match):
        obj.log_metadata_snapshot(*args, **kwargs)
