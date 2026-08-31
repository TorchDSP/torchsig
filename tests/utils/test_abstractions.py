"""Unit tests for HierarchicalMetadataObject."""

from __future__ import annotations

import logging
import pickle
from collections.abc import KeysView

import pytest

from torchsig.utils.abstractions import (
    HierarchicalMetadataObject,
    MetadataAttributeError,
    MetadataDebugConfig,
    MetadataDebugStatistics,
    MetadataResolution,
)


class ExampleMetadataObject(HierarchicalMetadataObject):
    """Concrete subclass used to verify subclass-specific behavior."""

    class_attribute = "class value"


class RequiredArgumentMetadataObject(HierarchicalMetadataObject):
    """Subclass using the copy hook to preserve required constructor state."""

    def __init__(self, required_value: str, **kwargs) -> None:
        """Initialize the object with required subclass state."""
        self.required_value = required_value
        super().__init__(**kwargs)

    def _copy_kwargs(self):
        return {"required_value": self.required_value}


def test_metadata_attribute_error_is_attribute_error():
    error = MetadataAttributeError("missing metadata")

    assert isinstance(error, AttributeError)
    assert str(error) == "missing metadata"


def test_initialization_without_metadata_creates_empty_metadata():
    obj = HierarchicalMetadataObject(seed=123)

    assert dict(obj.keys()) == {}
    assert obj["metadata"] == {}
    assert obj.rng_seed == 123


def test_initialization_copies_metadata_values():
    metadata = {"class_name": "bpsk", "snr_db": 10.0}

    obj = HierarchicalMetadataObject(metadata=metadata)

    assert obj["class_name"] == "bpsk"
    assert obj["snr_db"] == 10.0


def test_initialization_does_not_alias_input_metadata_dictionary():
    metadata = {"class_name": "bpsk"}

    obj = HierarchicalMetadataObject(metadata=metadata)
    metadata["class_name"] = "qpsk"
    metadata["new_field"] = 42

    assert obj["class_name"] == "bpsk"
    assert "new_field" not in obj.keys()


def test_keyword_metadata_overrides_metadata_dictionary():
    obj = HierarchicalMetadataObject(
        metadata={
            "class_name": "bpsk",
            "snr_db": 10.0,
        },
        snr_db=20.0,
        sample_rate=1_000_000,
    )

    assert obj["class_name"] == "bpsk"
    assert obj["snr_db"] == 20.0
    assert obj["sample_rate"] == 1_000_000


def test_metadata_can_be_accessed_as_attribute():
    obj = HierarchicalMetadataObject(
        metadata={
            "class_name": "bpsk",
            "snr_db": 10.0,
        }
    )

    assert obj.class_name == "bpsk"
    assert obj.snr_db == 10.0


def test_real_attribute_takes_precedence_over_metadata():
    obj = ExampleMetadataObject(metadata={"class_attribute": "metadata value"})

    assert obj.class_attribute == "class value"
    assert obj["class_attribute"] == "metadata value"


def test_metadata_property_returns_copy():
    obj = HierarchicalMetadataObject(metadata={"field": "original"})

    returned_metadata = obj.metadata
    returned_metadata["field"] = "modified"
    returned_metadata["new_field"] = "new value"

    assert obj["field"] == "original"
    assert "new_field" not in obj.keys()


def test_keys_returns_keys_view():
    """keys() should expose a mapping-key view rather than a list."""
    obj = HierarchicalMetadataObject(metadata={"field": "value"})

    keys = obj.keys()

    assert isinstance(keys, KeysView)
    assert set(keys) == {"field"}


def test_keys_view_reflects_later_metadata_changes():
    """A previously returned key view should remain linked to local metadata."""
    obj = HierarchicalMetadataObject(metadata={"original": 1})
    keys = obj.keys()

    obj["added"] = 2
    del obj["original"]

    assert set(keys) == {"added"}


def test_getitem_returns_local_metadata_value():
    obj = HierarchicalMetadataObject(metadata={"field": 123})

    assert obj["field"] == 123


def test_getitem_inherits_value_from_parent():
    parent = HierarchicalMetadataObject(
        metadata={
            "sample_rate": 1_000_000,
            "center_freq": 100_000_000,
        }
    )
    child = HierarchicalMetadataObject(parent=parent)

    assert child["sample_rate"] == 1_000_000
    assert child["center_freq"] == 100_000_000


def test_child_metadata_overrides_parent_metadata():
    parent = HierarchicalMetadataObject(
        metadata={
            "sample_rate": 1_000_000,
            "center_freq": 100_000_000,
        }
    )
    child = HierarchicalMetadataObject(
        parent=parent,
        metadata={"center_freq": 101_000_000},
    )

    assert child["sample_rate"] == 1_000_000
    assert child["center_freq"] == 101_000_000


def test_metadata_inheritance_works_across_multiple_levels():
    grandparent = HierarchicalMetadataObject(
        metadata={
            "grandparent_only": 1,
            "overridden": "grandparent",
        }
    )
    parent = HierarchicalMetadataObject(
        parent=grandparent,
        metadata={
            "parent_only": 2,
            "overridden": "parent",
        },
    )
    child = HierarchicalMetadataObject(
        parent=parent,
        metadata={
            "child_only": 3,
            "overridden": "child",
        },
    )

    assert child.grandparent_only == 1
    assert child.parent_only == 2
    assert child.child_only == 3
    assert child.overridden == "child"


def test_get_full_metadata_combines_parent_and_child_metadata():
    parent = HierarchicalMetadataObject(
        metadata={
            "field_1": 4,
            "field_2": 5,
        }
    )
    child = HierarchicalMetadataObject(
        parent=parent,
        metadata={
            "field_2": 6,
            "field_3": 7,
        },
    )

    assert child.get_full_metadata() == {
        "field_1": 4,
        "field_2": 6,
        "field_3": 7,
    }


def test_get_full_metadata_combines_multiple_hierarchy_levels():
    grandparent = HierarchicalMetadataObject(
        metadata={
            "a": 1,
            "shared": "grandparent",
        }
    )
    parent = HierarchicalMetadataObject(
        parent=grandparent,
        metadata={
            "b": 2,
            "shared": "parent",
        },
    )
    child = HierarchicalMetadataObject(
        parent=parent,
        metadata={
            "c": 3,
            "shared": "child",
        },
    )

    assert child.get_full_metadata() == {
        "a": 1,
        "b": 2,
        "c": 3,
        "shared": "child",
    }


def test_get_full_metadata_returns_new_dictionary():
    obj = HierarchicalMetadataObject(metadata={"field": "original"})

    full_metadata = obj.get_full_metadata()
    full_metadata["field"] = "modified"

    assert obj["field"] == "original"


def test_explain_metadata_reports_local_key():
    obj = HierarchicalMetadataObject(metadata={"field": "value"})

    resolution = obj.explain_metadata("field")

    assert resolution == MetadataResolution(
        key="field",
        found=True,
        source="local",
        depth=0,
        owner_type="HierarchicalMetadataObject",
        overrides_parent=False,
        cycle_detected=False,
        path=("HierarchicalMetadataObject",),
    )


def test_explain_metadata_reports_inherited_key_and_depth():
    grandparent = ExampleMetadataObject(metadata={"field": "value"})
    parent = HierarchicalMetadataObject(parent=grandparent)
    child = HierarchicalMetadataObject(parent=parent)

    resolution = child.explain_metadata("field")

    assert resolution.found is True
    assert resolution.source == "inherited"
    assert resolution.depth == 2
    assert resolution.owner_type == "ExampleMetadataObject"
    assert resolution.overrides_parent is False
    assert resolution.cycle_detected is False
    assert resolution.path == (
        "HierarchicalMetadataObject",
        "HierarchicalMetadataObject",
        "ExampleMetadataObject",
    )


def test_explain_metadata_reports_parent_override():
    grandparent = HierarchicalMetadataObject(metadata={"field": "grandparent"})
    parent = HierarchicalMetadataObject(
        parent=grandparent,
        metadata={"field": "parent"},
    )
    child = HierarchicalMetadataObject(parent=parent)

    resolution = child.explain_metadata("field")

    assert resolution.source == "inherited"
    assert resolution.depth == 1
    assert resolution.overrides_parent is True


def test_explain_metadata_reports_local_override():
    parent = HierarchicalMetadataObject(metadata={"field": "parent"})
    child = HierarchicalMetadataObject(
        parent=parent,
        metadata={"field": "child"},
    )

    resolution = child.explain_metadata("field")

    assert resolution.source == "local"
    assert resolution.depth == 0
    assert resolution.overrides_parent is True


def test_explain_metadata_reports_missing_key():
    parent = ExampleMetadataObject(metadata={"other": "value"})
    child = HierarchicalMetadataObject(parent=parent)

    resolution = child.explain_metadata("missing")

    assert resolution == MetadataResolution(
        key="missing",
        found=False,
        source="missing",
        depth=None,
        owner_type=None,
        overrides_parent=False,
        cycle_detected=False,
        path=("HierarchicalMetadataObject", "ExampleMetadataObject"),
    )


def test_explain_metadata_detects_parent_cycle():
    parent = ExampleMetadataObject(metadata={"field": "value"})
    child = HierarchicalMetadataObject(parent=parent)
    parent.parent = child

    resolution = child.explain_metadata("field")

    assert resolution.found is True
    assert resolution.source == "inherited"
    assert resolution.depth == 1
    assert resolution.cycle_detected is True
    assert resolution.path == (
        "HierarchicalMetadataObject",
        "ExampleMetadataObject",
    )


def test_explain_metadata_rejects_non_string_key():
    obj = HierarchicalMetadataObject()

    with pytest.raises(TypeError, match="metadata key must be a string"):
        obj.explain_metadata(123)


def test_metadata_debug_is_disabled_by_default(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    obj = HierarchicalMetadataObject(metadata={"field": "value"})

    assert obj["field"] == "value"

    assert obj.metadata_debug_enabled is False
    assert caplog.records == []


def test_metadata_debug_logs_structured_local_lookup(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    obj = HierarchicalMetadataObject(metadata={"field": "secret-value"})
    obj.enable_metadata_debug()

    assert obj["field"] == "secret-value"

    record = caplog.records[-1]
    assert record.metadata_event == "lookup"
    assert record.metadata_key == "field"
    assert record.metadata_source == "local"
    assert record.metadata_found is True
    assert record.metadata_depth == 0
    assert record.metadata_owner_type == "HierarchicalMetadataObject"
    assert record.metadata_overrides_parent is False
    assert record.metadata_cycle_detected is False
    assert record.metadata_path == ("HierarchicalMetadataObject",)
    assert not hasattr(record, "metadata_value")
    assert "secret-value" not in record.getMessage()
    assert "secret-value" not in vars(record).values()


def test_metadata_debug_logs_inherited_and_missing_lookups(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    parent = ExampleMetadataObject(metadata={"field": "value"})
    child = HierarchicalMetadataObject(parent=parent)
    child.enable_metadata_debug()

    assert child["field"] == "value"
    with pytest.raises(MetadataAttributeError):
        _ = child["missing"]

    inherited_record, missing_record = caplog.records[-2:]
    assert inherited_record.metadata_source == "inherited"
    assert inherited_record.metadata_depth == 1
    assert inherited_record.metadata_owner_type == "ExampleMetadataObject"
    assert missing_record.metadata_source == "missing"
    assert missing_record.metadata_found is False
    assert missing_record.metadata_depth is None
    assert missing_record.metadata_owner_type is None


def test_metadata_debug_logs_set_and_delete_events(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    parent = HierarchicalMetadataObject(metadata={"field": "parent"})
    child = HierarchicalMetadataObject(parent=parent)
    child.enable_metadata_debug()

    child["field"] = "child"
    del child["field"]

    set_record, delete_record = caplog.records[-2:]
    assert set_record.metadata_event == "set"
    assert set_record.metadata_source == "local"
    assert set_record.metadata_overrides_parent is True
    assert delete_record.metadata_event == "delete"
    assert delete_record.metadata_source == "inherited"
    assert delete_record.metadata_depth == 1


def test_metadata_debug_context_restores_previous_state(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    obj = HierarchicalMetadataObject(metadata={"field": "value"})

    with obj.metadata_debug() as debug_obj:
        assert debug_obj is obj
        assert obj.metadata_debug_enabled is True
        assert obj["field"] == "value"

    assert obj.metadata_debug_enabled is False
    assert obj["field"] == "value"
    assert [record.metadata_event for record in caplog.records] == [
        "lookup",
        "summary",
    ]


def test_metadata_debug_context_restores_enabled_state_after_exception():
    obj = HierarchicalMetadataObject()
    obj.enable_metadata_debug(
        keys={"original"},
        events={"lookup"},
    )
    previous_config = obj.metadata_debug_config

    with (
        pytest.raises(RuntimeError, match="test error"),
        obj.metadata_debug(
            keys={"temporary"},
            events={"set"},
        ),
    ):
        raise RuntimeError("test error")

    assert obj.metadata_debug_enabled is True
    assert obj.metadata_debug_config == previous_config


def test_metadata_debug_state_survives_pickle_round_trip():
    obj = HierarchicalMetadataObject(metadata={"field": "value"})
    obj.enable_metadata_debug(
        keys={"field"},
        events={"lookup"},
        max_events=5,
        include_values=True,
        value_repr_limit=50,
    )

    restored = pickle.loads(pickle.dumps(obj))  # noqa: S301

    assert restored.metadata_debug_enabled is True
    assert restored.metadata_debug_config == MetadataDebugConfig(
        keys=frozenset({"field"}),
        events=frozenset({"lookup"}),
        max_events=5,
        include_values=True,
        value_repr_limit=50,
    )
    assert restored["field"] == "value"


def test_metadata_debug_filters_keys_and_events(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    obj = HierarchicalMetadataObject(metadata={"field": "value", "other": "other-value"})
    obj.enable_metadata_debug(keys={"field"}, events={"lookup"})

    assert obj["field"] == "value"
    assert obj["other"] == "other-value"
    obj["field"] = "new-value"
    del obj["field"]

    assert obj.metadata_debug_statistics == MetadataDebugStatistics(
        emitted_events=1,
        suppressed_events=0,
        filtered_events=3,
    )
    obj.disable_metadata_debug()

    assert [record.metadata_event for record in caplog.records] == [
        "lookup",
        "summary",
    ]
    summary = caplog.records[-1]
    assert summary.metadata_emitted_events == 1
    assert summary.metadata_suppressed_events == 0
    assert summary.metadata_filtered_events == 3
    assert summary.metadata_debug_keys == frozenset({"field"})
    assert summary.metadata_debug_events == frozenset({"lookup"})


def test_metadata_debug_rate_limit_counts_suppressed_events(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    obj = HierarchicalMetadataObject(metadata={"field": "value"})
    obj.enable_metadata_debug(max_events=2)

    for _ in range(4):
        assert obj["field"] == "value"

    assert obj.metadata_debug_statistics == MetadataDebugStatistics(
        emitted_events=2,
        suppressed_events=2,
    )
    obj.disable_metadata_debug()

    assert [record.metadata_event for record in caplog.records] == [
        "lookup",
        "lookup",
        "summary",
    ]


def test_metadata_debug_can_include_bounded_value_representations(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    obj = HierarchicalMetadataObject(metadata={"field": "sensitive-value"})
    obj.enable_metadata_debug(include_values=True, value_repr_limit=10)

    assert obj["field"] == "sensitive-value"

    record = caplog.records[-1]
    assert record.metadata_value == "'sensit..."
    assert record.metadata_value_truncated is True
    assert len(record.metadata_value) == 10


def test_metadata_debug_value_logging_supports_inherited_and_missing(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    parent = HierarchicalMetadataObject(metadata={"field": "parent-value"})
    child = HierarchicalMetadataObject(parent=parent)
    child.enable_metadata_debug(include_values=True)

    assert child["field"] == "parent-value"
    with pytest.raises(MetadataAttributeError):
        _ = child["missing"]

    inherited_record, missing_record = caplog.records[-2:]
    assert inherited_record.metadata_value == "'parent-value'"
    assert inherited_record.metadata_value_truncated is False
    assert missing_record.metadata_value == "<missing>"
    assert missing_record.metadata_value_truncated is False


def test_metadata_debug_delete_logs_deleted_value(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    parent = HierarchicalMetadataObject(metadata={"field": "parent-value"})
    child = HierarchicalMetadataObject(
        parent=parent,
        metadata={"field": "child-value"},
    )
    child.enable_metadata_debug(events={"delete"}, include_values=True)

    del child["field"]

    record = caplog.records[-1]
    assert record.metadata_source == "inherited"
    assert record.metadata_value == "'child-value'"


@pytest.mark.parametrize(
    ("kwargs", "exception_type", "match"),
    [
        ({"keys": ["field"]}, TypeError, "keys must be a set"),
        ({"keys": {1}}, TypeError, "keys must be a set"),
        ({"events": ["lookup"]}, TypeError, "events must be a set"),
        ({"events": {"unknown"}}, ValueError, "unknown metadata debug events"),
        ({"max_events": -1}, ValueError, "max_events must be"),
        ({"max_events": True}, ValueError, "max_events must be"),
        ({"include_values": 1}, TypeError, "include_values must be"),
        ({"value_repr_limit": 0}, ValueError, "value_repr_limit must be"),
        ({"value_repr_limit": True}, ValueError, "value_repr_limit must be"),
    ],
)
def test_metadata_debug_rejects_invalid_configuration(
    kwargs,
    exception_type,
    match,
):
    obj = HierarchicalMetadataObject()

    with pytest.raises(exception_type, match=match):
        obj.enable_metadata_debug(**kwargs)


def test_keys_returns_only_local_metadata_keys():
    parent = HierarchicalMetadataObject(metadata={"parent_field": 1})
    child = HierarchicalMetadataObject(
        parent=parent,
        metadata={"child_field": 2},
    )

    assert set(child.keys()) == {"child_field"}
    assert "parent_field" not in child.keys()


def test_setitem_adds_metadata():
    obj = HierarchicalMetadataObject()

    obj["field"] = "value"

    assert obj["field"] == "value"
    assert obj.field == "value"


def test_setitem_overrides_inherited_metadata_locally():
    parent = HierarchicalMetadataObject(metadata={"field": "parent"})
    child = HierarchicalMetadataObject(parent=parent)

    child["field"] = "child"

    assert child["field"] == "child"
    assert parent["field"] == "parent"
    assert "field" in child.keys()


def test_delitem_removes_local_metadata():
    obj = HierarchicalMetadataObject(metadata={"field": "value"})

    del obj["field"]

    assert "field" not in obj.keys()

    with pytest.raises(MetadataAttributeError):
        _ = obj["field"]


def test_delitem_reveals_inherited_parent_value():
    parent = HierarchicalMetadataObject(metadata={"field": "parent"})
    child = HierarchicalMetadataObject(
        parent=parent,
        metadata={"field": "child"},
    )

    del child["field"]

    assert child["field"] == "parent"
    assert "field" not in child.keys()


def test_delitem_missing_local_key_raises_key_error():
    parent = HierarchicalMetadataObject(metadata={"field": "parent"})
    child = HierarchicalMetadataObject(parent=parent)

    with pytest.raises(KeyError):
        del child["field"]


def test_getitem_for_missing_key_raises_metadata_attribute_error():
    obj = HierarchicalMetadataObject()

    with pytest.raises(
        MetadataAttributeError,
        match="key: 'missing' could not be found in metadata",
    ):
        _ = obj["missing"]


def test_attribute_access_for_missing_key_raises_metadata_attribute_error():
    obj = HierarchicalMetadataObject()

    with pytest.raises(
        MetadataAttributeError,
        match="key: 'missing' could not be found in metadata",
    ):
        _ = obj.missing


def test_key_lookup_returns_metadata_value():
    obj = HierarchicalMetadataObject(metadata={"field": 123})

    assert obj.key_lookup("field") == 123


def test_key_lookup_reports_missing_key():
    obj = HierarchicalMetadataObject()

    with pytest.raises(
        MetadataAttributeError,
        match=r"key missing: 'missing'",
    ):
        obj.key_lookup("missing")


def test_attribute_lookup_reports_missing_key():
    obj = HierarchicalMetadataObject()

    with pytest.raises(
        MetadataAttributeError,
        match=r"key missing: 'missing'",
    ):
        _ = obj.missing


def test_direct_metadata_getitem_is_rejected():
    obj = HierarchicalMetadataObject(metadata={"field": 123})

    with pytest.raises(KeyError, match="check metadata field names"):
        _ = obj["_metadata"]


def test_internal_metadata_attribute_remains_accessible():
    obj = HierarchicalMetadataObject(metadata={"field": 123})

    assert obj._metadata == {"field": 123}


def test_copy_creates_distinct_object():
    obj = HierarchicalMetadataObject(
        seed=123,
        metadata={"field": "value"},
    )

    copied = obj.copy()

    assert copied is not obj
    assert copied.get_full_metadata() == obj.get_full_metadata()
    assert copied.rng_seed == obj.rng_seed


def test_copy_has_independent_metadata_dictionary():
    obj = HierarchicalMetadataObject(metadata={"field": "original"})

    copied = obj.copy()
    copied["field"] = "modified"
    copied["new_field"] = "new value"

    assert obj["field"] == "original"
    assert "new_field" not in obj.keys()
    assert copied["field"] == "modified"


def test_copy_is_shallow():
    nested_value = {"items": [1, 2, 3]}
    obj = HierarchicalMetadataObject(metadata={"nested": nested_value})

    copied = obj.copy()

    assert copied["nested"] is obj["nested"]


def test_copy_preserves_parent_by_default():
    parent = HierarchicalMetadataObject(metadata={"parent_field": 1})
    obj = HierarchicalMetadataObject(
        parent=parent,
        metadata={"child_field": 2},
    )

    copied = obj.copy()

    assert copied.parent is parent
    assert copied.get_full_metadata() == {
        "parent_field": 1,
        "child_field": 2,
    }


def test_copy_can_detach_from_parent():
    parent = HierarchicalMetadataObject(metadata={"parent_field": 1})
    obj = HierarchicalMetadataObject(
        parent=parent,
        metadata={"child_field": 2},
    )

    copied = obj.copy(preserve_parent=False)

    assert copied.parent is None
    assert copied.get_full_metadata() == {"child_field": 2}

    with pytest.raises(MetadataAttributeError):
        _ = copied["parent_field"]


def test_copy_preserves_runtime_subclass():
    obj = ExampleMetadataObject(
        seed=123,
        metadata={"field": "value"},
    )

    copied = obj.copy()

    assert type(copied) is ExampleMetadataObject
    assert copied["field"] == "value"


def test_copy_hook_preserves_required_subclass_constructor_state():
    obj = RequiredArgumentMetadataObject(
        required_value="required",
        seed=123,
        metadata={"field": "value"},
    )

    copied = obj.copy()

    assert type(copied) is RequiredArgumentMetadataObject
    assert copied is not obj
    assert copied.required_value == "required"
    assert copied.rng_seed == 123
    assert copied["field"] == "value"


def test_copy_hook_preserves_parent_option_and_shallow_metadata_copy():
    parent = HierarchicalMetadataObject(metadata={"inherited": "value"})
    nested_value = {"items": [1, 2, 3]}
    obj = RequiredArgumentMetadataObject(
        required_value="required",
        parent=parent,
        metadata={"nested": nested_value},
    )

    attached = obj.copy()
    detached = obj.copy(preserve_parent=False)

    assert attached.parent is parent
    assert detached.parent is None
    assert attached["nested"] is nested_value
    assert detached["nested"] is nested_value
    attached["copy_only"] = True
    detached["detached_only"] = True
    assert "copy_only" not in obj.keys()
    assert "detached_only" not in obj.keys()


def test_setstate_updates_instance_dictionary():
    obj = HierarchicalMetadataObject(metadata={"old": "value"})

    obj.__setstate__(
        {
            "_metadata": {"new": "value"},
            "additional_attribute": 123,
        }
    )

    assert obj["new"] == "value"
    assert obj.additional_attribute == 123

    with pytest.raises(MetadataAttributeError):
        _ = obj["old"]


def test_setstate_initializes_missing_metadata_debug_state():
    obj = HierarchicalMetadataObject()

    obj.__setstate__(
        {
            "_metadata": {"field": "value"},
        }
    )

    assert obj.metadata_debug_enabled is False
    assert obj.metadata_debug_config is None
    assert obj.metadata_debug_statistics == MetadataDebugStatistics(
        emitted_events=0,
        suppressed_events=0,
    )


def test_object_can_be_pickled_and_unpickled():
    parent = HierarchicalMetadataObject(
        seed=100,
        metadata={"parent_field": 1},
    )
    child = HierarchicalMetadataObject(
        seed=200,
        parent=parent,
        metadata={"child_field": 2},
    )

    restored = pickle.loads(pickle.dumps(child))

    assert restored is not child
    assert restored.parent is not parent
    assert restored.rng_seed == child.rng_seed
    assert restored.get_full_metadata() == {
        "parent_field": 1,
        "child_field": 2,
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (False, False),
        (0, 0),
        ("", ""),
        ([], []),
        ({}, {}),
    ],
)
def test_metadata_supports_falsy_values(value, expected):
    obj = HierarchicalMetadataObject(metadata={"field": value})

    assert obj["field"] == expected


def test_key_lookup_missing_key_has_descriptive_error():
    obj = HierarchicalMetadataObject()

    with pytest.raises(
        MetadataAttributeError,
        match=r"key missing: 'missing'",
    ):
        obj.key_lookup("missing")


def test_metadata_debug_context_restores_previous_session():
    obj = HierarchicalMetadataObject(metadata={"field": "value"})
    obj.enable_metadata_debug(
        keys={"field"},
        events={"lookup"},
        max_events=10,
    )
    previous_config = obj.metadata_debug_config

    with obj.metadata_debug(
        keys={"other"},
        events={"set"},
        max_events=2,
    ):
        assert obj.metadata_debug_config == MetadataDebugConfig(
            keys=frozenset({"other"}),
            events=frozenset({"set"}),
            max_events=2,
            include_values=False,
            value_repr_limit=200,
        )

    assert obj.metadata_debug_enabled is True
    assert obj.metadata_debug_config == previous_config


def test_metadata_debug_snapshot_accepts_subclass(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    controller = HierarchicalMetadataObject()
    target = ExampleMetadataObject(metadata={"field": "value"})
    controller.enable_metadata_debug(events={"snapshot"})

    controller.log_metadata_snapshot(target)

    assert caplog.records[-1].metadata_event == "snapshot"
