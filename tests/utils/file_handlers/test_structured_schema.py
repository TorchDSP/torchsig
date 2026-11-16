"""Tests for structured homogeneous sample schema inference and validation."""

import json

import numpy as np
import pytest
import torch

from torchsig.utils.file_handlers import (
    StructuredSampleSchema,
    infer_structured_sample_schema,
)


def test_infers_nested_input_and_target_fields() -> None:
    sample = {
        "inputs": (
            np.zeros((2, 4), dtype=np.complex64),
            torch.ones(3, dtype=torch.float32),
        ),
        "targets": {
            "class_index": np.int16(4),
            "mask": np.array([True, False], dtype=np.bool_),
        },
    }

    schema = infer_structured_sample_schema(sample)

    assert [field.path for field in schema.fields] == [
        ("inputs", 0),
        ("inputs", 1),
        ("targets", "class_index"),
        ("targets", "mask"),
    ]
    assert [field.shape for field in schema.fields] == [(2, 4), (3,), (), (2,)]
    assert [np.dtype(field.dtype) for field in schema.fields] == [
        np.dtype(np.complex64),
        np.dtype(np.float32),
        np.dtype(np.int16),
        np.dtype(np.bool_),
    ]
    assert [field.dataset_path for field in schema.fields] == [
        "/fields/0",
        "/fields/1",
        "/fields/2",
        "/fields/3",
    ]

    arrays = schema.validate(sample)
    assert len(arrays) == 4
    assert all(isinstance(array, np.ndarray) for array in arrays)


def test_schema_document_round_trips_through_json() -> None:
    expected = infer_structured_sample_schema((np.arange(4, dtype=np.uint8), {"score": np.float64(0.5)}))

    payload = json.loads(json.dumps(expected.to_dict()))
    actual = StructuredSampleSchema.from_dict(payload)

    assert actual == expected


def test_mapping_key_order_does_not_affect_validation() -> None:
    schema = infer_structured_sample_schema({"first": np.array([1], dtype=np.int8), "second": np.array([2], dtype=np.int8)})

    arrays = schema.validate({"second": np.array([3], dtype=np.int8), "first": np.array([4], dtype=np.int8)})

    np.testing.assert_array_equal(arrays[0], [4])
    np.testing.assert_array_equal(arrays[1], [3])


@pytest.mark.parametrize(
    ("sample", "message"),
    [
        ({}, "must not be empty"),
        ([], "must not be empty"),
        ({1: np.ones(1)}, "keys must be strings"),
        (np.array([object()], dtype=object), "Unsupported dtype"),
        (np.array(["text"]), "Unsupported dtype"),
        ("text", "Unsupported value"),
    ],
)
def test_rejects_unsupported_samples(sample, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        infer_structured_sample_schema(sample)


def test_rejects_cyclic_structure() -> None:
    sample = []
    sample.append(sample)

    with pytest.raises(ValueError, match=r"Cyclic structured sample at \$\[0\]"):
        infer_structured_sample_schema(sample)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (np.ones((2, 3), dtype=np.float32), r"Shape mismatch at \$\['input'\]"),
        (np.ones((2, 2), dtype=np.float64), r"Dtype mismatch at \$\['input'\]"),
    ],
)
def test_validation_reports_leaf_path(replacement, message) -> None:
    schema = infer_structured_sample_schema({"input": np.ones((2, 2), dtype=np.float32), "target": np.int64(1)})

    with pytest.raises(ValueError, match=message):
        schema.validate({"input": replacement, "target": np.int64(1)})


def test_validation_rejects_structure_mismatches() -> None:
    schema = infer_structured_sample_schema({"input": (np.ones(2, dtype=np.float32),), "target": np.int64(1)})

    with pytest.raises(ValueError, match="Mapping keys mismatch"):
        schema.validate({"input": (np.ones(2, dtype=np.float32),)})
    with pytest.raises(TypeError, match=r"Container mismatch at \$\['input'\]"):
        schema.validate({"input": [np.ones(2, dtype=np.float32)], "target": np.int64(1)})
    with pytest.raises(ValueError, match=r"Container length mismatch at \$\['input'\]"):
        schema.validate({"input": (np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32)), "target": np.int64(1)})


def test_validation_accepts_python_numeric_scalar_with_matching_dtype() -> None:
    schema = infer_structured_sample_schema(3.5)

    arrays = schema.validate(1.25)

    assert arrays[0].shape == ()
    assert arrays[0].dtype == np.asarray(3.5).dtype


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (lambda value: value.update(format="other"), "Unsupported structured sample format"),
        (lambda value: value.update(schema_major=2), "schema major version"),
        (lambda value: value["fields"][0].update(shape=[-1]), "Invalid structured sample schema"),
        (lambda value: value["fields"][0].update(dataset_path="/other/0"), "Invalid structured sample schema"),
        (lambda value: value["root"].update(field_index=4), "Invalid structured sample schema"),
    ],
)
def test_rejects_invalid_schema_documents(update, message) -> None:
    value = infer_structured_sample_schema(np.ones(2, dtype=np.float32)).to_dict()
    update(value)

    with pytest.raises(ValueError, match=message):
        StructuredSampleSchema.from_dict(value)
