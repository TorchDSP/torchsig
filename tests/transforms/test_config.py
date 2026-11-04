"""Tests for declarative transform configuration."""

import json

import pytest

from torchsig.transforms.base_transforms import RandomApply, Transform
from torchsig.transforms.config import TransformRegistry, TransformSpec
from torchsig.transforms.transforms import CarrierPhaseOffset, Quantize


def test_spec_builds_fresh_transform_instances() -> None:
    spec = TransformSpec.from_mapping(
        {
            "transform": "Quantize",
            "parameters": {"num_bits": [4, 8], "rounding_mode": ["floor"]},
        }
    )

    first = spec.build()
    second = spec.build()

    assert isinstance(first, Quantize)
    assert first is not second
    assert first.random_generator is not second.random_generator


def test_probability_wraps_transform_with_random_apply() -> None:
    transform = TransformSpec.from_mapping({"transform": "CarrierPhaseOffset", "probability": 0.25}).build()

    assert isinstance(transform, RandomApply)
    assert isinstance(transform.transform, CarrierPhaseOffset)
    assert transform.probability == 0.25


def test_nested_configuration_is_copied_immutable_and_canonical() -> None:
    source = {"transform": "Quantize", "parameters": {"num_bits": [4, 8]}}
    spec = TransformSpec.from_mapping(source)
    source["parameters"]["num_bits"][0] = 2

    assert spec.to_mapping()["parameters"]["num_bits"] == [4, 8]
    assert json.loads(spec.canonical_json()) == spec.to_mapping()
    with pytest.raises(TypeError):
        spec.parameters["num_bits"] = (2, 8)


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        ({"transform": "Missing"}, ValueError, "unknown transform"),
        ({"transform": "Quantize", "parameters": {"missing": 1}}, ValueError, "unknown parameters"),
        ({"transform": "Quantize", "parameters": {"rounding_mode": "invalid"}}, ValueError, "invalid parameters"),
        ({"transform": "Quantize", "probability": 1.1}, ValueError, "probability"),
        ({"transform": "Quantize", "probability": True}, TypeError, "probability"),
        ({"transform": "Quantize", "extra": 1}, ValueError, "unknown transform specification fields"),
    ],
)
def test_invalid_specifications_fail_early(value, error, message) -> None:
    with pytest.raises(error, match=message):
        TransformSpec.from_mapping(value)


def test_registry_supports_explicit_transform_sets() -> None:
    registry = TransformRegistry({"phase": CarrierPhaseOffset})
    spec = TransformSpec.from_mapping({"transform": "phase"}, registry=registry)

    assert isinstance(spec.build(), Transform)
    with pytest.raises(ValueError, match="unknown transform"):
        TransformSpec.from_mapping({"transform": "Quantize"}, registry=registry)
