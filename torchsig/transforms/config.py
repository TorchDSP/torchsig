"""Validated declarative configuration for signal transforms."""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from torchsig.transforms import transforms as signal_transforms
from torchsig.transforms.base_transforms import RandomApply, Transform

__all__ = ["TransformRegistry", "TransformSpec", "default_transform_registry"]


def _freeze(value: Any) -> Any:
    """Copy YAML-compatible data into immutable containers."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"transform configuration value {value!r} is not YAML/JSON compatible")


def _thaw(value: Any) -> Any:
    """Return mutable JSON-compatible data from frozen configuration data."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _constructor_parameters(transform_type: type[Transform]) -> set[str]:
    """Collect explicit constructor parameters across a transform's MRO."""
    parameters: set[str] = set()
    for cls in transform_type.mro():
        if cls is object:
            continue
        for name, parameter in inspect.signature(cls.__init__).parameters.items():
            if name not in {"self", "kwargs", "seed", "parent", "required_metadata", "data_dtype"} and parameter.kind is not inspect.Parameter.VAR_KEYWORD:
                parameters.add(name)
    return parameters


@dataclass(frozen=True)
class TransformRegistry:
    """Immutable mapping from declarative names to transform implementations."""

    transforms: Mapping[str, type[Transform]]

    def __post_init__(self) -> None:
        """Validate and defensively copy the registry mapping."""
        copied: dict[str, type[Transform]] = {}
        for name, transform_type in self.transforms.items():
            if not isinstance(name, str) or not name:
                raise ValueError("transform registry names must be non-empty strings")
            if not inspect.isclass(transform_type) or not issubclass(transform_type, Transform):
                raise TypeError(f"registered transform {name!r} must be a Transform class")
            copied[name] = transform_type
        object.__setattr__(self, "transforms", MappingProxyType(copied))

    def resolve(self, name: str) -> type[Transform]:
        """Resolve a registered transform name."""
        try:
            return self.transforms[name]
        except KeyError as error:
            available = ", ".join(sorted(self.transforms))
            raise ValueError(f"unknown transform {name!r}; available transforms: {available}") from error


def _builtin_transforms() -> dict[str, type[Transform]]:
    registry = {}
    for name in signal_transforms.__all__:
        candidate = getattr(signal_transforms, name)
        if inspect.isclass(candidate) and issubclass(candidate, Transform) and candidate is not signal_transforms.SignalTransform:
            registry[name] = candidate
    return registry


default_transform_registry = TransformRegistry(_builtin_transforms())


@dataclass(frozen=True)
class TransformSpec:
    """Immutable, validated specification for constructing a fresh transform."""

    transform: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    probability: float = 1.0
    _registry: TransformRegistry = field(default=default_transform_registry, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate and freeze specification values."""
        if not isinstance(self.transform, str) or not self.transform:
            raise ValueError("transform must be a non-empty string")
        if isinstance(self.probability, bool) or not isinstance(self.probability, (int, float)):
            raise TypeError("transform probability must be a real number")
        probability = float(self.probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("transform probability must be finite and between 0 and 1")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("transform parameters must be a mapping")

        transform_type = self._registry.resolve(self.transform)
        frozen_parameters = _freeze(self.parameters)
        unknown = set(frozen_parameters) - _constructor_parameters(transform_type)
        if unknown:
            raise ValueError(f"unknown parameters for {self.transform}: {sorted(unknown)}")
        try:
            transform_type(**_thaw(frozen_parameters))
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid parameters for {self.transform}: {error}") from error

        object.__setattr__(self, "parameters", frozen_parameters)
        object.__setattr__(self, "probability", probability)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        registry: TransformRegistry = default_transform_registry,
    ) -> TransformSpec:
        """Construct and validate a transform specification from a mapping."""
        if not isinstance(value, Mapping):
            raise TypeError("transform specification must be a mapping")
        unknown = set(value) - {"transform", "parameters", "probability"}
        if unknown:
            raise ValueError(f"unknown transform specification fields: {sorted(unknown)}")
        if "transform" not in value:
            raise ValueError("transform specification requires a transform name")
        return cls(
            transform=value["transform"],
            parameters=value.get("parameters", {}),
            probability=value.get("probability", 1.0),
            _registry=registry,
        )

    def build(self) -> Transform:
        """Construct a fresh transform, optionally wrapped with ``RandomApply``."""
        transform = self._registry.resolve(self.transform)(**_thaw(self.parameters))
        return transform if self.probability == 1.0 else RandomApply(transform, probability=self.probability)

    def to_mapping(self) -> dict[str, Any]:
        """Return YAML/JSON-compatible canonical configuration data."""
        return {
            "transform": self.transform,
            "parameters": _thaw(self.parameters),
            "probability": self.probability,
        }

    def canonical_json(self) -> str:
        """Serialize deterministically for configuration fingerprints."""
        return json.dumps(self.to_mapping(), allow_nan=False, separators=(",", ":"), sort_keys=True)
