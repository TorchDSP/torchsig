"""Schema inference and validation for structured homogeneous samples."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Number
from typing import Any, Literal

import numpy as np
import torch

__all__ = [
    "STRUCTURED_FORMAT",
    "STRUCTURED_SCHEMA_MAJOR",
    "STRUCTURED_SCHEMA_MINOR",
    "StructuredField",
    "StructuredNode",
    "StructuredSampleSchema",
    "infer_structured_sample_schema",
]

STRUCTURED_FORMAT = "torchsig-structured-homogeneous"
STRUCTURED_SCHEMA_MAJOR = 1
STRUCTURED_SCHEMA_MINOR = 0

PathElement = str | int
NodeKind = Literal["leaf", "tuple", "list", "mapping"]


def _format_path(path: tuple[PathElement, ...]) -> str:
    result = "$"
    for part in path:
        result += f"[{part}]" if isinstance(part, int) else f"[{part!r}]"
    return result


def _numeric_array(value: Any, path: tuple[PathElement, ...]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        try:
            array = value.detach().cpu().numpy()
        except (RuntimeError, TypeError) as error:
            raise TypeError(f"Unsupported tensor dtype at {_format_path(path)}: {value.dtype}") from error
    elif isinstance(value, (np.ndarray, np.generic, Number)):
        array = np.asarray(value)
    else:
        raise TypeError(f"Unsupported value at {_format_path(path)}: expected a numeric array, tensor, scalar, or supported container; got {type(value).__name__}")

    dtype = array.dtype
    if dtype.hasobject or not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
        raise TypeError(f"Unsupported dtype at {_format_path(path)}: {dtype}; expected a non-object numeric or boolean dtype")
    return array


@dataclass(frozen=True)
class StructuredField:
    """Description of one fixed-shape numeric leaf in a structured sample."""

    path: tuple[PathElement, ...]
    shape: tuple[int, ...]
    dtype: str
    dataset_path: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible field description."""
        return {
            "path": list(self.path),
            "shape": list(self.shape),
            "dtype": self.dtype,
            "dataset_path": self.dataset_path,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StructuredField:
        """Construct and validate a field description."""
        try:
            path = tuple(value["path"])
            shape = tuple(int(size) for size in value["shape"])
            dtype = np.dtype(value["dtype"])
            dataset_path = str(value["dataset_path"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid structured field description") from error
        if any(not isinstance(part, (str, int)) or isinstance(part, bool) for part in path):
            raise ValueError("Structured field paths may contain only strings and integers")
        if any(size < 0 for size in shape):
            raise ValueError("Structured field shapes must be non-negative")
        if dtype.hasobject or not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
            raise ValueError(f"Unsupported structured field dtype: {dtype}")
        if not dataset_path.startswith("/fields/"):
            raise ValueError("Structured field dataset paths must be under /fields")
        return cls(path=path, shape=shape, dtype=dtype.str, dataset_path=dataset_path)


@dataclass(frozen=True)
class StructuredNode:
    """One container or leaf node in a structured sample definition."""

    kind: NodeKind
    children: tuple[tuple[PathElement, StructuredNode], ...] = ()
    field_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible node description."""
        if self.kind == "leaf":
            return {"kind": self.kind, "field_index": self.field_index}
        return {
            "kind": self.kind,
            "children": [{"key": key, "node": child.to_dict()} for key, child in self.children],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StructuredNode:
        """Construct and validate a node description."""
        try:
            kind = value["kind"]
            if kind == "leaf":
                field_index = int(value["field_index"])
            else:
                children = tuple((child["key"], cls.from_dict(child["node"])) for child in value["children"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid structured node description") from error

        if kind == "leaf":
            if field_index < 0:
                raise ValueError("Structured leaf field indices must be non-negative")
            return cls(kind="leaf", field_index=field_index)
        if kind not in ("tuple", "list", "mapping"):
            raise ValueError(f"Invalid structured node kind: {kind!r}")

        expected_keys: tuple[PathElement, ...]
        if kind in ("tuple", "list"):
            expected_keys = tuple(range(len(children)))
        else:
            expected_keys = tuple(key for key, _ in children)
            if any(not isinstance(key, str) for key in expected_keys) or len(set(expected_keys)) != len(expected_keys):
                raise ValueError("Structured mapping node keys must be unique strings")
        if tuple(key for key, _ in children) != expected_keys:
            raise ValueError("Structured sequence node keys must be consecutive integers")
        if not children:
            raise ValueError("Structured containers must not be empty")
        return cls(kind=kind, children=children)


@dataclass(frozen=True)
class StructuredSampleSchema:
    """Versioned structure, shape, and dtype contract for homogeneous samples."""

    root: StructuredNode
    fields: tuple[StructuredField, ...]
    format: str = STRUCTURED_FORMAT
    schema_major: int = STRUCTURED_SCHEMA_MAJOR
    schema_minor: int = STRUCTURED_SCHEMA_MINOR

    def __post_init__(self) -> None:
        """Validate relationships between the structure and field records."""
        self._validate_definition()

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible on-disk schema representation."""
        return {
            "format": self.format,
            "schema_major": self.schema_major,
            "schema_minor": self.schema_minor,
            "root": self.root.to_dict(),
            "fields": [field.to_dict() for field in self.fields],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StructuredSampleSchema:
        """Construct a schema document and validate its supported version."""
        try:
            schema = cls(
                format=str(value["format"]),
                schema_major=int(value["schema_major"]),
                schema_minor=int(value["schema_minor"]),
                root=StructuredNode.from_dict(value["root"]),
                fields=tuple(StructuredField.from_dict(field) for field in value["fields"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid structured sample schema document") from error
        if schema.format != STRUCTURED_FORMAT:
            raise ValueError(f"Unsupported structured sample format: {schema.format!r}")
        if schema.schema_major != STRUCTURED_SCHEMA_MAJOR:
            raise ValueError(f"Unsupported structured sample schema major version: {schema.schema_major}; supported: {STRUCTURED_SCHEMA_MAJOR}")
        return schema

    def _validate_definition(self) -> None:
        leaf_indices: list[int] = []

        def visit(node: StructuredNode, path: tuple[PathElement, ...]) -> None:
            if node.kind == "leaf":
                if node.field_index is None:
                    raise ValueError(f"Structured leaf at {_format_path(path)} has no field index")
                leaf_indices.append(node.field_index)
                if node.field_index >= len(self.fields) or self.fields[node.field_index].path != path:
                    raise ValueError(f"Structured leaf at {_format_path(path)} does not match its field description")
                return
            if node.field_index is not None:
                raise ValueError(f"Structured container at {_format_path(path)} has a field index")
            for key, child in node.children:
                visit(child, (*path, key))

        visit(self.root, ())
        if leaf_indices != list(range(len(self.fields))):
            raise ValueError("Structured schema field indices must be unique and consecutive")
        if tuple(field.dataset_path for field in self.fields) != tuple(f"/fields/{index}" for index in range(len(self.fields))):
            raise ValueError("Structured schema field dataset paths must be unique and consecutive")

    def validate(self, sample: Any) -> tuple[np.ndarray, ...]:
        """Validate a sample and return its numeric leaves in schema order."""
        arrays: list[np.ndarray | None] = [None] * len(self.fields)
        active: set[int] = set()

        def visit(value: Any, node: StructuredNode, path: tuple[PathElement, ...]) -> None:
            if node.kind == "leaf":
                array = _numeric_array(value, path)
                field = self.fields[node.field_index]  # type: ignore[index]
                if array.shape != field.shape:
                    raise ValueError(f"Shape mismatch at {_format_path(path)}: expected {field.shape}, got {array.shape}")
                if array.dtype.str != field.dtype:
                    raise ValueError(f"Dtype mismatch at {_format_path(path)}: expected {np.dtype(field.dtype)}, got {array.dtype}")
                arrays[node.field_index] = array  # type: ignore[index]
                return

            identity = id(value)
            if identity in active:
                raise ValueError(f"Cyclic structured sample at {_format_path(path)}")
            active.add(identity)
            try:
                if node.kind == "mapping":
                    if not isinstance(value, Mapping):
                        raise TypeError(f"Container mismatch at {_format_path(path)}: expected mapping, got {type(value).__name__}")
                    expected = tuple(key for key, _ in node.children)
                    if set(value) != set(expected):
                        raise ValueError(f"Mapping keys mismatch at {_format_path(path)}: expected {list(expected)}, got {list(value)}")
                    for key, child in node.children:
                        visit(value[key], child, (*path, key))
                else:
                    expected_type = tuple if node.kind == "tuple" else list
                    if not isinstance(value, expected_type):
                        raise TypeError(f"Container mismatch at {_format_path(path)}: expected {node.kind}, got {type(value).__name__}")
                    if len(value) != len(node.children):
                        raise ValueError(f"Container length mismatch at {_format_path(path)}: expected {len(node.children)}, got {len(value)}")
                    for (index, child), item in zip(node.children, value, strict=True):
                        visit(item, child, (*path, index))
            finally:
                active.remove(identity)

        visit(sample, self.root, ())
        return tuple(array for array in arrays if array is not None)


def infer_structured_sample_schema(sample: Any) -> StructuredSampleSchema:
    """Infer a fixed-shape numeric schema from one representative sample."""
    fields: list[StructuredField] = []
    active: set[int] = set()

    def visit(value: Any, path: tuple[PathElement, ...]) -> StructuredNode:
        if isinstance(value, Mapping):
            if not value:
                raise ValueError(f"Structured containers must not be empty at {_format_path(path)}")
            if any(not isinstance(key, str) for key in value):
                raise TypeError(f"Structured mapping keys must be strings at {_format_path(path)}")
            kind: NodeKind = "mapping"
            items = tuple(value.items())
        elif isinstance(value, tuple):
            if not value:
                raise ValueError(f"Structured containers must not be empty at {_format_path(path)}")
            kind = "tuple"
            items = tuple(enumerate(value))
        elif isinstance(value, list):
            if not value:
                raise ValueError(f"Structured containers must not be empty at {_format_path(path)}")
            kind = "list"
            items = tuple(enumerate(value))
        else:
            array = _numeric_array(value, path)
            index = len(fields)
            fields.append(
                StructuredField(
                    path=path,
                    shape=array.shape,
                    dtype=array.dtype.str,
                    dataset_path=f"/fields/{index}",
                )
            )
            return StructuredNode(kind="leaf", field_index=index)

        identity = id(value)
        if identity in active:
            raise ValueError(f"Cyclic structured sample at {_format_path(path)}")
        active.add(identity)
        try:
            children = tuple((key, visit(item, (*path, key))) for key, item in items)
        finally:
            active.remove(identity)
        return StructuredNode(kind=kind, children=children)

    return StructuredSampleSchema(root=visit(sample, ()), fields=tuple(fields))
