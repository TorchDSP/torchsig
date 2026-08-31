"""Frozen schema definitions for TorchSig's packed HDF5 format.

Readers accept schema major version 1 and reject unknown required features.
Minor-version additions remain compatible when they use known required
features and metadata encodings.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import h5py

__all__ = [
    "DatasetSpec",
    "MetadataEncoding",
    "PackedHDF5Schema",
    "default_packed_schema",
    "read_schema",
    "write_schema",
]

SCHEMA_DATASET_PATH = "/schema"
SUPPORTED_FORMAT = "torchsig-packed"
SUPPORTED_SCHEMA_MAJOR = 1
SUPPORTED_SCHEMA_MINOR = 0
SUPPORTED_METADATA_ENCODINGS = {("torchsig-json", 1)}
SUPPORTED_FEATURES = {
    "mixed_dtypes",
    "variable_shapes",
    "component_signals",
    "hierarchical_metadata",
}


@dataclass(frozen=True)
class MetadataEncoding:
    """Encoding used for open-ended Signal metadata values."""

    name: str
    version: int


@dataclass(frozen=True)
class DatasetSpec:
    """Physical location and interpretation of one packed-format dataset."""

    path: str
    role: str
    dtype: str | None = None
    fields: dict[str, str] | None = None
    encoding: MetadataEncoding | None = None


@dataclass(frozen=True)
class PackedHDF5Schema:
    """Versioned vocabulary embedded in every packed TorchSig HDF5 file."""

    format: str
    schema_major: int
    schema_minor: int
    required_features: tuple[str, ...]
    datasets: dict[str, DatasetSpec]
    sentinels: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Convert the schema to its JSON-compatible representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PackedHDF5Schema:
        """Construct and minimally validate a schema dictionary."""
        try:
            datasets = {}
            for name, specification in value["datasets"].items():
                specification = dict(specification)
                encoding = specification.get("encoding")
                if encoding is not None:
                    specification["encoding"] = MetadataEncoding(**encoding)
                datasets[name] = DatasetSpec(**specification)
            return cls(
                format=str(value["format"]),
                schema_major=int(value["schema_major"]),
                schema_minor=int(value["schema_minor"]),
                required_features=tuple(value["required_features"]),
                datasets=datasets,
                sentinels={str(name): int(sentinel) for name, sentinel in value["sentinels"].items()},
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid packed HDF5 schema document") from error


def default_packed_schema() -> PackedHDF5Schema:
    """Return frozen packed schema version 1.0 emitted by the writer."""
    metadata_encoding = MetadataEncoding(name="torchsig-json", version=1)
    return PackedHDF5Schema(
        format=SUPPORTED_FORMAT,
        schema_major=SUPPORTED_SCHEMA_MAJOR,
        schema_minor=SUPPORTED_SCHEMA_MINOR,
        required_features=(
            "mixed_dtypes",
            "variable_shapes",
            "component_signals",
            "hierarchical_metadata",
        ),
        datasets={
            "index": DatasetSpec(path="/index", dtype="uint64", role="top_level_record_ids"),
            "records": DatasetSpec(
                path="/records",
                role="signal_record_table",
                fields={
                    "data_offset": "data_offset",
                    "data_length": "data_length",
                    "dtype_id": "dtype_id",
                    "shape_offset": "shape_offset",
                    "shape_count": "shape_count",
                    "component_offset": "component_offset",
                    "component_count": "component_count",
                    "parent_id": "parent_id",
                },
            ),
            "data": DatasetSpec(path="/data", role="dtype_stream_group"),
            "dtypes": DatasetSpec(path="/dtypes", dtype="utf8", role="numpy_dtype_descriptors"),
            "shapes": DatasetSpec(path="/shapes", dtype="uint64", role="flattened_record_shapes"),
            "components": DatasetSpec(
                path="/components",
                dtype="uint64",
                role="flattened_component_record_ids",
            ),
            "metadata": DatasetSpec(
                path="/metadata",
                dtype="utf8",
                role="record_metadata",
                encoding=metadata_encoding,
            ),
            "parent_records": DatasetSpec(
                path="/parent_records",
                role="parent_relationship_table",
                fields={"parent_id": "parent_id"},
            ),
            "parent_metadata": DatasetSpec(
                path="/parent_metadata",
                dtype="utf8",
                role="parent_metadata",
                encoding=metadata_encoding,
            ),
        },
        sentinels={"no_parent": (1 << 64) - 1},
    )


def write_schema(file: h5py.File, schema: PackedHDF5Schema) -> None:
    """Write an authoritative JSON schema into an open HDF5 file."""
    payload = json.dumps(schema.to_dict(), separators=(",", ":"), sort_keys=True)
    file.create_dataset(
        SCHEMA_DATASET_PATH,
        data=payload,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )


def read_schema(file: h5py.File) -> PackedHDF5Schema:
    """Read a packed schema and validate its compatibility with this reader."""
    if SCHEMA_DATASET_PATH not in file:
        raise ValueError("Not a packed TorchSig HDF5 file: missing /schema")
    payload = file[SCHEMA_DATASET_PATH][()]
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    try:
        schema = PackedHDF5Schema.from_dict(json.loads(payload))
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Invalid JSON in packed HDF5 /schema") from error

    if schema.format != SUPPORTED_FORMAT:
        raise ValueError(f"Unsupported HDF5 format: {schema.format!r}")
    if schema.schema_major != SUPPORTED_SCHEMA_MAJOR:
        raise ValueError(f"Unsupported packed HDF5 schema major version: {schema.schema_major}; supported: {SUPPORTED_SCHEMA_MAJOR}")
    unsupported_features = set(schema.required_features) - SUPPORTED_FEATURES
    if unsupported_features:
        raise ValueError(f"Unsupported required packed HDF5 features: {sorted(unsupported_features)}")
    for logical_name in ("metadata", "parent_metadata"):
        encoding = schema.datasets.get(logical_name, DatasetSpec("", "")).encoding
        if encoding is None or (encoding.name, encoding.version) not in (SUPPORTED_METADATA_ENCODINGS):
            raise ValueError(f"Unsupported metadata encoding for {logical_name}: {encoding}")
    return schema
