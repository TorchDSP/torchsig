"""Typed, portable manifests for logical dataset split collections.

Version 1 manifests are additive collection artifacts. They live beside, not
inside, the existing ``dataset_info.yaml`` and ``writer_info.yaml`` files.
Paths are canonical POSIX paths relative to the manifest's directory so a
collection can be moved without rewriting it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml

__all__ = [
    "SPLIT_MANIFEST_FILENAME",
    "SPLIT_MANIFEST_SCHEMA_VERSION",
    "DatasetSplitManifest",
    "ManifestProvenance",
    "MembershipReference",
    "SplitManifestEntry",
    "UnsupportedSplitManifestVersionError",
    "dump_split_manifest",
    "load_split_manifest",
    "save_split_manifest",
]

SPLIT_MANIFEST_FILENAME = "split_manifest.yaml"
SPLIT_MANIFEST_SCHEMA_VERSION = 1
_SPLIT_ROLES = ("train", "validation", "test")

SplitLayout = Literal["physical", "shared"]


class UnsupportedSplitManifestVersionError(ValueError):
    """Raised when a manifest uses a schema version this reader cannot parse."""


def _nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{location} must be a non-empty string")
    return value


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} must be a mapping")
    return value


def _fields(value: Any, location: str, expected: set[str]) -> Mapping[str, Any]:
    result = _mapping(value, location)
    missing = expected.difference(result)
    if missing:
        raise ValueError(f"{location} is missing required fields: {sorted(missing)}")
    unknown = set(result).difference(expected)
    if unknown:
        raise ValueError(f"{location} has unknown fields: {sorted(unknown)}")
    return result


def _relative_path(value: Any, location: str, *, allow_current: bool = False) -> str:
    path_string = _nonempty_string(value, location)
    if "\\" in path_string or "\x00" in path_string or ":" in path_string:
        raise ValueError(f"{location} must be a portable relative POSIX path")

    path = PurePosixPath(path_string)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{location} must be a portable relative POSIX path")
    if str(path) != path_string:
        raise ValueError(f"{location} must be a normalized POSIX path")
    if path_string == "." and not allow_current:
        raise ValueError(f"{location} cannot refer to the manifest directory")
    return path_string


@dataclass(frozen=True)
class ManifestProvenance:
    """Producer identity used to assess manifest compatibility."""

    producer: str
    producer_version: str

    def __post_init__(self) -> None:
        """Validate provenance fields."""
        _nonempty_string(self.producer, "provenance.producer")
        _nonempty_string(self.producer_version, "provenance.producer_version")

    def to_dict(self) -> dict[str, str]:
        """Return the canonical serialized representation."""
        return {
            "producer": self.producer,
            "producer_version": self.producer_version,
        }

    @classmethod
    def from_dict(cls, value: Any) -> ManifestProvenance:
        """Parse provenance from its serialized representation."""
        fields = _fields(value, "provenance", {"producer", "producer_version"})
        return cls(
            producer=fields["producer"],
            producer_version=fields["producer_version"],
        )


@dataclass(frozen=True)
class MembershipReference:
    """Portable reference to persisted membership for one shared split."""

    path: str
    format: str

    def __post_init__(self) -> None:
        """Validate the membership artifact reference."""
        _relative_path(self.path, "membership.path")
        _nonempty_string(self.format, "membership.format")

    def to_dict(self) -> dict[str, str]:
        """Return the canonical serialized representation."""
        return {"path": self.path, "format": self.format}

    @classmethod
    def from_dict(cls, value: Any) -> MembershipReference:
        """Parse a membership reference from its serialized representation."""
        fields = _fields(value, "membership", {"path", "format"})
        return cls(path=fields["path"], format=fields["format"])


@dataclass(frozen=True)
class SplitManifestEntry:
    """Dataset and optional membership metadata for one logical split."""

    name: str
    dataset_id: str
    path: str
    storage_backend: str
    length: int
    generation_seed: int | None
    membership: MembershipReference | None = None

    def __post_init__(self) -> None:
        """Validate split metadata independent of collection layout."""
        _nonempty_string(self.name, "split.name")
        _nonempty_string(self.dataset_id, "split.dataset_id")
        _relative_path(self.path, "split.path", allow_current=True)
        _nonempty_string(self.storage_backend, "split.storage_backend")
        if isinstance(self.length, bool) or not isinstance(self.length, int):
            raise TypeError("split.length must be an integer")
        if self.length < 0:
            raise ValueError("split.length cannot be negative")
        if self.generation_seed is not None and (isinstance(self.generation_seed, bool) or not isinstance(self.generation_seed, int)):
            raise TypeError("split.generation_seed must be an integer or null")
        if self.membership is not None and not isinstance(self.membership, MembershipReference):
            raise TypeError("split.membership must be a MembershipReference or null")

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical serialized representation."""
        return {
            "name": self.name,
            "dataset_id": self.dataset_id,
            "path": self.path,
            "storage_backend": self.storage_backend,
            "length": self.length,
            "generation_seed": self.generation_seed,
            "membership": None if self.membership is None else self.membership.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any, role: str) -> SplitManifestEntry:
        """Parse one logical split from its serialized representation."""
        fields = _fields(
            value,
            f"splits.{role}",
            {"name", "dataset_id", "path", "storage_backend", "length", "generation_seed", "membership"},
        )
        membership_value = fields["membership"]
        membership = None if membership_value is None else MembershipReference.from_dict(membership_value)
        return cls(
            name=fields["name"],
            dataset_id=fields["dataset_id"],
            path=fields["path"],
            storage_backend=fields["storage_backend"],
            length=fields["length"],
            generation_seed=fields["generation_seed"],
            membership=membership,
        )


@dataclass(frozen=True)
class DatasetSplitManifest:
    """Version 1 manifest for a train/validation/test dataset collection."""

    collection_id: str
    layout: SplitLayout
    provenance: ManifestProvenance
    train: SplitManifestEntry
    validation: SplitManifestEntry
    test: SplitManifestEntry
    schema_version: int = SPLIT_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate schema version and cross-split layout invariants."""
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError("schema_version must be an integer")
        if self.schema_version != SPLIT_MANIFEST_SCHEMA_VERSION:
            raise UnsupportedSplitManifestVersionError(f"unsupported split manifest schema_version {self.schema_version}; expected {SPLIT_MANIFEST_SCHEMA_VERSION}")
        _nonempty_string(self.collection_id, "collection_id")
        if self.layout not in {"physical", "shared"}:
            raise ValueError("layout must be 'physical' or 'shared'")
        if not isinstance(self.provenance, ManifestProvenance):
            raise TypeError("provenance must be ManifestProvenance")

        entries = (self.train, self.validation, self.test)
        if not all(isinstance(entry, SplitManifestEntry) for entry in entries):
            raise TypeError("train, validation, and test must be SplitManifestEntry values")
        if self.layout == "physical":
            self._validate_physical(entries)
        else:
            self._validate_shared(entries)

    @staticmethod
    def _validate_physical(entries: tuple[SplitManifestEntry, ...]) -> None:
        if any(entry.membership is not None for entry in entries):
            raise ValueError("physical split entries cannot have membership references")
        if any(entry.path == "." for entry in entries):
            raise ValueError("physical split paths cannot refer to the manifest directory")
        if len({entry.path for entry in entries}) != len(entries):
            raise ValueError("physical split paths must be distinct")

    @staticmethod
    def _validate_shared(entries: tuple[SplitManifestEntry, ...]) -> None:
        if any(entry.membership is None for entry in entries):
            raise ValueError("shared split entries require membership references")
        backing_identities = {(entry.dataset_id, entry.path, entry.storage_backend, entry.generation_seed) for entry in entries}
        if len(backing_identities) != 1:
            raise ValueError("shared split entries must identify the same backing dataset")
        membership_paths = {entry.membership.path for entry in entries if entry.membership is not None}
        if len(membership_paths) != len(entries):
            raise ValueError("shared split membership paths must be distinct")

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned canonical representation for YAML output."""
        return {
            "schema_version": self.schema_version,
            "collection_id": self.collection_id,
            "layout": self.layout,
            "provenance": self.provenance.to_dict(),
            "splits": {
                "train": self.train.to_dict(),
                "validation": self.validation.to_dict(),
                "test": self.test.to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, value: Any) -> DatasetSplitManifest:
        """Parse and validate a manifest mapping."""
        root = _mapping(value, "split manifest root")
        if "schema_version" not in root:
            raise ValueError("split manifest root is missing required field: schema_version")
        version = root["schema_version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise TypeError("schema_version must be an integer")
        if version != SPLIT_MANIFEST_SCHEMA_VERSION:
            raise UnsupportedSplitManifestVersionError(f"unsupported split manifest schema_version {version}; expected {SPLIT_MANIFEST_SCHEMA_VERSION}")

        fields = _fields(root, "split manifest root", {"schema_version", "collection_id", "layout", "provenance", "splits"})
        splits = _fields(fields["splits"], "splits", set(_SPLIT_ROLES))
        return cls(
            schema_version=version,
            collection_id=fields["collection_id"],
            layout=fields["layout"],
            provenance=ManifestProvenance.from_dict(fields["provenance"]),
            train=SplitManifestEntry.from_dict(splits["train"], "train"),
            validation=SplitManifestEntry.from_dict(splits["validation"], "validation"),
            test=SplitManifestEntry.from_dict(splits["test"], "test"),
        )


def dump_split_manifest(manifest: DatasetSplitManifest) -> str:
    """Serialize a manifest to deterministic, human-readable YAML."""
    if not isinstance(manifest, DatasetSplitManifest):
        raise TypeError("manifest must be a DatasetSplitManifest")
    return yaml.safe_dump(manifest.to_dict(), sort_keys=False, allow_unicode=True)


def save_split_manifest(manifest: DatasetSplitManifest, path: str | Path) -> None:
    """Serialize a manifest to ``path`` using UTF-8 text."""
    Path(path).write_text(dump_split_manifest(manifest), encoding="utf-8")


def load_split_manifest(path: str | Path) -> DatasetSplitManifest:
    """Load and validate a split manifest from a YAML file."""
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return DatasetSplitManifest.from_dict(loaded)
