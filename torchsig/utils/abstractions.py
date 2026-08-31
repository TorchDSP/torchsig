"""Some classes that define abstract data structures in other class relationships
This code is used behind the scenes in several places, and sensitive to errors; modify with caution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from torchsig.utils.metadata_debug_mixin import MetadataDebugMixin
from torchsig.utils.metadata_logging import (
    MetadataDebugConfig,
    MetadataDebugStatistics,
)
from torchsig.utils.random import Seedable

if TYPE_CHECKING:
    from collections.abc import KeysView

__all__ = [
    "HierarchicalMetadataObject",
    "MetadataAttributeError",
    "MetadataDebugConfig",
    "MetadataDebugStatistics",
    "MetadataResolution",
]


@dataclass(frozen=True)
class MetadataResolution:
    """Describe how a metadata key resolves through an object hierarchy.

    The result intentionally excludes the metadata value so diagnostic output
    can be logged or displayed without exposing large or sensitive values.

    Attributes:
        key: Metadata key that was inspected.
        found: Whether the key resolves to a value.
        source: Whether the winning value is local, inherited, or missing.
        depth: Number of parent links to the winning value, or ``None`` when
            the key is missing.
        owner_type: Class name of the object owning the winning value, or
            ``None`` when the key is missing.
        overrides_parent: Whether the winning value shadows another value for
            the same key farther up the hierarchy.
        cycle_detected: Whether parent traversal encountered a cycle.
        path: Class names visited during parent traversal.
    """

    key: str
    found: bool
    source: Literal["local", "inherited", "missing"]
    depth: int | None
    owner_type: str | None
    overrides_parent: bool
    cycle_detected: bool
    path: tuple[str, ...]


class MetadataAttributeError(AttributeError):
    """Custom exception for metadata attribute errors.

    This exception is raised when there are issues accessing or manipulating metadata fields.
    """

    def __init__(self, message: str, **kwargs: Any) -> None:
        """Initialize the MetadataAttributeError.

        Args:
            message: Error message describing the issue.
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            AttributeError: Base class for attribute-related errors.
        """
        super().__init__(message, **kwargs)


class HierarchicalMetadataObject(MetadataDebugMixin, Seedable):
    """A class for representing objects which have metadata in a hierarchical relationship.

    Metadata can be accessed directly (e.g., obj["some_field"]), or through the metadata field (e.g., obj.metadata["some_field"]).
    Metadata fields can be treated as class fields for access; i.e., obj.some_field is equivalent to obj["some_field"] or obj.metadata["some_field"] as long as some_field is not already a class field of obj.
    Metadata fields are inherited in a parent/child relationship such that if parent.metadata = {"field_1":4,"field_2":5}, and child.metadata = {"field_2":6} then child.field_1==4 and child.field_2==6.
    The parent of a HierarchicalMetadataObject (as defined in the Seedable class) should always be another HierarchicalMetadataObject.

    Attributes:
        _metadata: Dictionary containing the object's metadata.
    """

    def __init__(
        self,
        seed: int | None = None,
        parent: HierarchicalMetadataObject | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the HierarchicalMetadataObject.

        Args:
            seed: Random seed for reproducibility. Defaults to None.
            parent: Parent object in the hierarchy. Defaults to None.
            metadata: Initial metadata dictionary. Defaults to None.
            **kwargs: Additional metadata fields to set.

        Note:
            This will override fields in the object passed in with arguments directly given to the generator; useful for making multiple similar but not identical objects.
        """
        self._initialize_metadata_debug()
        self._metadata = {}
        Seedable.__init__(self, seed=seed, parent=parent)
        if metadata is not None and len(metadata.keys()) > 0:
            for key in metadata:
                self._metadata[key] = metadata[key]
        for key in kwargs:
            self._metadata[key] = kwargs[
                key
            ]  # this will override fields in the object passed in with arguments directly given to the generator; useful for making multiple similar but not identical objects

    def get_full_metadata(self) -> dict[str, Any]:
        """Function for modifying and returning a new metadata with all the fields in parent or child, with child overriding parent in conflicts.

        Returns:
            Dictionary containing all metadata from parent and child, with child values overriding parent values in case of conflicts.

        Example:
            >>> parent = HierarchicalMetadataObject(metadata={"field_1": 4, "field_2": 5})
            >>> child = HierarchicalMetadataObject(parent=parent, metadata={"field_2": 6})
            >>> child.get_full_metadata()
            {'field_1': 4, 'field_2': 6}
        """
        full_metadata = {}
        if self.parent is not None:
            for key in self.parent.get_full_metadata():
                full_metadata[key] = self.parent[key]
        for key in self.keys():
            full_metadata[key] = self[key]
        return full_metadata

    def explain_metadata(self, key: str) -> MetadataResolution:
        """Explain where a metadata key resolves without returning its value.

        This diagnostic method walks the object's metadata-parent hierarchy and
        reports the first object defining ``key``. It also reports whether that
        definition overrides another parent definition and safely terminates if
        a malformed hierarchy contains a parent cycle. Ordinary item and
        attribute lookup behavior is not modified.

        Args:
            key: Metadata key to inspect.

        Returns:
            Structured information describing the key's resolution.

        Raises:
            TypeError: If ``key`` is not a string.
        """
        if not isinstance(key, str):
            raise TypeError("metadata key must be a string")

        current: HierarchicalMetadataObject | None = self
        visited: set[int] = set()
        path: list[str] = []
        owner: HierarchicalMetadataObject | None = None
        owner_depth: int | None = None
        overrides_parent = False
        cycle_detected = False
        depth = 0

        while isinstance(current, HierarchicalMetadataObject):
            current_id = id(current)
            if current_id in visited:
                cycle_detected = True
                break

            visited.add(current_id)
            path.append(type(current).__name__)

            current_metadata = object.__getattribute__(current, "_metadata")
            defines_key = key == "metadata" and depth == 0
            defines_key = defines_key or key in current_metadata
            if defines_key:
                if owner is None:
                    owner = current
                    owner_depth = depth
                else:
                    overrides_parent = True

            parent = current.parent
            if not isinstance(parent, HierarchicalMetadataObject):
                break
            current = parent
            depth += 1

        if owner is None:
            return MetadataResolution(
                key=key,
                found=False,
                source="missing",
                depth=None,
                owner_type=None,
                overrides_parent=False,
                cycle_detected=cycle_detected,
                path=tuple(path),
            )

        return MetadataResolution(
            key=key,
            found=True,
            source="local" if owner_depth == 0 else "inherited",
            depth=owner_depth,
            owner_type=type(owner).__name__,
            overrides_parent=overrides_parent,
            cycle_detected=cycle_detected,
            path=tuple(path),
        )

    def keys(self) -> KeysView[str]:
        """Get a dynamic view of the local metadata keys.

        Returns:
            Dynamic view of the local metadata keys. The view reflects later
            additions to and removals from this object's metadata.

        Example:
            >>> obj = HierarchicalMetadataObject(metadata={"key1": 1, "key2": 2})
            >>> list(obj.keys())
            ['key1', 'key2']
        """
        return self._metadata.keys()

    def copy(
        self,
        *,
        preserve_parent: bool = True,
    ) -> HierarchicalMetadataObject:
        """Create a copy of the metadata object.

        Creates a new instance of the same class with a shallow copy of its
        metadata. By default, the copied object preserves the same parent
        relationship as the original, but this behavior can be disabled to
        create a detached copy with no parent. Subclasses with additional
        required constructor arguments can provide them by overriding
        :meth:`_copy_kwargs` without replacing the rest of the copy behavior.

        Args:
            preserve_parent: If ``True`` (default), preserve the parent
                relationship in the copied object. If ``False``, the copied
                object is created without a parent.

        Returns:
            A new instance of the same class with copied metadata and the
            requested parent relationship.
        """
        return self.__class__(
            parent=self.parent if preserve_parent else None,
            seed=self.rng_seed,
            metadata=self._metadata.copy(),
            **self._copy_kwargs(),
        )

    def _copy_kwargs(self) -> dict[str, Any]:
        """Return subclass-specific constructor arguments used by ``copy``.

        Subclasses whose constructors require arguments beyond ``parent``,
        ``seed``, and ``metadata`` should override this method and return those
        arguments as a new dictionary. The base copy implementation continues
        to control metadata copying and parent preservation.

        Returns:
            Additional keyword arguments for constructing the copied object.
        """
        return {}

    def __getitem__(self, key: str) -> Any:
        """Get a metadata value by key.

        Args:
            key: The metadata key to retrieve.

        Returns:
            The value associated with the key.

        Raises:
            KeyError: If trying to access the _metadata field directly.
            MetadataAttributeError: If the key is not found in the metadata or parent metadata.

        Example:
            >>> obj = HierarchicalMetadataObject(metadata={"key": "value"})
            >>> obj["key"]
            'value'
        """
        debug_enabled = object.__getattribute__(self, "__dict__").get(
            "_metadata_debug_enabled",
            False,
        )
        if debug_enabled and isinstance(key, str):
            self._log_metadata_event("lookup", key)

        if key == "_metadata":
            raise KeyError("unknown bug occured for:" + str(self.__class__.__name__) + "  ---   " + str(self.__dict__.keys()) + "; check metadata field names?")

        if key == "metadata":  # TODO: reconsider this; workaround to make getattr play nice
            return self._metadata.copy()
        if key in self._metadata:
            return self._metadata[key]
        if self.parent is not None:
            return self.parent[key]
        raise MetadataAttributeError("key: '" + str(key) + "' could not be found in metadata")

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a metadata value by key.

        Args:
            key: The metadata key to set.
            value: The value to associate with the key.

        Example:
            >>> obj = HierarchicalMetadataObject()
            >>> obj["key"] = "value"
            >>> obj["key"]
            'value'
        """
        self._metadata[key] = value
        debug_enabled = object.__getattribute__(self, "__dict__").get(
            "_metadata_debug_enabled",
            False,
        )
        if debug_enabled and isinstance(key, str):
            self._log_metadata_event("set", key, value)

    def __delitem__(self, key: str) -> None:
        """Delete a metadata value by key.

        Args:
            key: The metadata key to delete.

        Example:
            >>> obj = HierarchicalMetadataObject(metadata={"key": "value"})
            >>> del obj["key"]
            >>> "key" in obj.keys()
            False
        """
        deleted_value = self._metadata[key]
        del self._metadata[key]
        debug_enabled = object.__getattribute__(self, "__dict__").get(
            "_metadata_debug_enabled",
            False,
        )
        if debug_enabled and isinstance(key, str):
            self._log_metadata_event("delete", key, deleted_value)

    def key_lookup(self, key: str) -> Any:
        """Lookup a metadata key with enhanced error reporting.

        Args:
            key: The metadata key to lookup.

        Returns:
            The value associated with the key.

        Raises:
            MetadataAttributeError: If the key is not found in the metadata or parent metadata.

        Example:
            >>> obj = HierarchicalMetadataObject(metadata={"key": "value"})
            >>> obj.key_lookup("key")
            'value'
        """
        try:
            return self[key]
        except MetadataAttributeError as exc:
            message = f"{exc}; key missing: {key!r}"
            raise MetadataAttributeError(message) from exc

    def __setstate__(self, data):
        """Workaround pickling with multiple workers."""
        self.__dict__.update(data)
        self.__dict__.setdefault("_metadata_debug_enabled", False)
        self.__dict__.setdefault("_metadata_debug_session", None)

    def __getattribute__(self, key: str) -> Any:
        """Get an attribute, falling back to metadata lookup if not found.

        Args:
            key: The attribute or metadata key to retrieve.

        Returns:
            The attribute value or metadata value.

        Raises:
            MetadataAttributeError: If the attribute or metadata key is not found.

        Example:
            >>> obj = HierarchicalMetadataObject(metadata={"key": "value"})
            >>> obj.key
            'value'
        """
        try:
            return super().__getattribute__(key)
        except MetadataAttributeError:
            raise
        except AttributeError:
            return self.key_lookup(key)

    def _is_metadata_debug_target(self, target: Any) -> bool:
        """Return whether an object supports hierarchical metadata debugging."""
        return isinstance(target, HierarchicalMetadataObject)
