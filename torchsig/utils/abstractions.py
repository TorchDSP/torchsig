"""Some classes that define abstract data structures in other class relationships
This code is used behind the scenes in several places, and sensitive to errors; modify with caution
"""
from __future__ import annotations

from typing import Any, Optional

from torchsig.utils.random import Seedable


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


class HierarchicalMetadataObject(Seedable):
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
        parent: Optional["HierarchicalMetadataObject"] = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any
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

    def keys(self) -> list[str]:
        """Get all metadata keys.

        Returns:
            List of all metadata keys.

        Example:
            >>> obj = HierarchicalMetadataObject(metadata={"key1": 1, "key2": 2})
            >>> list(obj.keys())
            ['key1', 'key2']
        """
        return self._metadata.keys()

    def copy(self) -> "HierarchicalMetadataObject":
        """Create a copy of the object.

        Returns:
            A new instance of the same class with the same metadata and parent.

        Example:
            >>> obj = HierarchicalMetadataObject(metadata={"key": "value"})
            >>> copy_obj = obj.copy()
            >>> copy_obj["key"]
            'value'
        """
        return self.__class__(
            parent=self.parent, seed=self.seed, metadata=self._metadata
        )

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
        if key == "_metadata":
            raise KeyError(
                "unknown bug occured for:"
                + str(self.__class__.__name__)
                + "  ---   "
                + str(self.__dict__.keys())
                + "; check metadata field names?"
            )

        if (
            key == "metadata"
        ):  # TODO: reconsider this; workaround to make getattr play nice
            return self._metadata.copy()
        if key in self._metadata:
            return self._metadata[key]
        if self.parent is not None:
            return self.parent[key]
        raise MetadataAttributeError(
            "key: '" + str(key) + "' could not be found in metadata"
        )

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
        del self._metadata[key]

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
        except MetadataAttributeError as e:
            e.add_note("key missing: '" + str(key) + "'; ")
            raise e

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
