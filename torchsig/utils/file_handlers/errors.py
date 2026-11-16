"""Public exception categories shared by file-handler backends."""

__all__ = [
    "CorruptDatasetError",
    "FileHandlerError",
    "FileHandlerIndexError",
    "IncompatibleOptionsError",
    "IncompleteDatasetError",
    "UnsupportedSchemaError",
]


class FileHandlerError(Exception):
    """Base class for storage-backend failures."""


class FileHandlerIndexError(IndexError, FileHandlerError):
    """A record index is invalid for the selected dataset."""


class IncompleteDatasetError(ValueError, FileHandlerError):
    """A dataset was not completely published or finalized."""


class UnsupportedSchemaError(ValueError, FileHandlerError):
    """A stored schema or format version is unsupported."""


class CorruptDatasetError(ValueError, FileHandlerError):
    """Stored data violates its declared schema or integrity contract."""


class IncompatibleOptionsError(ValueError, FileHandlerError):
    """Requested backend options or reader/writer pair are incompatible."""
