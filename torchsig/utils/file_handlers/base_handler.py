"""File Handler Base and Utility Classes for reading and writing datasets to/from disk."""

import pathlib
import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import suppress
from operator import index as integer_index
from types import TracebackType
from typing import Any, Self

from torchsig.utils.printing import generate_repr_str

from .errors import FileHandlerIndexError

__all__ = ["BaseFileHandler", "FileReader", "FileWriter", "reset_folder"]


def reset_folder(path: str) -> None:
    """Resets a folder by deleting it if it exists and recreating it.

    Args:
        path (str): Path to the folder to reset.

    Raises:
        ValueError: If the path exists but is not a directory.
    """
    folder_path = pathlib.Path(path)

    if folder_path.exists():
        if folder_path.is_dir():
            # To delete non-empty folder, use shutil.rmtree
            shutil.rmtree(folder_path)
            print(f"Deleted folder: {folder_path}")
        else:
            # folder is not a directory
            raise ValueError(f"Path is not a directory: {path}")

    # folder does not exists / is deleted

    # Recreate the folder
    folder_path.mkdir(parents=True, exist_ok=True)  # 'parents=True' allows creation of intermediate dirs if needed


class FileWriter(ABC):
    """Base class for writing datasets to disk.

    Attributes:
        root (pathlib.Path): Location on disk to write dataset.
    """

    def __init__(self, root: str | pathlib.Path, **kwargs: Any) -> None:
        """Initializes the FileWriter.

        Args:
            root (str): Location on disk to write dataset.
        """
        _ = kwargs  # Accepted for a uniform file-handler constructor API.
        self.root: pathlib.Path = pathlib.Path(root).resolve()

    def _setup(self) -> None:  # noqa: B027 - optional lifecycle hook
        """Hook for subclasses to perform setup after folder reset."""

    def setup(self) -> None:
        """Prepare resources before writing begins.

        This resets the root folder and then calls the subclass `_setup`.
        """
        reset_folder(self.root)
        self._setup()

    def teardown(self) -> None:  # noqa: B027 - optional lifecycle hook
        """Hook for cleaning up resources after writing is complete."""

    @abstractmethod
    def write(self, batch_idx: int, data: Any) -> None:
        """Write a single batch to disk.

        Args:
            batch_idx (int): Index of the batch being written.
            data (Any): Data to be written.

        Raises:
            NotImplementedError: Must be implemented in subclasses.
        """
        raise NotImplementedError

    def exists(self) -> bool:
        """Check if the dataset directory already exists.

        Returns:
            bool: True if `self.root` exists on disk, False otherwise.
        """
        return self.root.exists()

    def close(self) -> None:
        """Release writer resources; repeated calls must be safe."""
        self.teardown()

    def __del__(self) -> None:
        """Destructor to ensure clean resource cleanup"""
        with suppress(Exception):  # Destructors must not propagate cleanup failures.
            self.teardown()

    def __str__(self) -> str:
        """Return the writer class name."""
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        """Return a detailed representation of the writer."""
        return generate_repr_str(self)

    @abstractmethod
    def __len__(self) -> int:
        """Return the dataset length supplied by a concrete writer."""
        raise NotImplementedError

    def __enter__(self) -> Self:
        """Set up the writer and enter its context."""
        self.setup()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> bool:
        """Tear down the writer when leaving its context."""
        self.teardown()
        return False


class FileReader(ABC):
    """Base class for reading datasets from disk.

    Attributes:
        root (pathlib.Path): Dataset location on disk.
        dataset_info_filepath (pathlib.Path): Path to dataset info file.
    """

    def __init__(self, root: str | pathlib.Path, **kwargs: Any) -> None:
        """File reader base class

        Args:
            root (str): Dataset location on disk.
        """
        _ = kwargs  # Accepted for a uniform file-handler constructor API.
        self.root = pathlib.Path(root).resolve()
        self.dataset_info_filepath = self.root.joinpath("dataset_info.yaml")

    @abstractmethod
    def read(self, idx: int) -> Any:
        """Load data from disk

        Args:
            idx (int): data item to load

        Raises:
            NotImplementedError: Subclasses must implement this method

        Returns:
            Any: data and targets
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """Return the reader class name."""
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        """Return a detailed representation of the reader."""
        return generate_repr_str(self)

    @abstractmethod
    def __len__(self) -> int:
        """Return the dataset length supplied by a concrete reader."""
        raise NotImplementedError

    def setup(self) -> None:  # noqa: B027 - optional lifecycle hook
        """Prepare reader resources after construction or teardown."""

    def teardown(self) -> None:  # noqa: B027 - optional lifecycle hook
        """Release reader resources; concrete implementations must be idempotent."""

    def close(self) -> None:
        """Release reader resources; repeated calls are safe by contract."""
        self.teardown()

    def validate_index(self, idx: int) -> int:
        """Validate and return a non-boolean integer record index."""
        if isinstance(idx, bool):
            raise TypeError(f"{self.__class__.__name__} index must be an integer")
        try:
            idx = integer_index(idx)
        except TypeError as error:
            raise TypeError(f"{self.__class__.__name__} index must be an integer") from error
        size = len(self)
        if idx < 0 or idx >= size:
            raise FileHandlerIndexError(f"{self.__class__.__name__} index {idx} out of range for {self.root} (size={size})")
        return idx

    def read_many(self, indices: Iterable[int]) -> list[Any]:
        """Read records in requested order using the portable scalar fallback.

        Duplicate and shuffled indices are preserved. Returned values have the
        same ownership semantics as independent :meth:`read` calls.
        """
        if isinstance(indices, (str, bytes)):
            raise TypeError("read_many() indices must be an iterable of integers")
        try:
            requested = list(indices)
        except TypeError as error:
            raise TypeError("read_many() indices must be an iterable of integers") from error
        normalized = [self.validate_index(idx) for idx in requested]
        return [self.read(idx) for idx in normalized]

    def __enter__(self) -> Self:
        """Prepare and return the reader as a context manager."""
        self.setup()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> bool:
        """Close reader resources without suppressing exceptions."""
        self.close()
        return False


class BaseFileHandler:
    """File handler base class. Not be instantiated.

    Usage:
        >>> BaseFileHandler.create_handler(mode = "r", root = "./) # create a reader
        >>> BaseFileHandler.create_handler(mode = "w", root = "./) # create a writer
    """

    reader_class: type[FileReader] = FileReader
    writer_class: type[FileWriter] = FileWriter

    @classmethod
    def create_handler(cls, mode: str, root: str | pathlib.Path, **kwargs: Any) -> FileWriter | FileReader:
        """Creates FileWriter or FileReader

        Args:
            mode (str): read or write mode
            root (str): where file handler will be running

        Raises:
            ValueError: invalid model

        Returns:
            FileWriter | FileReader: FileHandler's reader or writer.
        """
        if mode == "r":
            return cls.reader_class(root, **kwargs)
        if mode == "w":
            return cls.writer_class(root, **kwargs)
        raise ValueError(f"Invalid File Handler mode: {mode}")

    def __str__(self) -> str:
        """Return the handler class name."""
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        """Return a detailed representation of the handler."""
        return generate_repr_str(self)
