"""File Handler Base and Utility Classes for reading and writing datasets to/from disk."""

# TorchSig
# Third Party
# Built-In
import pathlib
import shutil
from typing import Any

from torchsig.utils.printing import generate_repr_str


def reset_folder(path: str) -> None:
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
    folder_path.mkdir(
        parents=True, exist_ok=True
    )  # 'parents=True' allows creation of intermediate dirs if needed


class FileWriter:

    def __init__(self, root: str, **kwargs):
        """File writer base clas

        Args:
            root (str): Location on disk to write dataset
        """
        self.root: pathlib.Path = pathlib.Path(root)

    def _setup(self) -> None:
        """Hook for subclasses to perform setup after folder reset."""

    def setup(self) -> None:
        """Prepare resources before writing begins.

        This resets the root folder and then calls the subclass `_setup`.
        """
        reset_folder(self.root)
        self._setup()

    def teardown(self) -> None:
        """Hook for cleaning up resources after writing is complete."""

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

    def __del__(self):
        """Destructor to ensure clean resource cleanup"""
        try:
            self.teardown()
        except Exception:
            pass  # Ignore errors during cleanup

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        return generate_repr_str(self)

    def __len__(self) -> int:
        raise NotImplementedError

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.teardown()
        return False


class FileReader:

    def __init__(self, root: str, **kwargs):
        """File reader base class

        Args:
            root (str): Dataset location on disk.
        """
        self.root = pathlib.Path(root)
        self.dataset_info_filepath = self.root.joinpath("dataset_info.yaml")

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
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        return generate_repr_str(self)

    def __len__(self) -> int:
        raise NotImplementedError


class BaseFileHandler:
    """File handler base class. Not be instantiated.

    Usage:
        >>> BaseFileHandler.create_handler(mode = "r", root = "./) # create a reader
        >>> BaseFileHandler.create_handler(mode = "w", root = "./) # create a writer
    """

    reader_class: FileReader = FileReader
    writer_class: FileWriter = FileWriter

    @staticmethod
    def create_handler(mode: str, root: str, **kwargs) -> FileWriter | FileReader:
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
            return BaseFileHandler.reader_class(root, **kwargs)
        if mode == "w":
            return BaseFileHandler.writer_class(root, **kwargs)
        raise ValueError(f"Invalid File Handler mode: {mode}")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        return generate_repr_str(self)
