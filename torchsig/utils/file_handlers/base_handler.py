"""File Handlers for writing and reading datasets to/from disk

Only write one item from a TorchSigDataset's `__getitem__` method
"""

from __future__ import annotations

# TorchSig
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.dataset_utils import dataset_full_path, writer_yaml_name
from torchsig.utils.printing import generate_repr_str

# Third Party
import numpy as np

# Built-In
from typing import Any, Tuple, List, Dict, TYPE_CHECKING
import os
import shutil
import yaml

# Imports for type checking
if TYPE_CHECKING:
    from torchsig.datasets.datasets import NewTorchSigDataset


class BaseFileHandler():
    def __init__(
        self,
        root: str
    ):
        self.root = root

    def _reset_folder(self, filepath: str) -> None:
        if os.path.exists(filepath):
            shutil.rmtree(filepath)
        os.makedirs(filepath, exist_ok=True)
    
    def _setup(self) -> None:
        pass

    def setup(self) -> None:
        # Prepares any necessary resources or configurations before writing.
        # dataset either does not exist or we want to overwrite it
        # ensures we have empty directory
        self._reset_folder(self.root)

        self._setup()
        
    def teardown(self) -> None:
        # cleans up resources after writing
        pass

    def exists(self) -> bool:
        # check whether dataset already exists on disk
        if os.path.exists(self.root):
            return True
        else:
            return False

    def write(self, batch_idx: int, batch: Any) -> None:
        # writes a batch from dataset's __getitem__
        raise NotImplementedError

    def load(self, idx: int) -> Any:
        # loads sample `idx` from disk into memory
        raise NotImplementedError

    @staticmethod
    def static_load(filename:str, idx: int) -> Any:
        # loads sample `idx` from `filename` into memory
        # method can be used without instantiating class
        # used for just reading
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        return generate_repr_str(self)

class TorchSigFileHandler(BaseFileHandler):
    
    def __init__(
        self,
        root: str,
        batch_size: int = 1
    ):
        super().__init__(
            root = root,
        )

        self.batch_size = batch_size

    def write(self, batch_idx: int, batch: Any) -> None:
        # writes a batch from dataset's __getitem__
        raise NotImplementedError

    @staticmethod
    def size(dataset_path: str) -> int:
        # given path to dataset on disk
        # return dataset size
        raise NotImplementedError

    @staticmethod
    def static_load(filename:str, idx: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        # loads sample `idx` from `filename` into memory
        # method can be used without instantiating class
        # used for just reading
        raise NotImplementedError

    def load(self, idx: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        # loads sample `idx` from disk into memory
        # uses instantiated class
        return self.static_load(self.root, idx)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        return generate_repr_str(self)

    @staticmethod
    def _calculate_batch_size(root: str) -> int:

        writer_yaml = f"{root}/{writer_yaml_name}"
        with open(writer_yaml, 'r') as f:
            writer_dict = yaml.load(f, Loader=yaml.FullLoader)
            # extract batch size
            batch_size = writer_dict['batch_size']

        return batch_size
