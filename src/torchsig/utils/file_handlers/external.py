"""External File Handler base class for user imported data
"""

from __future__ import annotations

# Third Party
import numpy as np

# Built-In
from typing import TYPE_CHECKING, Tuple, List

if TYPE_CHECKING:
    from torchsig.datasets.dataset_metadata import ExternalDatasetMetadata


class ExternalFileHandler:
    """Abstract base for user-provided file handlers in ExternalTorchSigDataset.

    Users should subclass this and implement `size`, `load_dataset_metadata`,
    and `load` to adapt external datasets into the TorchSig pipeline.
    """   
    def __init__(
        self,
        root: str,
    ):
        """Initialize with the external dataset root directory.

        Args:
            root (str): Path to the external dataset.
        """
        self.root = root

    def size(self) -> int:
        """Compute the number of samples in the external dataset.

        Returns:
            int: Total number of samples.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """   
        raise NotImplementedError

    def load_dataset_metadata(self) -> ExternalDatasetMetadata:
        """Load in dataset information into a `ExternalDatasetMetadata`.
        Raises:
            NotImplementedError: Subclasses must implement this method.

        Returns:
            ExternalDatasetMetadata: Dataset metadata.
        """        
        raise NotImplementedError

    def load(self, idx: int) -> Tuple[np.ndarray, List[Any]]:
        """Load a single sample from dataset on disk

        Args:
            idx (int): index of sample to load.

        Raises:
            NotImplementedError: Subclasses must implement this method.

        Returns:
            Tuple[np.ndarray, List[Any]]: data, targets
        """        
        raise NotImplementedError