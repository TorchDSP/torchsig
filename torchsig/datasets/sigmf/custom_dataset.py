"""Custom SigMF Static TorchSig Dataset

This module provides a specialized Dataset class for loading pre-converted SigMF data
from TorchSig's internal storage format. It supports both raw IQ data with transforms
and pre-processed data loading for efficient inference and training.
"""

from pathlib import Path
from typing import Tuple
import yaml
import numpy as np

from torch.utils.data import Dataset
from torchsig.signals.signal_types import DatasetSignal, DatasetDict
from torchsig.utils.verify import verify_transforms, verify_target_transforms
from torchsig.utils.file_handlers.base_handler import TorchSigFileHandler
from torchsig.utils.file_handlers.zarr import ZarrFileHandler
from torchsig.datasets.dataset_utils import dataset_yaml_name, writer_yaml_name, to_dataset_metadata


class CustomSigmfStaticTorchSigDataset(Dataset):
    """Static Dataset class for loading pre-generated SigMF data from disk.

    This class provides access to SigMF datasets that have been converted to TorchSig's
    internal format using SigMFDatasetConverter. It supports both raw IQ data loading
    with on-the-fly transforms and pre-processed data loading for faster access.

    The dataset handles both wideband (multiple overlapping signals) and narrowband
    (single signal per sample) formats, automatically adjusting target processing
    based on the dataset type.

    Args:
        root (str): Root directory where the converted dataset is stored.
            Should contain both data files and configuration YAML files.
        dataset_type (str): Type of dataset, either "narrowband" or "wideband".
            Affects how targets are processed and returned.
        transforms (list, optional): List of transforms to apply to signal data.
            Only used when loading raw IQ data. Defaults to [].
        target_transforms (list, optional): List of target transforms to apply
            to signal annotations. Only used with raw data. Defaults to [].
        file_handler_class (TorchSigFileHandler, optional): File handler class
            for reading dataset files. Defaults to ZarrFileHandler.
        train (bool, optional): Whether this is a training dataset. Currently
            not used but maintained for compatibility. Defaults to None.


    """

    def __init__(
        self,
        root: str,
        dataset_type: str,
        transforms: list = [],
        target_transforms: list = [],
        file_handler_class: TorchSigFileHandler = ZarrFileHandler,
        train: bool = None,
    ):
        """Initialize the custom SigMF static dataset.

        Args:
            root: Root directory where the converted dataset is stored
            dataset_type: Type of dataset ("narrowband" or "wideband")
            transforms: List of transforms to apply to signal data
            target_transforms: List of target transforms to apply to annotations
            file_handler_class: File handler class for reading dataset files
            train: Whether this is a training dataset (compatibility parameter)
        """
        self.root = Path(root)
        self.dataset_type = dataset_type
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.file_handler = file_handler_class
        self.train = train

        # Check dataset data type from writer_info.yaml to determine if raw or processed
        with open(f"{self.root}/{writer_yaml_name}", 'r') as f:
            writer_info = yaml.load(f, Loader=yaml.FullLoader)
            self.raw = writer_info['save_type'] == "raw"

        # Load dataset metadata from configuration file
        self.dataset_metadata = to_dataset_metadata(
            f"{self.root}/{dataset_yaml_name}")

        # Get total number of samples in the dataset
        self.num_samples = self.file_handler.size(self.root)

        # Verify transforms are compatible
        self._verify()

    def _verify(self):
        """Verify that transforms and target transforms are valid.

        This method ensures that all provided transforms are compatible with
        TorchSig's transform system and properly formatted.
        """
        # Verify and standardize transforms
        self.transforms = verify_transforms(self.transforms)

        # Verify and standardize target transforms
        self.target_transforms = verify_target_transforms(
            self.target_transforms)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Total number of samples available in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Tuple]:
        """Retrieve a sample from the dataset by index.

        This method handles both raw and pre-processed data loading. For raw data,
        it applies transforms and processes targets according to the dataset type.
        For pre-processed data, it directly returns the stored data and targets.

        Args:
            idx (int): Index of the sample to retrieve (0 <= idx < len(dataset)).

        Returns:
            Tuple[np.ndarray, Tuple]: A tuple containing:
                - data (np.ndarray): Signal data (may be IQ samples or spectrogram)
                - targets: Target annotations, format depends on dataset type:
                    - Narrowband: Single target tuple or value
                    - Wideband: List of target tuples for multiple signals

        Raises:
            IndexError: If the index is out of bounds.

        """
        if idx >= 0 and idx < self.__len__():

            if self.raw:
                # Loading raw IQ data and signal metadata for on-the-fly processing
                data, signal_metadatas = self.file_handler.static_load(
                    self.root, idx)

                # Convert to TorchSig DatasetSignal format
                sample = DatasetSignal(
                    data=data,
                    signals=signal_metadatas,
                    dataset_metadata=self.dataset_metadata,
                )

                # Apply user-specified data transforms (e.g., spectrogram, normalization)
                for transform in self.transforms:
                    sample = transform(sample)

                # Convert to DatasetDict format for target processing
                sample = DatasetDict(signal=sample)

                # Apply target transforms and collect outputs
                targets = []
                for target_transform in self.target_transforms:
                    # Apply transform to all signal metadata
                    sample.metadata = target_transform(sample.metadata)

                    # Extract target outputs for this transform
                    target_transform_output = []
                    for signal_metadata in sample.metadata:
                        # Extract required fields from metadata
                        signal_output = []
                        for field in target_transform.targets_metadata:
                            signal_output.append(signal_metadata[field])

                        signal_output = tuple(signal_output)
                        target_transform_output.append(signal_output)

                    targets.append(target_transform_output)

                # Reorganize targets: from transform-major to signal-major ordering
                # [(transform_1_all_signals), (transform_2_all_signals)] ->
                # [(signal_1_all_transforms), (signal_2_all_transforms)]
                targets = list(zip(*targets)) if targets else []

                # Process targets based on dataset type and transform count
                if len(self.target_transforms) == 0:
                    # No target transforms applied - return raw metadata
                    targets = sample.metadata
                elif self.dataset_type == 'narrowband':
                    # Narrowband: single signal per sample, unwrap from lists
                    targets = [item[0] if len(item) == 1 else item
                               for row in targets for item in row]
                    # Further unwrap if only one target transform was applied
                    targets = targets[0] if len(
                        targets) == 1 else tuple(targets)
                else:
                    # Wideband: multiple signals per sample, keep as list
                    targets = [
                        tuple([item[0] if len(item) ==
                              1 else item for item in row])
                        for row in targets
                    ]
                    # Unwrap single-transform case
                    targets = [row[0] if len(
                        row) == 1 else row for row in targets]

                return sample.data, targets

            else:
                # Loading pre-processed data and targets directly from storage
                data, targets = self.file_handler.static_load(self.root, idx)
                return data, targets

        else:
            raise IndexError(
                f"Index {idx} is out of bounds. Must be in range [0, {self.__len__()})")

    def __str__(self) -> str:
        """Return a string representation of the dataset.

        Returns:
            str: Simple string showing class name and root directory.
        """
        return f"{self.__class__.__name__}: {self.root}"

    def __repr__(self) -> str:
        """Return a detailed string representation of the dataset.

        Returns:
            str: Detailed representation including all configuration parameters.
        """
        return (
            f"{self.__class__.__name__}"
            f"(root={self.root}, "
            f"dataset_type={self.dataset_type}, "
            f"transforms={self.transforms.__repr__()}, "
            f"target_transforms={self.target_transforms.__repr__()}, "
            f"file_handler_class={self.file_handler}, "
            f"train={self.train})"
        )
