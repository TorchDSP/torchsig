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
    """Static Dataset class, which loads pre-generated data from a directory.

    This class assumes that the dataset has already been generated and saved to disk using a subclass of `NewTorchSigDataset`. 
    It allows loading raw or processed data from disk for inference or analysis.

    Args:
        root (str): The root directory where the dataset is stored.
        dataset_type (str): Type of the dataset, either "narrowband" or "wideband".
        transforms (list, optional): Transforms to apply to the data (default: []).
        target_transforms (list, optional): Target transforms to apply (default: []).
        file_handler_class (TorchSigFileHandler, optional): Class used for reading the dataset (default: ZarrFileHandler).
    """

    def __init__(
        self,
        root: str,
        dataset_type: str,
        transforms: list = [],
        target_transforms: list = [],
        file_handler_class: TorchSigFileHandler = ZarrFileHandler,
        train: bool = None,
        # **kwargs
    ):
        self.root = Path(root)
        self.dataset_type = dataset_type
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.file_handler = file_handler_class
        self.train = train

        # check dataset data type from writer_info.dataset_yaml_name
        with open(f"{self.root}/{writer_yaml_name}", 'r') as f:
            writer_info = yaml.load(f, Loader=yaml.FullLoader)
            self.raw = writer_info['save_type'] == "raw"

        self.dataset_metadata = to_dataset_metadata(
            f"{self.root}/{dataset_yaml_name}")

        # dataset size
        self.num_samples = self.file_handler.size(self.root)

        self._verify()

    def _verify(self):
        # Transforms
        self.transforms = verify_transforms(self.transforms)

        # Target Transforms
        self.target_transforms = verify_target_transforms(
            self.target_transforms)
        # print(self.target_transforms)
        # print("verify")

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Tuple]:
        """Retrieves a sample from the dataset by index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[np.ndarray, Tuple]: The data and targets for the sample.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if idx >= 0 and idx < self.__len__():
            # load data and metadata
            # data: np.ndarray
            # signal_metadatas: List[dict]
            if self.raw:
                # loading in raw IQ data and signal metadata
                data, signal_metadatas = self.file_handler.static_load(
                    self.root, idx)

                # convert to DatasetSignal
                sample = DatasetSignal(
                    data=data,
                    signals=signal_metadatas,
                    dataset_metadata=self.dataset_metadata,
                )

                # apply user transforms
                for t in self.transforms:
                    sample = t(sample)

                # convert to DatasetDict
                sample = DatasetDict(signal=sample)

                # apply target transforms
                targets = []
                for target_transform in self.target_transforms:
                    # apply transform to all metadatas
                    sample.metadata = target_transform(sample.metadata)
                    # get target outputs
                    target_transform_output = []
                    for signal_metadata in sample.metadata:
                        # extract output from metadata
                        # as required by TT target output field name
                        signal_output = []
                        for field in target_transform.targets_metadata:
                            signal_output.append(signal_metadata[field])

                        signal_output = tuple(signal_output)
                        target_transform_output.append(signal_output)

                    targets.append(target_transform_output)

                # convert targets as a list of target transform output ordered by transform
                # to ordered by signal
                # e.g., [(transform 1 output for all signals), (transform 2 output for all signals), ... ] ->
                # [signal 1 outputs, signal 2 outputs, ... ]
                targets = list(zip(*targets))

                if len(self.target_transforms) == 0:
                    # no target transform applied
                    targets = sample.metadata
                elif self.dataset_type == 'narrowband':
                    # only one signal in list for narrowband
                    # unwrap targets
                    targets = [item[0] if len(
                        item) == 1 else item for row in targets for item in row]
                    # unwrap any target transform output that produced a tuple
                    targets = targets[0] if len(
                        targets) == 1 else tuple(targets)
                else:
                    # wideband
                    targets = [
                        tuple([item[0] if len(item) == 1 else item for item in row]) for row in targets]
                    # unwrap any target transform output that produced a tuple
                    targets = [row[0] if len(
                        row) == 1 else row for row in targets]

                return sample.data, targets
            # else:
            # loading in transformed data and targets from target transform
            data, targets = self.file_handler.static_load(self.root, idx)

            return data, targets

        else:
            raise IndexError(
                f"Index {idx} is out of bounds. Must be [0, {self.__len__()}]")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.root}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(root={self.root}, "
            f"transforms={self.transforms.__repr__()}, "
            f"target_transforms={self.target_transforms.__repr__()}, "
            f"file_handler_class={self.file_handler}, "
            f"train={self.train})"
        )
