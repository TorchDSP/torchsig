"""TorchSig Dataset generation code for command line."""

# TorchSig
# Third Party
from torch.utils.data import DataLoader

from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.utils.writer import DatasetCreator


def generate(
    root: str,
    size: int,
    dataset_metadata: DatasetMetadata,
    batch_size: int,
    num_workers: int,
    transforms: list = [],
    target_labels: list | None = None,
):
    """Generates and saves a dataset to disk.

    This function calls the `DatasetCreator` class to generate the
    dataset and save it to disk. It writes the dataset in batches
    using the specified batch size and number of workers.

    Args:
        root (str): The root directory where the dataset will be saved.
        size (int): Dataset size.
        dataset_metadata (DatasetMetadata): Metadata that defines the dataset
                                             type and properties.
        batch_size (int): The number of samples per batch to process.
        num_workers (int): The number of worker threads to use for loading
                           the data in parallel.

    Raises:
        ValueError: If the dataset type is unknown or invalid.
    """
    create_dataset = TorchSigIterableDataset(
        dataset_metadata=dataset_metadata,
        transforms=transforms,
        target_labels=target_labels,
    )

    create_loader = DataLoader(
        dataset=create_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    creator = DatasetCreator(
        dataloader=create_loader,
        root=root,
        dataset_length=size,
        overwrite=True,
    )
    creator.create()
