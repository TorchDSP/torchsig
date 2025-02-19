"""TorchSig Dataset generation code for command line
"""

# TorchSig
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.narrowband import NewNarrowband
from torchsig.datasets.wideband import NewWideband
from torchsig.utils.writer import DatasetCreator

# Third Party

# Built-In

# generates a dataset, writes to disk
def generate(
    root: str,
    dataset_metadata: DatasetMetadata,
    batch_size: int,
    num_workers: int,
):
    """Generates and saves a dataset to disk.

    This function selects the dataset type ('narrowband' or 'wideband') based 
    on the provided metadata and then calls the `DatasetCreator` class to 
    generate the dataset and save it to disk. It writes the dataset in batches 
    using the specified batch size and number of workers.

    Args:
        root (str): The root directory where the dataset will be saved.
        dataset_metadata (DatasetMetadata): Metadata that defines the dataset 
                                             type and properties.
        batch_size (int): The number of samples per batch to process.
        num_workers (int): The number of worker threads to use for loading 
                           the data in parallel.

    Raises:
        ValueError: If the dataset type is unknown or invalid.
    """
    
    create_dataset = None
    if dataset_metadata.dataset_type == "narrowband":
        create_dataset = NewNarrowband(dataset_metadata=dataset_metadata)
    elif dataset_metadata.dataset_type == "wideband":
        create_dataset = NewWideband(dataset_metadata=dataset_metadata)

    creator = DatasetCreator(
        dataset=create_dataset,
        root = root,
        overwrite = True,
        batch_size=batch_size,
        num_workers=num_workers
    )
    creator.create()


