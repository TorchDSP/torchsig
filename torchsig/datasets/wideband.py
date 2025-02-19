"""WidebandMetadata and NewWideband Class
"""

# TorchSig
from torchsig.datasets.datasets import NewTorchSigDataset, StaticTorchSigDataset
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.utils.file_handlers.base_handler import TorchSigFileHandler
from torchsig.utils.file_handlers.zarr import ZarrFileHandler


class NewWideband(NewTorchSigDataset):
    """Creates a Wideband dataset.

    This class is responsible for creating a Wideband dataset, including the metadata 
    and any transformations needed.

    Args:
        dataset_metadata (DatasetMetadata | str | dict): Metadata for the Wideband dataset. 
            This can be a `DatasetMetadata` object, a string (path to the metadata file), or a dictionary.
        **kwargs: Additional keyword arguments passed to the parent class (`NewTorchSigDataset`).
    """
    def __init__(self, dataset_metadata: DatasetMetadata | str | dict, **kwargs):
        """Initializes the Wideband dataset.

        Args:
            dataset_metadata (DatasetMetadata): Metadata specific to the Wideband dataset.
        """
        super().__init__(dataset_metadata=dataset_metadata, **kwargs)




class StaticWideband(StaticTorchSigDataset):
    """Loads and provides access to a pre-generated Wideband dataset.

    This class allows loading a pre-generated Wideband dataset from disk, and includes 
    options for applying transformations to both the data and target labels. The dataset 
    can be accessed in raw or impaired form, depending on the flags set.

    Args:
        root (str): The root directory where the dataset is stored.
        impaired (bool): Whether the dataset contains impaired signals. Defaults to False.
        transforms (list, optional): A transformation to apply to the data. Defaults to [].
        target_transforms (list, optional): A transformation to apply to the targets. Defaults to [].
        file_handler_class (TorchSigFileHandler, optional): The file handler class for reading the dataset. 
            Defaults to `ZarrFileHandler`.
        **kwargs: Additional keyword arguments passed to the parent class (`StaticTorchSigDataset`).
    """
    def __init__(
        self,
        root: str,
        impaired: bool,
        transforms: list = [],
        target_transforms: list = [],
        file_handler_class: TorchSigFileHandler = ZarrFileHandler,
        train: bool = None,
        **kwargs
    ):
        """Initializes the StaticWideband dataset.

        Args:
            root (str): The root directory where the dataset is stored.
            impaired (bool): Whether the dataset is impaired or not.
            transforms (list, optional): Transforms to apply to the data.
            target_transforms (list, optional): Target Transforms to apply.
            file_handler_class (TorchSigFileHandler, optional): The file handler class for reading the dataset.
            **kwargs: Additional arguments passed to the parent class initialization.
        """
        super().__init__(
            root = root,
            impaired = impaired,
            dataset_type = "wideband",
            transforms = transforms,
            target_transforms = target_transforms,
            file_handler_class = file_handler_class,
            train=train,
            **kwargs
        )
