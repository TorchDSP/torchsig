from typing import Any, Callable, List, Optional, Tuple, Union
from torchsig.utils.types import SignalCapture, SignalData
from copy import deepcopy
import numpy as np
import torch


class SignalDataset(torch.utils.data.Dataset):
    """An abstract dataset class to be sub-classed by SignalDatasets

    Args:
        transform:
            Transforms to be applied to SignalData Objects

        target_transform:
            Transforms to be applied to dataset targets

    """

    def __init__(
        self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ) -> None:
        super(SignalDataset, self).__init__()
        self.random_generator = np.random.default_rng(seed=seed)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(
        self,
        index: int,
    ):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class SignalFileDataset(SignalDataset):
    """SignalFileDataset is meant to make a mappable (index-able) dataset from
    a set of files

    Args:
        root:
            Root file path to search recursively for files

        indexer:
            Using root, constructs an index of data/meta-data

        reader:
            Given a file path, produces an SignalData object

        index_filter:
            Given an index, remove certain elements

        *\\*kwargs:**
            Keyword arguments

    """

    def __init__(
        self,
        root: str,
        indexer: Callable[[str], List[Tuple[Any, SignalCapture]]],
        reader: Callable[[SignalCapture], SignalData],
        index_filter: Optional[Callable[[Tuple[Any, SignalCapture]], bool]] = None,
        **kwargs,
    ):
        super(SignalFileDataset, self).__init__(**kwargs)
        self.reader = reader
        self.index = indexer(root)
        if index_filter:
            self.index = list(filter(index_filter, self.index))

    def __getitem__(self, item: int) -> Tuple[np.ndarray, Any]:  # type: ignore
        target = self.index[item][0]
        signal_data = self.reader(self.index[item][1])

        if self.transform:
            signal_data = self.transform(signal_data)

        if self.target_transform:
            target = self.target_transform(target)

        return signal_data.iq_data, target  # type: ignore

    def __len__(self) -> int:
        return len(self.index)


class SignalTensorDataset(torch.utils.data.TensorDataset):
    """SignalTensorDataset converts Tensors to dataset of SignalData

    Args:
        transform:
            Transforms to be applied to SignalData Objects

        target_transform:
            Transforms to be applied to dataset targets

        ***args:**
            Args

        ***kwargs:**
            *tensors is passed on to the TensorDataset superclass

    """

    def __init__(
        self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        super(SignalTensorDataset, self).__init__(*args, **kwargs)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Tuple[SignalData, Any]:  # type: ignore
        # We assume that single-precision Tensors are provided we return
        # double-precision numpy arrays for usage in the transform pipeline.
        signal_data = SignalData(
            data=deepcopy(self.tensors[0].numpy().tobytes()),
            item_type=np.dtype(np.float32),
            data_type=np.dtype(np.float64)
            if self.tensors[0].dtype == torch.float
            else np.dtype(np.complex128),
        )
        target = tuple(self.tensors[idx][index] for idx in range(1, len(self.tensors)))

        if self.transform:
            signal_data = self.transform(signal_data)

        if self.target_transform:
            target = self.target_transform(target)

        return signal_data, target  # type: ignore
