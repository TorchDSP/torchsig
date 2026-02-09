"""Unit Tests for writer utilies."""

from torchsig.datasets.datasets import TorchSigIterableDataset, StaticTorchSigDataset
from torchsig.utils.writer import (
    DatasetCreator,
    default_collate_fn,
)
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.defaults import default_dataset

# Third Party
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import os
import shutil
import pytest

from typing import List, Any, Tuple
from collections.abc import Iterable
import itertools


RTOL = 1e-6
wb_data_dir = Path.joinpath(Path(__file__).parent, "data/dataset_data/")
wb_image_dir = Path.joinpath(Path(__file__).parent, "data/dataset_images/")
getitem_dir = Path.joinpath(Path(__file__).parent, "data/getitem_data/")


# directory for test data
def setup_module(module):
    if os.path.exists(wb_data_dir):
        shutil.rmtree(wb_data_dir)
    if os.path.exists(wb_image_dir):
        shutil.rmtree(wb_image_dir)

    os.makedirs(wb_data_dir)
    os.makedirs(wb_image_dir)


@pytest.mark.parametrize(
    "params, is_error",
    [
        (
            {
                "dataset_length": None,
            },
            True,
        ),
        (
            {
                "dataset_length": 10,
            },
            False,
        ),
    ],
)
def test_DatasetCreator(params: dict, is_error: bool) -> None:
    """Test DatasetCreator with pytest.

    Args:
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    seed = 1234567890
    dataset_length = params["dataset_length"]
    batch_size = 4
    ds = default_dataset(num_signals_max=3, num_signals_min=0)
    dl = WorkerSeedingDataLoader(ds, batch_size=batch_size)
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            dc = DatasetCreator(dataloader=dl, dataset_length=dataset_length, root=wb_data_dir, overwrite=True, multithreading=False)
            dc.create()
    else:
        # save dataset to disk
        dc = DatasetCreator(dataloader=dl, dataset_length=dataset_length, root=wb_data_dir, overwrite=True, multithreading=False)
        dc.create()
