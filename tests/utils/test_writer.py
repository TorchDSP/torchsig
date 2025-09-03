"""Unit Tests for writer utilies.
"""
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.datasets import TorchSigIterableDataset, StaticTorchSigDataset
from torchsig.utils.writer import (
    DatasetCreator, 
    default_collate_fn, 
)
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.transforms.transforms import Spectrogram

# Third Party
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
import os
import shutil
import pytest

from typing import List, Any, Tuple
from collections.abc import Iterable
import itertools


RTOL = 1E-6
wb_data_dir =  Path.joinpath(Path(__file__).parent,'data/dataset_data/')
wb_image_dir = Path.joinpath(Path(__file__).parent,'data/dataset_images/')
getitem_dir = Path.joinpath(Path(__file__).parent,'data/getitem_data/')

# directory for test data
def setup_module(module):
    if os.path.exists(wb_data_dir):
        shutil.rmtree(wb_data_dir)
    if os.path.exists(wb_image_dir):
        shutil.rmtree(wb_image_dir) 

    os.makedirs(wb_data_dir)
    os.makedirs(wb_image_dir)


@pytest.mark.parametrize("params, is_error", [
    ({'dataset_length': None,}, True),
    ({'dataset_length': 10,}, False)
])
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

    md = DatasetMetadata(
        num_iq_samples_dataset = 4096,
        fft_size = 64,
        sample_rate = 10e6,
        num_signals_max = 3,
        num_signals_min = 0,
    )

    ds = TorchSigIterableDataset(
        dataset_metadata = md,
        seed = seed,
        transforms=[YOLOLabel()],
        # target_labels=None,
        # target_labels=[],
        target_labels=["class_name"]
        # target_labels=["class_index", "class_name"]
        # target_labels=["yolo_label"],
        # target_labels=["class_name", "yolo_label"]
    )
    dl = WorkerSeedingDataLoader(
        ds, 
        batch_size = batch_size, 
        collate_fn=default_collate_fn
    )
    if is_error:
        with pytest.raises(Exception, match=r".*"):
            dc = DatasetCreator(
                dataloader=dl,
                dataset_length = dataset_length,
                root = wb_data_dir,
                overwrite = True,
                multithreading=False   
            )
            dc.create()
    else:
        # save dataset to disk
        dc = DatasetCreator(
            dataloader=dl,
            dataset_length = dataset_length,
            root = wb_data_dir,
            overwrite = True,
            multithreading=False   
        )
        dc.create()


test_DatasetCreator_targets_params = list(itertools.product(
    # transforms
    [[YOLOLabel()], [YOLOLabel(), Spectrogram(fft_size=64)]],
    # target_labels
    [["class_name"], ["class_index"], ["class_name", "class_index"], ["yolo_label"], ["class_name", "yolo_label"]],
    # num_signals_min
    [0, 1],
    # num_signals_max
    [1, 2, 3]
))

@pytest.mark.parametrize(
    "transforms, target_labels, num_signals_min, num_signals_max", 
    test_DatasetCreator_targets_params
)
def test_DatasetCreator_targets(
    transforms,
    target_labels,
    num_signals_min,
    num_signals_max,
) -> None:
    """Test DatasetCreator with pytest.

    Args:
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    
    seed = 1234567890
    dataset_length = 6
    batch_size = 4

    md = DatasetMetadata(
        num_iq_samples_dataset = 4096,
        fft_size = 64,
        sample_rate = 10e6,
        num_signals_max = num_signals_max,
        num_signals_min = num_signals_min,
    )

    ds = TorchSigIterableDataset(
        dataset_metadata = md,
        seed = seed,
        transforms=transforms,
        target_labels=None
    )
    dl = WorkerSeedingDataLoader(
        ds, 
        batch_size = batch_size, 
        collate_fn=lambda x: x
    )
    
    # save dataset to disk
    dc = DatasetCreator(
        dataloader=dl,
        dataset_length = dataset_length,
        root = wb_data_dir,
        overwrite = True,
        multithreading=False   
    )
    dc.create()

    # breakpoint()

    # load dataset from disk
    sds = StaticTorchSigDataset(
        root = wb_data_dir,
        target_labels=target_labels
    )

    assert isinstance(dc, DatasetCreator)
    assert isinstance(dc.root, Path)
    assert isinstance(dc.overwrite, bool)
    assert isinstance(dc.multithreading, bool)
    assert dc.overwrite
    assert isinstance(dc.get_writing_info_dict(), dict)


if __name__ == "__main__":
    # test_DatasetCreator(params={'dataset_length': 10}, is_error=False)
    transforms = [[YOLOLabel()]]
    target_labels = [
        ["class_name"], # ofdm-300
        ["class_index"], # 41
        ["class_name", "class_index"], # ('ofdm-300', 41)
        ["yolo_label"], # (41, 0.2279052734375, 0.25370860860317335, 0.199951171875, 0.078125)
        ["class_name", "yolo_label"] #('ofdm-300', (41, 0.2279052734375, 0.25370860860317335, 0.199951171875, 0.078125))
    ]
    num_signals_min = [0]
    num_signals_max = [1, 3]
    params = itertools.product(
        transforms,
        target_labels,
        num_signals_min,
        num_signals_max
    )
    # breakpoint()
    for p in params:
        print(f"{p} -------")
        try:
            test_DatasetCreator_targets(*p)
        except ImportError:
            pass
        # breakpoint()
    # test_padding()
