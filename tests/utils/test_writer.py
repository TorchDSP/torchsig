"""Unit Tests for writer utilies.
"""
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.datasets import TorchSigIterableDataset, StaticTorchSigDataset
from torchsig.utils.writer import (
    DatasetCreator, 
    default_collate_fn, 
    handle_non_numpy_datatypes,
    batch_as_signal_list
)
from torchsig.signals.signal_types import Signal, targets_as_metadata
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
from torch import Tensor

from typing import List, Any, Tuple
from collections.abc import Iterable
import itertools
from time import time


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

def test_handle_non_numpy_datatypes():
    # Test with a Tensor
    tensor_data = Tensor([1, 2, 3, 4, 5])
    converted_tensor = handle_non_numpy_datatypes(tensor_data)
    expected_numpy_array = np.array([1, 2, 3, 4, 5])
    assert isinstance(converted_tensor, np.ndarray), "Converted data should be a NumPy array"
    np.testing.assert_array_equal(converted_tensor, expected_numpy_array)

    # Test with a NumPy array (no conversion should take place)
    numpy_data = np.array([6, 7, 8, 9, 10])
    no_conversion_case = handle_non_numpy_datatypes(numpy_data)
    assert isinstance(no_conversion_case, np.ndarray), "Data should remain a NumPy array"
    np.testing.assert_array_equal(no_conversion_case, numpy_data)

    # Test with a regular list (no conversion should take place)
    list_data = [11, 12, 13, 14, 15]
    no_conversion_list = handle_non_numpy_datatypes(list_data)
    assert no_conversion_list == list_data, "List data should not be converted"

    # Test with a tuple (no conversion should take place)
    tuple_data = (16, 17, 18, 19, 20)
    no_conversion_tuple = handle_non_numpy_datatypes(tuple_data)
    assert no_conversion_tuple == tuple_data, "Tuple data should not be converted"

def test_batch_as_signal_list():
    # Test with a tuple of data and targets
    batch = (
        [Tensor([1, 2, 3]), np.array([4, 5, 6])],
        [{"target": 0}, {"target": 1}]
    )
    # dataset_metadata = DatasetMetadata(num_signals_max=1)
    # target_labels = ["label0", "label1"]

    # signal_list = batch_as_signal_list(batch, target_labels, dataset_metadata)

    # assert len(signal_list) == 2
    # assert isinstance(signal_list[0].data, np.ndarray)
    # assert isinstance(signal_list[1].data, np.ndarray)
    # assert signal_list[0].metadata == targets_as_metadata({"target": 0}, target_labels, dataset_metadata)
    # assert signal_list[1].metadata == targets_as_metadata({"target": 1}, target_labels, dataset_metadata)

    # Test with a numpy array
    batch_array = np.array([[7, 8, 9], [10, 11, 12]])
    signal_list_array = batch_as_signal_list(batch_array)

    assert len(signal_list_array) == 2
    assert isinstance(signal_list_array[0].data, np.ndarray)
    assert isinstance(signal_list_array[1].data, np.ndarray)
    assert signal_list_array[0].data.tolist() == [7, 8, 9]
    assert signal_list_array[1].data.tolist() == [10, 11, 12]

    # Test with a list of Signal objects
    signal_list_input = [
        Signal(data=np.array([13, 14, 15]), metadata=None, component_signals=[]),
        Signal(data=np.array([16, 17, 18]), metadata=None, component_signals=[])
    ]
    signal_list_signals = batch_as_signal_list(signal_list_input)

    assert len(signal_list_signals) == 2
    assert signal_list_signals[0].data.tolist() == [13, 14, 15]
    assert signal_list_signals[1].data.tolist() == [16, 17, 18]

    # Test with an invalid input
    with pytest.raises(ValueError, match="could not parse batch input as signals"):
        batch_as_signal_list("invalid")


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

def test_DatasetCreator_tqdm():
    seed = 1234567890
    dataset_length = 500
    batch_size = 2

    md = DatasetMetadata(
        num_iq_samples_dataset = 4096,
        fft_size = 64,
        sample_rate = 10e6,
        num_signals_max = 1,
        num_signals_min = 1,
    )

    ds = TorchSigIterableDataset(
        dataset_metadata = md,
        seed = seed,
        transforms=[],
        target_labels=None
    )
    dl = WorkerSeedingDataLoader(
        ds, 
        batch_size = batch_size, 
        collate_fn=lambda x: x
    )
    
    # save dataset to disk
    print("Single threaded")
    dc = DatasetCreator(
        dataloader=dl,
        dataset_length = dataset_length,
        root = wb_data_dir,
        overwrite = True,
        multithreading=False   
    )
    dc.create()

    print("Multithreading")
    dc2 = DatasetCreator(
        dataloader=dl,
        dataset_length = dataset_length,
        root = wb_data_dir,
        overwrite = True,
        multithreading=True   
    )
    dc2.create()


if __name__ == "__main__":
    test_DatasetCreator_tqdm()
    # test_DatasetCreator(params={'dataset_length': 10}, is_error=False)
    # transforms = [[YOLOLabel()]]
    # target_labels = [
    #     ["class_name"], # ofdm-300
    #     ["class_index"], # 41
    #     ["class_name", "class_index"], # ('ofdm-300', 41)
    #     ["yolo_label"], # (41, 0.2279052734375, 0.25370860860317335, 0.199951171875, 0.078125)
    #     ["class_name", "yolo_label"] #('ofdm-300', (41, 0.2279052734375, 0.25370860860317335, 0.199951171875, 0.078125))
    # ]
    # num_signals_min = [0]
    # num_signals_max = [1, 3]
    # params = itertools.product(
    #     transforms,
    #     target_labels,
    #     num_signals_min,
    #     num_signals_max
    # )
    # # breakpoint()
    # for p in params:
    #     print(f"{p} -------")
    #     try:
    #         test_DatasetCreator_targets(*p)
    #     except ImportError:
    #         pass
        # breakpoint()
    # test_padding()
