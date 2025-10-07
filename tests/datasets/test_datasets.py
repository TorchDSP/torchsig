"""Unit Tests for datasets
"""
from torchsig.datasets.datasets import TorchSigIterableDataset, StaticTorchSigDataset
from torchsig.utils.writer import DatasetCreator, default_collate_fn
from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.transforms.transforms import Spectrogram, ComplexTo2D
from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import TorchSigRealDataType
from torchsig.utils.file_handlers import FileReader
from torchsig.datasets.dataset_metadata import DatasetMetadata

# Third Party
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import pytest


from typing import List, Any, Tuple, Dict
from collections.abc import Iterable
import itertools
import csv
import math
import json
import pprint
from copy import deepcopy
from pathlib import Path
import os
import random


RTOL = 1E-6

wb_data_dir =  Path.joinpath(Path(__file__).parent,'datasets/dataset_data')
wb_image_dir = Path.joinpath(Path(__file__).parent,'datasets/dataset_images')
getitem_dir = Path.joinpath(Path(__file__).parent,'datasets/getitem_data')

# directory for test data
def setup_module(module):
    if os.path.exists(wb_data_dir):
        shutil.rmtree(wb_data_dir)
    if os.path.exists(wb_image_dir):
        shutil.rmtree(wb_image_dir) 

    os.makedirs(wb_data_dir)
    os.makedirs(wb_image_dir)


test_dataset_getitem_params = list(itertools.product(
    # num_signals_max
    [1, 2, 3],
    # target transforms to test
    [
#        None,
#        [],
        ["class_name"],
        ["yolo_label"],
        ["class_name", "snr_db"],
        ["class_name", "yolo_label"],
        ["class_name", "class_index", "start", "stop", "snr_db"]
    ],
))
num_check = 5
    

def verify_getitem_targets(num_signals_max: int, target_labels: List[str], sample: Any) -> None:
    """ Verfies target labels applied correctly

    Target Labels Table
    
    | Case      | target_labels                  | num_signals_max = 1          | num_signals_max > 1                                               |
    |-----------|--------------------------------|------------------------------|-------------------------------------------------------------------|
    | Case 1    | None                           | nothing, just Signal object  | nothing, just Signal object                                       |
    | Case 2    | []                             | nothing, just returns data   | nothing, just returns data                                        |
    | Case 3    | ["class_name"]                 | '8msk'                       | ['8msk', 'ofdm-600']                                              |
    | Case 4    | ["class_name", "class_index"]  | ('8msk', 0)                  | [('8msk', 0), ('ofdm-600', 1)]                                    |
    | Case 5    | ["class_name", "yolo_label"]   | ('8msk', (idx, x, y, w, h))  | [('8msk', (idx, x, y, w, h)), ('ofdm-600', (idx, x, y, w, h))]    |
    | Case 6    | ["yolo_label"]                 | (idx, x, y, w, h)            | [(idx, x, y, w, h), (idx, x, y, w, h)]                            |

    
    """
    # target_labels are None or []
    # just return data
    if target_labels is None:
        # Case 1
        assert isinstance(sample, Signal)
    elif len(target_labels) == 0:
        # Case 2
        assert isinstance(sample, np.ndarray)
    else:
        # Case 3-6
        # target_labels has at least 1 item
        data, targets = sample
        print(targets)

        if num_signals_max == 1:
            # one signal
            assert isinstance(targets, tuple) or isinstance(targets, list) or isinstance(targets, float) or isinstance(targets, int) or isinstance(targets, str)
        else:
            # sample has more than one signal
            # targets should be a list
            assert isinstance(targets, list)
            for t in targets:
                assert isinstance(targets, tuple) or isinstance(targets, list) or isinstance(targets, float) or isinstance(targets, int) or isinstance(targets, str)
    

@pytest.mark.parametrize("num_signals_max, target_labels", test_dataset_getitem_params)
def test_IterableDataset_getitem(num_signals_max: int, target_labels: List[str],):
    """ Tests targets from target transform are properly returned from dataset's getitem

    >>> pytest test_datasets.py -s

    Args:
        num_signals_max (str): Maximum number of signals.
        target_labels: List[str] (List[TargetTransform]): target labels to test.
    """    
    print(f"\nnum_signals_max={num_signals_max}, target_labels={target_labels}")
    dataset = None
    fft_size = 64

    dm = DatasetMetadata(
        num_iq_samples_dataset=fft_size**2,
        fft_size=fft_size,
        num_signals_max=num_signals_max,
    )
    dataset = TorchSigIterableDataset(
        dataset_metadata=dm,
        transforms = [YOLOLabel()],
        target_labels=target_labels
    )

    for i in range(num_check):
        sample = next(dataset)
        # data, targets = sample.data, [x.to_dict() for x in sample.get_full_metadata()]

        verify_getitem_targets(num_signals_max, target_labels, sample)

@pytest.mark.parametrize("num_signals_max, target_labels", test_dataset_getitem_params)
def test_StaticDataset_getitem(num_signals_max: int, target_labels):
    """ Tests targets from target transform are properly returned from dataset's getitem

    >>> pytest test_datasets.py -s

    Args:
        num_signals_max (int): Maximum number of signals.
        target_labels (List[TargetTransform]): target labels to test.
    """    
    print(f"\nnum_signals_max={num_signals_max}, target_labels={target_labels}")
    if target_labels is None or len(target_labels) == 0:
        # skip
        return
    new_dataset = None
    fft_size = 64
    root = getitem_dir
    num_generate = num_check * 2

    dm = DatasetMetadata(
        num_iq_samples_dataset=fft_size**2,
        fft_size=fft_size,
        num_signals_max=num_signals_max,
        num_signals_min = 1,
    )
    new_dataset = TorchSigIterableDataset(
        dataset_metadata=dm, 
        transforms = [YOLOLabel()],
        target_labels=target_labels
    )
    new_dataloader = WorkerSeedingDataLoader(new_dataset, collate_fn=default_collate_fn)
    dc = DatasetCreator(
        dataloader=new_dataloader,
        root = root,
        overwrite = True,
        dataset_length=num_generate
    )
    
    dc.create()

    static_dataset = StaticTorchSigDataset(
        root = root,
    )

    for i in range(num_check):
        idx = np.random.randint(len(static_dataset))
        sample = static_dataset[idx]

        # verify_getitem_targets(num_signals_max, target_labels, sample)

@pytest.mark.parametrize("params, is_error", [
    (
        {'dataset_length': 10},
        False
    )
])
def test_datasets(params: dict, is_error: bool) -> None:
    """Test datasets with pytest - TorchSigIterableDataset and StaticTorchSigDataset.

    Args:
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    seed = 123456789
    rng = np.random.default_rng(seed)
    
    # signals to simulate
    class_list = TorchSigSignalLists.all_signals

    # distribution of classes
    class_dist = np.ones(len(class_list))/len(class_list)    
    
    # number of samples to test generation
    dataset_length = params["dataset_length"]
    save_num_signals = 5
    num_signals_min = 1 # always generate a signal
    num_signals_max = 1

    # FFT/spectrogram params
    fft_size = rng.integers(128,1024, dtype=int)
    #num_iq_samples_dataset = fft_size * rng.integers(128,1024,dtype=int)
    num_iq_samples_dataset = fft_size**2

    # testing to handle cases in which number of samples is not an integer multiple of FFT size
    num_iq_samples_dataset = int(num_iq_samples_dataset + rng.integers(0,fft_size,dtype=int))

    # works for variable sample rates
    sample_rate = rng.uniform(100e6,200e6)

    # minimum and maximum SNR for signals
    snr_db_max = 50
    snr_db_min = 0

    # probability for each sample to contain N signals where N is the index,
    # for example, num_signals_dist = [0.15, 0.5, 0.35] is 25% probability to 
    # generate 0 signals, 50% probability to generate 1 signal, 35% 
    # probability to generate 2 signals
    num_signals_dist = np.ones(num_signals_max - num_signals_min+1)/(num_signals_max-num_signals_min+1)

    # define transforms
    transforms = [Spectrogram(fft_size=fft_size)]
    
    # build the dataset metadata
    md = DatasetMetadata(
        num_iq_samples_dataset=num_iq_samples_dataset,
        sample_rate=sample_rate,
        fft_size=fft_size,
        dataset_length=dataset_length,
        num_signals_max=num_signals_max,
        num_signals_min=num_signals_min,
        num_signals_distribution=num_signals_dist,
        snr_db_max=snr_db_max,
        snr_db_min=snr_db_min,
        class_list=class_list,
        class_distribution=class_dist,
    )

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            DS = TorchSigIterableDataset(dataset_metadata=md, target_labels=["class_index"], transforms=transforms)
            DL = WorkerSeedingDataLoader(DS, collate_fn=default_collate_fn)
            DL.seed(seed)
            dc = DatasetCreator(
                dataloader=DL,
                root = wb_data_dir,
                dataset_length=dataset_length,
                overwrite = True
            )
            dc.create()
            SDS = StaticTorchSigDataset(
                root = wb_data_dir,
            )
    else:
        # create the dataset object, derived from the metadata object
        DS0 = TorchSigIterableDataset(dataset_metadata=deepcopy(md), target_labels=None, seed=seed, transforms=transforms)
        DS1 = TorchSigIterableDataset(dataset_metadata=deepcopy(md), target_labels=None, seed=seed, transforms=transforms) # reproducible copy

        # save dataset to disk
        DL0 = WorkerSeedingDataLoader(DS0, collate_fn=lambda x: x)
        DL0.seed(seed)
        dc = DatasetCreator(
            dataloader=DL0,
            root = wb_data_dir,
            dataset_length=dataset_length,
            overwrite = True
        )
        dc.create()

        # load dataset from disk
        SDS0 = StaticTorchSigDataset(
            root = wb_data_dir,
            target_labels=["class_index"]
        )
        SDS1 = StaticTorchSigDataset(
            root = wb_data_dir,
            target_labels=["class_index"]
        )
            
        # dataset
        assert isinstance(DS0, TorchSigIterableDataset)

        # static dataset
        assert isinstance(SDS0, StaticTorchSigDataset)
        assert len(SDS0) == dataset_length
        for i in range(dataset_length):
            data0, meta0 = SDS0[i]
            data1, meta1 = SDS1[i] # reproducible copy
            
            assert type(data0) == np.ndarray
            assert data0.dtype == TorchSigRealDataType
            assert meta0 == meta1
            assert np.allclose(data0, data1, RTOL)
        
