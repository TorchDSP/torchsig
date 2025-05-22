"""Unit Tests for narrowband and wideband datasets
"""
from torchsig.datasets.dataset_metadata import WidebandMetadata
from torchsig.datasets.wideband import NewWideband, StaticWideband
from torchsig.utils.writer import DatasetCreator
from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.transforms.target_transforms import (
    TargetTransform,
    ClassName,
    Start,
    Stop,
    LowerFreq,
    UpperFreq,
    SNR,
    YOLOLabel,
    FamilyName,
)
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.dsp import (
    torchsig_complex_data_type,
    torchsig_real_data_type
)

# Third Party
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
import os
import shutil
import pytest

from typing import List, Any
from collections.abc import Iterable
import itertools


RTOL = 1E-6

nb_data_dir =  Path.joinpath(Path(__file__).parent,'data/narrowband_data')
nb_image_dir = Path.joinpath(Path(__file__).parent,'data/narrowband_images')
wb_data_dir =  Path.joinpath(Path(__file__).parent,'data/wideband_data')
wb_image_dir = Path.joinpath(Path(__file__).parent,'data/wideband_images')
getitem_dir = Path.joinpath(Path(__file__).parent,'data/getitem_data')

# directory for test data
def setup_module(module):
    if os.path.exists(nb_data_dir):
        shutil.rmtree(nb_data_dir)
    if os.path.exists(nb_image_dir):
        shutil.rmtree(nb_image_dir) 
    if os.path.exists(wb_data_dir):
        shutil.rmtree(wb_data_dir)
    if os.path.exists(wb_image_dir):
        shutil.rmtree(wb_image_dir) 

    os.makedirs(nb_data_dir)
    os.makedirs(nb_image_dir)
    os.makedirs(wb_data_dir)
    os.makedirs(wb_image_dir)


test_dataset_getitem_params = list(itertools.product(
    # datasets
    ["narrowband", "wideband"],
    # target transforms to test
    [
        [],
        [ClassName()],
        [YOLOLabel()],
        [ClassName(), SNR()],
        [ClassName(), YOLOLabel()],
        [ClassName(), FamilyName(), Start(), Stop(), SNR()]
    ],
    # impairement level
    [0, 2]
))
num_check = 5


def verify_getitem_targets(dataset_type: str, target_transforms: List[TargetTransform], targets: Any) -> None:
    # no TT -> list of dicts
    if len(target_transforms) == 0:
        required_keys = [
            "center_freq",
            "bandwidth",
            "start_in_samples",
            "duration_in_samples",
            "snr_db",
            "class_name",
            "class_index",
            "sample_rate",
            "num_samples",
            "start",
            "stop",
            "duration",
            "stop_in_samples",
            "upper_freq",
            "lower_freq",
            "oversampling_rate"
        ]
        assert isinstance(targets, list)
        for t in targets:
            assert isinstance(t, dict)
            assert set(required_keys) == set(t.keys())

    # 1 TT
    # narrowband: single item
    #   signal 1 output
    # wideband: list of single items 
    # [
    #   signal 1 output, 
    #   signal 2 output
    # ]
    if len(target_transforms) == 1:
        if dataset_type == 'narrowband':
            assert not isinstance(targets, dict)
            assert not isinstance(targets, list)
            if isinstance(targets, tuple):
                for item in targets:
                    assert isinstance(item, str) or not isinstance(item, Iterable)
        else:
            assert isinstance(targets, list)
            for t in targets:
                # can be a signle tuple
                if isinstance(t, tuple):
                    for item in t:
                        # should not be nested tuple
                        assert isinstance(t, str) or not isinstance(item, Iterable)
                else:
                    # primitive or some other datatype
                    # should not be iterable/nested lists
                    assert isinstance(t, str) or not isinstance(t, Iterable)

    # 2+ TT
    # narrowband: a (sorta) single tuple
    #   (signal 1 output)
    #   (class name, (x, y, w, h))
    # wideband: list of tuples 
    # [
    #   (signal 1 output),
    #   (signal 2 output), 
    #   ...
    # ]
    # [ 
    #   (class name 1, (x1, y1, w1, h1)),
    #   (class name 2, (x2, y2, w2, h2)),
    #   ...
    # ]
    if len(target_transforms) > 1:
        if dataset_type == 'narrowband':
            assert isinstance(targets, tuple)
            for item in targets:
                if isinstance(item, tuple):
                    for i in item:
                        # should not be nested tuple
                        assert isinstance(i, str) or not isinstance(i, Iterable)
                else:
                    # primitive or some other datatype
                    # should not be iterable/nested lists
                    assert isinstance(item, str) or not isinstance(item, Iterable)
        else:
            assert isinstance(targets, list)
            for t in targets:
                assert isinstance(t, tuple)
                for item in t:
                    if isinstance(item, tuple):
                        for i in item:
                            # should not be nested tuple
                            assert isinstance(i, str) or not isinstance(i, Iterable)
                    else:
                        # primitive or some other datatype
                        # should not be iterable/nested lists
                        assert isinstance(item, str) or not isinstance(item, Iterable)

# @pytest.mark.skip(reason="ere")
@pytest.mark.parametrize("dataset_type, target_transforms, impairment_level", test_dataset_getitem_params)
def test_NewDataset_getitem(dataset_type: str, target_transforms: List[TargetTransform], impairment_level: int):
    """ Tests targets from target transform are properly returned from dataset's getitem

    >>> pytest test_datasets.py -s

    Args:
        dataset_type (str): Dataset to test. Either "narrowband" or "wideband".
        target_transforms (List[TargetTransform]): target transforms to test.
    """    
    print(f"\n{dataset_type}, {target_transforms}, level {impairment_level}")
    dataset = None
    fft_size = 64

    if dataset_type == 'wideband':
        dm = WidebandMetadata(
            num_iq_samples_dataset=fft_size**2,
            fft_size=fft_size,
            impairment_level=impairment_level,
            num_signals_max=3,
            target_transforms=target_transforms
        )
        dataset = NewWideband(dataset_metadata=dm)
    else:
        raise ValueError('invalid dataset type')

    for i in range(num_check):
        data, targets = dataset[i]

        verify_getitem_targets(dataset_type, target_transforms, targets)

# @pytest.mark.skip(reason="ere")
@pytest.mark.parametrize("dataset_type, target_transforms, impairment_level", test_dataset_getitem_params)
def test_StaticDataset_getitem(dataset_type: str, target_transforms: List[TargetTransform], impairment_level: int):
    """ Tests targets from target transform are properly returned from dataset's getitem

    >>> pytest test_datasets.py -s

    Args:
        dataset_type (str): Dataset to test. Either "narrowband" or "wideband".
        target_transforms (List[TargetTransform]): target transforms to test.
    """    
    print(f"\n{dataset_type}, {target_transforms}, level {impairment_level}")
    new_dataset = None
    fft_size = 64
    root = getitem_dir
    num_generate = num_check * 2

    if dataset_type == 'wideband':
        dm = WidebandMetadata(
            num_iq_samples_dataset=fft_size**2,
            fft_size=fft_size,
            impairment_level=impairment_level,
            num_signals_max=3,
            num_samples=num_generate
        )
        new_dataset = NewWideband(dataset_metadata=dm)
    else:
        raise ValueError('unknown dataset type.')

    dc = DatasetCreator(
        new_dataset,
        root = root,
        overwrite = True
    )
    
    dc.create()

    static_dataset = None

    if dataset_type == 'wideband':
        static_dataset = StaticWideband(
            root = root,
            impairment_level = impairment_level,
            target_transforms=target_transforms,
        )
    else:
        raise ValueError('unsupported dataset type.')

    for i in range(num_check):
        idx = np.random.randint(len(static_dataset))
        data, targets = static_dataset[idx]

        verify_getitem_targets(dataset_type, target_transforms, targets)
    


# @pytest.mark.skip(reason="ere")
@pytest.mark.parametrize("params, is_error", [
    (
        {'num_samples': 10, 'impairment_level': 2},
        False
    )
])
def test_WidebandDatasets(params: dict, is_error: bool) -> None:
    """Test Wideband datasets with pytest - NewWideband and StaticWideband.

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
    num_samples = params["num_samples"]
    save_num_signals = 5
    num_signals_min = 1 # always generate a signal
    num_signals_max = 1
    
    # define impairment level
    impairment_level = params["impairment_level"]

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
    target_transform = [
        ClassName(),
        Start(),
        Stop(),
        LowerFreq(),
        UpperFreq(),
        SNR()
    ]
    
    # build the wideband metadata
    md = WidebandMetadata(
        num_iq_samples_dataset=num_iq_samples_dataset,
        sample_rate=sample_rate,
        fft_size=fft_size,
        num_samples=num_samples,
        num_signals_max=num_signals_max,
        num_signals_min=num_signals_min,
        num_signals_distribution=num_signals_dist,
        snr_db_max=snr_db_max,
        snr_db_min=snr_db_min,
        transforms=transforms,
        target_transforms=target_transform,
        impairment_level=impairment_level,
        class_list=class_list,
        class_distribution=class_dist,
    )

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            WB = NewWideband(dataset_metadata=md)
            dc = DatasetCreator(
                WB,
                root = wb_data_dir,
                overwrite = True
            )
            dc.create()
            WBS = StaticWideband(
                root = wb_data_dir,
                impairment_level = impairment_level,
            )
    else:
        # create the wideband object, derived from the metadata object
        WB0 = NewWideband(dataset_metadata=deepcopy(md), seed=seed)
        WB1 = NewWideband(dataset_metadata=deepcopy(md), seed=seed) # reproducible copy

        # save dataset to disk
        dc = DatasetCreator(
            WB0,
            root = wb_data_dir,
            overwrite = True
        )
        dc.create()

        # load dataset from disk
        WBS0 = StaticWideband(
            root = wb_data_dir,
            impairment_level = impairment_level,
        )
        WBS1 = StaticWideband(
            root = wb_data_dir,
            impairment_level = impairment_level,
        )
            
        # wideband dataset
        assert isinstance(WB0, NewWideband)
        assert len(WB0) == num_samples
        for i in range(num_samples):
            data0, meta0 = WB0[i]
            data1, meta1 = WB1[i] # reproducible copy
            
            assert type(data0) == np.ndarray
            assert data0.dtype == torchsig_real_data_type
            assert type(meta0) == list
            assert meta0 == meta1
            assert np.allclose(data0, data1, RTOL)

        # static wideband dataset
        assert isinstance(WBS0, StaticWideband)
        assert len(WBS0) == num_samples
        for i in range(num_samples):
            data0, meta0 = WBS0[i]
            data1, meta1 = WBS1[i] # reproducible copy
            
            assert type(data0) == np.ndarray
            assert data0.dtype == torchsig_real_data_type
            assert type(meta0) == list
            assert meta0 == meta1
            assert np.allclose(data0, data1, RTOL)
        
