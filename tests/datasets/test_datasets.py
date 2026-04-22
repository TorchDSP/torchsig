"""Unit Tests for datasets"""

from torchsig.datasets.datasets import TorchSigIterableDataset, StaticTorchSigDataset
from torchsig.utils.writer import DatasetCreator, default_collate_fn
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.transforms.transforms import Spectrogram
from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import TorchSigRealDataType

from torchsig.utils.defaults import TorchSigDefaults
from torchsig.transforms.impairments import Impairments

# Third Party
import numpy as np
from typing import Any, List
import itertools
from copy import deepcopy
import yaml
import pytest


test_dataset_getitem_params = list(
    itertools.product(
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
            ["class_name", "class_index", "start", "stop", "snr_db"],
        ],
        # num_workers
        [0, 2]
    )
)
num_check = 5


def verify_getitem_targets(num_signals_max: int, target_labels: List[str], sample: Any) -> None:
    """Verfies target labels applied correctly

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

def test_IterableDataset_transforms():
    seed = 83843293432
    impairments = Impairments(level=2)
    burst_impairments = impairments.signal_transforms
    whole_signal_impairments = impairments.dataset_transforms

    md = TorchSigDefaults().default_dataset_metadata
    md["fft_size"] = 64
    md["fft_stride"] = 64
    md["num_iq_samples_dataset"] = 64**2

    dataset_unimpaired = TorchSigIterableDataset(
        metadata=md,
        transforms=[
            # whole_signal_impairments,
            Spectrogram(fft_size=md["fft_size"]),
        ],
        target_labels=[],
    )
    dataset_whole_impaired = TorchSigIterableDataset(
        metadata=md,
        transforms=[
            whole_signal_impairments,
            Spectrogram(fft_size=md["fft_size"]),
        ],
        target_labels=[],
    )
    dataset_component_impaired = TorchSigIterableDataset(
        metadata=md,
        transforms=[
            Spectrogram(fft_size=md["fft_size"]),
        ],
        component_transforms=[burst_impairments],
        target_labels=[],
    )
    dataset_impaired = TorchSigIterableDataset(
        metadata=md,
        transforms=[
            whole_signal_impairments,
            Spectrogram(fft_size=md["fft_size"]),
        ],
        component_transforms=[burst_impairments],
        target_labels=[],
    )

    datasets = [
        dataset_whole_impaired,
        dataset_unimpaired,
        dataset_component_impaired,
        dataset_impaired
    ]

    for d in datasets:
        d.seed(seed)

    # check they are all different
    datas = [next(d) for d in datasets]

    for i, j in itertools.combinations(range(len(datas)), 2):
        if np.array_equal(datas[i], datas[j]):
            raise AssertionError(f"Datasets {i} and {j} are identical")




@pytest.mark.parametrize("num_signals_max, target_labels, num_workers", test_dataset_getitem_params)
def test_IterableDataset_getitem(
    num_signals_max: int,
    target_labels: List[str],
    num_workers: int
):
    """Tests targets from target transform are properly returned from dataset's getitem

    >>> pytest test_datasets.py -s

    Args:
        num_signals_max (str): Maximum number of signals.
        target_labels: List[str] (List[TargetTransform]): target labels to test.
        num_workers (int): Number of worker processes for the dataloader.
    """
    print(f"\nnum_signals_max={num_signals_max}, target_labels={target_labels}, num_workers={num_workers}")
    dm = TorchSigDefaults().default_dataset_metadata
    dataset = TorchSigIterableDataset(metadata=dm, transforms=[YOLOLabel()])

    for _ in range(num_check):
        sample = next(dataset)
        # data, targets = sample.data, [x.to_dict() for x in sample.get_full_metadata()]

        verify_getitem_targets(num_signals_max, None, sample)


@pytest.mark.parametrize("num_signals_max, target_labels, num_workers", test_dataset_getitem_params)
def test_StaticDataset_getitem(tmp_path, num_signals_max: int, target_labels: List[str], num_workers: int):
    """Tests targets from target transform are properly returned from dataset's getitem

    >>> pytest test_datasets.py -s

    Args:
        num_signals_max (int): Maximum number of signals.
        target_labels (List[TargetTransform]): target labels to test.
        num_workers (int): Number of worker processes for the dataloader.
    """
    print(f"\nnum_signals_max={num_signals_max}, target_labels={target_labels}, num_workers={num_workers}")
    if target_labels is None or len(target_labels) == 0:
        # skip
        return
    root = tmp_path / "run0"
    num_generate = num_check * 2

    dm = TorchSigDefaults().default_dataset_metadata
    new_dataset = TorchSigIterableDataset(metadata=dm, transforms=[YOLOLabel()])
    new_dataloader = WorkerSeedingDataLoader(new_dataset, num_workers=num_workers)
    dc = DatasetCreator(dataloader=new_dataloader, root=root, overwrite=True, dataset_length=num_generate)

    dc.create()

    static_dataset = StaticTorchSigDataset(root=root)

    for i in range(num_check):
        idx = np.random.randint(len(static_dataset))
        sample = static_dataset[idx]

        # verify_getitem_targets(num_signals_max, target_labels, sample)


@pytest.mark.parametrize("params, is_error", 
                        [({"dataset_length": 10, "num_workers": 0}, False),
                         ({"dataset_length": 10, "num_workers": 2}, False)]
    )
def test_datasets(tmp_path, params: dict, is_error: bool) -> None:
    """Test datasets with pytest - TorchSigIterableDataset and StaticTorchSigDataset.

    Args:
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    root0 = tmp_path / "run0"

    seed = 123456789
    rng = np.random.default_rng(seed)
    dataset_length = params["dataset_length"]
    num_workers = params["num_workers"]
    fft_size = rng.integers(128, 1024, dtype=int)
    transforms = [Spectrogram(fft_size=fft_size)]

    md = TorchSigDefaults().default_dataset_metadata

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            DS = TorchSigIterableDataset(metadata=md, target_labels=["class_index"], transforms=transforms)
            DL = WorkerSeedingDataLoader(DS, num_workers=num_workers, collate_fn=default_collate_fn)
            DL.seed(seed)
            dc = DatasetCreator(dataloader=DL, root=root0, dataset_length=dataset_length, overwrite=True)
            dc.create()
            SDS = StaticTorchSigDataset(
                root=root0,
            )
    else:
        # create the dataset object, derived from the metadata object
        DS0 = TorchSigIterableDataset(metadata=deepcopy(md), target_labels=None, seed=seed, transforms=transforms)

        # save dataset to disk
        DL0 = WorkerSeedingDataLoader(DS0, num_workers=num_workers, collate_fn=lambda x: x)
        DL0.seed(seed)
        dc = DatasetCreator(dataloader=DL0, root=root0, dataset_length=dataset_length, overwrite=True)
        dc.create()

        # load dataset from disk
        SDS0 = StaticTorchSigDataset(root=root0, target_labels=["class_index"])
        SDS1 = StaticTorchSigDataset(root=root0, target_labels=["class_index"])

        # dataset
        assert isinstance(DS0, TorchSigIterableDataset)

        # static dataset
        assert isinstance(SDS0, StaticTorchSigDataset)
        assert len(SDS0) == dataset_length
        for i in range(dataset_length):
            data0, meta0 = SDS0[i]
            data1, meta1 = SDS1[i]  # reproducible copy

            assert type(data0) == np.ndarray
            assert data0.dtype == TorchSigRealDataType
            assert meta0 == meta1
            assert np.allclose(data0, data1, 1e-6)

        ds_yaml = yaml.safe_load(open(root0 / "dataset_info.yaml", "r")) or {}
        assert ds_yaml["dataset_length"] == dataset_length
