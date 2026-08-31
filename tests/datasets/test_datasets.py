"""Unit Tests for datasets"""

import itertools
import logging
import warnings
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import yaml

from torchsig.datasets.datasets import (
    SafeTorchSigIterableDataset,
    StaticTorchSigDataset,
    TorchSigIterableDataset,
    apply_label_to_signal,
    apply_transforms_and_labels_to_signal,
)
from torchsig.signals.builder import BaseSignalGenerator
from torchsig.signals.signal_types import Signal
from torchsig.transforms.metadata_transforms import MultiHotLabel, YOLOLabel
from torchsig.transforms.transforms import ComplexTo2D, Spectrogram
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.dsp import TorchSigRealDataType
from torchsig.utils.metadata_logging import (
    MetadataLoggingContext,
    get_metadata_logging_context,
    metadata_logging_context,
)
from torchsig.utils.writer import DatasetCreator

# =============================================================================
# Helpers
# =============================================================================


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
        [0, 2],
    )
)
num_check = 5


class DummyGenerator(BaseSignalGenerator):
    def __call__(self):
        return Signal(class_name=self.class_name)


class ValidatingGenerator(BaseSignalGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.validate_metadata_fields = MagicMock()

    def __call__(self):
        return Signal()


class NonValidatingGenerator(BaseSignalGenerator):
    def __call__(self):
        return Signal()


class MetadataSignalGenerator(BaseSignalGenerator):
    """Generate a signal that inherits metadata from this generator."""

    def __init__(self, class_name, **kwargs):
        super().__init__(**kwargs)
        self.set_default_class_name(class_name)

    def generate(self):
        return Signal(data=np.ones(8, dtype=np.complex64))


def _dataset_with_empty_generators():
    return TorchSigIterableDataset(
        metadata=TorchSigDefaults().default_dataset_metadata.copy(),
        signal_generators=[],
        validate_init=False,
    )


def _safe_dataset():
    return SafeTorchSigIterableDataset(
        metadata=TorchSigDefaults().default_dataset_metadata.copy(),
        signal_generators=[],
        validate_init=False,
    )


def _small_metadata():
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": 16,
            "fft_size": 4,
            "fft_stride": 4,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "sample_rate": 16,
            "frequency_min": -8,
            "frequency_max": 8,
            "signal_center_freq_min": -4,
            "signal_center_freq_max": 4,
            "bandwidth_min": 1,
            "bandwidth_max": 4,
        }
    )
    return metadata


def test_per_signal_metadata_overrides_dataset_ranges():
    metadata = _small_metadata()
    metadata.update(
        {
            "snr_db_min": -5,
            "snr_db_max": 40,
            "signal_duration_in_samples_min": 2,
            "signal_duration_in_samples_max": 16,
            "bandwidth_min": 1,
            "bandwidth_max": 8,
        }
    )
    qpsk = MetadataSignalGenerator("qpsk")
    am_dsb_sc = MetadataSignalGenerator("am-dsb-sc")
    dataset = TorchSigIterableDataset(
        metadata=metadata,
        signal_generators=[qpsk, am_dsb_sc],
        per_signal_metadata={
            "qpsk": {
                "snr_db_min": 0,
                "snr_db_max": 10,
                "signal_duration_in_samples_min": 4,
                "signal_duration_in_samples_max": 6,
                "bandwidth_min": 2,
                "bandwidth_max": 3,
            },
            "am-dsb-sc": {"snr_db_min": 15, "snr_db_max": 30},
        },
        validate_init=False,
    )

    qpsk_signal = dataset.signal_generators[0]()
    am_signal = dataset.signal_generators[1]()

    assert (qpsk_signal.snr_db_min, qpsk_signal.snr_db_max) == (0, 10)
    assert (
        qpsk_signal.signal_duration_in_samples_min,
        qpsk_signal.signal_duration_in_samples_max,
    ) == (4, 6)
    assert (qpsk_signal.bandwidth_min, qpsk_signal.bandwidth_max) == (2, 3)
    assert (am_signal.snr_db_min, am_signal.snr_db_max) == (15, 30)
    assert (
        am_signal.signal_duration_in_samples_min,
        am_signal.signal_duration_in_samples_max,
    ) == (2, 16)
    assert (am_signal.bandwidth_min, am_signal.bandwidth_max) == (1, 8)


def test_per_signal_metadata_supports_string_generator_configuration():
    dataset = TorchSigIterableDataset(
        metadata=_small_metadata(),
        signal_generators=["qpsk", "am-dsb-sc"],
        per_signal_metadata={
            "qpsk": {"snr_db_min": 0, "snr_db_max": 10},
            "am-dsb-sc": {"snr_db_min": 15, "snr_db_max": 30},
        },
        validate_init=False,
    )

    generators = {generator.class_name: generator for generator in dataset.signal_generators}
    assert (generators["qpsk"].snr_db_min, generators["qpsk"].snr_db_max) == (
        0,
        10,
    )
    assert (
        generators["am-dsb-sc"].snr_db_min,
        generators["am-dsb-sc"].snr_db_max,
    ) == (15, 30)


@pytest.mark.parametrize(
    ("per_signal_metadata", "error", "message"),
    [
        ([], TypeError, "must be a dictionary"),
        ({"qpsk": []}, TypeError, "must be a dictionary"),
        ({"qpsk": {"center_freq": 1}}, ValueError, "unsupported"),
        ({"qpsk": {"bandwidth_min": 0}}, ValueError, "must be positive"),
        ({"qpsk": {"snr_db_min": np.nan}}, ValueError, "must be finite"),
    ],
)
def test_per_signal_metadata_rejects_invalid_configuration(per_signal_metadata, error, message):
    with pytest.raises(error, match=message):
        TorchSigIterableDataset(
            metadata=_small_metadata(),
            signal_generators=[MetadataSignalGenerator("qpsk")],
            per_signal_metadata=per_signal_metadata,
            validate_init=False,
        )


def test_per_signal_metadata_rejects_unknown_class_and_reversed_range():
    with pytest.raises(ValueError, match="classes not configured"):
        TorchSigIterableDataset(
            metadata=_small_metadata(),
            signal_generators=[MetadataSignalGenerator("qpsk")],
            per_signal_metadata={"am-dsb-sc": {"snr_db_min": 15}},
            validate_init=False,
        )

    with pytest.raises(ValueError, match="snr_db_min must be less than"):
        TorchSigIterableDataset(
            metadata=_small_metadata(),
            signal_generators=[MetadataSignalGenerator("qpsk")],
            per_signal_metadata={"qpsk": {"snr_db_min": 20, "snr_db_max": 10}},
            validate_init=False,
        )


def _parent_with_components(num_signals_max=2):
    parent = Signal(
        data=np.zeros(100, dtype=np.complex64),
        component_signals=[
            Signal(
                data=np.zeros(20, dtype=np.complex64),
                class_name="bpsk",
                class_index=3,
                start_in_samples=10,
                center_freq=100.0,
                bandwidth=40.0,
                _lower_frequency=80.0,
                _upper_frequency=120.0,
            ),
            Signal(
                data=np.zeros(30, dtype=np.complex64),
                class_name="qpsk",
                class_index=4,
                start_in_samples=50,
                center_freq=-200.0,
                bandwidth=60.0,
                _lower_frequency=-230.0,
                _upper_frequency=-170.0,
            ),
        ],
        num_iq_samples_dataset=100,
        num_signals_max=num_signals_max,
        class_names=np.array(["ook", "fm", "am", "bpsk", "qpsk"]),
    )

    for component in parent.component_signals:
        component.add_parent(parent, register=False)

    return parent


# =============================================================================
# Tests
# =============================================================================


def verify_getitem_targets(num_signals_max: int, target_labels: list[str], sample: Any) -> None:
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


def test_iterable_dataset_applies_component_and_dataset_transforms(monkeypatch):
    """Verify component transforms run before dataset transforms."""
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": 16,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "signal_duration_in_samples_min": 16,
            "signal_duration_in_samples_max": 16,
            "sample_rate": 16,
            "frequency_min": -8,
            "frequency_max": 8,
            "signal_center_freq_min": 0,
            "signal_center_freq_max": 0,
            "bandwidth_min": 1,
            "bandwidth_max": 1,
        }
    )

    class FixedGenerator(BaseSignalGenerator):
        def __call__(self) -> Signal:
            return Signal(
                data=np.ones(16, dtype=np.complex64),
                center_freq=0,
                bandwidth=1,
                class_name="dummy",
                class_index=0,
                snr_db_min=0,
                snr_db_max=0,
            )

    transform_order: list[str] = []
    dataset_transform_inputs: list[np.ndarray] = []

    def multiply_by_two(signal: Signal) -> Signal:
        transform_order.append("component")
        signal.data = signal.data * 2
        return signal

    def add_three(signal: Signal) -> Signal:
        transform_order.append("dataset")
        dataset_transform_inputs.append(signal.data.copy())
        signal.data = signal.data + 3
        return signal

    dataset = TorchSigIterableDataset(
        metadata=metadata,
        signal_generators=[FixedGenerator()],
        component_transforms=[multiply_by_two],
        transforms=[add_three],
        target_labels=None,
        seed=123,
        validate_init=False,
    )

    monkeypatch.setattr(
        dataset,
        "_build_noise_floor",
        lambda: np.zeros(16, dtype=np.complex64),
    )
    monkeypatch.setattr(
        dataset,
        "_choose_start_sample",
        lambda iq_samples, signal: 0,
    )

    sample = next(dataset)

    assert transform_order == ["component", "dataset"]
    assert len(dataset_transform_inputs) == 1

    np.testing.assert_array_equal(
        sample.data,
        dataset_transform_inputs[0] + 3,
    )


@pytest.mark.parametrize("num_signals_max", [1, 3])
def test_iterable_dataset_applies_yolo_label(
    num_signals_max: int,
) -> None:
    """Verify YOLOLabel is applied to generated component signals."""
    fft_size = 16
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": fft_size**2,
            "fft_size": fft_size,
            "fft_stride": fft_size,
            "num_signals_min": 1,
            "num_signals_max": num_signals_max,
            "signal_duration_in_samples_min": fft_size,
            "signal_duration_in_samples_max": fft_size**2,
        }
    )

    dataset = TorchSigIterableDataset(
        metadata=metadata,
        transforms=[
            Spectrogram(fft_size=fft_size),
            YOLOLabel(),
        ],
        target_labels=None,
        seed=123,
    )

    sample = next(dataset)

    assert isinstance(sample, Signal)
    assert isinstance(sample.data, np.ndarray)
    assert 1 <= len(sample.component_signals) <= num_signals_max

    for component in sample.component_signals:
        assert "yolo_label" in component.metadata


def test_iterable_dataset_applies_component_transforms():
    """Verify component transforms are applied during sample generation."""
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": 16,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "signal_duration_in_samples_min": 16,
            "signal_duration_in_samples_max": 16,
            "sample_rate": 16,
            "frequency_min": -8,
            "frequency_max": 8,
            "signal_center_freq_min": 0,
            "signal_center_freq_max": 0,
            "bandwidth_min": 1,
            "bandwidth_max": 1,
        }
    )

    class FixedGenerator(BaseSignalGenerator):
        def __call__(self) -> Signal:
            return Signal(
                data=np.ones(16, dtype=np.complex64),
                center_freq=0,
                bandwidth=1,
                class_name="dummy",
                class_index=0,
                snr_db_min=0,
                snr_db_max=0,
            )

    transform_inputs: list[np.ndarray] = []
    transform_outputs: list[np.ndarray] = []

    def multiply_by_two(signal: Signal) -> Signal:
        transform_inputs.append(signal.data.copy())

        transformed = signal.copy()
        transformed.data = signal.data * 2

        transform_outputs.append(transformed.data.copy())
        return transformed

    dataset = TorchSigIterableDataset(
        metadata=metadata,
        signal_generators=[FixedGenerator()],
        component_transforms=[multiply_by_two],
        target_labels=None,
        seed=123,
        validate_init=False,
    )

    next(dataset)

    assert len(transform_inputs) == 1
    assert len(transform_outputs) == 1
    np.testing.assert_allclose(
        transform_outputs[0],
        transform_inputs[0] * 2,
    )


@pytest.mark.parametrize(
    "num_signals_max,target_labels",
    [
        (1, ["class_index"]),
        (2, ["class_name", "class_index"]),
    ],
)
def test_static_dataset_getitem_returns_requested_targets(
    tmp_path,
    num_signals_max: int,
    target_labels: list[str],
) -> None:
    """Verify that static dataset samples return the requested target labels."""
    seed = 123
    dataset_length = 2
    fft_size = 16
    root = tmp_path / "static_dataset"

    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": fft_size**2,
            "fft_size": fft_size,
            "fft_stride": fft_size,
            "num_signals_min": 1,
            "num_signals_max": num_signals_max,
            "signal_duration_in_samples_min": fft_size,
            "signal_duration_in_samples_max": fft_size**2,
        }
    )

    iterable_dataset = TorchSigIterableDataset(
        metadata=metadata,
        target_labels=None,
        seed=seed,
    )

    dataloader = WorkerSeedingDataLoader(
        iterable_dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )
    dataloader.seed(seed)

    DatasetCreator(
        dataloader=dataloader,
        root=root,
        overwrite=True,
        dataset_length=dataset_length,
    ).create()

    static_dataset = StaticTorchSigDataset(
        root=root,
        target_labels=target_labels,
    )

    assert len(static_dataset) == dataset_length

    for index in range(dataset_length):
        sample = static_dataset[index]
        verify_getitem_targets(num_signals_max, target_labels, sample)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_dataset_creation_and_static_loading(tmp_path, num_workers: int) -> None:
    """Create a small dataset, reload it, and verify stored samples."""
    seed = 123456789
    dataset_length = 2
    fft_size = 16
    root = tmp_path / f"workers_{num_workers}"

    metadata = deepcopy(TorchSigDefaults().default_dataset_metadata)
    metadata.update(
        {
            "num_iq_samples_dataset": fft_size**2,
            "fft_size": fft_size,
            "fft_stride": fft_size,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "signal_duration_in_samples_min": fft_size,
            "signal_duration_in_samples_max": fft_size**2,
        }
    )

    iterable_dataset = TorchSigIterableDataset(
        metadata=metadata,
        target_labels=None,
        seed=seed,
        transforms=[Spectrogram(fft_size=fft_size)],
    )

    dataloader = WorkerSeedingDataLoader(
        iterable_dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=lambda batch: batch,
    )
    dataloader.seed(seed)

    DatasetCreator(
        dataloader=dataloader,
        root=root,
        dataset_length=dataset_length,
        overwrite=True,
    ).create()

    static_dataset = StaticTorchSigDataset(
        root=root,
        target_labels=["class_index"],
    )
    duplicate_reader = StaticTorchSigDataset(
        root=root,
        target_labels=["class_index"],
    )

    assert len(static_dataset) == dataset_length

    for index in range(dataset_length):
        data, target = static_dataset[index]
        duplicate_data, duplicate_target = duplicate_reader[index]

        assert isinstance(data, np.ndarray)
        assert data.dtype == TorchSigRealDataType
        assert target == duplicate_target
        np.testing.assert_allclose(data, duplicate_data, rtol=1e-6, atol=1e-6)

    with (root / "dataset_info.yaml").open(encoding="utf-8") as file:
        dataset_info = yaml.safe_load(file) or {}

    assert dataset_info["dataset_length"] == dataset_length


# =============================================================================
# apply_label_to_signal / apply_transforms_and_labels_to_signal
# =============================================================================


def test_apply_label_to_signal_uses_signal_properties():
    sample = _parent_with_components()

    assert apply_label_to_signal(sample, "class_name") == ["bpsk", "qpsk"]
    assert apply_label_to_signal(sample, "class_index") == [3, 4]
    assert apply_label_to_signal(sample, "start") == [0.1, 0.5]
    assert apply_label_to_signal(sample, "stop") == [0.3, 0.8]
    assert apply_label_to_signal(sample, "lower_freq") == [80.0, -230.0]
    assert apply_label_to_signal(sample, "upper_freq") == [120.0, -170.0]


def test_apply_label_to_signal_does_not_duplicate_parent_class_index():
    sample = _parent_with_components()
    sample["class_index"] = 99

    assert apply_label_to_signal(sample, "class_index") == [3, 4]


def test_apply_label_to_signal_prefers_direct_sample_level_label():
    sample = _parent_with_components()
    sample["multi_hot_label"] = np.array([0, 1, 0, 1], dtype=np.float32)

    values = apply_label_to_signal(sample, "multi_hot_label")

    assert len(values) == 1
    np.testing.assert_array_equal(values[0], sample.multi_hot_label)


def test_apply_label_to_signal_leaf_signal_fallback():
    sample = Signal(
        data=np.zeros(50, dtype=np.complex64),
        class_name="bpsk",
        class_index=3,
        start_in_samples=25,
        num_iq_samples_dataset=100,
        num_signals_max=1,
        _lower_frequency=-10.0,
        _upper_frequency=10.0,
    )

    assert apply_label_to_signal(sample, "class_name") == ["bpsk"]
    assert apply_label_to_signal(sample, "class_index") == [3]
    assert apply_label_to_signal(sample, "start") == [0.25]
    assert apply_label_to_signal(sample, "stop") == [0.75]
    assert apply_label_to_signal(sample, "lower_freq") == [-10.0]
    assert apply_label_to_signal(sample, "upper_freq") == [10.0]


def test_apply_transforms_and_labels_none_returns_signal():
    sample = _parent_with_components()

    assert apply_transforms_and_labels_to_signal(sample, [], None) is sample


def test_apply_transforms_and_labels_empty_returns_data():
    sample = _parent_with_components()
    result = apply_transforms_and_labels_to_signal(sample, [], [])

    assert isinstance(result, np.ndarray)
    assert result.shape == sample.data.shape


def test_apply_transforms_and_labels_single_target_multi_signal_returns_list():
    sample = _parent_with_components(num_signals_max=2)

    data, targets = apply_transforms_and_labels_to_signal(
        sample,
        [],
        ["class_index"],
    )

    assert data is sample.data
    assert targets == [3, 4]


def test_apply_transforms_and_labels_returns_wideband_multi_hot_vector():
    sample = _parent_with_components(num_signals_max=2)

    data, target = apply_transforms_and_labels_to_signal(
        sample,
        [MultiHotLabel(num_classes=6)],
        ["multi_hot_label"],
    )

    assert data is sample.data
    np.testing.assert_array_equal(
        target,
        np.array([0, 0, 0, 1, 1, 0], dtype=np.float32),
    )


def test_apply_transforms_and_labels_single_signal_squeezes_single_target():
    sample = Signal(
        data=np.zeros(100, dtype=np.complex64),
        component_signals=[
            Signal(
                data=np.zeros(20, dtype=np.complex64),
                class_index=3,
                num_iq_samples_dataset=100,
            )
        ],
        num_iq_samples_dataset=100,
        num_signals_max=1,
    )
    sample.component_signals[0].add_parent(sample, register=False)

    _, target = apply_transforms_and_labels_to_signal(sample, [], ["class_index"])

    assert target == 3


def test_apply_transforms_and_labels_multiple_targets_parallel_lists():
    sample = _parent_with_components(num_signals_max=2)

    _, targets = apply_transforms_and_labels_to_signal(
        sample,
        [],
        ["class_name", "class_index", "start", "stop", "lower_freq", "upper_freq"],
    )

    assert targets == [
        ["bpsk", "qpsk"],
        [3, 4],
        [0.1, 0.5],
        [0.3, 0.8],
        [80.0, -230.0],
        [120.0, -170.0],
    ]


def test_apply_label_to_signal_class_index_from_class_name_fallback():
    sample = Signal(
        data=np.zeros(100, dtype=np.complex64),
        component_signals=[
            Signal(
                data=np.zeros(20, dtype=np.complex64),
                class_name="qpsk",
                start_in_samples=0,
            )
        ],
        num_iq_samples_dataset=100,
        num_signals_max=1,
        class_names=np.array(["bpsk", "qpsk", "8psk"]),
    )
    sample.component_signals[0].add_parent(sample, register=False)

    assert apply_label_to_signal(sample, "class_index") == [1]


def test_apply_label_to_signal_leaf_class_index_from_class_name_fallback():
    sample = Signal(
        data=np.zeros(100, dtype=np.complex64),
        class_name="8psk",
        num_iq_samples_dataset=100,
        num_signals_max=1,
        class_names=np.array(["bpsk", "qpsk", "8psk"]),
    )

    assert apply_label_to_signal(sample, "class_index") == [2]


# =============================================================================
# TorchSigIterableDataset generation helpers
# =============================================================================


def test_insert_component_signal_uses_relative_signal_slice():
    dataset = TorchSigIterableDataset(
        metadata=_small_metadata(),
        signal_generators=[],
        validate_init=False,
    )

    iq_samples = np.zeros(8, dtype=np.complex64)
    signal = Signal(
        data=np.arange(10, dtype=np.float32).astype(np.complex64),
        center_freq=0,
        bandwidth=1,
    )

    dataset._insert_component_signal(iq_samples, signal, start_sample=5)

    expected = np.zeros(8, dtype=np.complex64)
    expected[5:8] = np.array([0, 1, 2], dtype=np.complex64)

    assert np.array_equal(iq_samples, expected)
    assert signal.start_in_samples == 5
    assert signal.duration_in_samples == 3


def test_insert_component_signal_does_not_truncate_when_signal_fits():
    dataset = TorchSigIterableDataset(
        metadata=_small_metadata(),
        signal_generators=[],
        validate_init=False,
    )

    iq_samples = np.zeros(8, dtype=np.complex64)
    signal = Signal(
        data=np.array([1, 2, 3], dtype=np.complex64),
        center_freq=0,
        bandwidth=1,
    )

    dataset._insert_component_signal(iq_samples, signal, start_sample=2)

    expected = np.zeros(8, dtype=np.complex64)
    expected[2:5] = np.array([1, 2, 3], dtype=np.complex64)

    assert np.array_equal(iq_samples, expected)
    assert signal.start_in_samples == 2
    assert signal.duration_in_samples == 3


def test_choose_start_sample_warns_when_signal_is_too_large():
    dataset = TorchSigIterableDataset(
        metadata=_small_metadata(),
        signal_generators=[],
        validate_init=False,
    )

    iq_samples = np.zeros(8, dtype=np.complex64)
    signal = Signal(data=np.zeros(10, dtype=np.complex64), center_freq=0, bandwidth=1)

    with pytest.warns(UserWarning, match="too large"):
        start_sample = dataset._choose_start_sample(iq_samples, signal)

    assert start_sample == 0


def test_generate_new_signal_sets_component_start_and_clips_duration(monkeypatch):
    dataset = TorchSigIterableDataset(
        metadata=_small_metadata(),
        signal_generators=[],
        validate_init=False,
    )
    dataset["cochannel_overlap_probability"] = 1.0

    component = Signal(
        data=np.ones(20, dtype=np.complex64),
        center_freq=0,
        bandwidth=1,
        class_name="dummy",
        class_index=0,
    )

    monkeypatch.setattr(dataset, "_build_noise_floor", lambda: np.zeros(8, dtype=np.complex64))
    monkeypatch.setattr(dataset, "_generate_component_signal", lambda: component.copy())
    monkeypatch.setattr(dataset, "_choose_start_sample", lambda iq_samples, signal: 5)
    monkeypatch.setattr(dataset, "_map_to_coordinates", lambda signal, start_sample: object())
    monkeypatch.setattr(dataset, "_check_if_overlap", lambda rectangle, rectangles: False)

    sample = dataset.__generate_new_signal__()

    placed = sample.component_signals[0]
    assert placed.start_in_samples == 5
    assert placed.duration_in_samples == 3
    assert np.all(sample.data[5:8] == 1)


def test_choose_start_sample_allows_last_valid_position(monkeypatch):
    dataset = TorchSigIterableDataset(
        metadata=_small_metadata(),
        signal_generators=[],
        validate_init=False,
    )

    iq_samples = np.zeros(8, dtype=np.complex64)
    signal = Signal(
        data=np.ones(3, dtype=np.complex64),
        center_freq=0,
        bandwidth=1,
    )

    class StubRandomGenerator:
        def integers(self, *, low, high):
            assert low == 0
            assert high == 6
            return high - 1

    monkeypatch.setattr(
        dataset,
        "random_generator",
        StubRandomGenerator(),
    )

    start_sample = dataset._choose_start_sample(
        iq_samples,
        signal,
    )

    assert start_sample == 5


def _insert_component_signal(
    self,
    iq_samples: np.ndarray,
    signal: Signal,
    start_sample: int,
) -> None:
    """Insert a component signal into the dataset IQ buffer."""
    stop_sample = min(
        start_sample + len(signal.data),
        len(iq_samples),
    )
    num_samples_to_add = stop_sample - start_sample

    if num_samples_to_add < len(signal.data):
        signal.data = signal.data[:num_samples_to_add]
        signal["duration_in_samples"] = num_samples_to_add

    iq_samples[start_sample:stop_sample] += signal.data
    signal["start_in_samples"] = start_sample


def test_insert_component_signal_truncates_component_data():
    dataset = TorchSigIterableDataset(
        metadata=_small_metadata(),
        signal_generators=[],
        validate_init=False,
    )

    iq_samples = np.zeros(8, dtype=np.complex64)
    signal = Signal(
        data=np.arange(10, dtype=np.float32).astype(np.complex64),
        center_freq=0,
        bandwidth=1,
    )

    dataset._insert_component_signal(
        iq_samples,
        signal,
        start_sample=5,
    )

    expected_component_data = np.array(
        [0, 1, 2],
        dtype=np.complex64,
    )
    expected_iq_samples = np.zeros(8, dtype=np.complex64)
    expected_iq_samples[5:8] = expected_component_data

    np.testing.assert_array_equal(
        signal.data,
        expected_component_data,
    )
    np.testing.assert_array_equal(
        iq_samples,
        expected_iq_samples,
    )
    assert signal.start_in_samples == 5
    assert signal.duration_in_samples == 3
    assert len(signal.data) == signal.duration_in_samples


def test_iterable_dataset_warns_when_max_signal_duration_exceeds_sample_length():
    metadata = _small_metadata()
    metadata["num_iq_samples_dataset"] = 4096
    metadata["signal_duration_in_samples_max"] = 262144

    with pytest.warns(
        UserWarning,
        match=("signal_duration_in_samples_max exceeds num_iq_samples_dataset"),
    ):
        TorchSigIterableDataset(
            metadata=metadata,
            signal_generators=[],
            validate_init=True,
        )


def test_iterable_dataset_does_not_warn_when_max_signal_duration_equals_sample_length():
    metadata = _small_metadata()
    metadata["num_iq_samples_dataset"] = 4096
    metadata["signal_duration_in_samples_max"] = 4096

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        TorchSigIterableDataset(
            metadata=metadata,
            signal_generators=[],
            validate_init=True,
        )

    matching_warnings = [warning for warning in caught_warnings if ("signal_duration_in_samples_max exceeds num_iq_samples_dataset") in str(warning.message)]

    assert matching_warnings == []


def test_validate_signal_duration_limits_warns_when_max_exceeds_sample_length():
    metadata = _small_metadata()
    metadata["num_iq_samples_dataset"] = 4096
    metadata["signal_duration_in_samples_max"] = 262144

    dataset = TorchSigIterableDataset(
        metadata=metadata,
        signal_generators=[],
        validate_init=False,
    )

    with pytest.warns(
        UserWarning,
        match=("signal_duration_in_samples_max exceeds num_iq_samples_dataset"),
    ):
        dataset._validate_signal_duration_limits()


def test_validate_signal_duration_limits_allows_equal_sample_length():
    metadata = _small_metadata()
    metadata["num_iq_samples_dataset"] = 4096
    metadata["signal_duration_in_samples_max"] = 4096

    dataset = TorchSigIterableDataset(
        metadata=metadata,
        signal_generators=[],
        validate_init=False,
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        dataset._validate_signal_duration_limits()

    assert caught_warnings == []


# Sampling weights / probabilities
# =============================================================================


@pytest.mark.parametrize(
    "value, expected",
    [(1, 1.0), (1.5, 1.5), (np.int64(2), 2.0), (np.float64(2.5), 2.5)],
)
def test_validate_positive_weight_accepts_positive_real_numbers(value, expected):
    assert TorchSigIterableDataset._validate_positive_weight(value, "likelihood") == expected


@pytest.mark.parametrize("value", [0, -1, -0.5, np.int64(0), np.float64(-1.0)])
def test_validate_positive_weight_rejects_nonpositive_values(value):
    with pytest.raises(ValueError, match="likelihood must be > 0"):
        TorchSigIterableDataset._validate_positive_weight(value, "likelihood")


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_validate_positive_weight_rejects_nonfinite_values(value):
    with pytest.raises(ValueError, match="probability must be finite"):
        TorchSigIterableDataset._validate_positive_weight(value, "probability")


@pytest.mark.parametrize("value", ["1.0", None, object(), [1.0]])
def test_validate_positive_weight_rejects_non_numeric_values(value):
    with pytest.raises(TypeError, match="probability must be a real number"):
        TorchSigIterableDataset._validate_positive_weight(value, "probability")


def test_validate_signal_sampling_configuration_allows_empty_dataset():
    _dataset_with_empty_generators()._validate_signal_sampling_configuration()


def test_validate_signal_sampling_configuration_accepts_valid_likelihoods():
    dataset = _dataset_with_empty_generators()
    dataset.add_signal_generator(DummyGenerator(class_name="a"), likelihood=1.0)
    dataset.add_signal_generator(DummyGenerator(class_name="b"), likelihood=2.0)

    dataset._validate_signal_sampling_configuration()


def test_validate_signal_sampling_configuration_rejects_likelihood_count_mismatch():
    dataset = _dataset_with_empty_generators()
    dataset.add_signal_generator(DummyGenerator(class_name="a"), likelihood=1.0)
    dataset.signal_likelihoods = []

    with pytest.raises(ValueError, match="signal likelihood count does not match"):
        dataset._validate_signal_sampling_configuration()


def test_validate_signal_sampling_configuration_rejects_nonpositive_likelihoods():
    dataset = _dataset_with_empty_generators()
    dataset.add_signal_generator(DummyGenerator(class_name="a"), likelihood=1.0)
    dataset.signal_likelihoods = [0.0]

    with pytest.raises(ValueError, match="all signal likelihoods must be > 0"):
        dataset._validate_signal_sampling_configuration()


def test_validate_signal_sampling_configuration_accepts_complete_probabilities():
    dataset = _dataset_with_empty_generators()
    dataset.add_signal_generator(DummyGenerator(class_name="a"), probability=0.25)
    dataset.add_signal_generator(DummyGenerator(class_name="b"), probability=0.75)

    dataset._validate_signal_sampling_configuration(require_complete=True)

    np.testing.assert_array_equal(
        dataset.signal_probabilities,
        np.array([0.25, 0.75]),
    )


def test_validate_signal_sampling_configuration_accepts_incomplete_probabilities_when_not_required():
    dataset = _dataset_with_empty_generators()
    dataset.add_signal_generator(DummyGenerator(class_name="a"), probability=0.25)
    dataset.add_signal_generator(DummyGenerator(class_name="b"), probability=0.25)

    dataset._validate_signal_sampling_configuration(require_complete=False)


def test_validate_signal_sampling_configuration_rejects_incomplete_probabilities_when_required():
    dataset = _dataset_with_empty_generators()
    dataset.add_signal_generator(DummyGenerator(class_name="a"), probability=0.25)
    dataset.add_signal_generator(DummyGenerator(class_name="b"), probability=0.25)

    with pytest.raises(ValueError, match="must sum to 1.0 before sampling"):
        dataset._validate_signal_sampling_configuration(require_complete=True)


def test_validate_signal_sampling_configuration_rejects_probability_sum_greater_than_one():
    dataset = _dataset_with_empty_generators()
    dataset._signal_probability_mode = "probability"
    dataset.signal_generators = [DummyGenerator(class_name="a"), DummyGenerator(class_name="b")]
    dataset.signal_probabilities = np.array([0.75, 0.50])

    with pytest.raises(ValueError, match="signal probabilities must sum to 1.0"):
        dataset._validate_signal_sampling_configuration(require_complete=True)


def test_validate_signal_sampling_configuration_rejects_probability_count_mismatch():
    dataset = _dataset_with_empty_generators()
    dataset._signal_probability_mode = "probability"
    dataset.signal_generators = [DummyGenerator(class_name="a"), DummyGenerator(class_name="b")]
    dataset.signal_probabilities = np.array([1.0])

    with pytest.raises(ValueError, match="signal probability count does not match"):
        dataset._validate_signal_sampling_configuration()


def test_validate_signal_sampling_configuration_rejects_negative_probabilities():
    dataset = _dataset_with_empty_generators()
    dataset._signal_probability_mode = "probability"
    dataset.signal_generators = [DummyGenerator(class_name="a"), DummyGenerator(class_name="b")]
    dataset.signal_probabilities = np.array([1.5, -0.5])

    with pytest.raises(ValueError, match="all signal probabilities must be >= 0"):
        dataset._validate_signal_sampling_configuration()


def test_validate_signal_sampling_configuration_accepts_zero_probability():
    dataset = _dataset_with_empty_generators()
    dataset._signal_probability_mode = "probability"
    dataset.signal_generators = [DummyGenerator(class_name="a"), DummyGenerator(class_name="b")]
    dataset.signal_probabilities = np.array([1.0, 0.0])

    dataset._validate_signal_sampling_configuration()


def test_refresh_signal_probabilities_empty():
    dataset = _dataset_with_empty_generators()
    dataset.signal_probabilities = np.array([1.0])

    dataset._refresh_signal_probabilities()

    assert dataset.signal_probabilities.shape == (0,)
    assert dataset.signal_probabilities.dtype == float


def test_refresh_signal_probabilities_probability_mode():
    dataset = _dataset_with_empty_generators()
    dataset.add_signal_generator(DummyGenerator(class_name="a"), probability=0.25)
    dataset.add_signal_generator(DummyGenerator(class_name="b"), probability=0.75)

    dataset._refresh_signal_probabilities()

    np.testing.assert_array_equal(dataset.signal_probabilities, np.array([0.25, 0.75]))


def test_refresh_signal_probabilities_likelihood_mode():
    dataset = _dataset_with_empty_generators()
    dataset.add_signal_generator(DummyGenerator(class_name="a"), likelihood=1.0)
    dataset.add_signal_generator(DummyGenerator(class_name="b"), likelihood=3.0)

    dataset._refresh_signal_probabilities()

    np.testing.assert_allclose(dataset.signal_probabilities, np.array([0.25, 0.75]))


def test_add_signal_generator_rejects_likelihood_and_probability():
    dataset = _dataset_with_empty_generators()

    with pytest.raises(ValueError, match="Specify only one of likelihood or probability"):
        dataset.add_signal_generator(
            DummyGenerator(class_name="a"),
            likelihood=1.0,
            probability=0.5,
        )


def test_add_signal_generator_rejects_missing_probability_after_probability_mode():
    dataset = _dataset_with_empty_generators()
    dataset.add_signal_generator(DummyGenerator(class_name="a"), probability=0.5)

    with pytest.raises(ValueError, match="All signal generators must specify probability"):
        dataset.add_signal_generator(DummyGenerator(class_name="b"))


def test_add_signal_generator_rejects_probability_after_likelihood_mode():
    dataset = _dataset_with_empty_generators()
    dataset.add_signal_generator(DummyGenerator(class_name="a"), likelihood=1.0)

    with pytest.raises(ValueError, match="Cannot mix explicit probability"):
        dataset.add_signal_generator(DummyGenerator(class_name="b"), probability=0.5)


def test_add_signal_generator_rejects_probability_sum_greater_than_one():
    dataset = _dataset_with_empty_generators()
    dataset.add_signal_generator(DummyGenerator(class_name="a"), probability=0.75)

    with pytest.raises(ValueError, match="signal probabilities must sum to 1.0 or less"):
        dataset.add_signal_generator(DummyGenerator(class_name="b"), probability=0.50)


@pytest.mark.parametrize("bad_likelihood", [0.0, -1.0])
def test_add_signal_generator_rejects_nonpositive_likelihood(bad_likelihood):
    dataset = _dataset_with_empty_generators()

    with pytest.raises(ValueError, match="likelihood must be > 0"):
        dataset.add_signal_generator(DummyGenerator(class_name="a"), likelihood=bad_likelihood)


@pytest.mark.parametrize("bad_probability", [0.0, -1.0])
def test_add_signal_generator_rejects_nonpositive_probability(bad_probability):
    dataset = _dataset_with_empty_generators()

    with pytest.raises(ValueError, match="probability must be > 0"):
        dataset.add_signal_generator(DummyGenerator(class_name="a"), probability=bad_probability)


def test_add_signal_generator_defaults_likelihood_to_one():
    dataset = _dataset_with_empty_generators()

    dataset.add_signal_generator(DummyGenerator(class_name="a"))
    dataset.add_signal_generator(DummyGenerator(class_name="b"), likelihood=3.0)

    assert dataset.signal_likelihoods == [1.0, 3.0]
    np.testing.assert_allclose(
        dataset.signal_probabilities,
        [0.25, 0.75],
    )


# =============================================================================
# TorchSigIterableDataset misc behavior
# =============================================================================


def test_iterable_dataset_call_returns_next_sample(monkeypatch):
    dataset = _dataset_with_empty_generators()
    expected = Signal(data=np.ones(8, dtype=np.complex64))

    monkeypatch.setattr(TorchSigIterableDataset, "__next__", lambda self: expected)

    assert dataset() is expected


def test_iterable_dataset_repr_includes_core_fields():
    dataset = _dataset_with_empty_generators()

    result = repr(dataset)

    assert result.startswith("TorchSigIterableDataset(")
    assert "metadata=" in result
    assert "transforms=" in result
    assert "signal_generators=" in result
    assert result.endswith(")")


def test_iterable_dataset_adds_sample_and_stage_logging_context(monkeypatch):
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata["dataset_id"] = "debug-dataset"
    observed_contexts = []

    def generate_signal(self):
        observed_contexts.append(("generate", get_metadata_logging_context()))
        return Signal(data=np.ones(8, dtype=np.complex64))

    def transform_signal(signal):
        observed_contexts.append(("transform", get_metadata_logging_context()))
        return signal

    dataset = TorchSigIterableDataset(
        metadata=metadata,
        signal_generators=[],
        transforms=[transform_signal],
        target_labels=None,
        validate_init=False,
    )
    dataset.enable_metadata_debug(keys={"unused"})
    monkeypatch.setattr(
        TorchSigIterableDataset,
        "__generate_new_signal__",
        generate_signal,
    )

    next(dataset)
    next(dataset)

    assert [stage for stage, _ in observed_contexts] == [
        "generate",
        "transform",
        "generate",
        "transform",
    ]
    contexts = [context for _, context in observed_contexts]
    assert [context.sample_index for context in contexts] == [0, 0, 1, 1]
    assert all(context.dataset_id == "debug-dataset" for context in contexts)
    assert all(context.worker_id == 0 for context in contexts)
    assert [dict(context.fields)["stage"] for context in contexts] == [
        "generate",
        "transform",
        "generate",
        "transform",
    ]
    assert contexts[0].session_id == contexts[1].session_id
    assert contexts[2].session_id == contexts[3].session_id
    assert contexts[0].session_id != contexts[2].session_id
    assert get_metadata_logging_context() == MetadataLoggingContext()


def test_iterable_dataset_logs_completed_metadata_snapshot(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata["dataset_id"] = "snapshot-dataset"
    component = Signal(
        data=np.ones(4, dtype=np.complex64),
        class_name="qpsk",
    )

    def generate_signal(self):
        return Signal(
            data=np.ones(8, dtype=np.complex64),
            component_signals=[component],
            parent=self,
        )

    def mark_complete(signal):
        signal["complete"] = True
        return signal

    dataset = TorchSigIterableDataset(
        metadata=metadata,
        signal_generators=[],
        transforms=[mark_complete],
        target_labels=None,
        validate_init=False,
    )
    dataset.enable_metadata_debug(
        keys={"complete", "class_name"},
        events={"snapshot"},
        include_values=True,
    )
    monkeypatch.setattr(
        TorchSigIterableDataset,
        "__generate_new_signal__",
        generate_signal,
    )

    next(dataset)

    snapshots = [record for record in caplog.records if record.metadata_event == "snapshot"]
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.metadata_snapshot == {"complete": "True"}
    assert snapshot.metadata_component_snapshots == ({"class_name": "'qpsk'"},)
    assert snapshot.metadata_sample_index == 0
    assert snapshot.metadata_dataset_id == "snapshot-dataset"
    assert snapshot.metadata_worker_id == 0
    assert snapshot.metadata_correlation_fields["stage"] == "transform"


def test_iterable_dataset_inherits_outer_logging_session(monkeypatch):
    dataset = _dataset_with_empty_generators()
    observed_contexts = []

    def generate_signal(self):
        observed_contexts.append(get_metadata_logging_context())
        return Signal(data=np.ones(8, dtype=np.complex64))

    monkeypatch.setattr(
        TorchSigIterableDataset,
        "__generate_new_signal__",
        generate_signal,
    )

    with metadata_logging_context(
        session_id="outer-session",
        dataset_id="outer-dataset",
        worker_id=9,
    ):
        next(dataset)

    context = observed_contexts[0]
    assert context.session_id == "outer-session"
    assert context.dataset_id == "outer-dataset"
    assert context.sample_index == 0
    assert context.worker_id == 0


def test_iterable_dataset_skips_context_when_debugging_is_inactive(monkeypatch):
    dataset = _dataset_with_empty_generators()
    observed_contexts = []

    def generate_signal(self):
        observed_contexts.append(get_metadata_logging_context())
        return Signal(data=np.ones(8, dtype=np.complex64))

    monkeypatch.setattr(
        TorchSigIterableDataset,
        "__generate_new_signal__",
        generate_signal,
    )

    next(dataset)

    assert observed_contexts == [MetadataLoggingContext()]
    assert dataset._metadata_logging_sample_index == 1


def test_iterable_dataset_logging_indices_are_independent_per_worker_copy(
    monkeypatch,
):
    datasets = [_dataset_with_empty_generators(), _dataset_with_empty_generators()]
    for dataset in datasets:
        dataset.enable_metadata_debug(keys={"unused"})
    current_worker = [0]
    observed_contexts = []

    def generate_signal(self):
        observed_contexts.append(get_metadata_logging_context())
        return Signal(data=np.ones(8, dtype=np.complex64))

    monkeypatch.setattr(
        "torchsig.datasets.datasets.get_worker_info",
        lambda: SimpleNamespace(id=current_worker[0]),
    )
    monkeypatch.setattr(
        TorchSigIterableDataset,
        "__generate_new_signal__",
        generate_signal,
    )

    for _ in range(3):
        for worker_id, dataset in enumerate(datasets):
            current_worker[0] = worker_id
            next(dataset)

    worker_indices = {0: [], 1: []}
    for context in observed_contexts:
        worker_indices[context.worker_id].append(context.sample_index)
    assert worker_indices == {0: [0, 1, 2], 1: [0, 1, 2]}


def test_iterable_dataset_logging_context_is_restored_after_failure(monkeypatch):
    dataset = _dataset_with_empty_generators()
    dataset.enable_metadata_debug(keys={"unused"})

    def generate_signal(self):
        assert get_metadata_logging_context().sample_index == 0
        raise RuntimeError("generation failed")

    monkeypatch.setattr(
        TorchSigIterableDataset,
        "__generate_new_signal__",
        generate_signal,
    )

    with pytest.raises(RuntimeError, match="generation failed"):
        next(dataset)

    assert get_metadata_logging_context() == MetadataLoggingContext()
    assert dataset._metadata_logging_sample_index == 1


# =============================================================================
# SafeTorchSigIterableDataset
# =============================================================================


def test_safe_iterable_dataset_set_fallback_policy_defaults_to_original():
    dataset = _safe_dataset()

    dataset.set_fallback_policy()

    assert dataset.pipeline_fallback == "original"


@pytest.mark.parametrize("fallback", ["original", "zero"])
def test_safe_iterable_dataset_set_fallback_policy_without_retries(fallback):
    dataset = _safe_dataset()

    dataset.set_fallback_policy(fallback=fallback)

    assert dataset.pipeline_fallback == fallback


def test_safe_iterable_dataset_set_fallback_policy_retry_sets_max_retries():
    dataset = _safe_dataset()

    dataset.set_fallback_policy(fallback="retry", max_retries=5)

    assert dataset.pipeline_fallback == "retry"
    assert dataset.pipeline_max_retries == 5


def test_safe_iterable_dataset_set_fallback_policy_retry_without_max_retries_preserves_existing_value():
    dataset = _safe_dataset()
    dataset.pipeline_max_retries = 7

    dataset.set_fallback_policy(fallback="retry")

    assert dataset.pipeline_fallback == "retry"
    assert dataset.pipeline_max_retries == 7


@pytest.mark.parametrize("fallback", ["original", "zero"])
def test_safe_iterable_dataset_set_fallback_policy_rejects_retries_without_retry_mode(fallback):
    dataset = _safe_dataset()

    with pytest.raises(ValueError, match="max_retries is only allowed with fallback='retry'"):
        dataset.set_fallback_policy(fallback=fallback, max_retries=3)


# =============================================================================
# StaticTorchSigDataset
# =============================================================================


def test_static_dataset_getitem_raises_index_error_for_out_of_bounds(tmp_path):
    root = tmp_path / "static_dataset"
    root.mkdir()

    dataset = StaticTorchSigDataset.__new__(StaticTorchSigDataset)
    dataset.root = root
    dataset.dataset_length = 3

    with pytest.raises(IndexError, match=r"Index -1 is out of bounds"):
        dataset[-1]

    with pytest.raises(IndexError, match=r"Index 3 is out of bounds"):
        dataset[3]


def test_static_dataset_uses_native_contiguous_batch_reader():
    class FakeReader:
        def __init__(self):
            self.batch_calls = []
            self.read_calls = []

        def read_signals_batch(self, start, stop):
            self.batch_calls.append((start, stop))
            return [Signal(data=np.full(4, idx, dtype=np.complex64), index=idx) for idx in range(start, stop)]

        def read(self, idx):
            self.read_calls.append(idx)
            return Signal(data=np.full(4, idx, dtype=np.complex64), index=idx)

    dataset = StaticTorchSigDataset.__new__(StaticTorchSigDataset)
    dataset.dataset_length = 5
    dataset.reader = FakeReader()
    dataset.transforms = []
    dataset.target_labels = None

    batch = dataset.__getitems__([1, 2, 3])

    assert [signal["index"] for signal in batch] == [1, 2, 3]
    assert dataset.reader.batch_calls == [(1, 4)]
    assert dataset.reader.read_calls == []


def test_static_dataset_noncontiguous_batch_falls_back_to_single_reads():
    class FakeReader:
        def __init__(self):
            self.batch_calls = []
            self.read_calls = []

        def read_signals_batch(self, start, stop):
            self.batch_calls.append((start, stop))
            return []

        def read(self, idx):
            self.read_calls.append(idx)
            return Signal(data=np.full(4, idx, dtype=np.complex64), index=idx)

    dataset = StaticTorchSigDataset.__new__(StaticTorchSigDataset)
    dataset.dataset_length = 5
    dataset.reader = FakeReader()
    dataset.transforms = []
    dataset.target_labels = None

    batch = dataset.__getitems__([3, 0, 2])

    assert [signal["index"] for signal in batch] == [3, 0, 2]
    assert dataset.reader.batch_calls == []
    assert dataset.reader.read_calls == [3, 0, 2]


def test_static_dataset_batch_rejects_out_of_bounds_index():
    dataset = StaticTorchSigDataset.__new__(StaticTorchSigDataset)
    dataset.dataset_length = 3

    with pytest.raises(IndexError, match=r"Index 3 is out of bounds"):
        dataset.__getitems__([1, 3])


def test_static_dataset_verify_raises_for_missing_root(tmp_path):
    dataset = StaticTorchSigDataset.__new__(StaticTorchSigDataset)
    dataset.root = tmp_path / "does_not_exist"

    with pytest.raises(ValueError, match=r"root does not exist:"):
        dataset._verify()


def test_static_dataset_verify_accepts_existing_root(tmp_path):
    dataset = StaticTorchSigDataset.__new__(StaticTorchSigDataset)
    dataset.root = tmp_path

    dataset._verify()


def test_static_dataset_str():
    dataset = StaticTorchSigDataset.__new__(StaticTorchSigDataset)
    dataset.root = Path("/tmp/test_dataset")

    assert str(dataset) == "StaticTorchSigDataset: /tmp/test_dataset"


def test_static_dataset_repr():
    dataset = StaticTorchSigDataset.__new__(StaticTorchSigDataset)
    dataset.root = Path("/tmp/test_dataset")
    dataset.reader = "DummyReader"

    assert repr(dataset) == ("StaticTorchSigDataset(root=/tmp/test_dataset, file_handler_class=DummyReader)")


# =============================================================================
# Slow static dataset integration/regression tests
# =============================================================================


def test_static_dataset_preserves_property_backed_target_labels(tmp_path):
    sample = _parent_with_components(num_signals_max=3)

    class FakeReader:
        def __init__(self, root):
            self.root = root

        def __len__(self):
            return 1

        def read(self, idx):
            assert idx == 0
            return sample

    static_dataset = StaticTorchSigDataset(
        root=tmp_path,
        file_handler_class=FakeReader,
        target_labels=["class_name", "start", "stop", "lower_freq", "upper_freq"],
    )

    _, targets = static_dataset[0]
    class_names, starts, stops, lower_freqs, upper_freqs = targets

    assert class_names == ["bpsk", "qpsk"]
    assert starts == [0.1, 0.5]
    assert stops == [0.3, 0.8]
    assert lower_freqs == [80.0, -230.0]
    assert upper_freqs == [120.0, -170.0]


def test_static_dataset_class_index_is_single_label_when_one_signal(tmp_path):
    sample = Signal(
        data=np.zeros(16, dtype=np.complex64),
        component_signals=[
            Signal(
                data=np.ones(8, dtype=np.complex64),
                class_index=7,
                num_iq_samples_dataset=16,
            )
        ],
        num_iq_samples_dataset=16,
        num_signals_max=1,
    )
    sample["class_index"] = 99
    sample.component_signals[0].add_parent(sample, register=False)

    class FakeReader:
        def __init__(self, root):
            self.root = root

        def __len__(self):
            return 1

        def read(self, idx):
            assert idx == 0
            return sample

    static_dataset = StaticTorchSigDataset(
        root=tmp_path,
        file_handler_class=FakeReader,
        target_labels=["class_index"],
    )

    _, target = static_dataset[0]

    assert target == 7
    assert isinstance(target, int)


@pytest.mark.slow
def test_static_iq_dataset_class_index_labels_are_valid(tmp_path):
    seed = 123
    dataset_length = 200
    batch_size = 32
    fft_size = 256

    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": fft_size**2,
            "fft_size": fft_size,
            "fft_stride": fft_size,
            "num_signals_max": 1,
            "num_signals_min": 1,
            "noise_power_db": 1,
            "signal_center_freq_min": 1000,
            "signal_center_freq_max": 2000,
            "sample_rate": 10000,
            "frequency_min": 1000,
            "frequency_max": 2000,
            "cochannel_overlap_probability": 0.2,
            "bandwidth_min": 1000,
            "bandwidth_max": 2000,
        }
    )

    iterable = TorchSigIterableDataset(
        metadata=metadata,
        transforms=[ComplexTo2D()],
        target_labels=None,
        signal_generators="all",
    )

    dataloader = WorkerSeedingDataLoader(
        iterable,
        batch_size=batch_size,
        collate_fn=lambda x: x,
        num_workers=1,
    )
    dataloader.seed(seed)

    root = tmp_path / "static_iq_dataset" / "train"

    DatasetCreator(
        dataloader=dataloader,
        root=root,
        overwrite=True,
        dataset_length=dataset_length,
    ).create()

    static_dataset = StaticTorchSigDataset(root=root, target_labels=["class_index"])
    label_counts = Counter()

    for idx in range(dataset_length):
        _, target = static_dataset[idx]

        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().reshape(-1).tolist()
        elif isinstance(target, np.ndarray):
            target = target.reshape(-1).tolist()

        assert not isinstance(target, (list, tuple))
        assert isinstance(target, (int, np.integer))
        assert 0 <= int(target) < len(iterable.class_names)

        label_counts[int(target)] += 1

    assert sum(label_counts.values()) == dataset_length


@pytest.mark.slow
def test_static_iq_dataset_target_labels_are_parallel_and_valid(tmp_path):
    seed = 123
    dataset_length = 200
    batch_size = 32
    fft_size = 256

    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": fft_size**2,
            "fft_size": fft_size,
            "fft_stride": fft_size,
            "num_signals_max": 5,
            "num_signals_min": 1,
            "noise_power_db": 0,
        }
    )

    iterable = TorchSigIterableDataset(
        metadata=metadata,
        target_labels=None,
        signal_generators="all",
    )

    dataloader = WorkerSeedingDataLoader(
        iterable,
        batch_size=batch_size,
        collate_fn=lambda x: x,
        num_workers=1,
    )
    dataloader.seed(seed)

    root = tmp_path / "static_multilabel_dataset" / "train"

    DatasetCreator(
        dataloader=dataloader,
        root=root,
        overwrite=True,
        dataset_length=dataset_length,
    ).create()

    target_labels = [
        "class_name",
        "class_index",
        "start",
        "stop",
        "lower_freq",
        "upper_freq",
        "snr_db",
    ]

    static_dataset = StaticTorchSigDataset(root=root, target_labels=target_labels)

    valid_class_names = set(iterable.class_names)
    num_classes = len(iterable.class_names)

    for idx in range(dataset_length):
        _, targets = static_dataset[idx]

        assert isinstance(targets, list)
        assert len(targets) == len(target_labels)

        class_names, class_indices, starts, stops, lower_freqs, upper_freqs, snrs = targets

        lengths = [len(values) for values in targets]
        assert len(set(lengths)) == 1

        num_labels = lengths[0]
        assert 1 <= num_labels <= metadata["num_signals_max"]

        assert all(name in valid_class_names for name in class_names)
        assert all(isinstance(label, (int, np.integer)) for label in class_indices)
        assert all(0 <= int(label) < num_classes for label in class_indices)
        assert all(0.0 <= float(start) <= 1.0 for start in starts)
        assert all(float(stop) >= float(start) for start, stop in zip(starts, stops))
        assert all(float(lower) <= float(upper) for lower, upper in zip(lower_freqs, upper_freqs))
        assert all(np.isfinite(float(snr)) for snr in snrs)


# =============================================================================
# Immutable Defaults
# =============================================================================


def test_iterable_dataset_transform_defaults_are_not_shared():
    first = _dataset_with_empty_generators()
    second = _dataset_with_empty_generators()

    first.transforms.append(object())
    first.component_transforms.append(object())

    assert second.transforms == []
    assert second.component_transforms == []


def test_iterable_dataset_accepts_explicit_transform_lists():
    transform = MagicMock()
    component_transform = MagicMock()

    dataset = TorchSigIterableDataset(
        metadata=TorchSigDefaults().default_dataset_metadata.copy(),
        signal_generators=[],
        transforms=[transform],
        component_transforms=[component_transform],
        validate_init=False,
    )

    assert dataset.transforms == [transform]
    assert dataset.component_transforms == [component_transform]


def test_string_signal_generators_support_class_name_grouping():
    dataset_metadata = TorchSigDefaults().default_dataset_metadata

    grouping = {
        "source": "class_name",
        "groups": [
            {
                "name": "psk",
                "values": ["bpsk", "qpsk"],
            },
            {
                "name": "fsk",
                "values": ["2fsk", "4fsk"],
            },
        ],
    }

    dataset = TorchSigIterableDataset(
        metadata=dataset_metadata,
        signal_generators=["bpsk", "qpsk", "2fsk", "4fsk"],
        sampling_grouping=grouping,
    )

    assert dataset.class_names == [
        "bpsk",
        "qpsk",
        "2fsk",
        "4fsk",
    ]
    np.testing.assert_allclose(
        dataset.signal_probabilities,
        [0.25, 0.25, 0.25, 0.25],
    )


def test_string_lookup_expands_concat_signal_generator():
    grouping = {
        "source": "class_name",
        "groups": [
            {
                "name": "analog",
                "values": ["fm"],
            },
            {
                "name": "digital",
                "values": ["bpsk"],
            },
        ],
    }

    dataset = TorchSigIterableDataset(
        metadata=TorchSigDefaults().default_dataset_metadata,
        signal_generators=["fm", "bpsk"],
        sampling_grouping=grouping,
    )

    assert [generator.class_name for generator in dataset.signal_generators] == ["fm", "bpsk"]

    np.testing.assert_allclose(
        dataset.signal_probabilities,
        [0.5, 0.5],
    )
