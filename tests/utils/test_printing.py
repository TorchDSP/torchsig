from dataclasses import dataclass

import numpy as np
import pytest

from torchsig.utils.printing import (
    dataset_metadata_repr,
    dataset_metadata_str,
    generate_repr_str,
)
from torchsig.utils.random import Seedable


class PlainObject:
    def __init__(self):
        self.alpha = 1
        self.beta = "two"
        self.gamma = [3, 4]


class DummySeedable(Seedable):
    def __init__(self):
        # Do not call super().__init__ so this test only targets repr filtering.
        self.children = []
        self.rng_seed = 123
        self.np_rng = object()
        self.random_generator = object()
        self.torch_rng = object()
        self.parent = object()

        self.alpha = 1
        self.beta = "two"


@dataclass
class DummyDatasetMetadata:
    num_iq_samples_dataset: int = 1024
    fft_size: int = 256
    sample_rate: float = 1_000_000.0
    num_signals_min: int = 1
    num_signals_max: int = 3
    num_signals_distribution: np.ndarray | None = None
    snr_db_min: float = -10.0
    snr_db_max: float = 30.0
    signal_duration_min: float = 0.1
    signal_duration_max: float = 0.9
    signal_bandwidth_min: float = 0.01
    signal_bandwidth_max: float = 0.2
    signal_center_freq_min: float = -0.4
    signal_center_freq_max: float = 0.4
    class_list: list[str] | None = None
    class_distribution: np.ndarray | None = None


def make_dataset_metadata(**overrides):
    defaults = {
        "num_signals_distribution": np.array([0.25, 0.75]),
        "class_list": ["bpsk", "qpsk"],
        "class_distribution": np.array([0.4, 0.6]),
    }
    defaults.update(overrides)
    return DummyDatasetMetadata(**defaults)


def test_generate_repr_str_includes_class_name_and_attributes():
    obj = PlainObject()

    result = generate_repr_str(obj)

    assert result == "PlainObject(alpha=1,beta=two,gamma=[3, 4])"


def test_generate_repr_str_excludes_requested_parameters():
    obj = PlainObject()

    result = generate_repr_str(obj, exclude_params=["beta"])

    assert result == "PlainObject(alpha=1,gamma=[3, 4])"
    assert "beta" not in result


def test_generate_repr_str_ignores_unknown_excluded_parameters():
    obj = PlainObject()

    result = generate_repr_str(obj, exclude_params=["does_not_exist"])

    assert result == "PlainObject(alpha=1,beta=two,gamma=[3, 4])"


def test_generate_repr_str_removes_seedable_internal_fields():
    obj = DummySeedable()

    result = generate_repr_str(obj)

    assert result == "DummySeedable(alpha=1,beta=two)"
    assert "children" not in result
    assert "rng_seed" not in result
    assert "np_rng" not in result
    assert "random_generator" not in result
    assert "torch_rng" not in result
    assert "parent" not in result


def test_generate_repr_str_raises_attribute_error_for_object_without_dict():
    with pytest.raises(AttributeError):
        generate_repr_str(1)


def test_dataset_metadata_repr_formats_all_fields_with_array_lists():
    metadata = make_dataset_metadata()

    result = dataset_metadata_repr(metadata)

    assert result == (
        "DummyDatasetMetadata("
        "num_iq_samples_dataset=1024,"
        "fft_size=256,"
        "num_signals_max=3,"
        "sample_rate=1000000.0,"
        "num_signals_min=1,"
        "num_signals_distribution=[0.25, 0.75],"
        "snr_db_min=-10.0,"
        "snr_db_max=30.0,"
        "signal_duration_min=0.1,"
        "signal_duration_max=0.9,"
        "signal_bandwidth_min=0.01,"
        "signal_bandwidth_max=0.2,"
        "signal_center_freq_min=-0.4,"
        "signal_center_freq_max=0.4,"
        "class_list=['bpsk', 'qpsk'],"
        "class_distribution=[0.4, 0.6]"
        ")"
    )


def test_dataset_metadata_repr_handles_none_distributions():
    metadata = make_dataset_metadata(
        num_signals_distribution=None,
        class_distribution=None,
    )

    result = dataset_metadata_repr(metadata)

    assert "num_signals_distribution=None" in result
    assert "class_distribution=None" in result


def assert_metadata_line(result: str, field: str, value: str):
    assert any(line.startswith(field) and value in line for line in result.splitlines())


def test_dataset_metadata_str_contains_header_separator_and_core_fields():
    metadata = make_dataset_metadata()

    result = dataset_metadata_str(metadata)

    assert result.startswith("\nDummyDatasetMetadata\n")
    assert "-" * 100 in result
    assert_metadata_line(result, "num_iq_samples_dataset", "1024")
    assert_metadata_line(result, "fft_size", "256")
    assert_metadata_line(result, "sample_rate", "1000000.0")
    assert_metadata_line(result, "num_signals_distribution", "[0.25, 0.75]")
    assert_metadata_line(result, "class_list", "['bpsk', 'qpsk']")
    assert_metadata_line(result, "class_distribution", "[0.4, 0.6]")


def test_dataset_metadata_str_handles_none_distributions():
    metadata = make_dataset_metadata(
        num_signals_distribution=None,
        class_distribution=None,
    )

    result = dataset_metadata_str(metadata)

    assert_metadata_line(result, "num_signals_distribution", "None")
    assert_metadata_line(result, "class_distribution", "None")


def test_dataset_metadata_str_wraps_long_class_list():
    metadata = make_dataset_metadata(
        class_list=[
            "very-long-class-name-0",
            "very-long-class-name-1",
            "very-long-class-name-2",
            "very-long-class-name-3",
        ],
    )

    result = dataset_metadata_str(metadata, max_width=60)

    assert "class_list" in result
    assert "very-long-class-name-0" in result
    assert "very-long-class-name-3" in result
    assert "\n" in result
