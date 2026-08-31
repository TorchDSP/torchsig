"""Unit tests for dataset utility functions."""

from unittest.mock import patch

import numpy as np
import pytest

from torchsig.datasets.dataset_utils import (
    frequency_shift_signal,
    save_type,
)
from torchsig.signals.signal_types import Signal


class DummySignal:
    """Minimal Signal-like object used to test frequency shifting."""

    def __init__(
        self,
        data: np.ndarray,
        *,
        center_freq: float = 0.0,
        bandwidth: float = 20.0,
    ) -> None:
        self.data = data
        self.metadata = {
            "center_freq": center_freq,
            "bandwidth": bandwidth,
            "center_freq_set": False,
        }

    def __getitem__(self, key: str):
        return self.metadata[key]

    def __setitem__(self, key: str, value) -> None:
        self.metadata[key] = value

    @property
    def upper_freq(self) -> float:
        return self["center_freq"] + self["bandwidth"] / 2

    @property
    def lower_freq(self) -> float:
        return self["center_freq"] - self["bandwidth"] / 2


@pytest.fixture
def signal() -> DummySignal:
    """Return a reusable dummy signal."""
    return DummySignal(
        data=np.ones(32, dtype=np.complex64),
        bandwidth=20.0,
    )


@pytest.mark.parametrize(
    ("transforms", "target_transforms", "expected"),
    [
        ([], [], True),
        ([object()], [], False),
        ([], [object()], False),
        ([object()], [object()], False),
    ],
)
def test_save_type(
    transforms: list,
    target_transforms: list,
    expected: bool,
) -> None:
    """Verify raw data is selected only when both transform lists are empty."""
    assert save_type(transforms, target_transforms) is expected


@patch("torchsig.datasets.dataset_utils.upconversion_anti_aliasing_filter")
@patch("torchsig.datasets.dataset_utils.frequency_shift")
def test_frequency_shift_signal_without_aliasing(
    frequency_shift_mock,
    anti_aliasing_filter_mock,
    signal,
) -> None:
    """Verify an in-band shift updates data and metadata without filtering."""
    shifted_data = np.full(32, 2 + 1j, dtype=np.complex64)
    frequency_shift_mock.return_value = shifted_data

    random_generator = np.random.default_rng(42)
    expected_generator = np.random.default_rng(42)
    expected_center_freq = expected_generator.uniform(low=-20.0, high=20.0)

    result = frequency_shift_signal(
        signal=signal,
        center_freq_min=-20.0,
        center_freq_max=20.0,
        sample_rate=100.0,
        frequency_max=50.0,
        frequency_min=-50.0,
        random_generator=random_generator,
    )

    assert result is signal
    np.testing.assert_array_equal(result.data, shifted_data)
    assert result["center_freq"] == pytest.approx(expected_center_freq)
    assert result["bandwidth"] == pytest.approx(20.0)
    assert result["center_freq_set"] is True

    frequency_shift_mock.assert_called_once()
    np.testing.assert_array_equal(
        frequency_shift_mock.call_args.args[0],
        np.ones(32, dtype=np.complex64),
    )
    assert frequency_shift_mock.call_args.args[1] == pytest.approx(expected_center_freq)
    assert frequency_shift_mock.call_args.args[2] == pytest.approx(100.0)

    anti_aliasing_filter_mock.assert_not_called()


@patch("torchsig.datasets.dataset_utils.upconversion_anti_aliasing_filter")
@patch("torchsig.datasets.dataset_utils.frequency_shift")
def test_frequency_shift_signal_filters_upper_edge_aliasing(
    frequency_shift_mock,
    anti_aliasing_filter_mock,
    signal,
) -> None:
    """Verify filtering occurs when the shifted upper edge exceeds the limit."""
    shifted_data = np.full(32, 2 + 1j, dtype=np.complex64)
    filtered_data = np.full(32, 3 + 2j, dtype=np.complex64)

    frequency_shift_mock.return_value = shifted_data
    anti_aliasing_filter_mock.return_value = (
        filtered_data,
        44.0,
        12.0,
    )

    result = frequency_shift_signal(
        signal=signal,
        center_freq_min=45.0,
        center_freq_max=45.0,
        sample_rate=100.0,
        frequency_max=50.0,
        frequency_min=-50.0,
        random_generator=np.random.default_rng(42),
    )

    assert result is signal
    np.testing.assert_array_equal(result.data, filtered_data)
    assert result["center_freq"] == pytest.approx(44.0)
    assert result["bandwidth"] == pytest.approx(12.0)
    assert result["center_freq_set"] is True

    anti_aliasing_filter_mock.assert_called_once()
    args = anti_aliasing_filter_mock.call_args.args

    np.testing.assert_array_equal(args[0], shifted_data)
    assert args[1] == pytest.approx(45.0)
    assert args[2] == pytest.approx(20.0)
    assert args[3] == pytest.approx(100.0)
    assert args[4] == pytest.approx(50.0)
    assert args[5] == pytest.approx(-50.0)


@patch("torchsig.datasets.dataset_utils.upconversion_anti_aliasing_filter")
@patch("torchsig.datasets.dataset_utils.frequency_shift")
def test_frequency_shift_signal_filters_lower_edge_aliasing(
    frequency_shift_mock,
    anti_aliasing_filter_mock,
    signal,
) -> None:
    """Verify filtering occurs when the shifted lower edge exceeds the limit."""
    shifted_data = np.full(32, 2 - 1j, dtype=np.complex64)
    filtered_data = np.full(32, 3 - 2j, dtype=np.complex64)

    frequency_shift_mock.return_value = shifted_data
    anti_aliasing_filter_mock.return_value = (
        filtered_data,
        -44.0,
        12.0,
    )

    result = frequency_shift_signal(
        signal=signal,
        center_freq_min=-45.0,
        center_freq_max=-45.0,
        sample_rate=100.0,
        frequency_max=50.0,
        frequency_min=-50.0,
        random_generator=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(result.data, filtered_data)
    assert result["center_freq"] == pytest.approx(-44.0)
    assert result["bandwidth"] == pytest.approx(12.0)
    assert result["center_freq_set"] is True

    anti_aliasing_filter_mock.assert_called_once()


@patch("torchsig.datasets.dataset_utils.upconversion_anti_aliasing_filter")
@patch("torchsig.datasets.dataset_utils.frequency_shift")
def test_frequency_shift_signal_recalculates_edges_after_alias_filtering(
    frequency_shift_mock,
    anti_aliasing_filter_mock,
) -> None:
    """Final edges must use the center and bandwidth returned by filtering."""
    signal = Signal(
        data=np.ones(32, dtype=np.complex64),
        center_freq=0.0,
        bandwidth=20.0,
    )
    frequency_shift_mock.return_value = signal.data.copy()
    anti_aliasing_filter_mock.return_value = (
        signal.data.copy(),
        44.0,
        12.0,
    )

    result = frequency_shift_signal(
        signal=signal,
        center_freq_min=45.0,
        center_freq_max=45.0,
        sample_rate=100.0,
        frequency_max=50.0,
        frequency_min=-50.0,
        random_generator=np.random.default_rng(42),
    )

    assert result.center_freq == pytest.approx(44.0)
    assert result.bandwidth == pytest.approx(12.0)
    assert result.lower_freq == pytest.approx(38.0)
    assert result.upper_freq == pytest.approx(50.0)
    assert "_lower_frequency" not in result.metadata
    assert "_upper_frequency" not in result.metadata


@patch("torchsig.datasets.dataset_utils.np.random.default_rng")
@patch("torchsig.datasets.dataset_utils.upconversion_anti_aliasing_filter")
@patch("torchsig.datasets.dataset_utils.frequency_shift")
def test_frequency_shift_signal_creates_default_random_generator(
    frequency_shift_mock,
    anti_aliasing_filter_mock,
    default_rng_mock,
    signal,
) -> None:
    """Verify a NumPy random generator is created when none is supplied."""
    generated_rng = default_rng_mock.return_value
    generated_rng.uniform.return_value = 10.0

    shifted_data = np.full(32, 4 + 1j, dtype=np.complex64)
    frequency_shift_mock.return_value = shifted_data

    result = frequency_shift_signal(
        signal=signal,
        center_freq_min=-20.0,
        center_freq_max=20.0,
        sample_rate=100.0,
        frequency_max=50.0,
        frequency_min=-50.0,
    )

    default_rng_mock.assert_called_once_with(seed=None)
    generated_rng.uniform.assert_called_once_with(low=-20.0, high=20.0)

    frequency_shift_mock.assert_called_once()
    assert frequency_shift_mock.call_args.args[1] == pytest.approx(10.0)

    anti_aliasing_filter_mock.assert_not_called()
    assert result["center_freq"] == pytest.approx(10.0)
    assert result["center_freq_set"] is True
