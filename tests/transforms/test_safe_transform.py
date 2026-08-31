import os
from unittest.mock import patch

import numpy as np
import pytest

from torchsig.transforms.transforms import transform_crash_logger


@pytest.fixture
def temp_dir(tmp_path):
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    yield
    os.chdir(original_dir)


def test_successful_transform():
    """Test that a successful transform returns the correct result and passes RNG."""

    def mock_transform(data, rng, **kwargs):
        assert isinstance(rng, np.random.Generator)
        assert "extra_param" in kwargs
        return data * 2

    data = np.array([1.0, 2.0, 3.0])
    result = transform_crash_logger(mock_transform, data, extra_param=42)
    assert np.array_equal(result, np.array([2.0, 4.0, 6.0]))


@patch("time.time")
def test_failed_transform_saves_state(mock_time, temp_dir):
    """Test that failed transforms save state to a .npz file."""
    mock_time.return_value = 1678901234.567
    timestamp = int(mock_time.return_value * 1000)
    expected_filename = f"crash_failing_transform_{timestamp}.npz"

    def failing_transform(data, **kwargs):
        raise ValueError("Simulated error")

    data = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        transform_crash_logger(failing_transform, data, param=42)

    assert os.path.exists(expected_filename)
    with np.load(expected_filename, allow_pickle=True) as f:
        assert np.array_equal(f["data"], data)
        assert isinstance(f["seed"].item(), (int, np.integer))
        saved_kwargs = f["kwargs"].item()
        assert saved_kwargs["param"] == 42
        assert "rng" not in saved_kwargs  # Verify rng was excluded


@patch("time.time")
def test_reproduce_failure_from_saved_state(mock_time, temp_dir):
    """Test that saved state can reproduce the exact failure."""
    mock_time.return_value = 1678901234.567
    timestamp = int(mock_time.return_value * 1000)
    filename = f"crash_failing_transform_{timestamp}.npz"

    def failing_transform(data, bad_param, **kwargs):
        if bad_param == "error":
            raise RuntimeError("Reproducible error")
        return data

    data = np.array([1, 2, 3])
    with pytest.raises(RuntimeError):
        transform_crash_logger(failing_transform, data, bad_param="error")

    with np.load(filename, allow_pickle=True) as f:
        saved_data = f["data"]
        saved_seed = f["seed"].item()
        saved_kwargs = f["kwargs"].item()

    # Recreate RNG from seed
    rng = np.random.default_rng(saved_seed)
    with pytest.raises(RuntimeError):
        failing_transform(saved_data, rng=rng, **saved_kwargs)


@patch("time.time")
def test_unique_filenames_rapid_failures(mock_time, temp_dir):
    """Test that rapid failures generate unique filenames."""

    def failing_transform(data, **kwargs):
        raise ValueError("Error")

    # Simulate 2 failures in the same millisecond
    mock_time.return_value = 1678901234.567
    data = np.array([1])
    with pytest.raises(ValueError):
        transform_crash_logger(failing_transform, data)
    with pytest.raises(ValueError):
        transform_crash_logger(failing_transform, data)

    # Verify transform_crash_logger handles collisions
    files = [f for f in os.listdir() if f.startswith("crash_failing_transform_")]
    assert len(files) == 2
