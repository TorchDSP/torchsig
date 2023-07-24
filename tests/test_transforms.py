from torchsig.transforms import RandomTimeShift, TimeCrop
import numpy as np
import pytest


def test_random_time_shift_right():
    rng = np.random.default_rng(0)
    data = (
        rng.random(
            16,
        )
        - 0.5
    ) + 1j * (
        rng.random(
            16,
        )
        - 0.5
    )
    shift = 5
    t = RandomTimeShift(
        shift=shift,
    )
    new_data = t(data)["data"]["samples"]
    assert np.allclose(data[:-shift], new_data[shift:])
    assert np.allclose(new_data[:shift], np.zeros(shift))


def test_random_time_shift_left():
    rng = np.random.default_rng(0)
    data = (
        rng.random(
            16,
        )
        - 0.5
    ) + 1j * (
        rng.random(
            16,
        )
        - 0.5
    )
    shift = -5
    t = RandomTimeShift(
        shift=shift,
    )
    new_data = t(data)["data"]["samples"]
    assert np.allclose(data[-shift:], new_data[:shift])
    assert np.allclose(new_data[shift:], np.zeros(np.abs(shift)))


def test_time_crop_start():
    rng = np.random.default_rng(0)
    num_iq_samples = 16
    data = (
        rng.random(
            num_iq_samples,
        )
        - 0.5
    ) + 1j * (
        rng.random(
            num_iq_samples,
        )
        - 0.5
    )
    length = 4
    t = TimeCrop(
        crop_type="start",
        crop_length=length,
    )
    new_data: np.ndarray = t(data)["data"]["samples"]
    assert np.allclose(data[:length], new_data)
    assert new_data.shape[0] == length


def test_time_crop_center():
    rng = np.random.default_rng(0)
    num_iq_samples = 16
    data = (
        rng.random(
            num_iq_samples,
        )
        - 0.5
    ) + 1j * (
        rng.random(
            num_iq_samples,
        )
        - 0.5
    )

    length = 4
    t = TimeCrop(crop_type="center", crop_length=length, signal_length=data.shape[0])
    new_data: np.ndarray = t(data)["data"]["samples"]
    extra_samples = num_iq_samples - length
    assert np.allclose(data[extra_samples // 2 : -extra_samples // 2], new_data)
    assert new_data.shape[0] == length


def test_time_crop_end():
    rng = np.random.default_rng(0)
    num_iq_samples = 16
    data = (
        rng.random(
            num_iq_samples,
        )
        - 0.5
    ) + 1j * (
        rng.random(
            num_iq_samples,
        )
        - 0.5
    )
    length = 4
    t = TimeCrop(crop_type="end", crop_length=length, signal_length=data.shape[0])
    new_data: np.ndarray = t(data)["data"]["samples"]
    assert np.allclose(data[-length:], new_data)
    assert new_data.shape[0] == length
