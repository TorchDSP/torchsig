from unittest import TestCase
from torchsig.transforms.system_impairment.si import RandomTimeShift, TimeCrop
import numpy as np


class RandomTimeShiftTestCase(TestCase):
    def test_random_time_shift_right(self):
        rng = np.random.RandomState(0)
        data = (
            rng.rand(
                16,
            )
            - 0.5
        ) + 1j * (
            rng.rand(
                16,
            )
            - 0.5
        )
        shift = 5
        t = RandomTimeShift(
            shift=shift,
        )
        new_data = t(data)
        self.assertTrue(np.allclose(data[:-shift], new_data[shift:]))
        self.assertTrue(np.allclose(new_data[:shift], np.zeros(shift)))

    def test_random_time_shift_left(self):
        rng = np.random.RandomState(0)
        data = (
            rng.rand(
                16,
            )
            - 0.5
        ) + 1j * (
            rng.rand(
                16,
            )
            - 0.5
        )
        shift = -5
        t = RandomTimeShift(
            shift=shift,
        )
        new_data = t(data)
        self.assertTrue(np.allclose(data[-shift:], new_data[:shift]))
        self.assertTrue(np.allclose(new_data[shift:], np.zeros(np.abs(shift))))


class TimeCropTestCase(TestCase):
    def test_time_crop_start(self):
        rng = np.random.RandomState(0)
        num_iq_samples = 16
        data = (
            rng.rand(
                num_iq_samples,
            )
            - 0.5
        ) + 1j * (
            rng.rand(
                num_iq_samples,
            )
            - 0.5
        )
        length = 4
        t = TimeCrop(
            crop_type="start",
            length=length,
        )
        new_data: np.ndarray = t(data)
        self.assertTrue(np.allclose(data[:length], new_data))
        self.assertTrue(new_data.shape[0] == length)

    def test_time_crop_center(self):
        rng = np.random.RandomState(0)
        num_iq_samples = 16
        data = (
            rng.rand(
                num_iq_samples,
            )
            - 0.5
        ) + 1j * (
            rng.rand(
                num_iq_samples,
            )
            - 0.5
        )
        length = 4
        t = TimeCrop(
            crop_type="center",
            length=length,
        )
        new_data: np.ndarray = t(data)
        extra_samples = num_iq_samples - length
        self.assertTrue(
            np.allclose(data[extra_samples // 2 : -extra_samples // 2], new_data)
        )
        self.assertTrue(new_data.shape[0] == length)

    def test_time_crop_end(self):
        rng = np.random.RandomState(0)
        num_iq_samples = 16
        data = (
            rng.rand(
                num_iq_samples,
            )
            - 0.5
        ) + 1j * (
            rng.rand(
                num_iq_samples,
            )
            - 0.5
        )
        length = 4
        t = TimeCrop(
            crop_type="end",
            length=length,
        )
        new_data: np.ndarray = t(data)
        self.assertTrue(np.allclose(data[-length:], new_data))
        self.assertTrue(new_data.shape[0] == length)
