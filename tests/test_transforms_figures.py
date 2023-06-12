from torchsig.datasets.synthetic import DigitalModulationDataset
from torchsig.transforms.transforms import *
from torchsig.utils.types import SignalData, SignalDescription
from matplotlib import pyplot as plt
import itertools
import numpy as np
import pytest


def generate_data(modulation_name):
    dataset = DigitalModulationDataset(
        [modulation_name],
        num_iq_samples=128,
        num_samples_per_class=1,
        iq_samples_per_symbol=2,
        random_pulse_shaping=False,
        random_data=False,
    )
    short_data = SignalData(
        dataset[0][0].tobytes(),
        item_type=np.float64,
        data_type=np.complex128,
        signal_description=SignalDescription(),
    )

    dataset = DigitalModulationDataset(
        [modulation_name],
        num_iq_samples=4096,
        num_samples_per_class=1,
        iq_samples_per_symbol=2,
        random_pulse_shaping=False,
        random_data=False,
    )
    long_data = SignalData(
        dataset[0][0].tobytes(),
        item_type=np.float64,
        data_type=np.complex128,
        signal_description=SignalDescription(),
    )
    return short_data, long_data


transforms_list = [
    (
        "random_resample_up",
        RandomResample(1.5, num_iq_samples=128, keep_samples=False),
        RandomResample(1.5, num_iq_samples=4096, keep_samples=False),
    ),
    (
        "random_resample_down",
        RandomResample(0.75, num_iq_samples=128, keep_samples=False),
        RandomResample(0.75, num_iq_samples=4096, keep_samples=False),
    ),
    ("add_noise", AddNoise(-10), AddNoise(-10)),
    ("time_varying_noise", TimeVaryingNoise(-30, -10), TimeVaryingNoise(-30, -10)),
    (
        "rayleigh_fading",
        RayleighFadingChannel(0.05, (1.0, 0.5, 0.1)),
        RayleighFadingChannel(0.05, (1.0, 0.5, 0.1)),
    ),
    ("phase_shift", RandomPhaseShift(0.5), RandomPhaseShift(0.5)),
    ("time_shift", RandomTimeShift(-100.5), RandomTimeShift(-2.5)),
    (
        "time_crop",
        TimeCrop("random", length=64),
        TimeCrop("random", length=2048),
    ),
    ("time_reversal", TimeReversal(False), TimeReversal(False)),
    ("frequency_shift", RandomFrequencyShift(-0.25), RandomFrequencyShift(-0.25)),
]

modulations = ["bpsk", "4fsk"]


@pytest.mark.parametrize(
    "transform, modulation_name", itertools.product(transforms_list, modulations)
)
def test_transform_figure(transform, modulation_name):
    short_data, long_data = generate_data(modulation_name)

    short_data_iq = short_data.iq_data
    long_data_iq = long_data.iq_data

    short_data_transform = transform[1](short_data).iq_data
    long_data_transform = transform[2](long_data).iq_data

    # IQ Data
    figure = plt.figure(figsize=(9, 4))
    figure.suptitle("{}_{}".format(transform[0], modulation_name))
    plt.title(transform[0])
    plt.subplot(4, 2, 1)
    plt.plot(short_data_iq.real)
    plt.plot(short_data_iq.imag)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel("Time")
    plt.title("Original")

    plt.subplot(4, 2, 2)
    plt.plot(short_data_transform.real)
    plt.plot(short_data_transform.imag)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.title("Transform")

    plt.subplot(4, 2, 3)
    _ = plt.scatter(long_data_iq.real, long_data_iq.imag)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel("Const")

    plt.subplot(4, 2, 4)
    _ = plt.scatter(long_data_transform.real, long_data_transform.imag)
    plt.xticks([])
    plt.yticks([])
    plt.title("")

    plt.subplot(4, 2, 5)
    _ = plt.psd(long_data_iq)
    plt.xticks([])
    plt.xlabel("")
    plt.yticks([])
    plt.ylabel("PSD")
    plt.title("")

    plt.subplot(4, 2, 6)
    _ = plt.psd(long_data_transform)
    plt.xticks([])
    plt.xlabel("")
    plt.yticks([])
    plt.ylabel("")
    plt.title("")

    plt.subplot(4, 2, 7)
    _ = plt.specgram(long_data_iq)
    plt.xticks([])
    plt.ylabel("Spectrogram")
    plt.yticks([])
    plt.title("")

    plt.subplot(4, 2, 8)
    _ = plt.specgram(long_data_transform)
    plt.xticks([])
    plt.yticks([])
    plt.title("")

    plt.savefig(
        "tests/figures/transform_{}_{}.jpg".format(transform[0], modulation_name)
    )
