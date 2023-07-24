from torchsig.datasets.synthetic import DigitalModulationDataset
from torchsig.transforms.transforms import *
from torchsig.utils.types import create_modulated_rf_metadata, create_signal, SignalData
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
    iq_data = dataset[0][0]
    short_data_iq = SignalData(samples=iq_data)
    short_data_meta = create_modulated_rf_metadata(num_samples=iq_data.shape[0])
    short_data = create_signal(short_data_iq, [short_data_meta])

    dataset = DigitalModulationDataset(
        [modulation_name],
        num_iq_samples=4096,
        num_samples_per_class=1,
        iq_samples_per_symbol=2,
        random_pulse_shaping=False,
        random_data=False,
    )
    iq_data = dataset[0][0]
    long_data_iq = SignalData(samples=iq_data)
    long_data_meta = create_modulated_rf_metadata(num_samples=iq_data.shape[0])
    long_data = create_signal(long_data_iq, [long_data_meta])
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
        TimeCrop("random", crop_length=64, signal_length=128),
        TimeCrop("random", crop_length=2048, signal_length=4096),
    ),
    ("time_reversal", TimeReversal(False), TimeReversal(False)),
    ("frequency_shift", RandomFrequencyShift(-0.25), RandomFrequencyShift(-0.25)),
    (
        "delayed_frequency_shift",
        RandomDelayedFrequencyShift(0.2, 0.25),
        RandomDelayedFrequencyShift(0.2, 0.25),
    ),
    (
        "oscillator_drift",
        LocalOscillatorDrift(0.01, 0.001),
        LocalOscillatorDrift(0.01, 0.001),
    ),
    ("gain_drift", GainDrift(0.01, 0.001, 0.1), GainDrift(0.01, 0.001, 0.1)),
    (
        "iq_imbalance",
        IQImbalance(3, np.pi / 180, 0.05),
        IQImbalance(3, np.pi / 180, 0.05),
    ),
    ("roll_off", RollOff(0.05, 0.98), RollOff(0.05, 0.98)),
    ("add_slope", AddSlope(), AddSlope()),
    ("spectral_inversion", SpectralInversion(), SpectralInversion()),
    ("channel_swap", ChannelSwap(), ChannelSwap()),
    ("magnitude_rescale", RandomMagRescale(0.5, 3), RandomMagRescale(0.5, 3)),
    (
        "drop_samples",
        RandomDropSamples(0.3, 50, ["zero"]),
        RandomDropSamples(0.3, 50, ["zero"]),
    ),
    ("quantize", Quantize(32, ["floor"]), Quantize(32, ["floor"])),
    ("clip", Clip(0.85), Clip(0.85)),
]

modulations = ["bpsk", "4fsk"]


@pytest.mark.parametrize(
    "transform, modulation_name", itertools.product(transforms_list, modulations)
)
def test_transform_figures(transform, modulation_name):
    short_data, long_data = generate_data(modulation_name)

    short_data_iq = short_data["data"]["samples"]
    long_data_iq = long_data["data"]["samples"]

    short_data_transform = transform[1](short_data)["data"]["samples"]
    long_data_transform = transform[2](long_data)["data"]["samples"]

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
