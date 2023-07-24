from torchsig.datasets.synthetic import DigitalModulationDataset
from torchsig.utils.types import create_modulated_rf_metadata, create_signal, SignalData
from torchsig.transforms.transforms import *
import numpy as np
import pytest


def generate_data():
    dataset = DigitalModulationDataset(
        ["bpsk"],
        num_iq_samples=4096,
        num_samples_per_class=1,
        iq_samples_per_symbol=2,
        random_pulse_shaping=False,
        random_data=False,
    )
    iq_data = dataset[0][0]
    data = SignalData(samples=iq_data)
    meta = create_modulated_rf_metadata(num_samples=iq_data.shape[0])
    signal = create_signal(data, [meta])
    return signal


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


def call_transform(transform, signal):
    transform(signal)


@pytest.mark.benchmark(group="transforms")
@pytest.mark.parametrize(
    "transform", transforms_list, ids=([t[0] for t in transforms_list])
)
def test_transform_benchmark(benchmark, transform):
    signal = generate_data()
    benchmark(call_transform, transform[2], signal)
