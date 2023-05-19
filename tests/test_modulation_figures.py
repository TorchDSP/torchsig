from torchsig.datasets.synthetic import (
    ConstellationDataset,
    FSKDataset,
    OFDMDataset,
    default_const_map,
    freq_map,
)
from matplotlib import pyplot as plt
import numpy as np
import pytest


@pytest.mark.parametrize("modulation_name", default_const_map.keys())
def test_can_generate_constellation_figures(modulation_name):
    dataset = ConstellationDataset(
        [modulation_name],
        num_iq_samples=4096,
        num_samples_per_class=1,
        iq_samples_per_symbol=2,
        pulse_shape_filter=None,
        random_pulse_shaping=False,
        random_data=False,
        use_gpu=False,
    )
    item = dataset[0]
    iq_data: np.ndarray = item[0]

    # IQ Data
    plt.figure(figsize=(9, 4))
    plt.subplot(2, 2, 1)
    plt.plot(iq_data.real)
    plt.plot(iq_data.imag)
    plt.legend(["real", "imaginary"])
    plt.title("IQ Data")

    plt.subplot(2, 2, 2)
    _ = plt.scatter(iq_data.real, iq_data.imag)
    plt.title("Constellation")

    plt.subplot(2, 2, 3)
    _ = plt.psd(iq_data)
    plt.title("PSD")

    plt.subplot(2, 2, 4)
    _ = plt.specgram(iq_data)
    plt.title("Spectrogram")
    plt.savefig("tests/figures/synthetic_{}.jpg".format(modulation_name))


@pytest.mark.parametrize("modulation_name", freq_map.keys())
def test_can_generate_fsk_figures(modulation_name):
    dataset = FSKDataset(
        [modulation_name],
        num_iq_samples=4096,
        num_samples_per_class=1,
        iq_samples_per_symbol=2,
        random_pulse_shaping=False,
        random_data=False,
        use_gpu=False,
    )
    item = dataset[0]
    iq_data: np.ndarray = item[0]

    # IQ Data
    plt.figure(figsize=(9, 4))
    plt.subplot(2, 2, (1, 2))
    plt.plot(iq_data.real[:256])
    plt.plot(iq_data.imag[:256])
    plt.legend(["real", "imaginary"])
    plt.title("IQ Data")

    plt.subplot(2, 2, 3)
    _ = plt.psd(iq_data)
    plt.title("PSD")

    plt.subplot(2, 2, 4)
    _ = plt.specgram(iq_data)
    plt.title("Spectrogram")
    plt.savefig("tests/figures/synthetic_{}.jpg".format(modulation_name))


num_subcarriers = (64, 72, 128, 180, 256, 300, 512, 600, 900, 1024, 1200, 2048)


@pytest.mark.parametrize("num_subcarriers", num_subcarriers)
def test_can_generate_ofdm_figures(num_subcarriers):
    constellations = ("bpsk", "qpsk", "16qam", "64qam", "256qam", "1024qam")
    sidelobe_suppression_methods = ("lpf", "win_start")
    dataset = OFDMDataset(
        constellations,
        num_subcarriers=(num_subcarriers,),
        num_iq_samples=4096,
        num_samples_per_class=1,
        sidelobe_suppression_methods=sidelobe_suppression_methods,
        use_gpu=False,
    )
    item = dataset[0]
    iq_data: np.ndarray = item[0]

    # IQ Data
    plt.figure(figsize=(9, 4))
    plt.subplot(2, 2, (1, 2))
    plt.plot(iq_data.real[:256])
    plt.plot(iq_data.imag[:256])
    plt.legend(["real", "imaginary"])
    plt.title("IQ Data")

    plt.subplot(2, 2, 3)
    _ = plt.psd(iq_data)
    plt.title("PSD")

    plt.subplot(2, 2, 4)
    _ = plt.specgram(iq_data)
    plt.title("Spectrogram")
    plt.savefig("tests/figures/synthetic_ofdm_{}.jpg".format(num_subcarriers))


if __name__ == "__main__":
    pytest.main()
