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


def iterate_one_epoch(dataset):
    for _ in dataset:
        pass


@pytest.mark.benchmark(group="constellation")
@pytest.mark.parametrize("modulation_name", default_const_map.keys())
def test_generate_constellation_benchmark(benchmark, modulation_name):
    dataset = ConstellationDataset(
        [modulation_name],
        num_iq_samples=4096,
        num_samples_per_class=100,
        iq_samples_per_symbol=2,
        pulse_shape_filter=None,
        random_pulse_shaping=False,
        random_data=False,
        use_gpu=False,
    )
    benchmark(iterate_one_epoch, dataset)


@pytest.mark.benchmark(group="fsk")
@pytest.mark.parametrize("modulation_name", freq_map.keys())
def test_generate_fsk_benchmark(benchmark, modulation_name):
    dataset = FSKDataset(
        [modulation_name],
        num_iq_samples=4096,
        num_samples_per_class=100,
        iq_samples_per_symbol=2,
        random_pulse_shaping=False,
        random_data=False,
        use_gpu=False,
    )
    benchmark(iterate_one_epoch, dataset)


num_subcarriers = (64, 72, 128, 180, 256, 300, 512, 600, 900, 1024, 1200, 2048)


@pytest.mark.benchmark(group="ofdm")
@pytest.mark.parametrize("num_subcarriers", num_subcarriers)
def test_generate_ofdm_benchmark(benchmark, num_subcarriers):
    constellations = ("bpsk", "qpsk", "16qam", "64qam", "256qam", "1024qam")
    sidelobe_suppression_methods = ("lpf", "win_start")
    dataset = OFDMDataset(
        constellations,
        num_subcarriers=(num_subcarriers,),
        num_iq_samples=4096,
        num_samples_per_class=100,
        sidelobe_suppression_methods=sidelobe_suppression_methods,
        use_gpu=False,
    )
    benchmark(iterate_one_epoch, dataset)


if __name__ == "__main__":
    pytest.main()
