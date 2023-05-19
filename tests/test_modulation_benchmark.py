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


@pytest.mark.benchmark()
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


if __name__ == "__main__":
    pytest.main()
