from torchsig.datasets.wideband import WidebandModulationsDataset
import pytest


def iterate_one_epoch(dataset):
    for idx in range(len(dataset)):
        _ = dataset[idx]


@pytest.mark.benchmark(group="wideband")
def test_generate_wideband_modulation_benchmark(benchmark):
    dataset = WidebandModulationsDataset(
        level=2,
        num_samples=10,
    )
    iterate_one_epoch(dataset)
    # benchmark(iterate_one_epoch, dataset)
