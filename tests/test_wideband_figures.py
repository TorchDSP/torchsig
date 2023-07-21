from torchsig.utils.visualize import (
    MaskClassVisualizer,
    mask_class_to_outline,
    complex_spectrogram_to_magnitude,
)
from torchsig.transforms import Compose, Spectrogram, Normalize
from torchsig.datasets.wideband import WidebandModulationsDataset
from torchsig.utils.writer import DatasetCreator, DatasetLoader
from torchsig.datasets.wideband_sig53 import WidebandSig53
from torchsig.transforms.target_transforms import (
    DescToMaskClass,
    DescToListTuple,
)
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
import shutil
import pytest
import os


def collate_fn(batch):
    return tuple(zip(*batch))


def generate_static_wideband_dataset(level: int):
    wideband_ds = WidebandModulationsDataset(
        level=level, num_samples=16, target_transform=DescToListTuple(), seed=12345678
    )

    dataset_loader = DatasetLoader(wideband_ds, seed=12345678, collate_fn=collate_fn)
    creator = DatasetCreator(
        wideband_ds,
        seed=12345678,
        path="tests/wideband_sig53_impaired_train/",
        loader=dataset_loader,
    )
    creator.create()


def setup_module(module):
    if os.path.exists("tests/wideband_sig53_impaired_train/"):
        shutil.rmtree("tests/wideband_sig53_impaired_train/")


@pytest.mark.serial
@pytest.mark.parametrize("level", (0, 1, 2), ids=("level_0", "level_1", "level_2"))
def test_generate_wideband_modulation_figures(level: int):
    generate_static_wideband_dataset(level)
    transform = Compose(
        [
            Spectrogram(nperseg=512, noverlap=0, nfft=512, mode="complex"),
            Normalize(norm=np.inf, flatten=True),
        ]
    )

    target_transform = Compose(
        [
            DescToMaskClass(num_classes=53, width=512, height=512),
        ]
    )

    # Instantiate the WidebandSig53 Dataset
    dataset = WidebandSig53(
        root="tests/",
        train=True,
        impaired=True,
        transform=transform,
        target_transform=target_transform,
        use_signal_data=True,
    )

    data_loader = DataLoader(dataset=dataset, shuffle=True)
    visualizer = MaskClassVisualizer(
        data_loader=data_loader,
        visualize_transform=complex_spectrogram_to_magnitude,
        visualize_target_transform=mask_class_to_outline,
        class_list=dataset.modulation_list,
    )

    for figure in iter(visualizer):
        figure.set_size_inches(16, 9)
        plt.savefig("tests/figures/wideband_level_{}.jpg".format(level))
        break
