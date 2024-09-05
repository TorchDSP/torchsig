from torchsig.transforms.target_transforms import DescToListTuple
from torchsig.utils.writer import DatasetCreator, DatasetLoader
from torchsig.datasets.wideband import WidebandModulationsDataset
from torchsig.datasets import conf
from torchsig.datasets.signal_classes import sig53
from torchsig.transforms.transforms import *
from torchsig.utils.dataset import collate_fn
from typing import List
import click
import os

import numpy as np


modulation_list = sig53.class_list

def generate(root: str, configs: List[conf.WidebandSig53Config], num_workers: int, num_samples_override: int = -1, num_iq_samples_override: int = -1, batch_size: int = 32):
    for config in configs:
        num_samples = config.num_samples if num_samples_override <=0 else num_samples_override
        num_iq_samples = config.num_iq_samples if num_iq_samples_override <= 0 else num_iq_samples_override
        # batch_size = 32 if num_workers > config.num_samples else int(np.min((num_samples // num_workers, 32)))
        prefetch_factor = None if num_workers <= 1 else 4
        print(f'batch_size -> {batch_size} num_samples -> {num_samples}, config -> {config}')

        wideband_ds = WidebandModulationsDataset(
            level=config.level,
            num_iq_samples=num_iq_samples,
            num_samples=num_samples,
            modulation_list=modulation_list,
            # target_transform=DescToListTuple(),
            seed=config.seed,
            overlap_prob=config.overlap_prob
        )

        dataset_loader = DatasetLoader(wideband_ds, seed=12345678, collate_fn=collate_fn, num_workers=num_workers, batch_size=batch_size, prefetch_factor=prefetch_factor)
        creator = DatasetCreator(wideband_ds, seed=12345678, num_workers=num_workers, path=os.path.join(root, config.name), loader=dataset_loader,)
        creator.create()


@click.command()
@click.option("--root", default="wideband_sig53", help="Path to generate wideband_sig53 datasets")
@click.option("--all", is_flag=True, default=False, help="Generate all versions of wideband_sig53 dataset.")
@click.option("--qa", is_flag=True, default=False, help="Generate only QA versions of wideband_sig53 dataset.")
@click.option("--num-iq-samples", "num_iq_samples", default=-1, help="Override number of iq samples in wideband_sig53 dataset.")
@click.option("--batch-size", "batch_size", default=32, help="Override batch size.")
@click.option("--num-samples", default=-1, help="Override for number of dataset samples.")
@click.option("--impaired", is_flag=True, default=False, help="Generate impaired dataset. Ignored if --all (default)",)
@click.option("--num-workers", "num_workers", default=os.cpu_count() // 2, help="Define number of workers for both DatasetLoader and DatasetCreator")
def main(root: str, all: bool, qa: bool, impaired: bool, num_workers: int, num_samples: int, num_iq_samples: int, batch_size: int):
    os.makedirs(root, exist_ok=True)

    configs = [
        conf.WidebandSig53CleanTrainConfig,
        conf.WidebandSig53CleanValConfig,
        conf.WidebandSig53ImpairedTrainConfig,
        conf.WidebandSig53ImpairedValConfig,
        conf.WidebandSig53CleanTrainQAConfig,
        conf.WidebandSig53CleanValQAConfig,
        conf.WidebandSig53ImpairedTrainQAConfig,
        conf.WidebandSig53ImpairedValQAConfig,
    ]

    impaired_configs = []
    impaired_configs.extend(configs[2:4])
    impaired_configs.extend(configs[-2:])
    
    if all:
        generate(root, configs, num_workers, num_samples, num_iq_samples, batch_size)
        return
    
    elif qa:
        generate(root, configs[-4:], num_workers, num_samples, num_iq_samples, batch_size)
        return

    elif impaired:
        generate(root, impaired_configs, num_workers, num_samples, num_iq_samples, batch_size)
        return

    else:
        generate(root, configs[:2], num_workers, num_samples, num_iq_samples, batch_size)


if __name__ == "__main__":
    main()
