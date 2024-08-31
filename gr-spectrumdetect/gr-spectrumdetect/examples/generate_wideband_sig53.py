from torchsig.transforms.target_transforms import DescToListTuple
from torchsig.utils.writer import DatasetCreator, DatasetLoader
from torchsig.datasets.wideband import WidebandModulationsDataset
from torchsig.datasets import conf
from torchsig.transforms.transforms import *
from torchsig.utils.dataset import collate_fn
from typing import List
import click
import os

import numpy as np


modulation_list = ["ook",
        "bpsk",
        "4pam",
        "4ask",
        "qpsk",
        "8pam",
        "8ask",
        "8psk",
        "16qam",
        "16pam",
        "16ask",
        "16psk",
        "32qam",
        "32qam_cross",
        "32pam",
        "32ask",
        "32psk",
        "64qam",
        "64pam",
        "64ask",
        "64psk",
        "128qam_cross",
        "256qam",
        "512qam_cross",
        "1024qam",
        "2fsk",
        "2gfsk",
        "2msk",
        "2gmsk",
        "4fsk",
        "4gfsk",
        "4msk",
        "4gmsk",
        "8fsk",
        "8gfsk",
        "8msk",
        "8gmsk",
        "16fsk",
        "16gfsk",
        "16msk",
        "16gmsk",
        "ofdm-64",
        "ofdm-72",
        "ofdm-128",
        "ofdm-180",
        "ofdm-256",
        "ofdm-300",
        "ofdm-512",
        "ofdm-600",
        "ofdm-900",
        "ofdm-1024",
        "ofdm-1200",
        "ofdm-2048",
    ]

def generate(root: str, configs: List[conf.WidebandSig53Config], num_workers: int, num_samples_override: int):
    for config in configs:
        num_samples = config.num_samples if num_samples_override <=0 else num_samples_override
        prefetch_factor=4

        wideband_ds = WidebandModulationsDataset(
            level=config.level,
            num_iq_samples=int(1024*1024),
            num_samples=num_samples,
            modulation_list=modulation_list,
            target_transform=DescToListTuple(),
            seed=config.seed,
        )

        dataset_loader = DatasetLoader(wideband_ds, seed=12345678, collate_fn=collate_fn, num_workers=16, batch_size=32, prefetch_factor=prefetch_factor)
        creator = DatasetCreator(wideband_ds, seed=12345678, num_workers=16, path=os.path.join(root, config.name), loader=dataset_loader,)
        creator.create()


@click.command()
@click.option("--root", default="wideband_sig53", help="Path to generate wideband_sig53 datasets")
@click.option("--all", default=False, help="Generate all versions of wideband_sig53 dataset.")
@click.option("--qa", default=False, help="Generate only QA versions of wideband_sig53 dataset.")
@click.option("--num-samples", default=-1, help="Override for number of dataset samples.")
@click.option("--impaired", default=False, help="Generate impaired dataset. Ignored if --all=True (default)",)
@click.option("--num-workers", "num_workers", default=os.cpu_count() // 2, help="Define number of workers for both DatasetLoader and DatasetCreator")
def main(root: str, all: bool, qa: bool, impaired: bool, num_workers: int, num_samples: int):
    if not os.path.isdir(root):
        os.mkdir(root)

    configs = [
        conf.WidebandSig53ImpairedTrainConfig,
        conf.WidebandSig53ImpairedValConfig,
    ]
    print(configs)
    
    impaired_configs = []
    impaired_configs.extend(configs[2:])
    impaired_configs.extend(configs[-2:])
    
    if all:
        generate(root, configs, num_workers, num_samples)
        return
    
    elif qa:
        generate(root, configs[4:], num_workers, num_samples)
        return

    elif impaired:
        generate(root, impaired_configs, num_workers, num_samples)
        return

    else:
        generate(root, configs[:2], num_workers, num_samples)


if __name__ == "__main__":
    main()
