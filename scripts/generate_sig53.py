from torchsig.utils.writer import DatasetCreator, DatasetLoader
from torchsig.datasets.modulations import ModulationsDataset
from torchsig.datasets import conf
from typing import List
import click
import os
import numpy as np


def generate(path: str, configs: List[conf.Sig53Config], num_workers: int, num_samples_override: int):
    num_samples = config.num_samples if num_samples == 0 else num_samples
    for config in configs:
        num_samples = config.num_samples if num_samples_override <=0 else num_samples_override
        batch_size = int(np.min((config.num_samples // num_workers, 32)))
        print(f'batch_size -> {batch_size} num_samples -> {num_samples}')
        ds = ModulationsDataset(
            level=config.level,
            num_samples=num_samples,
            num_iq_samples=config.num_iq_samples,
            use_class_idx=config.use_class_idx,
            include_snr=config.include_snr,
            eb_no=config.eb_no,
        )
        loader = DatasetLoader(
            ds,
            seed=12345678,
            num_workers=num_workers,
            batch_size=batch_size,
        )
        creator = DatasetCreator(
            ds,
            seed=12345678,
            path="{}".format(os.path.join(path, config.name)),
            loader=loader,
            num_workers=num_workers,
        )
        creator.create()


@click.command()
@click.option("--root", default="sig53", help="Path to generate sig53 datasets")
@click.option("--all", default=True, help="Generate all versions of sig53 dataset.")
@click.option("--qa", default=False, help="Generate only QA versions of sig53 dataset.")
@click.option("--qa", default=False, help="Generate only QA versions of sig53 dataset.")
@click.option("--num-samples", default=-1, help="Override for number of dataset samples.")
@click.option("--num-workers", "num_workers", default=os.cpu_count() // 2, help="Define number of workers for both DatasetLoader and DatasetCreator")
@click.option("--impaired", default=False, help="Generate impaired dataset. Ignored if --all=True (default)")
def main(root: str, all: bool, qa: bool, impaired: bool, num_workers: int, num_samples):
    if not os.path.isdir(root):
        os.mkdir(root)

    configs = [
        conf.Sig53CleanTrainConfig,
        conf.Sig53CleanValConfig,
        conf.Sig53ImpairedTrainConfig,
        conf.Sig53ImpairedValConfig,
        conf.Sig53CleanTrainQAConfig,
        conf.Sig53CleanValQAConfig,
        conf.Sig53ImpairedTrainQAConfig,
        conf.Sig53ImpairedValQAConfig,
    ]
    if all:
        generate(root, configs[:4], num_workers, num_samples)
        return
    
    elif qa:
        generate(root, configs[4:], num_workers, num_samples)
        return

    elif impaired:
        generate(root, configs[2:4], num_workers, num_samples)
        return

    else:
        generate(root, configs[:2], num_workers, num_samples)


if __name__ == "__main__":
    main()
