from torchsig.utils.writer import DatasetCreator, DatasetLoader
from torchsig.datasets.modulations import ModulationsDataset
from torchsig.datasets import conf
from torchsig.utils.dataset import collate_fn
from typing import List
import click
import os
import numpy as np


def generate(path: str, configs: List[conf.Sig53Config], num_workers: int, num_samples_override: int, num_iq_samples_override: int = -1, batch_size: int = 32):
    for config in configs:
        num_samples = config.num_samples if num_samples_override <=0 else num_samples_override
        num_iq_samples = config.num_iq_samples if num_iq_samples_override <= 0 else num_iq_samples_override
        # batch_size = 32 if num_workers > config.num_samples else int(np.min((config.num_samples // num_workers, 32)))
        print(f'batch_size -> {batch_size} num_samples -> {num_samples}, config -> {config}')
        ds = ModulationsDataset(
            level=config.level,
            num_samples=num_samples,
            num_iq_samples=config.num_iq_samples,
            use_class_idx=config.use_class_idx,
            include_snr=config.include_snr,
            eb_no=config.eb_no,
        )
        dataset_loader = DatasetLoader(ds, seed=12345678, collate_fn=collate_fn, num_workers=num_workers, batch_size=batch_size)
        creator = DatasetCreator(ds, seed=12345678, path="{}".format(os.path.join(path, config.name)), loader=dataset_loader, num_workers=num_workers)
        creator.create()


@click.command()
@click.option("--root", default="sig53", help="Path to generate sig53 datasets")
@click.option("--all", is_flag=True, default=False, help="Generate all versions of sig53 dataset.")
@click.option("--qa", is_flag=True, default=False, help="Generate only QA versions of sig53 dataset.")
@click.option("--num-iq-samples", "num_iq_samples", default=-1, help="Override number of iq samples in wideband_sig53 dataset.")
@click.option("--batch-size", "batch_size", default=32, help="Override batch size.")
@click.option("--num-samples", default=-1, help="Override for number of dataset samples.")
@click.option("--num-workers", "num_workers", default=os.cpu_count() // 2, help="Define number of workers for both DatasetLoader and DatasetCreator")
@click.option("--impaired", is_flag=True, default=False, help="Generate impaired dataset. Ignored if --all=True (default)")
def main(root: str, all: bool, qa: bool, impaired: bool, num_workers: int, num_samples: int, num_iq_samples: int, batch_size: int):
    os.makedirs(root, exist_ok=True)

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

    impaired_configs = []
    impaired_configs.extend(configs[2:])
    impaired_configs.extend(configs[-2:])
    if all:
        generate(root, configs[:4], num_workers, num_samples, num_iq_samples, batch_size)
        return
    
    elif qa:
        generate(root, configs[4:], num_workers, num_samples, num_iq_samples, batch_size)
        return

    elif impaired:
        generate(root, impaired_configs, num_workers, num_samples, num_iq_samples, batch_size)
        return

    else:
        generate(root, configs[:2], num_workers, num_samples, num_iq_samples, batch_size)


if __name__ == "__main__":
    main()
