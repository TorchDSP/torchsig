from torchsig.utils.writer import DatasetCreator, DatasetLoader
from torchsig.datasets.modulations import ModulationsDataset
from torchsig.datasets import conf
from typing import List
import click
import os


def generate(path: str, configs: List[conf.Sig53Config]):
    for config in configs:
        ds = ModulationsDataset(
            level=config.level,
            num_samples=config.num_samples,
            num_iq_samples=config.num_iq_samples,
            use_class_idx=config.use_class_idx,
            include_snr=config.include_snr,
            eb_no=config.eb_no,
        )
        loader = DatasetLoader(
            ds,
            seed=12345678,
            num_workers=os.cpu_count() // 2,
            batch_size=os.cpu_count() // 2,
        )
        creator = DatasetCreator(
            ds,
            seed=12345678,
            path="{}".format(os.path.join(path, config.name)),
            loader=loader,
        )
        creator.create()


@click.command()
@click.option("--root", default="sig53", help="Path to generate sig53 datasets")
@click.option("--all", default=True, help="Generate all versions of sig53 dataset.")
@click.option(
    "--impaired",
    default=False,
    help="Generate impaired dataset. Ignored if --all=True (default)",
)
def main(root: str, all: bool, impaired: bool):
    if not os.path.isdir(root):
        os.mkdir(root)

    configs = [
        conf.Sig53CleanTrainConfig,
        conf.Sig53CleanValConfig,
        conf.Sig53ImpairedTrainConfig,
        conf.Sig53ImpairedValConfig,
    ]
    if all:
        generate(root, configs)
        return

    if impaired:
        generate(root, configs[2:])
        return

    generate(root, configs[:2])


if __name__ == "__main__":
    main()
