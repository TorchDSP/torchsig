from torchsig.transforms.target_transforms import DescToListTuple
from torchsig.utils.writer import DatasetCreator, DatasetLoader
from torchsig.datasets.wideband import WidebandModulationsDataset
from torchsig.datasets import conf
from typing import List
import click
import os


def collate_fn(batch):
    return tuple(zip(*batch))


def generate(root: str, configs: List[conf.WidebandSig53Config]):
    for config in configs:
        wideband_ds = WidebandModulationsDataset(
            level=config.level,
            num_iq_samples=config.num_iq_samples,
            num_samples=config.num_samples,
            target_transform=DescToListTuple(),
            seed=config.seed,
        )

        dataset_loader = DatasetLoader(
            wideband_ds, seed=12345678, collate_fn=collate_fn
        )
        creator = DatasetCreator(
            wideband_ds,
            seed=12345678,
            path=os.path.join(root, config.name),
            loader=dataset_loader,
        )
        creator.create()


@click.command()
@click.option(
    "--root", default="wideband_sig53", help="Path to generate wideband_sig53 datasets"
)
@click.option(
    "--all", default=True, help="Generate all versions of wideband_sig53 dataset."
)
@click.option(
    "--qa", default=True, help="Generate only QA versions of wideband_sig53 dataset."
)
@click.option(
    "--impaired",
    default=False,
    help="Generate impaired dataset. Ignored if --all=True (default)",
)
def main(root: str, all: bool, qa: bool, impaired: bool):
    if not os.path.isdir(root):
        os.mkdir(root)

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
    if qa:
        generate(root, configs[4:])
        return

    if all:
        generate(root, configs[:4])
        return

    if impaired:
        generate(root, configs[2:4])
        return

    generate(root, configs[:2])


if __name__ == "__main__":
    main()
