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
            wideband_ds,
            num_workers=8,
            batch_size=8,
            seed=12345678,
            collate_fn=collate_fn,
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
    "--size",
    default="large",
    help="small, medium, or large (default)",
)
@click.option(
    "--type",
    default="all",
    help="clean, impaired, all, or qa",
)
def main(root: str, size: str, type: str):
    if not os.path.isdir(root):
        os.mkdir(root)

    configs = [
        conf.WidebandSig53CleanTrainConfig,
        conf.WidebandSig53CleanTrainMediumConfig,
        conf.WidebandSig53CleanTrainSmallConfig,
        conf.WidebandSig53CleanTrainQAConfig,
        conf.WidebandSig53CleanValConfig,
        conf.WidebandSig53CleanValMediumConfig,
        conf.WidebandSig53CleanValSmallConfig,
        conf.WidebandSig53CleanValQAConfig,
        conf.WidebandSig53ImpairedTrainConfig,
        conf.WidebandSig53ImpairedTrainMediumConfig,
        conf.WidebandSig53ImpairedTrainSmallConfig,
        conf.WidebandSig53ImpairedTrainQAConfig,
        conf.WidebandSig53ImpairedValConfig,
        conf.WidebandSig53ImpairedValMediumConfig,
        conf.WidebandSig53ImpairedValSmallConfig,
        conf.WidebandSig53ImpairedValQAConfig,
    ]
    if type == "qa":
        generate(root, configs[3::4])
        return

    if size == "small":
        if type == "clean" or type == "all":
            generate(root, configs[2:7:4])
        if type == "impaired" or type == "all":
            generate(root, configs[10::4])
        return

    if size == "medium":
        if type == "clean" or type == "all":
            generate(root, configs[1:6:4])
        if type == "impaired" or type == "all":
            generate(root, configs[9::4])

        return

    if type == "clean" or type == "all":
        generate(root, configs[0:5:4])

    if type == "impaired" or type == "all":
        generate(root, configs[8::4])


if __name__ == "__main__":
    main()
