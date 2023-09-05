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
        conf.Sig53CleanTrainConfig,
        conf.Sig53CleanTrainMediumConfig,
        conf.Sig53CleanTrainSmallConfig,
        conf.Sig53CleanTrainQAConfig,
        conf.Sig53CleanValConfig,
        conf.Sig53CleanValMediumConfig,
        conf.Sig53CleanValSmallConfig,
        conf.Sig53CleanValQAConfig,
        conf.Sig53ImpairedTrainConfig,
        conf.Sig53ImpairedTrainMediumConfig,
        conf.Sig53ImpairedTrainSmallConfig,
        conf.Sig53ImpairedTrainQAConfig,
        conf.Sig53ImpairedValConfig,
        conf.Sig53ImpairedValMediumConfig,
        conf.Sig53ImpairedValSmallConfig,
        conf.Sig53ImpairedValQAConfig,
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
