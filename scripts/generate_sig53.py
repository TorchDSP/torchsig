from torchsig.utils.writer import DatasetCreator
from torchsig.datasets.modulations import ModulationsDataset
from torchsig.datasets import conf
import click
import os


def generate(path: str, config: conf.Sig53Config):
    ds = ModulationsDataset(
        level=config.level,
        num_samples=config.num_samples,
        num_iq_samples=config.num_iq_samples,
        use_class_idx=config.use_class_idx,
        include_snr=config.include_snr,
        eb_no=config.eb_no,
    )
    creator = DatasetCreator(
        ds, seed=12345678, path="{}".format(os.path.join(path, config.name))
    )
    creator.create()


@click.command()
@click.option("--path", default="sig53", help="Path to generate sig53 datasets")
@click.option("--all", default=True, help="Generate all versions of sig53 dataset.")
@click.option(
    "--impaired",
    default=False,
    help="Generate impaired dataset. Ignored if --all=True (default)",
)
def main(path: str, all: bool, impaired: bool):
    if not os.path.isdir(path):
        os.mkdir(path)

    configs = [
        conf.Sig53CleanTrainConfig,
        conf.Sig53CleanValConfig,
        conf.Sig53ImpairedTrainConfig,
        conf.Sig53ImpairedValConfig,
    ]
    if all:
        for config in configs:
            generate(path, config)
        return

    if impaired:
        for config in configs[2:]:
            generate(path, config)
        return

    for config in configs[:2]:
        generate(path, config)
    return


if __name__ == "__main__":
    main()
