from torchsig.datasets.modulations import ModulationsDataset
from torchsig.datasets.sig53 import Sig53
from torchsig.datasets import conf
from torchsig.utils.writer import DatasetCreator
import shutil
import pytest
import os


def setup_module(module):
    os.mkdir("tests/test1")


def teardown_module(module):
    if os.path.exists("tests/test1/"):
        shutil.rmtree("tests/test1/")


@pytest.mark.serial
@pytest.mark.parametrize(
    "config",
    (
        conf.Sig53CleanTrainQAConfig,
        conf.Sig53CleanValQAConfig,
        conf.Sig53ImpairedTrainQAConfig,
        conf.Sig53ImpairedValQAConfig,
    ),
)
def test_can_generate_sig53(config: conf.Sig53Config):
    ds = ModulationsDataset(
        level=config.level,
        num_samples=53 * 10,
        num_iq_samples=config.num_iq_samples,
        use_class_idx=config.use_class_idx,
        include_snr=config.include_snr,
        eb_no=config.eb_no,
    )

    creator = DatasetCreator(
        ds, seed=12345678, path="tests/test1/{}".format(config.name)
    )
    creator.create()

    train = config in (conf.Sig53CleanTrainQAConfig, conf.Sig53ImpairedTrainQAConfig)
    impaired = config in (
        conf.Sig53ImpairedTrainQAConfig,
        conf.Sig53ImpairedValQAConfig,
    )
    Sig53(root="tests/test1", train=train, impaired=impaired)
