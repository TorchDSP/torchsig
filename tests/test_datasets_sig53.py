from torchsig.datasets.sig53 import Sig53
from unittest import TestCase
from torch.utils.data import Subset
from torchsig.datasets.modulations import ModulationsDataset
from torchsig.datasets import conf
from torchsig.utils.writer import (
    DatasetLoader,
    LMDBDatasetWriter,
    DatasetCreator,
)
import numpy as np
import shutil
import os


class GenerateSig53(TestCase):
    @staticmethod
    def clean_folders():
        if os.path.exists("tests/test1/"):
            shutil.rmtree("tests/test1/")

        if os.path.exists("tests/test2/"):
            shutil.rmtree("tests/test2/")

    def setUp(self) -> None:
        GenerateSig53.clean_folders()
        os.mkdir("tests/test1")
        os.mkdir("tests/test2")
        return super().setUp()

    def tearDown(self) -> None:
        GenerateSig53.clean_folders()
        return super().tearDown()

    def test_can_generate_sig53_clean_train(self):
        cfg = conf.Sig53CleanTrainConfig

        ds = ModulationsDataset(
            level=cfg.level,
            num_samples=cfg.num_samples,
            num_iq_samples=cfg.num_iq_samples,
            use_class_idx=cfg.use_class_idx,
            include_snr=cfg.include_snr,
            eb_no=cfg.eb_no,
        )

        loader = DatasetLoader(
            Subset(ds, np.arange(10000).tolist()),
            seed=12345678,
            num_workers=16,
            batch_size=16,
        )
        writer = LMDBDatasetWriter(path="tests/test1/sig53_clean_train")
        creator = DatasetCreator(loader, writer)
        creator.create()

        writer = LMDBDatasetWriter(path="tests/test2/sig53_clean_train")
        creator = DatasetCreator(loader, writer)
        creator.create()

        ds1 = Sig53(root="tests/test1", impaired=False)
        ds2 = Sig53(root="tests/test2", impaired=False)

        for idx in range(len(ds1)):
            self.assertEqual(ds1[idx][0].all(), ds2[idx][0].all())

    def test_can_generate_sig53_clean_val(self):
        cfg = conf.Sig53CleanValConfig

        ds = ModulationsDataset(
            level=cfg.level,
            num_samples=cfg.num_samples,
            num_iq_samples=cfg.num_iq_samples,
            use_class_idx=cfg.use_class_idx,
            include_snr=cfg.include_snr,
            eb_no=cfg.eb_no,
        )

        loader = DatasetLoader(
            Subset(ds, np.arange(10000).tolist()),
            seed=12345678,
            num_workers=16,
            batch_size=16,
        )
        writer = LMDBDatasetWriter(path="tests/test1/sig53_clean_val")
        creator = DatasetCreator(loader, writer)
        creator.create()

        writer = LMDBDatasetWriter(path="tests/test2/sig53_clean_val")
        creator = DatasetCreator(loader, writer)
        creator.create()

        ds1 = Sig53(root="tests/test1", impaired=False)
        ds2 = Sig53(root="tests/test2", impaired=False)

        for idx in range(len(ds1)):
            self.assertEqual(ds1[idx][0].all(), ds2[idx][0].all())

    def test_can_generate_sig53_impaired_train(self):
        cfg = conf.Sig53ImpairedTrainConfig

        ds = ModulationsDataset(
            level=cfg.level,
            num_samples=cfg.num_samples,
            num_iq_samples=cfg.num_iq_samples,
            use_class_idx=cfg.use_class_idx,
            include_snr=cfg.include_snr,
            eb_no=cfg.eb_no,
        )

        loader = DatasetLoader(
            Subset(ds, np.arange(10000).tolist()),
            seed=12345678,
            num_workers=16,
            batch_size=16,
        )
        writer = LMDBDatasetWriter(path="tests/test1/sig53_impaired_train")
        creator = DatasetCreator(loader, writer)
        creator.create()

        writer = LMDBDatasetWriter(path="tests/test2/sig53_impaired_train")
        creator = DatasetCreator(loader, writer)
        creator.create()

        ds1 = Sig53(root="tests/test1", impaired=False)
        ds2 = Sig53(root="tests/test2", impaired=False)

        for idx in range(len(ds1)):
            self.assertEqual(ds1[idx][0].all(), ds2[idx][0].all())

    def test_can_generate_sig53_impaired_val(self):
        cfg = conf.Sig53ImpairedValConfig

        ds = ModulationsDataset(
            level=cfg.level,
            num_samples=cfg.num_samples,
            num_iq_samples=cfg.num_iq_samples,
            use_class_idx=cfg.use_class_idx,
            include_snr=cfg.include_snr,
            eb_no=cfg.eb_no,
        )

        loader = DatasetLoader(
            Subset(ds, np.arange(10000).tolist()),
            seed=12345678,
            num_workers=16,
            batch_size=16,
        )
        writer = LMDBDatasetWriter(path="tests/test1/sig53_impaired_val")
        creator = DatasetCreator(loader, writer)
        creator.create()

        writer = LMDBDatasetWriter(path="tests/test2/sig53_impaired_val")
        creator = DatasetCreator(loader, writer)
        creator.create()

        ds1 = Sig53(root="tests/test1", impaired=False)
        ds2 = Sig53(root="tests/test2", impaired=False)

        for idx in range(len(ds1)):
            self.assertEqual(ds1[idx][0].all(), ds2[idx][0].all())
