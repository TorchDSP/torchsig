from torchsig.datasets.sig53 import Sig53
from unittest import TestCase
from torch.utils.data import Subset
from torchsig.datasets.modulations import ModulationsDataset
from torchsig.datasets import conf
import numpy as np
import shutil
import os


class GenerateSig53(TestCase):
    def setUp(self) -> None:
        if os.path.exists("tests/sig53_clean_train"):
            shutil.rmtree("tests/sig53_clean_train")

        if os.path.exists("tests/sig53_clean_val"):
            shutil.rmtree("tests/sig53_clean_val")

        if os.path.exists("tests/sig53_impaired_train"):
            shutil.rmtree("tests/sig53_impaired_train")

        if os.path.exists("tests/sig53_impaired_val"):
            shutil.rmtree("tests/sig53_impaired_val")

        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("tests/test1/sig53_clean_train"):
            shutil.rmtree("tests/test1/sig53_clean_train")

        if os.path.exists("tests/test2/sig53_clean_train"):
            shutil.rmtree("tests/test2/sig53_clean_train")

        if os.path.exists("tests/sig53_clean_val"):
            shutil.rmtree("tests/sig53_clean_val")

        if os.path.exists("tests/sig53_impaired_train"):
            shutil.rmtree("tests/sig53_impaired_train")

        if os.path.exists("tests/sig53_impaired_val"):
            shutil.rmtree("tests/sig53_impaired_val")

        return super().tearDown()

    def test_can_generate_sig53_clean_train(self):
        from torchsig.utils.writer import (
            DatasetLoader,
            LMDBDatasetWriter,
            DatasetCreator,
        )

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
            Subset(ds, np.arange(1000).tolist()),
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
        ds = Sig53(
            root="tests/",
            impaired=False,
            train=False,
            regenerate=True,
            generation_test=True,
        )

        first_data = ds[0][0]
        ds = Sig53(
            root="tests/",
            impaired=False,
            train=False,
            regenerate=True,
            generation_test=True,
        )
        second_data = ds[0][0]

        self.assertEqual(first_data.all(), second_data.all())

    def test_can_generate_sig53_impaired_train(self):
        ds = Sig53(root="tests/", impaired=True, regenerate=True, generation_test=True)
        first_data = ds[0][0]
        ds = Sig53(root="tests/", impaired=True, regenerate=True, generation_test=True)
        second_data = ds[0][0]

        self.assertEqual(first_data.all(), second_data.all())

    def test_can_generate_sig53_impaired_val(self):
        ds = Sig53(
            root="tests/",
            impaired=True,
            train=False,
            regenerate=True,
            generation_test=True,
        )
        first_data = ds[0][0]
        ds = Sig53(
            root="tests/",
            impaired=True,
            train=False,
            regenerate=True,
            generation_test=True,
        )
        second_data = ds[0][0]

        self.assertEqual(first_data.all(), second_data.all())
