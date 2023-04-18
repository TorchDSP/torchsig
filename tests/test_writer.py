from torchsig.transforms.target_transforms.target_transforms import DescToClassIndex
from torchsig.utils.writer import DatasetLoader, DatasetCreator, LMDBDatasetWriter
from torchsig.datasets.synthetic import DigitalModulationDataset
from torchsig.transforms.wireless_channel.wce import AddNoise
from unittest import TestCase
import pickle
import shutil
import torch
import lmdb
import os


class SeedModulationDataset(TestCase):
    def setUp(self) -> None:
        if os.path.exists("tests/test1"):
            shutil.rmtree("tests/test1")

        if os.path.exists("tests/test2"):
            shutil.rmtree("tests/test2")

        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("tests/test1"):
            shutil.rmtree("tests/test1")

        if os.path.exists("tests/test2"):
            shutil.rmtree("tests/test2")

        return super().tearDown()

    def test_can_seed_modulation_dataset(self):
        transform = AddNoise(noise_power_db=(5, 10))
        # Create first dataset
        dataset = DigitalModulationDataset(
            num_samples_per_class=1000,
            transform=transform,
            target_transform=DescToClassIndex(["bpsk", "2gfsk"]),
        )
        loader = DatasetLoader(dataset, seed=12345678, num_workers=16)
        writer = LMDBDatasetWriter(path="tests/test1")
        creator = DatasetCreator(loader, writer)
        creator.create()

        # Create second dataset
        dataset = DigitalModulationDataset(
            num_samples_per_class=1000,
            transform=transform,
            target_transform=DescToClassIndex(["bpsk", "2gfsk"]),
        )
        loader = DatasetLoader(dataset, seed=12345678, num_workers=8)
        writer = LMDBDatasetWriter(path="tests/test2")
        creator = DatasetCreator(loader, writer)
        creator.create()

        # See if they're the same
        env1 = lmdb.Environment("tests/test1", map_size=int(1e12), max_dbs=2)
        data_db1 = env1.open_db(b"data")
        env2 = lmdb.Environment("tests/test2", map_size=int(1e12), max_dbs=2)
        data_db2 = env2.open_db(b"data")

        with env1.begin(db=data_db1) as txn1:
            with env2.begin(db=data_db2) as txn2:
                for idx in range(txn1.stat()["entries"]):
                    item1 = pickle.loads(txn1.get(pickle.dumps(idx)))
                    data1: torch.complex128 = item1[0]
                    label1 = item1[1]
                    item2 = pickle.loads(txn2.get(pickle.dumps(idx)))
                    data2: torch.complex128 = item2[0]
                    label2: torch.Tensor = item2[1]

                    real_equal = data1.real.all() == data2.real.all()
                    imag_equal = data1.imag.all() == data2.imag.all()
                    label_equal = label1.all() == label2.all()
                    self.assertTrue(real_equal)
                    self.assertTrue(imag_equal)
                    self.assertTrue(label_equal)
