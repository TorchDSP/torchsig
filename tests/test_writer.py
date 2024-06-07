from torchsig.transforms.target_transforms import DescToClassIndex
from torchsig.utils.writer import DatasetCreator
from torchsig.datasets.synthetic import DigitalModulationDataset
from torchsig.transforms import AddNoise
import pickle
import shutil
import torch
import lmdb
import os


def setup_module(module) -> None:
    os.mkdir("tests/test1_writer")
    os.mkdir("tests/test2_writer")


def teardown_module(module) -> None:
    if os.path.exists("tests/test1_writer"):
        shutil.rmtree("tests/test1_writer")

    if os.path.exists("tests/test2_writer"):
        shutil.rmtree("tests/test2_writer")


def test_can_seed_modulation_dataset():
    transform = AddNoise(noise_power_db=(5, 10))
    # Create first dataset
    dataset = DigitalModulationDataset(
        num_samples_per_class=1060,
        num_iq_samples=512,
        transform=transform,
        target_transform=DescToClassIndex(["bpsk", "2gfsk"]),
    )
    creator = DatasetCreator(dataset, seed=12345678, path="tests/test1_writer")
    creator.create()

    creator = DatasetCreator(dataset, seed=12345678, path="tests/test2_writer")
    creator.create()

    # See if they're the same
    env1 = lmdb.Environment("tests/test1_writer", map_size=int(1e12), max_dbs=2)
    data_db1 = env1.open_db(b"data")
    env2 = lmdb.Environment("tests/test2_writer", map_size=int(1e12), max_dbs=2)
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
                assert real_equal
                assert imag_equal
                assert label_equal
