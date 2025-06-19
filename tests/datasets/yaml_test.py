
from torchsig.datasets.datasets import TorchSigIterableDataset, StaticTorchSigDataset
import yaml
import numpy as np
from torchsig.utils.writer import DatasetCreator
import pathlib


def compare(s1, s2):
    d1, t1 = s1
    d2, t2 = s2

    data_matches = np.all(d1 == d2)
    targets_match = t1 == t2

    return data_matches and targets_match

dataset_name = f"yaml_test_dataset"
directory_path = pathlib.Path(__file__).parent.resolve()
yaml_path = str(directory_path.joinpath(f"{dataset_name}.yaml"))
dataset_path = str(directory_path.joinpath(f"{dataset_name}"))

DS = TorchSigIterableDataset(yaml_path)

test_idx = np.random.randint(DS.dataset_metadata.num_samples)

dc = DatasetCreator(
    DS,
    dataset_path,
    overwrite=True
)

dc.create()

SDS = StaticTorchSigDataset(
    root = dataset_path,
    impairment_level = 0
)

SDS2 = StaticTorchSigDataset(
    root = dataset_path,
    impairment_level = 0
)

match = compare(SDS[test_idx], SDS2[test_idx])
if not match:
    print("Does not match.")
    breakpoint()
print("Success.")


    

