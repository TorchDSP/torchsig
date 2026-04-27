"""Prints out Seedable Dependency Tree

Helps debugging for developers. Outputs tree to `seedable_tree.txt`.
"""

from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.transforms.impairments import Impairments
from torchsig.utils.random import Seedable
from pathlib import Path

num_iq_samples_dataset = 4096 # 64^2
fft_size = 64
num_signals_max = 5
num_signals_min = 1
seed = 10
impairment_level = 2

THIS_DIR = Path(__file__).resolve().parent

def print_seedable_tree(seedable_obj: Seedable, file, prefix="", is_last=True):
    connector = "└── " if is_last else "├── "
    # write to terminal and file
    print(prefix + connector + seedable_obj.__class__.__name__)
    file.write(prefix + connector + seedable_obj.__class__.__name__ + "\n")

    new_prefix = prefix + ("    " if is_last else "│   ")
    child_count = len(seedable_obj.children)
    for i, child in enumerate(seedable_obj.children):
        is_last_child = (i == child_count - 1)
        print_seedable_tree(child, file, new_prefix, is_last_child)

def main():
    md = DatasetMetadata(
        num_iq_samples_dataset = num_iq_samples_dataset,
        fft_size = fft_size,
        num_signals_max = num_signals_max,
        num_signals_min = num_signals_min
    )

    impairments = Impairments(level=impairment_level)

    ds = TorchSigIterableDataset(
        dataset_metadata = md,
        transforms=[impairments.dataset_transforms],
        component_transforms=[impairments.signal_transforms],
        target_labels=["class_index"],
        seed=seed
    )
    print(ds)

    with open(THIS_DIR / "seedable_tree.txt", "w", encoding="utf-8") as f:
        print_seedable_tree(ds, f)

    print(f"Tree output written: {THIS_DIR / "seedable_tree.txt"}")

if __name__ == "__main__":
    main()