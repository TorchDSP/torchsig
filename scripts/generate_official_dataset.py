"""Official Dataset generation for command line

Example:
To generate the official clean train dataset:
    >>> python generate_official_datasets.py <path to root> --train

To generate the official impaired train dataset with 1 batch size and 32 workers:
    >>> python generate_official_datasets.py <path to root> --train --impaired --batch_size 1 --num_workers 32
"""
# TorchSig
from torchsig.datasets.default_configs.loader import get_default_yaml_config
from torchsig.utils.generate import generate
from torchsig.datasets.dataset_utils import to_dataset_metadata

# Built-In
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="Path to generate dataset.")
    parser.add_argument("--train", action="store_true", help="Generate train dataset (otherwise validation).")
    parser.add_argument("--impairment_level", type=int, default = 2, help="Impairment level. Defaults to 2.")
    parser.add_argument("--batch_size", type=int, default = 32, help="Batch size. Defaults to 32.")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() // 3, help="Number of workers to generate dataset. Defaults to a third of available CPU cores.")

    # Parse the arguments
    args = parser.parse_args()

    # get dataset metadata
    dataset_metadata = to_dataset_metadata(get_default_yaml_config(
        impairment_level = args.impairment_level,
        train = args.train
    ))

    generate(
        root=args.root,
        dataset_metadata=dataset_metadata,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

if __name__ == "__main__":
    main()
