"""Offidical Narrowband Dataset generation for command line

Example:
To generate the official clean Narrowband train dataset:
    >>> python generate_official_datasets.py <path to root> --type narrowband --train

To generate the official impaired Wideband train dataset with 1 batch size and 32 workers:
    >>> python generate_official_datasets.py <path to root> --type wideband --train --impaired --batch_size 1 --num_workers 32
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
    parser.add_argument("root", type=str, help="Path to generate Narrowband dataset.")
    parser.add_argument("--type", type=str, default='narrowband', help="Which dataset to generate. Must be narrowband or wideband. Defaults to narrowband.")
    parser.add_argument("--train", action="store_true", help="Generate train dataset (otherwise validation).")
    parser.add_argument("--impaired", action="store_true", help="Generate impaired dataset (otherwise clean).")
    parser.add_argument("--batch_size", type=int, default = 32, help="Batch size. Defaults to 32.")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() // 3, help="Number of workers to generate dataset. Defaults to a third of available CPU cores.")

    # Parse the arguments
    args = parser.parse_args()

    # get dataset metadata
    dataset_metadata = to_dataset_metadata(get_default_yaml_config(
        dataset_type = args.type.lower(),
        impairment_level = args.impaired,
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