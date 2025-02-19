"""Wideband generation code for command line

Example:
To generate Wideband with 10 samples, each sample having 100 IQ samples and 3 signals, and impaired:
    >>> python generate_wideband.py <path to root> --num_signals=3 --num_samples=10 --num_iq_samples=100 --impaired
"""

# TorchSig
from torchsig.datasets.dataset_metadata import WidebandMetadata
from torchsig.utils.generate import generate

# Third Party


# Built-In
import os
import argparse
import math
import subprocess
import sys

def main():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "zarr"])
    from torchsig.utils.generate import generate
    parser = argparse.ArgumentParser()

    parser.add_argument("root", type=str, help="Path to generate Wideband dataset.")
    parser.add_argument("--num_signals", type=int, default=3, help="Number of signals per sample. Defaults to 3.")
    parser.add_argument("--num_samples", type=int, default=100, help= "Dataset size. Defaults to 100")
    parser.add_argument("--num_iq_samples", type=int, default=100, help="Nmber of IQ samples per sample. Defaults to 100")
    parser.add_argument("--fft_size", type=int, default=-1, help="FFT Size. Defaults to sqrt(num_iq_samples)")
    parser.add_argument("--impaired", action="store_true", help="Generate impaired dataset.")
    parser.add_argument("--batch_size", type=int, default = 32, help="Batch size. Defaults to 32.")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() // 3, help="Number of workers to generate dataset. Defaults to a third of available CPU cores.")

    # Parse the arguments
    args = parser.parse_args()

    # create wideband metadata
    dataset_metadata = WidebandMetadata(
        num_signals_max=args.num_signals,
        num_samples = args.num_samples,
        num_iq_samples_dataset=args.num_iq_samples,
        fft_size= int(math.sqrt(args.num_iq_samples)) if args.fft_size == -1 else args.fft_size,
        impairment_level= 2 if args.impaired else 0,
    )

    generate(
        root=args.root,
        dataset_metadata=dataset_metadata,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

if __name__ == "__main__":
    main()

