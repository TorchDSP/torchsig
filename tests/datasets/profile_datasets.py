"""Profile Dataset generation and writing"""

# TorchSig
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.datasets import NewTorchSigDataset, StaticTorchSigDataset
from torchsig.utils.writer import DatasetCreator
from torchsig.transforms.transforms import Spectrogram
from torchsig.transforms.target_transforms import ClassName, Start, Stop, LowerFreq, UpperFreq, SNR


# Third Party
from tqdm import tqdm

# Built-In
import cProfile
import pstats
from pathlib import Path
import sys
import datetime

fft_size = 1024
num_iq_samples_dataset = fft_size**2
impairment_level = 2
num_signals_max = 10
num_signals_min = 0

dataset_length = 100

enable_tqdm = True

root = Path.joinpath(Path(__file__).parent, "profile")
batch_size = 2
num_workers = 2

target_transform = [ClassName(), Start(), Stop(), LowerFreq(), UpperFreq(), SNR()]


def dataset_infinite_generation(transforms=[]):

    print(f"\nProfiling infinite dataset for {dataset_length} samples..................")

    print(f"IQ Array Size: {num_iq_samples_dataset}")
    print(f"Impairment Level: {impairment_level}")
    print(f"Num Signals: {num_signals_min} - {num_signals_max}")

    # initialize profiler
    profiler = cProfile.Profile()

    # Ininitialize dataset
    md = DatasetMetadata(
        num_iq_samples_dataset=num_iq_samples_dataset, fft_size=fft_size, impairment_level=impairment_level, num_signals_max=num_signals_max, num_signals_min=num_signals_min, transforms=transforms
    )

    dataset = NewTorchSigDataset(dataset_metadata=md)

    profiler.enable()
    for i in tqdm(range(dataset_length), disable=not enable_tqdm):
        data, targets = dataset[i]
    profiler.disable()
    print("Profile done.")

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumtime")
    stats.print_stats(20)

    total_time = stats.total_tt
    print(f"Average rate: {dataset_length / total_time} samples/sec")


def dataset_finite_writing(transforms=[]):

    print(f"\nProfiling dataset writing finite dataset for {dataset_length} samples..................")

    print(f"IQ Array Size: {num_iq_samples_dataset}")
    print(f"Impairment Level: {impairment_level}")
    print(f"Num Signals: {num_signals_min} - {num_signals_max}")

    # initialize profiler
    profiler = cProfile.Profile()

    # Ininitialize dataset
    md = DatasetMetadata(
        num_iq_samples_dataset=num_iq_samples_dataset,
        fft_size=fft_size,
        impairment_level=impairment_level,
        num_signals_max=num_signals_max,
        num_signals_min=num_signals_min,
        dataset_length=dataset_length,
        transforms=transforms,
    )

    dataset = NewTorchSigDataset(dataset_metadata=md)

    dc = DatasetCreator(dataset=dataset, root=root, overwrite=True, batch_size=batch_size, num_workers=num_workers)

    profiler.enable()

    dc.create()

    profiler.disable()
    print("Profile done.")

    stats = pstats.Stats(profiler)
    stats.strip_dirs()

    stats.sort_stats("cumtime")
    stats.print_stats(20)

    total_time = stats.total_tt
    print(f"Average rate: {dataset_length / total_time} samples/sec")


def dataset_finite_reading(transforms=[]):

    print(f"\nProfiling dataset reading finite dataset for {dataset_length} samples..................")

    print(f"IQ Array Size: {num_iq_samples_dataset}")
    print(f"Impairment Level: {impairment_level}")
    print(f"Num Signals: {num_signals_min} - {num_signals_max}")

    # initialize profiler
    profiler = cProfile.Profile()

    # Ininitialize dataset
    static_dataset = StaticTorchSigDataset(
        root=root,
        impairment_level=impairment_level,
        transforms=Spectrogram(fft_size=fft_size),
        target_transforms=target_transform,
    )

    profiler.enable()

    for i in tqdm(range(dataset_length), disable=not enable_tqdm):
        data, targets = static_dataset[i]

    profiler.disable()
    print("Profile done.")

    stats = pstats.Stats(profiler)
    stats.strip_dirs()

    stats.sort_stats("cumtime")
    stats.print_stats(20)

    total_time = stats.total_tt
    print(f"Average rate: {dataset_length / total_time} samples/sec")


def main():

    print("\nOutput will be saved to profile_datasets.out")

    with open("profile_datasets.out", "w") as terminal_output:
        sys.stdout = terminal_output

        # Timestamp output file
        now = datetime.datetime.now()
        print("Profile Run at: ", now.strftime("%Y-%m-%d %H:%M:%S"))

        dataset_infinite_generation(transforms=Spectrogram(fft_size=fft_size))
        dataset_finite_writing(transforms=Spectrogram(fft_size=fft_size))
        dataset_finite_reading(transforms=Spectrogram(fft_size=fft_size))


if __name__ == "__main__":
    main()
