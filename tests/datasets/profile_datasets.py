"""Profile Dataset generation and writing
"""

# TorchSig
from torchsig.datasets.dataset_metadata import WidebandMetadata, NarrowbandMetadata
from torchsig.datasets.wideband import NewWideband, StaticWideband
from torchsig.datasets.narrowband import NewNarrowband, StaticNarrowband
from torchsig.utils.writer import DatasetCreator
from torchsig.transforms.dataset_transforms import Spectrogram
from torchsig.transforms.target_transforms import (
    ClassName,
    Start,
    Stop,
    LowerFreq,
    UpperFreq,
    SNR
)


# Third Party
from tqdm import tqdm
import zarr

# Built-In
import cProfile
import pstats
from pathlib import Path
import sys
import datetime

fft_size = 512
num_iq_samples_dataset = fft_size ** 2
impairment_level = 2
num_signals_max = 5
num_signals_min = 5

num_samples = 100

enable_tqdm = True

root = Path.joinpath(Path(__file__).parent,'profile')
batch_size = 1
num_workers = 1

target_transform = [
    ClassName(),
    Start(),
    Stop(),
    LowerFreq(),
    UpperFreq(),
    SNR()
]

def wideband_infinite_generation(transforms = []):

    print(f"\nProfiling wideband infinite dataset for {num_samples} samples..................")

    print(f"IQ Array Size: {num_iq_samples_dataset}")
    print(f"Impairment Level: {impairment_level}")
    print(f"Num Signals: {num_signals_min} - {num_signals_max}")

    # initialize profiler
    profiler = cProfile.Profile()

    # Ininitialize dataset
    md = WidebandMetadata(
        num_iq_samples_dataset=num_iq_samples_dataset,
        fft_size=fft_size,
        impairment_level=impairment_level,
        num_signals_max=num_signals_max,
        num_signals_min=num_signals_min,
        transforms=transforms
    )

    wideband = NewWideband(dataset_metadata=md)
    
    profiler.enable()
    for i in tqdm(range(num_samples), disable = not enable_tqdm):
        data, targets = wideband[i]
    profiler.disable()
    print("Profile done.")

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumtime')
    stats.print_stats(20)

    total_time = stats.total_tt
    print(f"Average rate: {num_samples / total_time} samples/sec")

def narrowband_infinite_generation(transforms = []):

    print(f"Profiling narrowband infinite dataset for {num_samples} samples..................")

    print(f"IQ Array Size: {num_iq_samples_dataset}")
    print(f"Impairment Level: {impairment_level}")

    # initialize profiler
    profiler = cProfile.Profile()

    # Ininitialize dataset
    md = NarrowbandMetadata(
        num_iq_samples_dataset=num_iq_samples_dataset,
        fft_size=fft_size,
        impairment_level=impairment_level,
        transforms=transforms
    )

    narrowband = NewNarrowband(dataset_metadata=md)

    profiler.enable()
    for i in tqdm(range(num_samples), disable = not enable_tqdm):
        data, targets = narrowband[i]
    profiler.disable()
    print("Profile done.")

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumtime')
    stats.print_stats(20)

    total_time = stats.total_tt
    print(f"Average rate: {num_samples / total_time} samples/sec")



def narrowband_finite_writing(transforms = []):

    print(f"\nProfiling narrowband writing finite dataset for {num_samples} samples..................")

    print(f"IQ Array Size: {num_iq_samples_dataset}")
    print(f"Impairment Level: {impairment_level}")

    # initialize profiler
    profiler = cProfile.Profile()

    # Ininitialize dataset
    md = NarrowbandMetadata(
        num_iq_samples_dataset=num_iq_samples_dataset,
        fft_size=fft_size,
        impairment_level=impairment_level,
        num_samples=num_samples,
        transforms = transforms,
    )

    narrowband = NewNarrowband(dataset_metadata=md)

    dc = DatasetCreator(
        dataset = narrowband,
        root = root,
        overwrite=True,
        batch_size=batch_size,
        num_workers=num_workers
    )

    profiler.enable()

    dc.create()

    profiler.disable()
    print("Profile done.")

    stats = pstats.Stats(profiler)
    stats.strip_dirs()

    stats.sort_stats('cumtime')
    stats.print_stats(20)

    total_time = stats.total_tt
    print(f"Average rate: {num_samples / total_time} samples/sec")

def wideband_finite_writing(transforms = []):

    print(f"\nProfiling wideband writing finite dataset for {num_samples} samples..................")

    print(f"IQ Array Size: {num_iq_samples_dataset}")
    print(f"Impairment Level: {impairment_level}")
    print(f"Num Signals: {num_signals_min} - {num_signals_max}")

    # initialize profiler
    profiler = cProfile.Profile()

    # Ininitialize dataset
    md = WidebandMetadata(
        num_iq_samples_dataset=num_iq_samples_dataset,
        fft_size=fft_size,
        impairment_level=impairment_level,
        num_signals_max=num_signals_max,
        num_signals_min=num_signals_min,
        num_samples=num_samples,
        transforms = transforms,
    )

    wideband = NewWideband(dataset_metadata=md)

    dc = DatasetCreator(
        dataset = wideband,
        root = root,
        overwrite=True,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    profiler.enable()

    dc.create()

    profiler.disable()
    print("Profile done.")

    stats = pstats.Stats(profiler)
    stats.strip_dirs()

    stats.sort_stats('cumtime')
    stats.print_stats(20)

    total_time = stats.total_tt
    print(f"Average rate: {num_samples / total_time} samples/sec")


def wideband_finite_reading(transforms = []):

    print(f"\nProfiling wideband reading finite dataset for {num_samples} samples..................")

    print(f"IQ Array Size: {num_iq_samples_dataset}")
    print(f"Impairment Level: {impairment_level}")
    print(f"Num Signals: {num_signals_min} - {num_signals_max}")

    # initialize profiler
    profiler = cProfile.Profile()

    # Ininitialize dataset
    static_wideband = StaticWideband(
        root = root,
        impairment_level = impairment_level,
        transforms = Spectrogram(fft_size = fft_size),
        target_transforms = target_transform,
    )

    profiler.enable()

    for i in tqdm(range(num_samples), disable = not enable_tqdm):
        data, targets = static_wideband[i]

    profiler.disable()
    print("Profile done.")

    stats = pstats.Stats(profiler)
    stats.strip_dirs()

    stats.sort_stats('cumtime')
    stats.print_stats(20)

    total_time = stats.total_tt
    print(f"Average rate: {num_samples / total_time} samples/sec")

def narrowband_finite_reading(transforms = []):

    print(f"\nProfiling narrowband reading finite dataset for {num_samples} samples..................")

    print(f"IQ Array Size: {num_iq_samples_dataset}")
    print(f"Impairment Level: {impairment_level}")

    # initialize profiler
    profiler = cProfile.Profile()

    # Ininitialize dataset
    static_narrowband = StaticNarrowband(
        root = root,
        impairment_level = impairment_level,
        transforms = Spectrogram(fft_size = fft_size),
        target_transforms = target_transform,
    )

    profiler.enable()

    for i in tqdm(range(num_samples), disable = not enable_tqdm):
        data, targets = static_narrowband[i]

    profiler.disable()
    print("Profile done.")

    stats = pstats.Stats(profiler)
    stats.strip_dirs()

    stats.sort_stats('cumtime')
    stats.print_stats(20)

    total_time = stats.total_tt
    print(f"Average rate: {num_samples / total_time} samples/sec")


def main():

    print("\nOutput will be saved to profile_datasets.out")

    with open("profile_datasets.out", 'w') as terminal_output:
        sys.stdout = terminal_output

        # Timestamp output file
        now = datetime.datetime.now()
        print("Profile Run at: ", now.strftime("%Y-%m-%d %H:%M:%S"))

        wideband_infinite_generation(transforms = Spectrogram(fft_size=fft_size))
        wideband_finite_writing(transforms = Spectrogram(fft_size=fft_size))
        wideband_finite_reading(transforms = Spectrogram(fft_size=fft_size))

        narrowband_infinite_generation(transforms = Spectrogram(fft_size=fft_size))
        narrowband_finite_writing(transforms = Spectrogram(fft_size=fft_size))
        narrowband_finite_reading(transforms = Spectrogram(fft_size=fft_size))


if __name__ == "__main__":
    main()
