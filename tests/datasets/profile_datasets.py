"""Profile Dataset generation and writing
"""

# TorchSig
from torchsig.datasets.dataset_metadata import WidebandMetadata, NarrowbandMetadata
from torchsig.datasets.wideband import NewWideband
from torchsig.datasets.narrowband import NewNarrowband
from torchsig.utils.writer import DatasetCreator
from torchsig.transforms.dataset_transforms import Spectrogram

# Third Party
from tqdm import tqdm
import zarr

# Built-In
import cProfile
import pstats
from pathlib import Path

fft_size = 512
num_iq_samples_dataset = fft_size ** 2
impairment_level = 2
num_signals_max = 5
num_signals_min = 5

num_test = 100
num_test_write = 2000

enable_tqdm = True

root = Path.joinpath(Path(__file__).parent,'profile')
batch_size = 1
num_workers = 2

def wideband_generation():
    # Ininitialize dataset
    md = WidebandMetadata(
        num_iq_samples_dataset=num_iq_samples_dataset,
        fft_size=fft_size,
        impairment_level=impairment_level,
        num_signals_max=num_signals_max,
        num_signals_min=num_signals_min,
    )

    wideband = NewWideband(dataset_metadata=md)
    
    # initialize profiler
    profiler = cProfile.Profile()

    # profile sample generation
    print(f"IQ Array Size: {num_iq_samples_dataset}")
    print(f"Impairment Level: {impairment_level}")
    print(f"Num Signals: {num_signals_min} - {num_signals_max}")
    print(f"Profiling wideband for {num_test} samples...")
    profiler.enable()
    for i in tqdm(range(num_test), disable = not enable_tqdm):
        data, targets = wideband[i]
    profiler.disable()
    print("Profile done.")

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumtime')
    stats.print_stats(20)

def narrowband_generation():
    # Ininitialize dataset
    md = NarrowbandMetadata(
        num_iq_samples_dataset=num_iq_samples_dataset,
        fft_size=fft_size,
        impairment_level=impairment_level,
    )

    narrowband = NewNarrowband(dataset_metadata=md)
    
    # initialize profiler
    profiler = cProfile.Profile()

    # profile sample generation
    print(f"IQ Array Size: {num_iq_samples_dataset}")
    print(f"Impairment Level: {impairment_level}")
    print(f"Num Signals: {num_signals_min} - {num_signals_max}")
    print(f"Profiling narrowband for {num_test} samples...")
    profiler.enable()
    for i in tqdm(range(num_test), disable = not enable_tqdm):
        data, targets = narrowband[i]
    profiler.disable()
    print("Profile done.")

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumtime')
    stats.print_stats(20)


# Profiling dataset writing to disk

def narrowband_writing(transforms = []):

    # Ininitialize dataset
    md = NarrowbandMetadata(
        num_iq_samples_dataset=num_iq_samples_dataset,
        fft_size=fft_size,
        impairment_level=impairment_level,
        num_samples=num_test_write,
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
    
    # initialize profiler
    profiler = cProfile.Profile()

    # profile sample generation
    print(f"IQ Array Size: {num_iq_samples_dataset}")
    print(f"Impairment Level: {impairment_level}")
    print(f"Num Signals: {num_signals_min} - {num_signals_max}")
    print(f"Profiling narrowband for {num_test_write} samples...")
    profiler.enable()

    dc.create()

    profiler.disable()
    print("Profile done.")

    stats = pstats.Stats(profiler)
    stats.strip_dirs()

    stats.sort_stats('cumtime')
    stats.print_stats(20)

    zarr_arr = zarr.open(f"profile/torchsig_narrowband_impaired/data.zarr", mode = 'r')
    # print(zarr_arr.info_complete())

def main():
    wideband_generation()
    narrowband_generation()
    narrowband_writing(transforms = Spectrogram(fft_size=fft_size))


if __name__ == "__main__":
    main()
