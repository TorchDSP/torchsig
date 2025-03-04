"""Profile Wideband dataset generation
"""

# TorchSig
from torchsig.datasets.dataset_metadata import WidebandMetadata
from torchsig.datasets.wideband import NewWideband

# Built-In
import cProfile
import pstats

fft_size = 1024
num_iq_samples_dataset = fft_size ** 2
impairment_level = 2
num_signals_max = 5
num_signals_min = 5

num_test = 2

def main():
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
    print(f"Profiling wideband for {num_test} samples...")
    profiler.enable()
    for i in range(num_test):
        data, targets = wideband[i]
    profiler.disable()
    print("Profile done.")

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumtime')
    stats.print_stats(20)


if __name__ == "__main__":
    main()