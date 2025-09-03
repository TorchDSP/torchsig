"""Profile Rust Functions
"""
from torchsig.utils.rust_functions import upfirdn as upfirdn_RUST

import numpy as np
from scipy.signal import firwin
from scipy.signal import upfirdn as upfirdn_SCIPY

import cProfile
import pstats
import os
import sys
import datetime

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def profile_upfirdn():
    print(f"\nProfiling upfirdn (resampler)...........")

    # initialize profiler
    profiler = cProfile.Profile()

    up_rate = 10000
    down_rate = 9700
    input_len = 5310

    print(f"up = {up_rate}, down = {down_rate}")
    print(f"num input samples = {input_len}")

    # build input signal
    cfo = 1/input_len
    in_array = np.array(np.exp(1j*2*np.pi*cfo*np.arange(input_len)),dtype=np.complex64)

    # number of iterations to test speed
    num_iter = 100

    # filter design params
    num_taps = 10*up_rate
    # num_taps = 197000 # ** test case breaks resampler **
    fs = 1
    cutoff = 0.5/up_rate
    weights = firwin(num_taps,cutoff,fs=fs)
    weights = weights.astype(np.float32)

    # profile resampler
    profiler.enable()
    for this_iter in range(num_iter):
        out_array_rust = upfirdn_RUST(weights,in_array,up_rate,down_rate)
    profiler.disable()
    print("Profile done.")

    stats = pstats.Stats(profiler)
    stats.strip_dirs()

    stats.sort_stats('cumtime')
    stats.print_stats(20)

    total_time_rust = stats.total_tt
    

    # profile scipy's resampler
    profiler_2 = cProfile.Profile()
    profiler_2.enable()
    for this_iter in range(num_iter):
        out_array_scipy = upfirdn_SCIPY(weights,in_array,up_rate,down_rate)
    profiler_2.disable()
    print("Profile done.")

    stats = pstats.Stats(profiler_2)
    stats.strip_dirs()

    stats.sort_stats('cumtime')
    stats.print_stats(20)

    total_time_scipy = stats.total_tt

    print(f"Rust Resampler Timer: {total_time_rust}s")
    print(f"Scipy Resampler Timer: {total_time_scipy}s")

    print(f"\nRust resampler is {total_time_scipy/total_time_rust}x faster than scipy.")

def main():
    print(f"\nOutput will be saved to {THIS_DIR}/profile_rust.out")

    with open(f"{THIS_DIR}/profile_rust.out", 'w') as terminal_output:
        sys.stdout = terminal_output

        # Timestamp output file
        now = datetime.datetime.now()
        print("Profile Run at: ", now.strftime("%Y-%m-%d %H:%M:%S"))

        profile_upfirdn()


if __name__ == "__main__":
    main()