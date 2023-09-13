from scipy import signal as sp
import numpy as np


def convolve(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    return sp.convolve(signal, taps, "same")


def low_pass(cutoff: float, transition_bandwidth: float) -> np.ndarray:
    """Basic low pass FIR filter design

    Args:
        cutoff (float): From 0.0 to .5
        transition_bandwidth (float): width of the transition region

    """
    num_taps = estimate_filter_length(transition_bandwidth)
    return sp.firwin(
        num_taps,
        cutoff,
        width=transition_bandwidth,
        window=sp.get_window("blackman", num_taps),
        scale=True,
        fs=1,
    )


def estimate_filter_length(
    transition_bandwidth: float, attenuation_db: int = 72, sample_rate: float = 1.0
) -> int:
    # estimate the length of an FIR filter using harris' approximaion,
    # N ~= (sampling rate/transition bandwidth)*(sidelobe attenuation in dB / 22)
    # fred harris, Multirate Signal Processing for Communication Systems,
    # Second Edition, p.59
    filter_length = int(
        np.round((sample_rate / transition_bandwidth) * (attenuation_db / 22))
    )

    # odd-length filters are desirable because they do not introduce a half-sample delay
    if np.mod(filter_length, 2) == 0:
        filter_length += 1

    return filter_length


def rrc_taps(
    iq_samples_per_symbol: int, size_in_symbols: int, alpha: float = 0.35
) -> np.ndarray:
    # this could be made into a transform
    M = size_in_symbols
    Ns = float(iq_samples_per_symbol)
    n = np.arange(-M * Ns, M * Ns + 1)
    taps = np.zeros(int(2 * M * Ns + 1))
    for i in range(int(2 * M * Ns + 1)):
        # handle the discontinuity at t=+-Ns/(4*alpha)
        if n[i] * 4 * alpha == Ns or n[i] * 4 * alpha == -Ns:
            taps[i] = (
                1
                / 2.0
                * (
                    (1 + alpha) * np.sin((1 + alpha) * np.pi / (4.0 * alpha))
                    - (1 - alpha) * np.cos((1 - alpha) * np.pi / (4.0 * alpha))
                    + (4 * alpha) / np.pi * np.sin((1 - alpha) * np.pi / (4.0 * alpha))
                )
            )
        else:
            taps[i] = 4 * alpha / (np.pi * (1 - 16 * alpha**2 * (n[i] / Ns) ** 2))
            taps[i] = taps[i] * (
                np.cos((1 + alpha) * np.pi * n[i] / Ns)
                + np.sinc((1 - alpha) * n[i] / Ns) * (1 - alpha) * np.pi / (4.0 * alpha)
            )
    return taps


def gaussian_taps(samples_per_symbol: int, BT: float = 0.35) -> np.ndarray:
    # pre-modulation Bb*T product which sets the bandwidth of the Gaussian lowpass filter
    M = 4  # duration in symbols
    n = np.arange(-M * samples_per_symbol, M * samples_per_symbol + 1)
    p = np.exp(
        -2 * np.pi**2 * BT**2 / np.log(2) * (n / float(samples_per_symbol)) ** 2
    )
    p = p / np.sum(p)
    return p
