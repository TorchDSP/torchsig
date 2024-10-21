"""Digital Signal Processing (DSP) Utils
"""

import scipy
from scipy import signal as sp
import numpy as np

def convolve(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """A modified version of scipy.signal.convolve, which discards trasitional regions

    Args:
        signal (np.ndarray): input signal to be filtered
        taps (np.ndarray): filter with which to colvolve the signal

    """
    filtered = sp.convolve(signal, taps, "full") 
    lidx = (len(filtered) - len(signal)) // 2
    ridx = lidx + len(signal)
    return filtered[lidx:ridx]

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

def polyphase_prototype_filter ( num_branches: int ) -> np.ndarray:
    # design a low-pass filter
    cutoff = 1/(2*num_branches)
    transitionBandwidth = 1/(4*num_branches)
    prototypeFilterPFB = low_pass(cutoff=cutoff, transition_bandwidth=transitionBandwidth)
    # increase gain to account for change in sample rate
    prototypeFilterPFB *= num_branches
    return prototypeFilterPFB

def rational_rate_resampler ( input_signal: np.ndarray, resampler_rate: float ) -> np.ndarray:
    numBranchesPFB = 10000
    resamplerUpRate = numBranchesPFB
    resamplerDownRate = int(np.round(numBranchesPFB/resampler_rate))
    # design the PFB prototype filter
    prototypeFilterPFB = polyphase_prototype_filter ( numBranchesPFB )
    # apply the PFB via upfirdn()
    output = sp.upfirdn(prototypeFilterPFB, input_signal, up=resamplerUpRate, down=resamplerDownRate)
    return output

def estimate_filter_length(
    transition_bandwidth: float, attenuation_db: int = 120, sample_rate: float = 1.0
) -> int:
    # estimate the length of an FIR filter using harris' approximaion,
    # N ~= (sampling rate/transition bandwidth)*(sidelobe attenuation in dB / 22)
    # fred harris, Multirate Signal Processing for Communication Systems,
    # Second Edition, p.59
    filter_length = int(np.round((sample_rate / transition_bandwidth) * (attenuation_db / 22)))

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
    p = np.exp(-2 * np.pi**2 * BT**2 / np.log(2) * (n / float(samples_per_symbol)) ** 2)
    p = p / np.sum(p)
    return p


def calculate_exponential_filter ( M=1, P=1, fc=.25, num_taps=None, K=4. ):
    """
        Function used to generate Single Band FIR filter using either the Remez
        algorithm or exponential filters.

        Calculate the filter-coefficients for the finite impulse response
        (FIR) filter whose transfer function minimizes the maximum error
        between the desired gain and the realized gain in the specified
        frequency bands using the Remez exchange algorithm.

        ==========
        Parameters
        ==========

        (Default values are given in parentheses)

        fc  : float (.25)
            Cutoff frequency in normalized discrete frequency (pi rad/sample)
        num_taps : int (default is determined by remez_ord method)
                    The desired number of taps in the filter. The number of taps is
                    the number of terms in the filter, or the filter order plus one.
        P : int (1)
            Upsampling rate of polyphase implementation.
        M : int (1)
            Downsampling rate of polyphase implementation
        K : float (9.396)
            K parameter for exponential filter design. Higher K results on
            smaller transition bandwidth but at the expense of less stopband
            attenuation.

    """
    fc_atten = 10 * np.log10(.5)
    paths = int(P if (P > M) else np.floor(M))

    hp_point = fc
    # root-raised error function K and MTerm parameters
    offset = .5
    MTerm = np.round(1. / hp_point)

    # for polyphase implementations -- pad n so that if gives the correct
    # number of taps for the number of phases.

    if num_taps % paths:
        num_taps += paths - (num_taps % paths)  

    #b = calc_exp_filter(num_taps, K, MTerm, offset, fc_atten)

    """
        using root raised erf function to generate filter prototype
        less control but much faster option for very large filters.
        Perfectly fine for standard low-pass filters. Link to code
        effectively use twiddle algorithm to get the correct cut-off frequency
        http://www.mathworks.com/matlabcentral/fileexchange/15813-near-perfect-reconstruction-polyphase-filterbank
    """
    F = np.arange(num_taps)
    F = np.double(F) / len(F)

    x = K * (MTerm * F - offset)  #offset parameter allows tuning of the cut-off frequency
    A = np.sqrt(0.5 * scipy.special.erfc(x))

    N = len(A)

    idx = np.arange(N // 2)
    A[N - idx - 1] = np.conj(A[1 + idx])
    A[N // 2] = 0

    # scale using
    db_diff = fc_atten - 10 * np.log10(.5)
    exponent = 10**(-db_diff / 10.)

    A = A**exponent

    b = np.fft.ifft(A)
    # the imaginary components should be tiny -- the error caused is negligible .
    b = (np.fft.fftshift(b)).real

    b /= np.sum(b)

    return b

# signal upper edge cannot exceed this number, calculated as center freq + (bandwidth/2)
#
# these number needs to be slightly less than 0.5 in order to account for transition bandwidth
# such that a filter can be designed with some transition bandwidth (see: function
# antiAliasingFilter())
#
# additionally, 0.48 is close enough to 0.5 such that signals can still press up against the 
# edge of the -fs/2 and +fs/2 boundary to simulate a receiver being offtuned.
# 
# these values need to be treated as constants!
# they cannot be changed on the fly at run-time
MAX_SIGNAL_UPPER_EDGE_FREQ = 0.48
MAX_SIGNAL_LOWER_EDGE_FREQ = -MAX_SIGNAL_UPPER_EDGE_FREQ

